import dataclasses
import pprint
from functools import partial
import re
import os
from threading import Lock


from tqdm import tqdm, trange
import numpy as np
import wandb
from flask import Flask, request

import absl.app
import absl.flags
import absl.logging
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.experimental import PartitionSpec as PS
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
import optax


from ..data import PretrainDataset
from ..jax_utils import (
    JaxRNG, ShardingHelper, get_jax_mp_mesh, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, named_tree_map, global_norm,
    optax_add_scheduled_weight_decay
)
from ..utils import (
    WandBLogger, define_flags_with_default, get_user_flags, set_random_seed,
    load_pickle, load_checkpoint
)
from ..models.gptj import (
    GPTJConfig, FlaxGPTJForCausalLMModule, FlaxGPTJForCausalLM
)
from ..serving import LMServer


FLAGS_DEF = define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mp_mesh_dim=1,
    dtype='',
    input_length=256,
    seq_length=1024,
    top_k=50,
    top_p=1.0,
    do_sample=False,
    num_beams=1,
    load_hf_pretrained='',
    load_checkpoint='',
    tokenizer=GPTJConfig.get_tokenizer_config(),
    lm_server=LMServer.get_default_config(),
)
FLAGS = absl.flags.FLAGS


def main(argv):
    FLAGS = absl.flags.FLAGS
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()
    set_random_seed(FLAGS.seed)

    tokenizer = GPTJConfig.get_tokenizer(FLAGS.tokenizer)

    with jax.default_device(jax.devices("cpu")[0]):
        if FLAGS.load_hf_pretrained != '':
            gptj_config = GPTJConfig.from_pretrained(FLAGS.load_hf_pretrained)
            params = gptj_config.load_pretrained(FLAGS.load_hf_pretrained)
        elif FLAGS.load_checkpoint != '':
            checkpoint, metadata = load_checkpoint(FLAGS.load_checkpoint, target=None)
            gptj_config = metadata['gptj_config']
            params = flax.core.frozen_dict.freeze(checkpoint['params'])
        else:
            raise ValueError('Params must be loaded from checkpoint or huggingface!')

        hf_model = FlaxGPTJForCausalLM(
            gptj_config,
            input_shape=(1, FLAGS.seq_length),
            seed=FLAGS.seed,
            _do_init=False
        )

        if FLAGS.dtype == 'fp32':
            params = hf_model.to_fp32(params)
        elif FLAGS.dtype == 'fp16':
            params = hf_model.to_fp16(params)
        elif FLAGS.dtype == 'bf16':
            params = hf_model.to_bf16(params)
        elif FLAGS.dtype == '':
            pass  # No dtype conversion
        else:
            raise ValueError(f'Unsupported dtype: {FLAGS.dtype}')

    model_ps = match_partition_rules(
        GPTJConfig.get_partition_rules(), params
    )
    sharding_helper = ShardingHelper(model_ps)

    @partial(
        pjit,
        in_axis_resources=(model_ps, PS(), PS()),
        out_axis_resources=(PS(), PS(), PS())
    )
    def forward_loglikelihood(params, rng, batch):
        batch = with_sharding_constraint(batch, PS('dp'))
        rng_generator = JaxRNG(rng)
        input_tokens = batch['input_tokens']
        output_tokens = batch['output_tokens']
        input_mask = batch['input_mask']
        output_mask = batch['output_mask']

        logits = hf_model.module.apply(
            params, input_tokens, attention_mask=input_mask,
            deterministic=True, rngs=rng_generator(gptj_config.rng_keys()),
        ).logits
        if gptj_config.n_real_tokens is not None:
          logits = logits.at[:, :, gptj_config.n_real_tokens:].set(-1e8)
        loglikelihood = jax.nn.log_softmax(logits, axis=-1)
        indices = jnp.expand_dims(output_tokens, axis=-1)
        loglikelihood = jnp.take_along_axis(loglikelihood, indices, axis=-1)
        loglikelihood = jnp.squeeze(loglikelihood, axis=-1)
        loglikelihood = jnp.sum(loglikelihood * output_mask, axis=-1)
        match_count = jnp.sum(
            (jnp.argmax(logits, axis=-1) == output_tokens) * output_mask,
            axis=-1
        )
        total = jnp.sum(output_mask, axis=-1)
        is_greedy = match_count == total
        return loglikelihood, is_greedy, rng_generator()


    @partial(
        pjit,
        in_axis_resources=(model_ps, PS(), PS(), PS()),
        out_axis_resources=(PS(), PS())
    )
    def forward_generate(params, rng, temperature, batch):
        batch = with_sharding_constraint(batch, PS('dp'))
        rng_generator = JaxRNG(rng)
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            params=params['params'],
            prng_key=rng_generator(),
            max_length=FLAGS.seq_length,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_k=FLAGS.top_k,
            top_p=FLAGS.top_p,
            num_beams=FLAGS.num_beams,
            do_sample=FLAGS.do_sample,
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output, rng_generator()

    @partial(
        pjit,
        in_axis_resources=(model_ps, PS(), PS()),
        out_axis_resources=(PS(), PS())
    )
    def forward_greedy_generate(params, rng, batch):
        batch = with_sharding_constraint(batch, PS('dp'))
        rng_generator = JaxRNG(rng)
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            params=params['params'],
            prng_key=rng_generator(),
            max_length=FLAGS.seq_length,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,
            do_sample=False,
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output, rng_generator()

    mesh = get_jax_mp_mesh(FLAGS.mp_mesh_dim)
    with mesh:
        params = sharding_helper.put(params)
        sharded_rng = next_rng()

    class GPTJServer(LMServer):

        @staticmethod
        def loglikelihood(prefix_text, text):
            nonlocal sharded_rng
            prefix = tokenizer(
                prefix_text,
                padding='max_length',
                truncation=True,
                max_length=FLAGS.input_length,
                return_tensors='np',
            )
            inputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=FLAGS.seq_length - FLAGS.input_length,
                return_tensors='np',
            )
            output_tokens = np.concatenate([prefix.input_ids, inputs.input_ids], axis=1)
            bos_tokens = np.full(
                (output_tokens.shape[0], 1), tokenizer.bos_token_id, dtype=np.int32
            )
            input_tokens = np.concatenate([bos_tokens, output_tokens[:, :-1]], axis=-1)
            input_mask = np.concatenate(
                [prefix.attention_mask, inputs.attention_mask], axis=1
            )
            output_mask = np.concatenate(
                [np.zeros_like(prefix.attention_mask), inputs.attention_mask], axis=1
            )
            loglikelihood_mask = np.concatenate(
                [np.zeros_like(prefix.attention_mask), np.ones_like(inputs.attention_mask)],
                axis=1
            )
            batch = dict(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_mask=input_mask,
                output_mask=output_mask,
            )
            with mesh:
                loglikelihood, is_greedy, sharded_rng = forward_loglikelihood(
                    params, sharded_rng, batch
                )
                loglikelihood, is_greedy = jax.device_get((loglikelihood, is_greedy))
            return loglikelihood, is_greedy

        @staticmethod
        def loglikelihood_rolling(text):
            nonlocal sharded_rng
            inputs = tokenizer(
                text,
                padding='longest',
                truncation=False,
                max_length=np.iinfo(np.int32).max,
                return_tensors='np',
            )
            batch_size = inputs.input_ids.shape[0]
            output_tokens = inputs.input_ids
            attention_mask = inputs.attention_mask

            if output_tokens.shape[1] < FLAGS.seq_length:
                padding_length = FLAGS.seq_length - output_tokens.shape[1]
                pad_tokens = np.full(
                    (batch_size, padding_length), tokenizer.pad_token_id, dtype=np.int32
                )
                output_tokens = np.concatenate([output_tokens, pad_tokens], axis=-1)
                pad_mask = np.zeros(
                    (batch_size, padding_length), dtype=inputs.attention_mask.dtype
                )
                attention_mask = np.concatenate([attention_mask, pad_mask], axis=-1)

            bos_tokens = np.full(
                (batch_size, 1), tokenizer.bos_token_id, dtype=np.int32
            )
            input_tokens = np.concatenate([bos_tokens, output_tokens[:, :-1]], axis=-1)
            bos_mask = np.ones((batch_size, 1), dtype=inputs.attention_mask.dtype)
            total_seq_length = output_tokens.shape[1]

            total_loglikelihood = 0.0
            total_is_greedy = True
            # Sliding window
            for i in range(0, total_seq_length, FLAGS.seq_length):
                # Last window
                if i + FLAGS.seq_length > total_seq_length:
                    last_output_mask = np.copy(attention_mask[:, -FLAGS.seq_length:])
                    last_output_mask[:, :i - total_seq_length] = 0.0

                    batch = dict(
                        input_tokens=input_tokens[:, -FLAGS.seq_length:],
                        output_tokens=output_tokens[:, -FLAGS.seq_length:],
                        input_mask=attention_mask[:, -FLAGS.seq_length:],
                        output_mask=last_output_mask,
                    )

                # Normal window
                else:
                    batch = dict(
                        input_tokens=input_tokens[:, i:i + FLAGS.seq_length],
                        output_tokens=output_tokens[:, i:i + FLAGS.seq_length],
                        input_mask=attention_mask[:, i:i + FLAGS.seq_length],
                        output_mask=attention_mask[:, i:i + FLAGS.seq_length],
                    )

                with mesh:
                    loglikelihood, is_greedy, sharded_rng = forward_loglikelihood(
                        params, sharded_rng, batch
                    )
                    loglikelihood, is_greedy = jax.device_get((loglikelihood, is_greedy))

                total_loglikelihood += loglikelihood
                total_is_greedy = np.logical_and(is_greedy, total_is_greedy)

            return total_loglikelihood, total_is_greedy

        @staticmethod
        def generate(text, temperature):
            nonlocal sharded_rng
            inputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=FLAGS.input_length,
                return_tensors='np',
            )
            batch = dict(
                input_tokens=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            with mesh:
                output, sharded_rng = forward_generate(
                    params, sharded_rng, temperature, batch
                )
                output = jax.device_get(output)
            output_text = list(tokenizer.batch_decode(output))
            return output_text

        @staticmethod
        def greedy_until(prefix_text, until, max_length):
            nonlocal sharded_rng
            all_outputs = []
            for pf, ut in zip(prefix_text, until):
                total_length = 0
                total_generated = ''

                while total_length < max_length:
                    pf_tokens = tokenizer(
                        pf,
                        padding=False,
                        truncation=False,
                        max_length=np.iinfo(np.int32).max,
                        return_tensors='np',
                    )
                    input_tokens = pf_tokens.input_ids
                    attention_mask = pf_tokens.attention_mask

                    if input_tokens.shape[1] < FLAGS.input_length:
                        extra = FLAGS.input_length - input_tokens.shape[1]
                        pad_tokens = np.full(
                            (1, extra), tokenizer.pad_token_id, dtype=np.int32
                        )
                        input_tokens = np.concatenate(
                            [pad_tokens, input_tokens], axis=1
                        )
                        pad_attention = np.zeros((1, extra), dtype=attention_mask.dtype)
                        attention_mask = np.concatenate(
                            [pad_attention, attention_mask], axis=1
                        )
                    elif input_tokens.shape[1] > FLAGS.input_length:
                        input_tokens = input_tokens[:, -FLAGS.input_length:]
                        attention_mask = attention_mask[:, -FLAGS.input_length:]

                    batch = dict(input_tokens=input_tokens, attention_mask=attention_mask)

                    with mesh:
                        output, sharded_rng = forward_greedy_generate(
                            params, sharded_rng, batch
                        )
                        output = jax.device_get(output)

                    total_length += output.shape[1]
                    output_text = tokenizer.batch_decode(output)[0]
                    total_generated = total_generated + output_text
                    pf = pf + output_text

                    if ut in total_generated:
                        break

                all_outputs.append(total_generated)

            return all_outputs


    server = GPTJServer(FLAGS.lm_server)
    server.run()


if __name__ == "__main__":
    absl.app.run(main)
