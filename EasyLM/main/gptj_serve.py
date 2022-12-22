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
from flax.training.checkpoints import restore_checkpoint
import optax


from ..data import PretrainDataset
from ..jax_utils import (
    JaxRNG, ShardingHelper, get_jax_mp_mesh, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, named_tree_map, global_norm,
    optax_add_scheduled_weight_decay
)
from ..utils import (
    WandBLogger, define_flags_with_default, get_user_flags, set_random_seed,
    load_pickle
)
from ..models.gptj import (
    GPTJConfig, FlaxGPTJForCausalLMModule, FlaxGPTJForCausalLM
)


FLAGS_DEF = define_flags_with_default(
    seed=42,
    host='0.0.0.0',
    port=5007,
    initialize_jax_distributed=False,
    mp_mesh_dim=1,
    dtype='',
    input_length=128,
    seq_length=512,
    top_k=50,
    top_p=1.0,
    do_sample=True,
    num_beams=10,
    load_hf_pretrained='',
    load_checkpoint='',
    load_config='',
    tokenizer=GPTJConfig.get_tokenizer_config(),
)
FLAGS = absl.flags.FLAGS


def main(argv):
    FLAGS = absl.flags.FLAGS
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()
    set_random_seed(FLAGS.seed)

    if FLAGS.load_checkpoint != '' and FLAGS.load_config == '':
        FLAGS.load_config = os.path.join(
            os.path.dirname(os.path.dirname(FLAGS.load_checkpoint)),
            'metadata.pkl'
        )

    tokenizer = GPTJConfig.get_tokenizer(FLAGS.tokenizer)

    with jax.default_device(jax.devices("cpu")[0]):
        if FLAGS.load_hf_pretrained != '':
            gptj_config = GPTJConfig.from_pretrained(FLAGS.load_hf_pretrained)
            params = gptj_config.load_pretrained(FLAGS.load_hf_pretrained)
        elif FLAGS.load_checkpoint != '':
            metadata = load_pickle(FLAGS.load_config)
            gptj_config = metadata['gptj_config']
            params = flax.core.frozen_dict.freeze(
                restore_checkpoint(FLAGS.load_checkpoint, target=None)['params']
            )
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
        in_axis_resources=(model_ps, PS(), PS(), PS(), PS()),
        out_axis_resources=(PS(), PS())
    )
    def generate_fn(params, rng, input_tokens, attention_mask, hparams):
        rng_generator = JaxRNG(rng)
        input_tokens = with_sharding_constraint(input_tokens, PS('dp'))
        attention_mask = with_sharding_constraint(attention_mask, PS('dp'))
        output = hf_model.generate(
            input_tokens,
            attention_mask=attention_mask,
            params=params['params'],
            prng_key=rng_generator(),
            max_length=FLAGS.seq_length,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=hparams['temperature'],
            top_k=FLAGS.top_k,
            top_p=FLAGS.top_p,
            num_beams=FLAGS.num_beams,
            do_sample=FLAGS.do_sample,
        ).sequences[:, input_tokens.shape[1]:]
        return output, rng_generator()

    @partial(
        pjit,
        in_axis_resources=(model_ps, PS(), PS(), PS()),
        out_axis_resources=(PS(), PS())
    )
    def log_likelihood_fn(params, rng, tokens, attention_mask):
        rng_generator = JaxRNG(rng)
        tokens = with_sharding_constraint(tokens, PS('dp'))
        attention_mask = with_sharding_constraint(attention_mask, PS('dp'))
        bos_tokens = jnp.full(
            (tokens.shape[0], 1), gptj_config.bos_token_id, dtype=jnp.int32
        )
        bos_attention_mask = jnp.full((tokens.shape[0], 1), 1, dtype=jnp.int32)
        input_tokens = jnp.concatenate([bos_tokens, tokens[:, :-1]], axis=1)
        input_attention_mask = jnp.concatenate(
            [bos_attention_mask, attention_mask[:, :-1]], axis=1
        )
        logits = hf_model.module.apply(
            params, input_tokens, attention_mask=input_attention_mask,
            deterministic=False, rngs=rng_generator(gptj_config.rng_keys()),
        ).logits
        log_likelihood = jax.nn.log_softmax(logits, axis=-1)
        indices = jnp.expand_dims(tokens, axis=-1)
        log_likelihood = jnp.take_along_axis(log_likelihood, indices, axis=-1)
        log_likelihood = jnp.squeeze(log_likelihood, axis=-1)
        log_likelihood = jnp.sum(log_likelihood * attention_mask, axis=-1)
        return log_likelihood, rng_generator()

    mesh = get_jax_mp_mesh(FLAGS.mp_mesh_dim)
    lock = Lock()
    app = Flask(__name__)

    @app.post('/generate')
    def generate():
        with lock:
            nonlocal sharded_rng
            data = request.get_json()
            absl.logging.info(
                '\n========= Serving Request ========= \n'
                + pprint.pformat(data) + '\n'
            )

            input_text = data['input_text']
            hparams = {'temperature': data.get('temperature', 1.0)}

            inputs = tokenizer(
                input_text,
                padding='max_length',
                truncation=True,
                max_length=FLAGS.input_length,
                return_tensors='np',
            )
            with mesh:
                output, sharded_rng = generate_fn(
                    params, sharded_rng, inputs.input_ids,
                    inputs.attention_mask, hparams
                )
                output = jax.device_get(output)
            output_text = list(tokenizer.batch_decode(output))
        output = {'output_text': output_text}
        absl.logging.info(
            '\n========= Output ========= \n'
            + pprint.pformat(output) + '\n'
        )
        return output

    @app.post('/loglikelihood')
    def loglikelihood():
        with lock:
            nonlocal sharded_rng
            data = request.get_json()
            absl.logging.info(
                '\n========= Serving Request ========= \n'
                + pprint.pformat(data) + '\n'
            )

            input_text = data['input_text']
            inputs = tokenizer(
                input_text,
                padding='max_length',
                truncation=True,
                max_length=FLAGS.seq_length,
                return_tensors='np',
            )
            with mesh:
                log_likelihood, sharded_rng = log_likelihood_fn(
                    params, sharded_rng, inputs.input_ids,
                    inputs.attention_mask,
                )
                log_likelihood = jax.device_get(log_likelihood)
        output = {'log_likelihood': log_likelihood.tolist()}
        absl.logging.info(
            '\n========= Output ========= \n'
            + pprint.pformat(output) + '\n'
        )
        return output

    with mesh:
        params = sharding_helper.put(params)
        sharded_rng = next_rng()
        app.run(host=FLAGS.host, port=FLAGS.port)


if __name__ == "__main__":
    absl.app.run(main)
