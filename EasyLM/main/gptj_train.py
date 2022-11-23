import dataclasses
import pprint
from functools import partial
import re

from tqdm import tqdm, trange
import numpy as np
import wandb

import absl.app
import absl.flags
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.experimental import PartitionSpec as PS
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
from flax.training.checkpoints import restore_checkpoint
import optax


from ..data import C4Dataset
from ..jax_utils import (
    JaxRNG, ShardingHelper, JaxMesh, next_rng, match_parition_rules,
    cross_entropy_loss_and_accuracy, named_tree_map
)
from ..utils import (
    WandBLogger, define_flags_with_default, get_user_flags, set_random_seed
)
from ..models.gptj import GPTJConfig, FlaxGPTJForCausalLMModule


FLAGS_DEF = define_flags_with_default(
    seed=42,
    lr=1e-4,
    opt_b1=0.9,
    opt_b2=0.95,
    weight_decay=1e-3,
    total_steps=10000,
    mp_mesh_dim=4,
    load_checkpoint='',
    log_freq=50,
    save_model_freq=0,
    data=C4Dataset.get_default_config(),
    gptj=GPTJConfig.get_default_config(),
    logger=WandBLogger.get_default_config(),
    log_all_worker=False,
)
FLAGS = absl.flags.FLAGS



def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    logger = WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    dataset = C4Dataset(FLAGS.data)
    seq_length = dataset.config.seq_length

    model_config = GPTJConfig.from_pretrained('EleutherAI/gpt-j-6B')
    model_config.update(FLAGS.gptj)
    model_config.vocab_size = dataset.vocab_size
    model = FlaxGPTJForCausalLMModule(model_config)

    def get_weight_decay_mask(params):
        def decay(name, _):
            for rule in GPTJConfig.get_weight_decay_exclusions():
                if re.search(rule, name) is not None:
                    return False
            return True
        return named_tree_map(decay, params, sep='/')

    optimizer = optax.adamw(
        learning_rate=FLAGS.lr, b1=FLAGS.opt_b1, b2=FLAGS.opt_b2,
        weight_decay=FLAGS.weight_decay, mask=get_weight_decay_mask
    )

    device_count = jax.device_count()
    assert device_count % FLAGS.mp_mesh_dim == 0
    mesh = JaxMesh(('dp', 'mp'), (device_count // FLAGS.mp_mesh_dim, FLAGS.mp_mesh_dim))

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs={
                'params': rng_generator(),
                'dropout': rng_generator(),
            }
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, tokens):
        def loss_and_accuracy(params):
            logits = model.apply(params, tokens).logits
            return cross_entropy_loss_and_accuracy(logits, tokens)
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = {'loss': loss, 'accuracy': accuracy}
        return train_state, metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_parition_rules(
        GPTJConfig.get_partition_rules(), train_state_shapes
    )

    sharding_helper = ShardingHelper(train_state_partition)

    sharded_init_fn = pjit(
        init_fn,
        in_axis_resources=(None,),
        out_axis_resources=train_state_partition
    )

    sharded_train_step = pjit(
        train_step,
        in_axis_resources=(train_state_partition, PS('dp')),
        out_axis_resources=(train_state_partition, None),
        donate_argnums=0,
    )

    if FLAGS.load_checkpoint != '':
        with jax.default_device(jax.devices("cpu")[0]):
            restored_checkpoint_state = restore_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes
            )
            start_step = restored_checkpoint_state.step
    else:
        start_step = 0

    with mesh:
        if FLAGS.load_checkpoint != '':
            train_state = sharding_helper.put(restored_checkpoint_state)
            del restored_checkpoint_state
        else:
            train_state = sharded_init_fn(next_rng())

        if FLAGS.save_model_freq > 0:
            logger.save_checkpoint(
                sharding_helper.get(train_state),
                step=train_state.step, overwrite=True
            )

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, batch in zip(step_counter, dataset):
            tokens = mesh.get_local_array_slice(batch['tokens'], 0, 'dp')
            train_state, metrics = sharded_train_step(train_state, tokens)

            if step % FLAGS.log_freq == 0:
                log_metrics = {"step": step}
                log_metrics.update(metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_model_freq > 0 and step % FLAGS.save_model_freq == 0:
                checkpoint_state = sharding_helper.get(train_state)
                if mesh.process_id == 0:
                    logger.save_checkpoint(
                        checkpoint_state, step=train_state.step,
                        overwrite=True
                    )
                del checkpoint_state

        if FLAGS.save_model_freq > 0:
            logger.save_checkpoint(
                sharding_helper.get(train_state),
                step=train_state.step, overwrite=True
            )


if __name__ == "__main__":
    absl.app.run(main)
