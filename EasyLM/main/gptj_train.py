import dataclasses
import pprint
from functools import partial

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
import optax


from ..data import C4Dataset
from ..jax_utils import (
    JaxRNG, get_metrics, next_rng, JaxMesh, match_parition_rules,
    cross_entropy_loss_and_accuracy
)
from ..utils import (
    WandBLogger, define_flags_with_default, get_user_flags, load_pickle,
    set_random_seed
)
from ..models.gptj import (
    GPTJConfig, FlaxGPTJForCausalLMModule, get_parition_rules
)


FLAGS_DEF = define_flags_with_default(
    seed=42,
    lr=1e-4,
    data=C4Dataset.get_default_config(),
    gptj=GPTJConfig.get_default_config(),
    logger=WandBLogger.get_default_config(),
)
FLAGS = absl.flags.FLAGS



def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    set_random_seed(FLAGS.seed)

    dataset = C4Dataset(FLAGS.data)
    seq_length = dataset.config.seq_length

    model_config = GPTJConfig.from_pretrained('EleutherAI/gpt-j-6B')
    model_config.update(FLAGS.gptj)
    model = FlaxGPTJForCausalLMModule(model_config)

    optimizer = optax.adam(FLAGS.lr)

    mesh = JaxMesh(('dp', 'mp'), (2, 4))

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
        return TrainState.create(
            params=params, tx=optimizer,
            apply_fn=None,
        )

    def train_step(train_state, tokens):
        def loss_and_accuracy(params):
            logits = model.apply(params, tokens).logits
            return cross_entropy_loss_and_accuracy(logits, tokens)
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = {'loss': loss, 'accuracy': accuracy}
        return train_state, metrics

    shapes = jax.eval_shape(init_fn, next_rng())
    model_partition = match_parition_rules(
        get_parition_rules(), shapes
    )

    sharded_init_fn = pjit(
        init_fn,
        in_axis_resources=(None,),
        out_axis_resources=model_partition
    )

    sharded_train_step = pjit(
        train_step,
        in_axis_resources=(model_partition, PS('dp')),
        out_axis_resources=(model_partition, None)
    )
    with mesh:
        train_state = sharded_init_fn(next_rng())

        for step, batch in tqdm(enumerate(dataset)):
            tokens = mesh.get_local_array_slice(batch['tokens'], 0, 'dp')
            train_state, metrics = sharded_train_step(train_state, tokens)


if __name__ == "__main__":
    absl.app.run(main)
