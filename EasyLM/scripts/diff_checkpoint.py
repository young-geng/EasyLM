# This script converts model checkpoint trained by EsayLM to a standard
# mspack checkpoint that can be loaded by huggingface transformers or
# flax.serialization.msgpack_restore. Such conversion allows models to be
# used by other frameworks that integrate with huggingface transformers.

import pprint
from functools import partial
import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization
import mlxu
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.jax_utils import float_to_dtype


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    recover_diff=False,
    load_base_checkpoint='',
    load_target_checkpoint='',
    output_file='',
    streaming=True,
    float_dtype='bf16',
)


def main(argv):
    assert FLAGS.load_base_checkpoint != '' and FLAGS.load_target_checkpoint != ''
    assert FLAGS.output_file != ''
    base_params = StreamingCheckpointer.load_trainstate_checkpoint(
        FLAGS.load_base_checkpoint, disallow_trainstate=True
    )[1]['params']

    target_params = StreamingCheckpointer.load_trainstate_checkpoint(
        FLAGS.load_target_checkpoint, disallow_trainstate=True
    )[1]['params']

    if FLAGS.recover_diff:
        params = jax.tree_util.tree_map(
            lambda b, t: b + t, base_params, target_params
        )
    else:
        params = jax.tree_util.tree_map(
            lambda b, t: t - b, base_params, target_params
        )

    if FLAGS.streaming:
        StreamingCheckpointer.save_train_state_to_file(
            params, FLAGS.output_file, float_dtype=FLAGS.float_dtype
        )
    else:
        params = float_to_dtype(params, FLAGS.float_dtype)
        with mlxu.open_file(FLAGS.output, 'wb') as fout:
            fout.write(flax.serialization.msgpack_serialize(params, in_place=True))


if __name__ == "__main__":
    mlxu.run(main)
