# This script converts model checkpoint trained by EsayLM to a standard
# mspack checkpoint that can be loaded by huggingface transformers or
# flax.serialization.msgpack_restore. Such conversion allows models to be
# used by other frameworks that integrate with huggingface transformers.

import pprint
from functools import partial
import os
import numpy as np
import mlxu
import jax.numpy as jnp
import flax.serialization
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.jax_utils import float_to_dtype


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    load_checkpoint='',
    output_file='',
    streaming=False,
    float_dtype='bf16',
)


def main(argv):
    assert FLAGS.load_checkpoint != '' and FLAGS.output_file != '', 'input and output must be specified'
    params = StreamingCheckpointer.load_trainstate_checkpoint(
        FLAGS.load_checkpoint, disallow_trainstate=True
    )[1]['params']

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
