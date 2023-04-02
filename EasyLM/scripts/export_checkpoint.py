# This script converts model checkpoint trained by EsayLM to a standard
# mspack checkpoint that can be loaded by huggingface transformers or
# flax.serialization.msgpack_restore. Such conversion allows models to be
# used by other frameworks that integrate with huggingface transformers.

import pprint
from functools import partial
import os
import numpy as np
import mlxu
import flax.serialization
import flax.training.checkpoints
from EasyLM.checkpoint import StreamingCheckpointer


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    input=('', 'input checkpoint file'),
    output=('', 'output Flax msgpack file'),
)


def main(argv):
    assert FLAGS.input != '' and FLAGS.output != '', 'input and output must be specified'
    params = StreamingCheckpointer.load_trainstate_checkpoint(
        FLAGS.input, disallow_trainstate=True
    )[1]['params']
    with mlxu.open_file(FLAGS.output, 'wb') as fout:
        fout.write(flax.serialization.msgpack_serialize(params, in_place=True))


if __name__ == "__main__":
    mlxu.run(main)
