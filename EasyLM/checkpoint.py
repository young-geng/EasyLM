import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlxu
import jax
import jax.numpy as jnp
import flax
from flax.serialization import from_bytes, to_bytes
import msgpack

from EasyLM.jax_utils import inplace_float_to_dtype


class StreamingCheckpointer(object):
    """ Custom msgpack checkpointer that saves large train states by serializing
        and saving tensors one by one in a streaming fashion. Avoids running
        out of memory or local TPU disk with default flax checkpointer. The
        checkpointer saves the train state in an asynchronous manner to avoid
        timing out on JAX barriers in multi-host training.
    """

    def __init__(self, checkpoint_dir, enable=True):
        self.checkpoint_dir = checkpoint_dir
        self.enable = enable
        self.async_manager = ThreadPoolExecutor(max_workers=1)

    def _save_checkpoint_worker(self, train_state, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        packer = msgpack.Packer()
        flattend_train_state = flax.traverse_util.flatten_dict(train_state)
        with mlxu.open_file(path, "wb") as fout:
            for key, value in flattend_train_state.items():
                fout.write(packer.pack((key, to_bytes(value))))

    def save_checkpoint(self, train_state, filename):
        train_state = flax.serialization.to_state_dict(train_state)
        if self.enable:
            self.async_manager.submit(
                self._save_checkpoint_worker, train_state, filename
            )

    @staticmethod
    def load_checkpoint(path, target=None, dtype=None):
        flattend_train_state = {}
        with mlxu.open_file(path) as fin:
            # 83886080 bytes = 80 MB, which is 16 blocks on GCS
            unpacker = msgpack.Unpacker(fin, read_size=83886080, max_buffer_size=0)
            for key, value in unpacker:
                flattend_train_state[tuple(key)] = from_bytes(None, value)

        if dtype is not None:
            inplace_float_to_dtype(flattend_train_state, dtype)

        train_state = flax.traverse_util.unflatten_dict(flattend_train_state)
        if target is None:
            return train_state
        return flax.serialization.from_state_dict(target, train_state)

    @staticmethod
    def load_flax_checkpoint(path, target=None, dtype=None):
        """ Load a standard flax checkpoint that's not saved with the
            msgpack streaming format.
        """
        with mlxu.open_file(path, "rb") as fin:
            encoded_bytes = fin.read()

        state_dict = flax.serialization.msgpack_restore(encoded_bytes)
        if dtype is not None:
            inplace_float_to_dtype(state_dict, dtype)
        if target is None:
            return state_dict
        return flax.serialization.from_state_dict(target, state_dict)

    def _save_pickle_worker(self, obj, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        mlxu.save_pickle(obj, path)

    def save_pickle(self, obj, filename):
        if self.enable:
            self.async_manager.submit(self._save_pickle_worker, obj, filename)
