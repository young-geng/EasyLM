import os
import pprint
import random
import tempfile
import time
import uuid
import inspect
from copy import copy
from socket import gethostname
from concurrent.futures import ThreadPoolExecutor

import absl.flags
import cloudpickle as pickle
import gcsfs
import numpy as np
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags
import jax
import jax.numpy as jnp
import flax
from flax.serialization import from_bytes, to_bytes
import msgpack

from .jax_utils import init_rng


class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = False
        config.prefix = "EasyLM"
        config.project = "easy_lm"
        config.output_dir = "/tmp/easy_lm"
        config.gcs_output_dir = ""
        config.random_delay = 0.0
        config.async_save = True
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)
        config.entity = config_dict.placeholder(str)
        config.prefix_to_id = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant, enable=True):
        self.enable = enable
        self.config = self.get_default_config(config)
        self.async_manager = ThreadPoolExecutor(max_workers=1)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != "":
            if self.config.prefix_to_id:
                self.config.experiment_id = "{}--{}".format(
                    self.config.prefix, self.config.experiment_id
                )
            else:
                self.config.project = "{}--{}".format(self.config.prefix, self.config.project)

        if self.enable:
            if self.config.output_dir == "":
                self.config.output_dir = tempfile.mkdtemp()
            else:
                self.config.output_dir = os.path.join(
                    self.config.output_dir, self.config.experiment_id
                )
                os.makedirs(self.config.output_dir, exist_ok=True)

            if self.config.gcs_output_dir != "":
                self.config.gcs_output_dir = os.path.join(
                    self.config.gcs_output_dir, self.config.experiment_id
                )

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        if self.enable:
            self.run = wandb.init(
                reinit=True,
                config=self._variant,
                project=self.config.project,
                dir=self.config.output_dir,
                id=self.config.experiment_id,
                anonymous=self.config.anonymous,
                notes=self.config.notes,
                entity=self.config.entity,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
                mode="online" if self.config.online else "offline",
            )
        else:
            self.run = None

    def log(self, *args, **kwargs):
        if self.enable:
            self.run.log(*args, **kwargs)

    def _save_pickle_worker(self, obj, filename):
        if self.enable:
            if self.config.gcs_output_dir != "":
                path = os.path.join(self.config.gcs_output_dir, filename)
            else:
                path = os.path.join(self.config.output_dir, filename)

            with open_file(path, "wb") as fout:
                pickle.dump(obj, fout)

    def save_pickle(self, obj, filename):
        if self.config.async_save:
            self.async_manager.submit(self._save_pickle_worker, obj, filename)
        else:
            self._save_pickle_worker(obj, filename)

    def _save_checkpoint_worker(self, train_state, filename):
        if self.enable:
            if self.config.gcs_output_dir != "":
                path = os.path.join(self.config.gcs_output_dir, filename)
            else:
                path = os.path.join(self.config.output_dir, filename)

            packer = msgpack.Packer()
            flattend_train_state = flax.traverse_util.flatten_dict(train_state)
            with open_file(path, "wb") as fout:
                for key, value in flattend_train_state.items():
                    fout.write(packer.pack((key, to_bytes(value))))

    def save_checkpoint(self, train_state, filename):
        train_state = flax.serialization.to_state_dict(train_state)
        if self.config.async_save:
            self.async_manager.submit(self._save_checkpoint_worker, train_state, filename)
        else:
            self._save_checkpoint_worker(train_state, filename)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, "automatically defined flag")
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, "automatically defined flag")
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, "automatically defined flag")
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, "automatically defined flag")
        else:
            raise ValueError("Incorrect value type")
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def print_flags(flags, flags_def):
    logging.info(
        "Running training with hyperparameters: \n{}".format(
            pprint.pformat(
                [
                    "{}: {}".format(key, val)
                    for key, val in get_user_flags(flags, flags_def).items()
                ]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def user_flags_to_config_dict(flags, flags_def):
    output = ConfigDict()
    for key in flags_def:
        output[key] = getattr(flags, key)

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


def open_file(path, mode='rb'):
    if path.startswith("gs://"):
        return gcsfs.GCSFileSystem().open(path, mode, cache_type='block')
    else:
        return open(path, mode)


def load_pickle(path):
    with open_file(path, 'rb') as fin:
        data = pickle.load(fin)
    return data


def load_checkpoint(path, target):
    flattend_train_state = {}
    with open_file(path) as fin:
        unpacker = msgpack.Unpacker(fin, max_buffer_size=0)
        for key, value in unpacker:
            flattend_train_state[tuple(key)] = from_bytes(None, value)

    return flax.traverse_util.unflatten_dict(flattend_train_state)


def function_args_to_config(fn, none_arg_types=None, exclude_args=None, override_args=None):
    config = ConfigDict()
    arg_spec = inspect.getargspec(fn)
    n_args = len(arg_spec.defaults)
    arg_names = arg_spec.args[-n_args:]
    default_values = arg_spec.defaults
    for name, value in zip(arg_names, default_values):
        if exclude_args is not None and name in exclude_args:
            continue
        elif override_args is not None and name in override_args:
            config[name] = override_args[name]
        elif none_arg_types is not None and value is None and name in none_arg_types:
            config[name] = config_dict.placeholder(none_arg_types[name])
        else:
            config[name] = value

    return config

