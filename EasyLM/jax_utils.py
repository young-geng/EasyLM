import os
import time
from typing import Any, Mapping, Text, Tuple, Union, NamedTuple
from functools import partial
import re
import dataclasses
import random

import dill
import flax
import jax
import jax.numpy as jnp
from jax.experimental import PartitionSpec as PS
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
import numpy as np
from absl import logging
from flax import jax_utils
from flax.training.train_state import TrainState
from flax.core import FrozenDict
import optax


class JaxRNG(object):
    """ A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside
        pure function.
    """

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


class ShardingHelper(object):
    """ A helper utility that handles gathering sharded pytree to host and
        shard host pytree to devices that supports multi-host environment.
        This utility does gather and shard one by one to avoid OOM on device.
    """

    def __init__(self, partition_specs):
        self.partition_specs = partition_specs
        def gather_tensor(partition_spec):
            return pjit(
                lambda x: x,
                in_axis_resources=partition_spec,
                out_axis_resources=None
            )

        def shard_tensor(partition_spec):
            return pjit(
                lambda x: x,
                in_axis_resources=None,
                out_axis_resources=partition_spec
            )

        self.gather_fns = jax.tree_util.tree_map(gather_tensor, partition_specs)
        self.shard_fns = jax.tree_util.tree_map(shard_tensor, partition_specs)

    def get(self, tree):
        def get_fn(gather_fn, tensor):
            return jax.device_get(gather_fn(tensor))

        return jax.tree_util.tree_map(get_fn, self.gather_fns, tree)

    def put(self, tree):
        def put_fn(shard_fn, tensor):
            return shard_fn(tensor).block_until_ready()

        return jax.tree_util.tree_map(put_fn, self.shard_fns, tree)


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def get_jax_mp_mesh(mp_axis_dim, mp_axis_name='mp', dp_axis_name='dp'):
    """ Return a 2D mesh for (MP, DP) partitioning. """
    device_count = jax.device_count()
    if mp_axis_dim <= 0:
        mp_axis_dim = device_count
    assert device_count % mp_axis_dim == 0
    return Mesh(
        np.array(jax.devices()).reshape(-1, mp_axis_dim),
        (dp_axis_name, mp_axis_name)
    )


def wrap_function_with_rng(rng):
    """ To be used as decorator, automatically bookkeep a RNG for the wrapped function. """
    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)
        return wrapped
    return wrap_function


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


def get_metrics(metrics, unreplicate=False, stack=False):
    if unreplicate:
        metrics = flax.jax_utils.unreplicate(metrics)
    metrics = jax.device_get(metrics)
    if stack:
        return jax.tree_map(lambda *args: np.stack(args), *metrics)
    else:
        return {key: float(val) for key, val in metrics.items()}


def mse_loss(val, target, valid=None):
    if valid is None:
        valid = jnp.ones((*target.shape[:2], 1))
    valid = valid.astype(jnp.float32)
    loss = jnp.mean(
        jnp.where(
            valid > 0.0,
            jnp.square(val - target),
            0.0
        )
    )
    return loss


def cross_entropy_loss(logits, labels, smoothing_factor=0.):
    num_classes = logits.shape[-1]
    if labels.dtype == jnp.int32 or labels.dtype == jnp.int64:
        labels = jax.nn.one_hot(labels, num_classes)
    if smoothing_factor > 0.:
        labels = labels * (1. - smoothing_factor) + smoothing_factor / num_classes
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(logp * labels, axis=-1))


def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)

    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy


def global_norm(tree):
    """ Return the global L2 norm of a pytree. """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = jax.flatten_util.ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))


def average_metrics(metrics):
    return jax.tree_map(
        lambda *args: jnp.mean(jnp.stack(args)),
        *metrics
    )


def get_float_dtype_by_name(dtype):
    return {
        'bf16': jnp.bfloat16,
        'fp16': jnp.float16,
        'fp32': jnp.float32,
        'fp64': jnp.float64,
    }[dtype]


def float_to_dtype(tree, dtype=jnp.float32):
    """ Convert all float(fp16, bf16, fp32, fp64) arrays in a pytree to a given
        dtype.
    """
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    assert dtype in float_dtypes, 'dtype must be a float type!'
    def to_dtype(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            if x.dtype in float_dtypes and x.dtype != dtype:
                x = x.astype(dtype)
        return x
    return jax.tree_util.tree_map(to_dtype, tree)


def inplace_float_to_dtype(tree, dtype=jnp.float32):
    """ Convert all float(fp16, bf16, fp32, fp64) arrays in a pytree to a given
        dtype inplace. Only supports nested dicts.
    """
    if isinstance(tree, FrozenDict):
        raise ValueError('Only supports nested dicts!')
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    assert dtype in float_dtypes, 'dtype must be a float type!'
    for key, val in tree.items():
        if isinstance(val, (np.ndarray, jnp.ndarray)) and val.dtype in float_dtypes:
            if val.dtype != dtype:
                tree[key] = val.astype(dtype)
        elif isinstance(val, dict):
            inplace_float_to_dtype(val, dtype)
        elif isinstance(val, FrozenDict):
            raise ValueError('Only supports nested dicts!')


def flatten_tree(xs, is_leaf=None, sep=None):
    """ A stronger version of flax.traverse_util.flatten_dict, supports
        dict, tuple, list and TrainState. Tuple and list indices will be
        converted to strings.
    """
    tree_node_classes = (FrozenDict, dict, tuple, list, TrainState)
    if not isinstance(xs, tree_node_classes):
        ValueError('fUnsupported node type: {type(xs)}')

    def _is_leaf(prefix, fx):
        if is_leaf is not None:
            return is_leaf(prefix, xs)
        return False

    def _key(path):
        if sep is None:
            return path
        return sep.join(path)

    def _convert_to_dict(xs):
        if isinstance(xs, (FrozenDict, dict)):
            return xs
        elif isinstance(xs, (tuple, list)):
            return {f'{i}': v for i, v in enumerate(xs)}
        elif isinstance(xs, TrainState):
            output = {}
            for field in dataclasses.fields(xs):
                if 'pytree_node' not in field.metadata or field.metadata['pytree_node']:
                    output[field.name] = getattr(xs, field.name)
            return output
        else:
            raise ValueError('fUnsupported node type: {type(xs)}')

    def _flatten(xs, prefix):
        if not isinstance(xs, tree_node_classes) or _is_leaf(prefix, xs):
            return {_key(prefix): xs}

        result = {}
        is_empty = True
        for (key, value) in _convert_to_dict(xs).items():
            is_empty = False
            path = prefix + (key, )
            result.update(_flatten(value, path))
        return result

    return _flatten(xs, ())


def named_tree_map(f, tree, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    flattened_tree = flatten_tree(tree, is_leaf=is_leaf, sep=sep)
    id_to_name = {id(val): key for key, val in flattened_tree.items()}
    def map_fn(leaf):
        name = id_to_name[id(leaf)]
        return f(name, leaf)
    return jax.tree_util.tree_map(map_fn, tree)


def match_partition_rules(rules, params):
    """ Returns a pytree of PartitionSpec according to rules. Supports handling
        Flax TrainState and Optax optimizer state.
    """
    def get_partition_spec(name, leaf):
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            """ Don't partition scalar values. """
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {name}')
    return named_tree_map(get_partition_spec, params, sep='/')
