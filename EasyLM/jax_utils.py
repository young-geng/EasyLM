import os
import time
from typing import Any, Mapping, Text, Tuple, Union
from functools import partial
import re
import dataclasses

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


class JaxMesh(object):
    """ A helper function for multihost mesh paritioned programs. """

    def __init__(self, axis_names, axis_shape):
        assert len(axis_shape) == len(axis_names)
        self._axis_shape = tuple(axis_shape)
        self._axis_names = tuple(axis_names)

        self._process_id = jax.process_index()
        self._local_device_count = jax.local_device_count()
        self._device_count = jax.device_count()
        self._mesh_ids = np.meshgrid(
            *[np.arange(x) for x in axis_shape],
            indexing='ij'
        )

        self._mesh_devices = np.array(jax.devices()).reshape(axis_shape)
        self._mesh = Mesh(self._mesh_devices, axis_names)

    def axis_length(self, axis_name):
        axis_id = self.axis_names.index(axis_name)
        return self.axis_shape[axis_id]

    def local_axis_ids(self, axis_name):
        """ Return an array of ids for local devices on the given axis. """
        axis_id = self.axis_names.index(axis_name)
        return np.unique(
            self._mesh_ids[axis_id].reshape(
                -1, self.local_device_count
            )[self.process_id]
        )

    def get_local_array_slice(self, x, axis, axis_name):
        """ Get the slice of array for local devices. """
        axis_length = self.axis_length(axis_name)
        local_axis_ids = self.local_axis_ids(axis_name)
        splits = np.split(x, axis_length, axis=axis)
        shards = [splits[i] for i in local_axis_ids]
        return np.concatenate(shards, axis=axis)

    def __enter__(self):
        return self.mesh.__enter__()

    def __exit__(self, *args, **kwargs):
        return self.mesh.__exit__(*args, **kwargs)

    @property
    def axis_shape(self):
        return self._axis_shape

    @property
    def axis_names(self):
        return self._axis_names

    @property
    def mesh_devices(self):
        return self._mesh_devices

    @property
    def mesh(self):
        return self._mesh

    @property
    def process_id(self):
        return self._process_id

    @property
    def local_device_count(self):
        return self._local_device_count

    @property
    def device_count(self):
        return self._device_count


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


def get_metrics(metrics, unreplicate=False):
    if unreplicate:
        metrics = flax.jax_utils.unreplicate(metrics)
    metrics = jax.device_get(metrics)
    return {key: float(val) for key, val in metrics.items()}


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))


def cross_entropy_loss(logits, labels, smoothing_factor=0.):
    num_classes = logits.shape[-1]
    if labels.dtype == jnp.int32 or labels.dtype == jnp.int64:
        labels = jax.nn.one_hot(labels, num_classes)
    if smoothing_factor > 0.:
        labels = labels * (1. - smoothing_factor) + smoothing_factor / num_classes
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(logp * labels, axis=-1))


@partial(jax.pmap, axis_name="pmap", donate_argnums=0)
def sync_state_across_devices(state):
    i = jax.lax.axis_index("pmap")

    def select(x):
        return jax.lax.psum(jnp.where(i == 0, x, jnp.zeros_like(x)), "pmap")

    return jax.tree_map(select, state)


def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
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


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0
    )


def get_2d_sincos_pos_embed(embed_dim, length):
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)


def flatten_tree(xs, keep_empty_nodes=False, is_leaf=None, sep=None):
    """ A stronger version of flax.traverse_util.flatten_dict, suppoerts
        dict, tuple, list and TrainState.
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
        if keep_empty_nodes and is_empty:
            if prefix == ():  # when the whole input is empty
                return {}
            return {_key(prefix): empty_node}
        return result

    return _flatten(xs, ())


def match_parition_rules(rules, params):
    flattened_params = flatten_tree(params, sep='/')
    id_to_name_map = {
        id(val): key for key, val in flattened_params.items()
    }

    def get_parition_spec(leaf):
        name = id_to_name_map[id(leaf)]
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Parition rule not found for param: {name}')

    return jax.tree_util.tree_map(get_parition_spec, params)

