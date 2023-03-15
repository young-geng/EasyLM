import os
import time
from typing import Any, Mapping, Text, Tuple, Union, NamedTuple
from functools import partial
import re
import dataclasses
import random

from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
import optax

from EasyLM.jax_utils import float_to_dtype


class OptimizerFactory(object):
    """ Configurable optax optimizer factory. """

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.accumulate_gradient_steps = 1
        config.bf16_accumulate_gradient = True
        config.type = 'adamw'
        config.palm_optimizer = PalmOptimizerFactory.get_default_config()
        config.adamw_optimizer = AdamWOptimizerFactory.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, weight_decay_mask=None):
        config = cls.get_default_config(config)
        if config.type == 'palm':
            optimizer, optimizer_info = PalmOptimizerFactory.get_optimizer(
                config.palm_optimizer, weight_decay_mask
            )
        elif config.type == 'adamw':
            optimizer, optimizer_info = AdamWOptimizerFactory.get_optimizer(
                config.adamw_optimizer, weight_decay_mask
            )
        else:
            raise ValueError(f'Unknown optimizer type: {config.type}')

        if config.accumulate_gradient_steps > 1:
            if config.bf16_accumulate_gradient:
                accumulator_class = AccumulateGradientBF16
            else:
                accumulator_class = optax.MultiSteps

            optimizer = accumulator_class(
                optimizer, config.accumulate_gradient_steps
            )

        return optimizer, optimizer_info


class PalmOptimizerFactory(object):
    """ PaLM optimizer factory. This optimizer implements the optimizer
        described in the PaLM paper: https://arxiv.org/abs/2204.02311
    """

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr = 0.01
        config.lr_warmup_steps = 10000
        config.b1 = 0.9
        config.b2 = 0.99
        config.clip_gradient = 1.0
        config.weight_decay = 1e-4
        config.bf16_momentum = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, weight_decay_mask=None):
        config = cls.get_default_config(config)

        def learning_rate_schedule(step):
            multiplier = config.lr / 0.01
            return multiplier / jnp.sqrt(jnp.maximum(step, config.lr_warmup_steps))

        def weight_decay_schedule(step):
            multiplier = config.weight_decay / 1e-4
            return -multiplier * jnp.square(learning_rate_schedule(step))

        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule,
            weight_decay_schedule=weight_decay_schedule,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.adafactor(
                learning_rate=learning_rate_schedule,
                multiply_by_parameter_scale=True,
                momentum=config.b1,
                decay_rate=config.b2,
                factored=False,
                clipping_threshold=None,
                dtype_momentum=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
            ),
            optax_add_scheduled_weight_decay(
                weight_decay_schedule, weight_decay_mask
            )
        )
        return optimizer, optimizer_info


class AdamWOptimizerFactory(object):
    """ AdamW optimizer with cosine schedule. """

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.init_lr = 0.0
        config.end_lr = 0.001
        config.lr = 0.01
        config.lr_warmup_steps = 2000
        config.lr_decay_steps = 500000
        config.b1 = 0.9
        config.b2 = 0.95
        config.clip_gradient = 1.0
        config.weight_decay = 1e-4
        config.bf16_momentum = True
        config.multiply_by_parameter_scale = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, weight_decay_mask=None):
        config = cls.get_default_config(config)

        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.init_lr,
            peak_value=config.lr,
            warmup_steps=config.lr_warmup_steps,
            decay_steps=config.lr_decay_steps,
            end_value=config.end_lr,
        )

        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule,
        )

        if config.multiply_by_parameter_scale:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.clip_gradient),
                optax.adafactor(
                    learning_rate=learning_rate_schedule,
                    multiply_by_parameter_scale=True,
                    momentum=config.b1,
                    decay_rate=config.b2,
                    factored=False,
                    clipping_threshold=None,
                    dtype_momentum=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
                ),
                optax_add_scheduled_weight_decay(
                    lambda step: -learning_rate_schedule(step) * config.weight_decay,
                    weight_decay_mask
                )
            )
        else:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.clip_gradient),
                optax.adamw(
                    learning_rate=learning_rate_schedule,
                    weight_decay=config.weight_decay,
                    b1=0.9,
                    b2=0.95,
                    mask=weight_decay_mask,
                    mu_dtype=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
                ),
            )

        return optimizer, optimizer_info


class OptaxScheduledWeightDecayState(NamedTuple):
    count: jnp.DeviceArray


def optax_add_scheduled_weight_decay(schedule_fn, mask=None):
    """ Apply weight decay with schedule. """

    def init_fn(params):
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError('Params cannot be None for weight decay!')

        weight_decay = schedule_fn(state.count)
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * p, updates, params
        )
        return updates, OptaxScheduledWeightDecayState(
            count=optax.safe_int32_increment(state.count)
        )

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)


class AccumulateGradientBF16(optax.MultiSteps):
    """ Customized optax MultiSteps to accumulate gradients with bf16 dtype. """

    def init(self, params):
        updates = jax.tree_util.tree_map(
            jnp.zeros_like, float_to_dtype(params, jnp.bfloat16)
        )
        gradient_step = jnp.zeros([], dtype=jnp.int32)
        _, skip_state = self._should_skip_update_fn(updates, gradient_step, params)
        init_state = optax.MultiStepsState(
            mini_step=jnp.zeros([], dtype=jnp.int32),
            gradient_step=gradient_step,
            inner_opt_state=self._opt.init(params),
            acc_grads=updates,
            skip_state=skip_state
        )
        return init_state

    def update(self, updates, state, params, **extra_args):
        k_steps = self._every_k_schedule(state.gradient_step)
        acc_grads = jax.tree_util.tree_map(
            partial(self._acc_update, n_acc=state.mini_step),
            float_to_dtype(updates, jnp.bfloat16), state.acc_grads
        )

        should_skip_update, skip_state = self._should_skip_update_fn(
            updates, state.gradient_step, params
        )

        def final_step(args):
            del args
            final_updates, new_inner_state = self._opt.update(
                acc_grads, state.inner_opt_state, params=params, **extra_args
            )
            new_state = optax.MultiStepsState(
                mini_step=jnp.zeros([], dtype=jnp.int32),
                gradient_step=optax._src.numerics.safe_int32_increment(state.gradient_step),
                inner_opt_state=new_inner_state,
                acc_grads=jax.tree_util.tree_map(jnp.zeros_like, acc_grads),
                skip_state=skip_state
            )
            return final_updates, new_state

        def mid_step(args):
            del args
            updates_shape_dtype, _ = jax.eval_shape(
                self._opt.update, acc_grads, state.inner_opt_state, params=params
            )
            mid_updates = jax.tree_util.tree_map(
                lambda sd: jnp.zeros(sd.shape, sd.dtype), updates_shape_dtype
            )
            new_state = optax.MultiStepsState(
                mini_step=optax._src.numerics.safe_int32_increment(state.mini_step),
                gradient_step=state.gradient_step,
                inner_opt_state=state.inner_opt_state,
                acc_grads=acc_grads,
                skip_state=skip_state
            )
            return mid_updates, new_state

        new_updates, new_state = jax.lax.cond(
            state.mini_step < k_steps - 1, (), mid_step, (), final_step
        )

        if (should_skip_update.dtype, should_skip_update.shape) != (jnp.bool_, ()):
            raise ValueError(
                'The `should_skip_update_fn` function should return a boolean scalar '
                f'array, but it returned an array of dtype {should_skip_update.dtype}'
                f' and shape {should_skip_update.shape}'
            )

        multi_state_when_skip = optax.MultiStepsState(
            mini_step=state.mini_step,
            gradient_step=state.gradient_step,
            inner_opt_state=state.inner_opt_state,
            acc_grads=state.acc_grads,
            skip_state=skip_state
        )
        zero_updates = jax.tree_util.tree_map(jnp.zeros_like, updates)
        new_updates, new_state = jax.lax.cond(
            should_skip_update,
            (), lambda args: (zero_updates, multi_state_when_skip),
            (), lambda args: (new_updates, new_state)
        )
        return new_updates, new_state
