from functools import partial
from time import time
import os
import numpy as np
import jax
import jax.flatten_util
import jax.numpy as jnp
import mlxu
from EasyLM.bpt import blockwise_attn
from EasyLM.jax_utils import (
    get_float_dtype_by_name, set_random_seed, next_rng, JaxRNG
)


FLAGS, _ = mlxu.define_flags_with_default(
    seed=42,
    dtype='fp32',
    embed_dim=2048,
    n_heads=16,
    ref_attn_seq_len=2048,
    eff_attn_seq_len=16384,
    batch_size=1,
    query_chunk_size=2048,
    key_chunk_size=2048,
    warmup_steps=40,
    steps=200,
)


def main(argv):

    def random_kqv(rng_key, seq_len):
        rng_generator = JaxRNG(rng_key)
        kqv = []
        for i in range(3):
            kqv.append(
                jax.random.normal(
                    rng_generator(),
                    (FLAGS.batch_size, seq_len, FLAGS.n_heads, FLAGS.embed_dim // FLAGS.n_heads),
                    dtype=get_float_dtype_by_name(FLAGS.dtype)
                )
            )
        return tuple(kqv)

    def reference_attn(query, key, value):
        dtype = get_float_dtype_by_name(FLAGS.dtype)
        query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
        logits = jnp.einsum("bqhc,bkhc->bhqk", query, key)
        mask_value = jnp.finfo(logits.dtype).min
        _, q_seq_len, _, _ = query.shape
        _, kv_seq_len, _, _ = key.shape
        mask_shape = (q_seq_len, kv_seq_len)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (row_ids < col_ids)[None, None, :, :]
        logits = logits + jnp.where(causal_mask, mask_value, 0.0)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bhqk,bkhc->bqhc", weights, value)
        return out

    def efficient_attention(query, key, value):
        dtype = get_float_dtype_by_name(FLAGS.dtype)
        return blockwise_attn(
            query, key, value,
            bias=None,
            deterministic=True,
            dropout_rng=None,
            attn_pdrop=0.0,
            causal=True,
            query_chunk_size=FLAGS.query_chunk_size,
            key_chunk_size=FLAGS.key_chunk_size,
            dtype=get_float_dtype_by_name(FLAGS.dtype),
            policy=jax.checkpoint_policies.nothing_saveable(),
            precision=None,
            float32_logits=True,
            prevent_cse=True,
        )


    @partial(jax.jit, static_argnums=(1,))
    def reference_attn_forward_backward(rng_key, seq_len):
        @partial(jax.grad, argnums=(0, 1, 2))
        @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable())
        def grad_fn(query, key, value):
            out = reference_attn(query, key, value)
            return jnp.mean(out)

        query, key, value = random_kqv(rng_key, seq_len)
        return jax.flatten_util.ravel_pytree(
            grad_fn(query, key, value)[1]
        )[0].mean()

    @partial(jax.jit, static_argnums=(1,))
    def efficient_attn_forward_backward(rng_key, seq_len):
        @partial(jax.grad, argnums=(0, 1, 2))
        def grad_fn(query, key, value):
            out = efficient_attention(query, key, value)
            return jnp.mean(out)

        query, key, value = random_kqv(rng_key, seq_len)
        return jax.flatten_util.ravel_pytree(
            grad_fn(query, key, value)[1]
        )[0].mean()


    set_random_seed(FLAGS.seed)

    jax.block_until_ready(reference_attn_forward_backward(next_rng(), FLAGS.ref_attn_seq_len))
    jax.block_until_ready(efficient_attn_forward_backward(next_rng(), FLAGS.eff_attn_seq_len))

    all_results = []
    for i in range(FLAGS.warmup_steps):
        all_results.append(reference_attn_forward_backward(next_rng(), FLAGS.ref_attn_seq_len))
    jax.block_until_ready(all_results)

    start_time = time()
    all_results = []
    for i in range(FLAGS.steps):
        all_results.append(reference_attn_forward_backward(next_rng(), FLAGS.ref_attn_seq_len))

    jax.block_until_ready(all_results)
    elapsed_time_ref_attn = time() - start_time
    print(f'Reference attention: {elapsed_time_ref_attn:.3f} seconds')


    all_results = []
    for i in range(FLAGS.warmup_steps):
        all_results.append(efficient_attn_forward_backward(next_rng(), FLAGS.eff_attn_seq_len))
    jax.block_until_ready(all_results)


    start_time = time()
    all_results = []
    for i in range(FLAGS.steps):
        all_results.append(efficient_attn_forward_backward(next_rng(), FLAGS.eff_attn_seq_len))

    jax.block_until_ready(all_results)
    elapsed_time_efficient_attn = time() - start_time
    print(f'Efficient attention: {elapsed_time_efficient_attn:.3f} seconds')

    flops_ratio = (FLAGS.eff_attn_seq_len / FLAGS.ref_attn_seq_len) ** 2
    efficiency = elapsed_time_ref_attn / elapsed_time_efficient_attn * flops_ratio
    print(f'Efficiency: {efficiency:.3f}')


if __name__ == '__main__':
    mlxu.run(main)



