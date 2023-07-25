# This file is taken from the flaxformer library at
# https://github.com/google/flaxformer

# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of memory-efficient attention.

Original version published here: https://arxiv.org/abs/2112.05682

Also known as Flash Attention: https://arxiv.org/abs/2205.14135
"""

import functools
from typing import Callable, NamedTuple, Optional, Any

import jax
from jax import lax
from jax import numpy as jnp
from jax import random

# from flaxformer.types import DType
# from flaxformer.types import PRNGKey

Array = jax.Array
PRNGKey = Any
DType = Any

def _causal_bias(
    q_len: int,
    k_len: int,
    offset: Optional[int] = None,
    mask_to_bias_factor: float = 1e6,
) -> Array:
    q_idxs = lax.broadcasted_iota(dtype=jnp.int32, shape=(q_len, 1), dimension=0)
    k_idxs = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, k_len), dimension=1)
    if offset is not None:
        q_idxs += offset
    inverted_mask = q_idxs < k_idxs  # broadcasts to shape (q_len, k_len)
    return inverted_mask * (-1 * mask_to_bias_factor)


def _local_causal_bias(
    q_len: int,
    k_len: int,
    query_offset: int,
    key_offset: int,
) -> Array:
    offset = query_offset - key_offset
    return _causal_bias(q_len, k_len, offset=offset)


class _AttentionSummary(NamedTuple):
    """The summary of the attention over a segment of keys and values."""

    # Sum of the values weighted by the exponentiated scores. Array of shape
    # `[batch, queries, heads, queries_per_head, value_features]`.
    numerator: Array
    # Sum of the exponentiated scores per query. Array of shape
    # `[batch, queries, heads, queries_per_head]`.
    denominator: Array
    # Maximum score encountered per query. Array of shape
    # `[batch, queries, heads, queries_per_head]`.
    max_so_far: Array


def _summarize_chunk(
    query: Array,
    key: Array,
    value: Array,
    current_summary: _AttentionSummary,
    bias: Optional[Array],
    precision=None,
) -> _AttentionSummary:
    """Attention for a segment of queries, keys, and values.

    Args:
        query: An array of shape `[batch, q_length, heads, queries_per_head,
            qk_depth_per_head]`.
        key: An array of shape `[batch, kv_length, heads, qk_depth_per_head]`.
        value: An array of shape `[batch, kv_length, heads, v_depth_per_head]`.
        current_summary: The partially summarized queries so far, before adding the
            summarization of this kv chunk.
        bias: bias for the attention weights. This should be broadcastable to the
            shape `[batch, heads, queries_per_head, q_length, kv_length]` This can be
            used for incorporating causal masks, padding masks, proximity bias, etc.
        precision: numerical precision of the computation see `jax.lax.Precision`
            for details.

    Returns:
        The summary for this segment, consisting of sum of the sum of the
        values weighted by their exponentiated attention scores, the exponentiated
        attention scores, and the maximum score of this segment.
    """
    batch, q_len, q_heads, queries_per_head, q_feat = query.shape
    del q_feat
    _, kv_len, _, v_feat = value.shape

    (numerator, denominator, max_so_far) = current_summary

    attn_weights = jnp.einsum(
        'bqhnd,bkhd->bqhnk', query, key, precision=precision
    )

    if bias is not None:
        bias = jnp.moveaxis(bias, -2, 1)  # move sequence length outside
        attn_weights += bias

    previous_max = max_so_far
    assert previous_max.shape == (batch, q_len, q_heads, queries_per_head)
    chunk_maxima = jnp.max(attn_weights, axis=-1)
    assert chunk_maxima.shape == (batch, q_len, q_heads, queries_per_head)
    max_so_far = jnp.maximum(max_so_far, chunk_maxima)
    max_so_far = jax.lax.stop_gradient(max_so_far)
    correction = jnp.exp(previous_max - max_so_far)
    assert correction.shape == (batch, q_len, q_heads, queries_per_head)

    corrected_weights = jnp.exp(
        attn_weights - max_so_far[:, :, :, :, jnp.newaxis]
    )

    assert corrected_weights.shape == (
        batch,
        q_len,
        q_heads,
        queries_per_head,
        kv_len,
    )

    expected_numerator_shape = (batch, q_len, q_heads, queries_per_head, v_feat)
    assert numerator.shape == expected_numerator_shape, (
            f'numerator.shape is {numerator.shape}, but expected'
            f' {expected_numerator_shape}.'
    )

    numerator = numerator * correction[:, :, :, :, jnp.newaxis]
    numerator = numerator + jnp.einsum(
            'bqhnv,bvhf->bqhnf', corrected_weights, value, precision=precision
    )

    assert denominator.shape == (batch, q_len, q_heads, queries_per_head)
    denominator = denominator * correction
    denominator = denominator + corrected_weights.sum(axis=-1)

    return _AttentionSummary(
        numerator,
        denominator,
        max_so_far,
    )


def _memory_efficient_attention(
    query,
    key,
    value,
    bias_fn: Callable[[int, int], Array],
    query_chunk_size: int,
    key_chunk_size: int,
    precision=None,
    dtype=jnp.float32,
    use_extra_logit: bool = False,
    causal_mask: bool = False,
):
    """Computes dot-product multiquery-attention given query, key, and value."""
    batch, num_q, heads, queries_per_head, q_feat = query.shape
    batch, num_kv, heads, k_features = key.shape
    batch, num_kv, heads, v_features = value.shape

    num_q_chunks = num_q // query_chunk_size
    num_kv_chunks = num_kv // key_chunk_size

    query = query.reshape(
            (batch, num_q_chunks, query_chunk_size, heads, queries_per_head, q_feat)
    )
    key = key.reshape((batch, num_kv_chunks, key_chunk_size, heads, k_features))
    value = value.reshape(
            (batch, num_kv_chunks, key_chunk_size, heads, v_features)
    )
    # We move the chunk_idx axis to the front to iterate over it with lax.map.
    query = jnp.moveaxis(query, 1, 0)
    key = jnp.moveaxis(key, 1, 0)
    value = jnp.moveaxis(value, 1, 0)

    # The zero_chunk is the output of _summarize_chunk when the inputs are zeros.
    # We define the zero_chunk outside of the loops to prevent the compiler from
    # re-creating these arrays in every loop iteration.
    zero_chunk = _AttentionSummary(
        # numerator
        jnp.zeros(
            (batch, query_chunk_size, heads, queries_per_head, v_features),
            dtype=dtype,
        ),
        # denominator
        jnp.zeros(
            (batch, query_chunk_size, heads, queries_per_head), dtype=dtype
        ),
        # max_so_far
        (-jnp.inf)
        * jnp.ones(
            (batch, query_chunk_size, heads, queries_per_head), dtype=dtype
        ),
    )

    def _query_chunk_attention(args):
        query_chunk, query_chunk_idx = args

        @functools.partial(jax.checkpoint, prevent_cse=False)
        def conditional_summarize_fn(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args

            skip_block = jnp.array(False)
            if causal_mask:
                skip_block = query_chunk_idx < key_chunk_idx

            def cond_fn(query, key, value, carry, key_chunk_idx):
                with jax.named_scope('compute_bias'):
                    chunk_bias = bias_fn(query_chunk_idx, key_chunk_idx)
                return (
                    _summarize_chunk(
                        query, key, value, carry, chunk_bias, precision=precision
                    ),
                    None,
                )

            return jax.lax.cond(
                skip_block,
                lambda a, b, c, carry, d: (carry, None),
                cond_fn,
                query_chunk,
                key_chunk,
                value_chunk,
                carry,
                key_chunk_idx,
            )

        (numerator, denominator, max_so_far), _ = jax.lax.scan(
            conditional_summarize_fn,
            zero_chunk,
            xs=(key, value, jnp.arange(0, num_kv_chunks)),
        )

        if use_extra_logit:
            denominator += jnp.exp(-max_so_far)

        return numerator / denominator[:, :, :, :, jnp.newaxis]

    res = lax.map(_query_chunk_attention, xs=(query, jnp.arange(0, num_q_chunks)))

    expected_res_shape = (
        num_q_chunks,
        batch,
        query_chunk_size,
        heads,
        queries_per_head,
        v_features,
    )
    assert (
        res.shape == expected_res_shape
    ), f'res.shape is {res.shape}, but expected {expected_res_shape}.'
    res = jnp.moveaxis(res, 0, 1)
    return res.reshape(batch, num_q, heads, queries_per_head, value.shape[-1])


def dot_product_attention_queries_per_head(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    broadcast_dropout: bool = True,
    rescale_logits: bool = False,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    enable_dropout: bool = True,
    dtype: DType = jnp.float32,
    precision: Optional[lax.Precision] = None,
    use_extra_logit: bool = False,
    float32_logits: bool = False,
    causal_mask: bool = False,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 2048,
) -> Array:
    """Computes dot-product attention given query, key, and value.

    This is a variant of attention that generalizes both multi-head and
    multi-query attention. It features an extra dimension for the query array,
    that specifies the number of queries per head.

    This function is improved by the memory-efficient attention algorithm
    (https://arxiv.org/abs/2112.05682), which is also called FlashAttention
    (https://arxiv.org/abs/2205.14135).

    Note: query, key, value needn't have any batch dimensions.

    Args:
        query: queries for calculating attention with shape of `[batch..., q_length,
            num_heads, queries_per_head, qk_depth_per_head]`.
        key: keys for calculating attention with shape of `[batch..., kv_length,
            num_heads, qk_depth_per_head]`.
        value: values to be used in attention with shape of `[batch..., kv_length,
            num_heads, v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
            shape `[batch..., num_heads, queries_per_head, q_length, kv_length]` This
            can be used for incorporating causal masks, padding masks, proximity bias,
            etc.
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
        dropout_rng: JAX PRNGKey: to be used for dropout
        dropout_rate: dropout rate
        enable_dropout: bool, whether to apply dropout
        dtype: the dtype of the computation (default: float32)
        precision: numerical precision of the computation see `jax.lax.Precision`
            for details.
        use_extra_logit: whether to include a virtual extra logit equal to zero.
        float32_logits: bool, if True then compute logits in float32 to avoid
            numerical issues with bfloat16.
        causal_mask: Apply a causal mask. This can be used alternatively or in
            addition to the given bias.
        query_chunk_size: Positive integer to control the size of the query chunks.
        key_chunk_size: Positive integer to control the size of the key chunks.

    Returns:
        Output of shape `[batch..., length, num_heads, queries_per_head,
        v_depth_per_head]`.
    """
    assert (
        key.ndim == value.ndim
    ), f'k, v must have same rank. key: {key.shape}, value: {value.shape}'
    assert (
        query.shape[:-4] == key.shape[:-3] == value.shape[:-3]
    ), f'q, k, v batch dim must match. query: {query.shape}'

    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

    # Ensure that we have exactly one batch dimension
    orig_batch_dims = query.shape[:-4]
    query = query.reshape(-1, *query.shape[-4:])
    key = key.reshape(-1, *key.shape[-3:])
    value = value.reshape(-1, *value.shape[-3:])
    if bias is not None:
        bias = bias.reshape(-1, *bias.shape[-4:])

    batch_size, query_length, heads, queries_per_head, _ = query.shape
    _, key_length, _, _ = key.shape

    # TODO: Consider automatic padding to remove this constraint.
    if query_length % query_chunk_size != 0 and query_length > query_chunk_size:
        raise ValueError(
            'Sequence length of the query vector %d needs to be less '
            'than, or a multiple of the query_chunk_size %d.'
            % (query_length, query_chunk_size)
        )
    if key_length % key_chunk_size != 0 and key_length > key_chunk_size:
        raise ValueError(
            'Sequence length of the key/value vector %d needs to be less '
            'than, or a multiple of the key_chunk_size %d.'
            % (key_length, key_chunk_size)
        )

    query_chunk_size = min(query_chunk_size, query_length)
    key_chunk_size = min(key_chunk_size, key_length)

    if bias is not None:
        broadcastable_to = (
            batch_size,
            heads,
            queries_per_head,
            query_length,
            key_length,
        )
        # Check that bias is broadcastable as expected:
        for bias_dim, broadcast_dim in zip(bias.shape, broadcastable_to):
            if bias_dim not in [1, broadcast_dim]:
                raise ValueError(
                    f'Expected bias dimensions {bias.shape} to be broadcastable to'
                    f' {broadcastable_to}.'
                )

    if enable_dropout and dropout_rate > 0.0:
        # Precompute dropout
        drop_shape = [batch_size, heads, queries_per_head, query_length, key_length]
        if broadcast_dropout:
            # We mimick the semantics of T5 and broadcast along the "length" dim.
            drop_shape[-2] = 1  # query_length dim
        precomputed_dropout = random.bernoulli(
                dropout_rng, dropout_rate, drop_shape
        )

    def bias_fn(
        query_chunk_idx: int,
        key_chunk_idx: int,
    ) -> Array:
        query_offset = query_chunk_idx * query_chunk_size
        key_offset = key_chunk_idx * key_chunk_size

        local_bias = jnp.zeros((1, 1, 1, 1, 1))
        if bias is not None:
            # If bias is not broadcasted yet, dynamic slice would fail with full slice
            # size. In this case we keep the bias unbroadcasted.
            slice_q_len = min(bias.shape[-2], query_chunk_size)
            slice_k_len = min(bias.shape[-1], key_chunk_size)
            local_bias = lax.dynamic_slice(
                bias,
                # query_offset and key_offset might be > 1 but bias dims might
                # not yet be broadcasted. We rely on the protection against
                # out-of-bounds array accesses built into dynamic_slice.
                start_indices=(0, 0, 0, query_offset, key_offset),
                slice_sizes=(*bias.shape[:3], slice_q_len, slice_k_len),
            )
        if causal_mask:
            causal = _local_causal_bias(
                query_chunk_size, key_chunk_size, query_offset, key_offset
            )
            # add batch, head, and queries_per_head dims
            local_bias += causal.reshape(1, 1, 1, *causal.shape)
        # We implement dropout as part of the bias, which is additive to the
        # attention scores. In some other implementations it is treated as a
        # multiplicative factor applied to the probabilities after softmax.
        if enable_dropout and dropout_rate > 0.0:
            with jax.named_scope('dropout'):
                # If dropout is not broadcasted yet, we need the collapsed dims.
                slice_q_len = min(precomputed_dropout.shape[-2], query_chunk_size)
                slice_k_len = min(precomputed_dropout.shape[-1], key_chunk_size)
                dropout_slice = lax.dynamic_slice(
                    precomputed_dropout,
                    # query_offset and key_offset might be > 1 but dropout dims might
                    # not yet be broadcasted. We rely on the protection against
                    # out-of-bounds array accesses built into dynamic_slice.
                    start_indices=(0, 0, 0, query_offset, key_offset),
                    slice_sizes=(
                        *precomputed_dropout.shape[:3],
                        slice_q_len,
                        slice_k_len,
                    ),
                )
                local_bias -= dropout_slice * 1e6
        return local_bias

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    if rescale_logits:
        depth = query.shape[-1]
        query = query / jnp.sqrt(depth).astype(dtype)

    # Casting logits and softmax computation for float32 for model stability.
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    result = _memory_efficient_attention(
        query,
        key,
        value,
        bias_fn,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
        precision=precision,
        dtype=dtype,
        use_extra_logit=use_extra_logit,
        causal_mask=causal_mask,
    )
    result = result.reshape(*orig_batch_dims, *result.shape[1:])
    return result


def dot_product_attention_multiquery(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    broadcast_dropout: bool = True,
    rescale_logits: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    enable_dropout: bool = True,
    dtype: DType = jnp.float32,
    precision: Optional[lax.Precision] = None,
    use_extra_logit: bool = False,
    float32_logits: bool = True,
    causal_mask: bool = False,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 2048,
) -> Array:
    """Computes dot-product multiquery-attention given query, key, and value.

    This is a variant of the multi-head dot product attention introduced in
    https://arxiv.org/abs/1706.03762 and implemented in `dot_product_attention`.
    In this function, the key and the value have 1 head whereas query has 1 or
    more heads. This variant is called "multi-query" attention.

    This implementation is equivalent to
    `dense_attention.dot_product_attention_multiquery`, but is improved by the
    memory-efficient attention algorithm (https://arxiv.org/abs/2112.05682),
    which is also called FlashAttention (https://arxiv.org/abs/2205.14135).

    Note: query, key, value needn't have any batch dimensions.

    Args:
        query: queries for calculating attention with shape of `[batch..., q_length,
            num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of `[batch..., kv_length,
            qk_depth_per_head]`.
        value: values to be used in attention with shape of `[batch..., kv_length,
            v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
            shape `[batch..., num_heads, q_length, kv_length]` This can be used for
            incorporating causal masks, padding masks, proximity bias, etc.
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
        dropout_rng: JAX PRNGKey: to be used for dropout
        dropout_rate: dropout rate
        enable_dropout: bool, whether to apply dropout
        dtype: the dtype of the computation (default: float32)
        precision: numerical precision of the computation see `jax.lax.Precision`
            for details.
        use_extra_logit: whether to include a virtual extra logit equal to zero.
        float32_logits: bool, if True then compute logits in float32 to avoid
            numerical issues with bfloat16.
        causal_mask: Apply a causal mask. This can be used alternatively or in
            addition to the given bias.
        query_chunk_size: Positive integer to control the size of the query chunks.
        key_chunk_size: Positive integer to control the size of the key chunks.

    Returns:
        Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
    """
    # num_heads, treat current num_heads as queries_per_head
    query = jnp.expand_dims(query, axis=-3)
    # add num_heads dim
    key = jnp.expand_dims(key, axis=-2)
    # add num_heads dim
    value = jnp.expand_dims(value, axis=-2)
    if bias is not None:
        # add num_heads, treat current num_heads dim as queries_per_head
        bias = jnp.expand_dims(bias, axis=-4)
    result = dot_product_attention_queries_per_head(
        query,
        key,
        value,
        bias,
        broadcast_dropout=broadcast_dropout,
        rescale_logits=rescale_logits,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        enable_dropout=enable_dropout,
        dtype=dtype,
        precision=precision,
        use_extra_logit=use_extra_logit,
        float32_logits=float32_logits,
        causal_mask=causal_mask,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
    )
    return jnp.squeeze(result, axis=-3)  # remove head dim


def dot_product_attention_multihead(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    broadcast_dropout: bool = True,
    rescale_logits: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    enable_dropout: bool = True,
    dtype: DType = jnp.float32,
    precision: Optional[lax.Precision] = None,
    use_extra_logit: bool = False,
    float32_logits: bool = True,
    causal_mask: bool = False,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 2048,
) -> Array:
    """Computes dot-product multi-head attention given query, key, and value.

    This implementation is equivalent to `dense_attention.dot_product_attention`,
    but is more memory-efficient.

    Note: query, key, value needn't have any batch dimensions.

    Args:
        query: queries for calculating attention with shape of `[batch..., q_length,
            num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of `[batch..., kv_length,
            num_heads, qk_depth_per_head]`.
        value: values to be used in attention with shape of `[batch..., kv_length,
            num_heads, v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
            shape `[batch..., num_heads, q_length, kv_length]` This can be used for
            incorporating causal masks, padding masks, proximity bias, etc.
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
        dropout_rng: JAX PRNGKey: to be used for dropout
        dropout_rate: dropout rate
        enable_dropout: bool, whether to apply dropout
        dtype: the dtype of the computation (default: float32)
        precision: numerical precision of the computation see `jax.lax.Precision`
            for details.
        use_extra_logit: whether to include a virtual extra logit equal to zero.
        float32_logits: bool, if True then compute logits in float32 to avoid
            numerical issues with bfloat16.
        causal_mask: Apply a causal mask. This can be used alternatively or in
            addition to the given bias.
        query_chunk_size: Positive integer to control the size of the query chunks.
        key_chunk_size: Positive integer to control the size of the key chunks.

    Returns:
        Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
    """
    # queries_per_head
    query = jnp.expand_dims(query, axis=-2)
    if bias is not None:
        # add num_heads, treat current num_heads dim as queries_per_head
        bias = jnp.expand_dims(bias, axis=-3)
    result = dot_product_attention_queries_per_head(
        query,
        key,
        value,
        bias,
        broadcast_dropout=broadcast_dropout,
        rescale_logits=rescale_logits,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        enable_dropout=enable_dropout,
        dtype=dtype,
        precision=precision,
        use_extra_logit=use_extra_logit,
        float32_logits=float32_logits,
        causal_mask=causal_mask,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
    )
    return jnp.squeeze(result, axis=-2)  # remove queries_per_head dim
