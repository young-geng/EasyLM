"""
Usage:
python convert_hf_to_easylm.py  \
       --hf_model     /path/hf_format_dir    \
       --output_file /path/easylm_format.easylm   \
       --llama.base_model llama_7b \
       --streaming
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import time
from pathlib import Path

import mlxu
import torch
import flax
from transformers import AutoModelForCausalLM

from EasyLM.models.llama.llama_model import LLaMAConfigurator
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.jax_utils import get_float_dtype_by_name


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    hf_model="",
    output_file="",
    streaming=True,
    float_dtype="bf16",
    llama=LLaMAConfigurator.get_default_config(),
)


def inverse_permute(w, n_heads, input_dim, output_dim):
    reshaped_w = w.reshape(n_heads, 2, output_dim // n_heads // 2, input_dim)
    transposed_w = reshaped_w.transpose(0, 2, 1, 3)
    inverted_w = transposed_w.reshape(output_dim, input_dim)
    return inverted_w


def main(argv):
    start = time.time()
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)
    hf_model = AutoModelForCausalLM.from_pretrained(FLAGS.hf_model)
    ckpt = hf_model.state_dict()

    print(f"Start convert weight to easylm format...")
    jax_weights = {
        "transformer": {
            "wte": {"embedding": ckpt["model.embed_tokens.weight"].numpy()},
            "ln_f": {"kernel": ckpt["model.norm.weight"].numpy()},
            "h": {
                "%d"
                % (layer): {
                    "attention": {
                        "wq": {
                            "kernel": inverse_permute(
                                ckpt[f"model.layers.{layer}.self_attn.q_proj.weight"].numpy(),
                                llama_config.num_attention_heads,
                                llama_config.hidden_size,
                                llama_config.hidden_size,
                            ).transpose()
                        },
                        "wk": {
                            "kernel": inverse_permute(
                                ckpt[f"model.layers.{layer}.self_attn.k_proj.weight"].numpy(),
                                llama_config.num_key_value_heads,
                                llama_config.hidden_size,
                                llama_config.hidden_size // (
                                    llama_config.num_attention_heads
                                    // llama_config.num_key_value_heads
                                ),
                            ).transpose()
                        },
                        "wv": {
                            "kernel": ckpt[f"model.layers.{layer}.self_attn.v_proj.weight"]
                            .numpy().transpose()
                        },
                        "wo": {
                            "kernel": ckpt[f"model.layers.{layer}.self_attn.o_proj.weight"]
                            .numpy().transpose()
                        },
                    },
                    "feed_forward": {
                        "w1": {
                            "kernel": ckpt[f"model.layers.{layer}.mlp.gate_proj.weight"]
                            .numpy().transpose()
                        },
                        "w2": {
                            "kernel": ckpt[f"model.layers.{layer}.mlp.down_proj.weight"]
                            .numpy().transpose()
                        },
                        "w3": {
                            "kernel": ckpt[f"model.layers.{layer}.mlp.up_proj.weight"]
                            .numpy().transpose()
                        },
                    },
                    "attention_norm": {
                        "kernel": ckpt[f"model.layers.{layer}.input_layernorm.weight"].numpy()
                    },
                    "ffn_norm": {
                        "kernel": ckpt[
                            f"model.layers.{layer}.post_attention_layernorm.weight"
                        ].numpy()
                    },
                }
                for layer in range(llama_config.num_hidden_layers)
            },
        },
        "lm_head": {"kernel": ckpt["lm_head.weight"].numpy().transpose()},
    }
    print(f"Convert weight to easylm format finished...")
    print(f"Start to save...")

    if FLAGS.streaming:
        StreamingCheckpointer.save_train_state_to_file(
            jax_weights,
            FLAGS.output_file,
            float_dtype=get_float_dtype_by_name(FLAGS.float_dtype),
        )
    else:
        with mlxu.open_file(FLAGS.output_file, "wb") as fout:
            fout.write(flax.serialization.msgpack_serialize(jax_weights, in_place=True))

    print(
        f"Save finished!!! take time: {time.time() - start} save path: {FLAGS.output_file}"
    )


if __name__ == "__main__":
    mlxu.run(main)
