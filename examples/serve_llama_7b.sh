#! /bin/bash

# This is the example script to serve a 7B LLaMA model on a GPU machine or
# single TPU v3-8 VM. The server will be listening on port 35009.


python -m EasyLM.models.llama.llama_serve \
    --load_llama_config='7b' \
    --load_checkpoint="params::/path/to/checkpoint/file" \
    --tokenizer.vocab_file='/path/to/llama/tokenizer/vocab/file' \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --input_length=1024 \
    --seq_length=2048 \
    --lm_server.batch_size=4 \
    --lm_server.port=35009 \
    --lm_server.pre_compile='all'

