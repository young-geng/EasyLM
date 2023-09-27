python3 -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='params::/home/pphuc/code/EasyLM/model_pretrained/llama/output_dir/0c3be9e05d4643ecb54b4b5cebb6b3bd/streaming_train_state_500' \
    --tokenizer_path='/home/pphuc/code/EasyLM/model_pretrained/llama/tokenizer.json' \
    --model_size='1b' \
    --output_dir='/home/pphuc/code/EasyLM/model_pretrained/llama/converted'