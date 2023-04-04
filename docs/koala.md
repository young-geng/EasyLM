# Koala
Koala is a language model fine-tuned on top of LLaMA.
[Check out the blogpost!](https://bair.berkeley.edu/blog/2023/04/03/koala/)
This documentation will describe the process of downloading, recovering the
Koala model weights, and running the Koala chatbot locally.


## Obtaining the Wegith Diff of Koala
Due to the licence of the LLaMA model, we cannot directly release the fine-tuned
Koala model weights. Instead, we release the diff of weights, which can be used
to recover the Koala model weights with the origina LLaMA model weights. [The diff
weights can be downloaded here.](https://drive.google.com/drive/folders/10f7wrlAFoPIy-TECHsx9DKIvbQYunCfl?usp=sharing).


## Recovering the Koala Model Weights
The first step of recovering the Koala model weights is to obtain the original
LLaMA model weights and convert it to EasyLM checkpoint format. To convert the weights,
use the following command:

``` shell
python -m EasyLM.models.llama.convert_torch_to_easylm.py \
    --checkpoint_dir='path/to/torch/llama/checkpoint' \
    --output_dir='path/to/output/easylm/checkpoint' \
    --streaming=True
```

This script will convert the official torch checkpoint from Meta to the
streaming checkpoint format used by EasyLM. For more information
about the checkpoint format of EasyLM, see [the checkpointing documentation](checkpointing.md).


After converting the original LLaMA model weights, you can recover the Koala
model weights with the following command:

``` shell
python -m EasyLM.scripts.diff_checkpoint \
    --recover_diff=True \
    --load_base_checkpoint='params::path/to/llama/checkpoint' \
    --load_target_checkpoint='params::path/to/koala/diff/checkpoint' \
    --output_file='path/to/output/checkpoint' \
    --streaming=True
```


## Serving the Koala Chatbot
You can serve the LLaMA model with the LMServer of EasyLM. To do so, use the
following command:

``` shell
python -m EasyLM.models.llama.llama_serve \
    --load_llama_config='13b' \
    --load_checkpoint="params::/path/to/recovered/checkpoint" \
    --tokenizer.vocab_file='/path/to/tokenizer.model' \
    --mp_mesh_dim='-1,1' \
    --dtype='bf16' \
    --input_length=1024 \
    --seq_length=2048 \
    --do_sample=True \
    --lm_server.batch_size=1 \
    --lm_server.port=5009 \
    --lm_server.pre_compile='chat' \
    --lm_server.chat_prepend_text='BEGINNING OF CONVERSATION: ' \
    --lm_server.chat_lm_prefix='GPT:' \
    --lm_server.chat_lm_suffix='</s>' \
    --lm_server.chat_user_prefix='USER: ' \
    --lm_server.chat_user_suffix=' '
```

Then navigate to `http://localhost:5009` to interact with the chatbot.


## Converting the Koala Weights to HuggingFace Transformers
You can also convert the Koala model weights to HuggingFace Transformers format,
so it can be used with the LLaMA implementation in transformers. To do so, use
the following command:

``` shell
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='params::path/to/koala/checkpoint' \
    --tokenizer_path='path/to/llama/tokenizer' \
    --model_size='13b' \  # '7b', '13b', '30b' or '65b'
    --output_dir='path/to/output/huggingface/koala/checkpoint'
```
