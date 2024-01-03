# LLaMA
LLaMA is a language model developed by Meta. The official implementation can
be found [here](https://github.com/facebookresearch/llama). EasyLM provides
a JAX implementation of LLaMA, located at [EasyLM/models/llama](/EasyLM/models/llama).


## Converting the Official LLaMA Checkpoint to EasyLM Format
If you are using our [OpenLLaMA](https://github.com/openlm-research/open_llama),
you can directly download the EasyLM checkpoints and skip this section.
If you are using the official LLaMA weights from Meta, the first step is to
convert the official LLaMA checkpoint to the EasyLM checkpoint format. To do so,
use the following command:

``` shell
python -m EasyLM.models.llama.convert_torch_to_easylm \
    --checkpoint_dir='path/to/torch/llama/checkpoint' \
    --output_file='path/to/output/easylm/checkpoint' \
    --streaming=True
```

This script will convert the official torch checkpoint from Meta to the
streaming checkpoint format used by EasyLM. If you set `--streaming` to `False`,
the script will output a standard flax checkpoint instead. For more information
about the checkpoint format of EasyLM, see [the checkpointing documentation](checkpointing.md).


## Fine-Tuning LLaMA
After converting the checkpoint and setting up the data, you can fine-tune
LLaMA with EasyLM. The training script is implemented in
[EasyLM/models/llama/llama_train.py](/EasyLM/models/llama/llama_train.py).
To fine-tune LLaMA, use the following command:

``` shell
python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --load_llama_config='13b' \
    --load_checkpoint='params::path/to/easylm/llama/checkpoint' \
    ...
```

The following command line options are supported for the training script:
* `seed`: The random seed to use for the training script.
* `initialize_jax_distributed`: Whether to call `jax.distributed.initialize()`.
* `mesh_dim`: The mesh dimensions for the data, fully sharded data, and model parallelism.
  LLaMA uses a 3D mesh so a comma-separated list of 3 values is required. See
  [the parallelism documentation](parallelism.md) for more details.
* `total_steps`: The total number of training steps.
* `load_llama_config`: the LLaMA configuration to use. Can be `1b`, `3b`, `7b`,
  `13b`, `30b` or `65b`.
* `update_llama_config`: A string containing a Python dictionary used to update the
  LLaMA configuration. For example, to set the dropout probability to 0.1, you
  can use the following value
  `{"resid_pdrop": 0.05, "embd_pdrop": 0.05, "attn_pdrop": 0.05}`.
* `load_checkpoint`: The checkpoint to load. See [the checkpointing documentation](checkpointing.md)
  for more details.
* `load_dataset_state`: The dataset state to load. Rarely used.
* `log_freq`: The frequency of logging the training metrics.
* `save_model_freq`: The frequency of saving the model checkpoint. The older
  checkpoints will be overwritten by the newest checkpoint.
* `save_milestone_freq`: The frequency of saving the milestones of the model checkpoint.
  The milestone checkpoints will not be overwritten.
* `eval_steps`: The number of evaluation steps to run to evaluate the model. Setting
  it to 0 will disable the evaluation. Using this requires the `eval_dataset` to be
  properly specified.
* `tokenizer`: The tokenizer configuration.
* `train_dataset`: The training dataset configuration. See [the dataset documentation](dataset.md)
  for more details.
* `eval_dataset`: The evaluation dataset configuration. See [the dataset documentation](dataset.md)
  for more details.
* `optimizer`: The optimizer configuration. See [the optimizer documentation](optimizer.md)
  for more details.
* `checkpointer`: The checkpointer configuration. See [the checkpointing documentation](checkpointing.md)
  for more details.
* `llama`: Manually specify the LLaMA configuration. The available configurations
  can be found in the [LLaMA model implementation](/EasyLM/models/llama/llama_model.py).
* `logger`: The logger configuration. For more details, see [the logger documentation](logger.md).
* `log_all_workers`: Whether to log the metrics from all workers in a multi-host
    setting. If set to `False`, only the metrics from the first worker will be logged.


## Serving LLaMA
You can serve the LLaMA model using the LMServer of EasyLM. To do so, use the
following command:

``` shell
python -m EasyLM.models.llama.llama_serve \
    --mesh_dim='1,1,-1' \
    --load_llama_config='13B' \
    --load_checkpoint='params::path/to/easylm/llama/checkpoint' \
    ...
```

The following command line options are supported for the serving script:
* `seed`: The random seed to use for the serving script.
* `initialize_jax_distributed`: Whether to call `jax.distributed.initialize()`.
* `mesh_dim`: The mesh dimensions for the data, fully sharded data, and model parallelism.
  LLaMA uses a 3D mesh so a comma-separated list of 3 values is required. See
  [the parallelism documentation](parallelism.md) for more details.
* `dtype`: The float dtype (data type) to use for the model. Can be `bf16` or `fp16` or `fp32`.
* `input_length`: The maximum length of the input sequence.
* `seq_length`: The maximum length of the total sequence (input and output).
* `top_k`: The number of top-k candidates to use for the sampling.
* `top_p`: The top-p sampling probability.
* `do_sample`: Thether to use sampling or greedy decoding.
* `num_beams`: The number of beams to use for beam search.
* `add_bos_token`: Whether to add the bos (Beginning Of String/Sentence) token for loglikelihood
  calculation and text generation.
* `load_llama_config`: the LLaMA configuration to use. Can be `1b`, `3b`, `7b`,
  `13b`, `30b` or `65b`.
* `load_checkpoint`: The checkpoint to load. See [the checkpointing documentation](checkpointing.md)
  for more details.
* `tokenizer`: The tokenizer configuration.
* `lm_server`: The LM server configuration. See [the LM server documentation](serving.md)
  for more details.


## LLaMA Tokenizer
LLaMA uses a custom tokenizer that needs to be loaded during training and serving.
Specifically, you need to set the `tokenizer.vocab_file` command line option to
the path of the `tokenizer.model` file that is in the official LLaMA checkpoint.


## Converting the EasyLM LLaMA Checkpoint to a Huggingface LLaMA Checkpoint
To facilitate the interoperability with Huggingface transformers, EasyLM also
provides a script to convert the EasyLM LLaMA checkpoint to the Huggingface
PyTorch LLaMA checkpoint. To do so, use the following command:

``` shell
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='params::path/to/easylm/checkpoint' \
    --tokenizer_path='path/to/llama/tokenizer' \
    --model_size='13b' \  # '7b', '13b', '30b' or '65b' \
    --output_dir='path/to/output/huggingface/llama/checkpoint' \
```
