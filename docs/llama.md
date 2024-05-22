# LLaMA
LLaMA is a language model developed by Meta. The official implementation can
be found [here](https://github.com/facebookresearch/llama). EasyLM provides
a JAX implementation of LLaMA, located at [EasyLM/models/llama](/EasyLM/models/llama).


## Converting the Official LLaMA Checkpoint to EasyLM Format
If you are using our [OpenLLaMA](https://github.com/openlm-research/open_llama),
you can directly download the EasyLM checkpoints and skip this section.
If you are using the official LLaMA weights from Meta, the first step of is to
convert the Huggingface transformers LLaMA checkpoint to the EasyLM checkpoint format. To do so,
use the following command:

``` shell
python -m EasyLM.models.llama.convert_hf_to_easylm \
    --hf_model='path/to/transformers/llama/checkpoint' \
    --output_file='path/to/output/easylm/checkpoint' \
    --streaming=True \
    --llama.base_model='llama_7b'
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
    --llama.base_model='llama_7b' \
    --load_checkpoint='params::path/to/easylm/llama/checkpoint' \
    ...
```

The following command line options are supported for the training script:
* `seed`: The random seed to use for the training script.
* `mesh_dim`: The mesh dimensions for the data, fully sharded data and model parallelism.
  LLaMA uses 3D mesh so a comma separated list of 3 values are required. See
  [the parallelism documentation](parallelism.md) for more details.
* `dtype`: the float dtype to use for the model activation. Can be `bf16` or `fp16` or `fp32`.
* 'params_dtype': the float dtype to use for the model parameters. Can be `bf16` or `fp16` or `fp32`.
* `total_steps`: The total number of training steps.
* `load_checkpoint`: the checkpoint to load. See [the checkpointing documentation](checkpointing.md)
  for more details.
* `load_dataset_state`: the dataset state to load. Rarely used.
* `log_freq`: the frequency of logging the training metrics.
* `save_model_freq`: the frequency of saving the model checkpoint. The older
  checkpoints will be overwritten by the newest checkpoint.
* `save_milestone_freq`: the frequency of saving the milestones of model checkpoint.
  The milestone checkpoints will not be overwritten.
* `eval_steps`: the number of evaluation steps to run to evaluate the model. Setting
  to 0 will disable the evaluation. Using this requires the `eval_dataset` to be
  properly specified.
* `tokenizer`: Huggingface transformers pretrained tokenizer.
* `train_dataset`: training dataset configuration. See [the dataset documentation](dataset.md)
  for more details.
* `eval_dataset`: evaluation dataset configuration. See [the dataset documentation](dataset.md)
  for more details.
* `optimizer`: optimizer configuration. See [the optimizer documentation](optimizer.md)
  for more details.
* `checkpointer`: checkpointer configuration. See [the checkpointing documentation](checkpointing.md)
  for more details.
* `llama`: Specify the LLaMA configuration by starting from a base model. The avaiable configurations
  can be found in the [LLaMA model implementation](/EasyLM/models/llama/llama_model.py).
* `logger`: logger configuration. For more details, see [the logger documentation](logger.md).
* `log_all_workers`: whether to log the metrics from all workers in a multi-host
    setting. If set to `False`, only the metrics from the first worker will be logged.
* `jax_distributed`: JAX distributed configuration. This only needs to be set when running
  multi-host training on GPU.


## Serving LLaMA
You can serve the LLaMA model with the LMServer of EasyLM. To do so, use the
following command:

``` shell
python -m EasyLM.models.llama.llama_serve \
    --mesh_dim='1,1,-1' \
    --llama.base_model='llama_7b' \
    --load_checkpoint='params::path/to/easylm/llama/checkpoint' \
    ...
```

The following command line options are supported for the serving script:
* `seed`: The random seed to use for the serving script.
* `mesh_dim`: The mesh dimensions for the data, fully sharded data and model parallelism.
  LLaMA uses 3D mesh so a comma separated list of 3 values are required. See
  [the parallelism documentation](parallelism.md) for more details.
* `dtype`: the float dtype to use for the model activation. Can be `bf16` or `fp16` or `fp32`.
* `params_dtype`: the float dtype to use for the model parameters. Can be `bf16` or `fp16` or `fp32`.
* `input_length`: the maximum length of the input sequence.
* `seq_length`: the maximum length of the total sequence (input and output).
* `top_k`: the number of top-k candidates to use for the sampling.
* `top_p`: the top-p sampling probability.
* `do_sample`: whether to use sampling or greedy decoding.
* `num_beams`: the number of beams to use for beam search.
* `add_bos_token`: whether to add the bos token for loglikelihood
  calculation and text generation.
* `llama`: the LLaMA configuration to use.
* `load_checkpoint`: the checkpoint to load. See [the checkpointing documentation](checkpointing.md)
  for more details.
* `tokenizer`: Huggingface transformers pretrained tokenizer.
* `lm_server`: the LM server configuration. See [the LM server documentation](serving.md)
  for more details.
* `jax_distributed`: JAX distributed configuration. This only needs to be set when running
  multi-host training on GPU.

## Converting the EasyLM LLaMA Checkpoint to Huggingface LLaMA Checkpoint
To facilitate the interoperability with Huggingface transformers, EasyLM also
provides a script to convert the EasyLM LLaMA checkpoint to the Huggingface
Pytorch LLaMA checkpoint. To do so, use the following command:

``` shell
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='params::path/to/easylm/checkpoint' \
    --output_dir='path/to/output/huggingface/llama/checkpoint' \
    --llama.base_model='llama_7b'
```
