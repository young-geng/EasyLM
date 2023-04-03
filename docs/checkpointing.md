# Checkpointing
To facilitate training very large language models that does not fit into the
main memory of a single machine, EasyLM adopt a streaming format of model
checkpoint. The streaming checkpointing format is implemented in
[checkpoint.py](/EasyLM/checkpoint.py). During checkpointing, the
StreamingCheckpointer simply flatten a nested state dictionary into a single
level dictionary, and stream the key, value pairs to a file one by one using
messagepack. Because it streams the tensors one by one, the checkpointer only
needs to gather one tensor from the distributed accelerators to the main memory
at a time, hence saving a lot of memory.


## Loading Checkpoint
While EasyLM mainly uses the streaming checkpointing format, it also supports
directly loading the standard flax checkpoint file created using
`flax.training.checkpoints.save_checkpoint`. The loading format can be specified
as part of the path passed into the training or serving script. For example, if
we want to serve a LLaMA model using the streaming checkpointing format, we can
use the following command:

``` shell
python -m EasyLM.models.llama.llama_serve \
    --load_checkpoint='params::path/to/checkpoint'
    ...
```

Note that the `params::` prefix is used to specify that the checkpoint is in
streaming format. The following prefix are supported for loading checkpoint:
* `params::`: Streaming checkpointing format.
* `flax::`: Standard flax checkpointing format.
* `trainstate::`: Loading an entire train state with optimizer state, this
    option is only supported for training script.
* `trainstate_params::`: Loading the params part from the entire train state.

By default, EasyLM does not save the optimizer state in the checkpoint, so
we will rarely need to use the `trainstate::` or `trainstate_params::` options.


## Saving Checkpoint
EasyLM will only save the checkpoint in the streaming format. By default, only
the model parameters are saved in the checkpoint file in the bfloat16 data type.
To configure the checkpointing behavior, you can use the following options:
* `float_dtype`: The float data type of the model parameters in the checkpoint file.
    The default value is `bf16`, other supported values are `fp32` and `fp16`.
* `save_optimizer_state`: Whether to save the entire train state in the checkpoint

Typically, we pass these optiosn into the training script. For example, for
LLaMA, we can use the following command to save the checkpoint in the fp32 data:
``` shell
python -m EasyLM.models.llama.llama_train \
    --checkpointer.float_dtype='fp32' \
    ...
```


## Converting Checkpoint to and from Standard Flax Format
To facilitate the use of EasyLM trained models with other Flax based libraries,
EasyLM provides a script to convert between the streaming checkpointing format
and the standard flax checkpointing format. The script can be found at
[EasyLM/scripts/convert_checkpoint.py](/EasyLM/scripts/convert_checkpoint.py).

To convert a checkpoint from the streaming format to the standard flax format,
use the following command:

``` shell
python -m EasyLM.scripts.convert_checkpoint \
    --load_checkpoint='params::path/to/checkpoint' \
    --output_file='path/to/output/checkpoint' \
    --streaming=False
```

To convert a standard flax checkpoint to the streaming format, use the following
command:

``` shell
python -m EasyLM.scripts.convert_checkpoint \
    --load_checkpoint='flax::path/to/checkpoint' \
    --output_file='path/to/output/checkpoint' \
    --streaming=True
```


## Diffing Checkpoint
To facilitate the release of fine-tuned model checkpoints that's based on
a non-public base model checkpoint, EasyLM provides a script to compute the
difference between two checkpoints. The script can be found at
[EasyLM/scripts/diff_checkpoint.py](/EasyLM/scripts/diff_checkpoint.py).

To compute the difference between a based checkpoint (based model) and a
target checkpoint (fine-tuned model), use the following command:

``` shell
python -m EasyLM.scripts.diff_checkpoint \
    --recover_diff=False \
    --load_base_checkpoint='params::path/to/based/checkpoint' \
    --load_target_checkpoint='params::path/to/target/checkpoint' \
    --output_file='path/to/output/checkpoint' \
    --streaming=True
```

The script will output a checkpoint that contains the difference between the
two checkpoints. You can use the `--streaming` flag to specify the format
(streaming or standard flax) of the output checkpoint. To recover a checkpoint
from a based checkpoint and a diff checkpoint, use the following command:

``` shell
python -m EasyLM.scripts.diff_checkpoint \
    --recover_diff=True \
    --load_base_checkpoint='params::path/to/base/checkpoint' \
    --load_target_checkpoint='params::path/to/diff/checkpoint' \
    --output_file='path/to/output/checkpoint' \
    --streaming=True
```
