# EasyLM Documentations
EasyLM is a framework for pre-training, fine-tuning, and evaluating and serving
large language models in JAX/Flax. EasyLM is designed to be easy to use by
hiding the complexities of distributed model/data parallelism but exposing the
core training and inference details of large language models, making it easy to
customize. EasyLM can scale up LLM training to hundreds of TPU/GPU accelerators
without the need to write complicated distributed training code.

## Installation
EasyLM supports both GPU and TPU training. The installation method differs by
the type of accelerator. The first step is to pull from GitHub.

``` shell
git clone https://github.com/young-geng/EasyLM.git
cd EasyLM
export PYTHONPATH="${PWD}:$PYTHONPATH"
```

#### Installing on GPU Host
The GPU environment can be installed via [Anaconda](https://www.anaconda.com/products/distribution).

``` shell
conda env create -f scripts/gpu_environment.yml
conda activate EasyLM
```

#### Installing on Cloud TPU Host
The TPU host VM comes with Python and PIP pre-installed. Simply run the following
script to set up the TPU host.

``` shell
./scripts/tpu_vm_setup.sh
```


## Modular Configuration
EasyLM is designed to be highly modular. Typically, the training or inference
script will combine various modules to form a complete training or
inference process. Building on top of [MLXU](https://github.com/young-geng/mlxu),
EasyLM training or inference scripts can directly plug in the configuration of
used modules into the command line flags of the script.

For example, if we have a training script `train.py` that uses the optimizer module,
we can directly plug in the configuration of the optimizer module into the FLAGS
of the training script in this way:

``` python
import mlxu
from EasyLM.optimizer import OptimizerFactory

# Defining the command line flags, flag data type will be inferred from the default value
FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,  # Defining a integer flag with default value 42
    optimizer=OptimizerFactory.get_default_config(), # Plugging in the default configuration of the optimizer module
)

def main(argv):
    seed = FLAGS.seed
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)
    ...

if __name__ == "__main__":
    mlxu.run(main)

```


By plugging in the configuration of the optimizer module into the FLAGS of the
training script, we can directly control the optimizer module from the command
line. For example, if we want to use the AdamW optimizer with learning rate 1e-4,
we can run the training script with the following command:

``` shell
python train.py \
    --seed=42 \
    --optimizer.type=adamw \
    --optimizer.adamw_optimizer.lr=1e-4
```

For more information about the configuration of each module, please refer to the
`get_default_config()` method of the module.


## Documentations for EasyLM Modules and Scripts
Here are the documentations for the common modules and scripts in EasyLM:
* [Parallelism](parallelism.md): model and data parallelism in EasyLM
* [Dataset](dataset.md): data loading and processing
* [Optimizer](optimizer.md): optimizer and gradient accumulation
* [Checkpointing](checkpointing.md): checkpointing
* [Serving](serving.md): serving the language model with an HTTP server
* [Logger](logger.md): logging metrics for training
* [Evaluation](evaluation.md): evaluation of language models on benchmarks



## Documentations for Language Models Supported by EasyLM
Currently, the following models are supported:
* [LLaMA](llama.md)
* GPT-J
* OPT
* RoBERTa


## Additional Examples and Tutorials
* [Running Koala locally](koala.md)