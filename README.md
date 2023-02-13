# EasyLM
Easy to use model parallel large language models training and evaluation in JAX/Flax with pjit
support on cloud TPU pods.

Building on top of Hugginface's [transformers](https://huggingface.co/docs/transformers/main/en/index)
and [datasets](https://huggingface.co/docs/datasets/index), this repo provides
an easy to use and easy to customize codebase for training large langauge models
without the complexity in many other frameworks.


EasyLM is built with JAX/Flax. By leveraging JAX's pjit utility, EasyLM is able
to train large model that doesn't fit on a single accelerator by sharding
the model weights and training data across multiple accelerators. Currently,
EasyLM supports multiple TPU/GPU training in a single host as well as multi-host
training on Google Cloud TPU Pods.

Currently, the following models are supported:
* [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B), with support for
[Forgetful Causal Masking (FCM)](https://arxiv.org/abs/2210.13432)
* [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)


## Installation
The installation method differs between GPU hosts and Cloud TPU hosts. The first
step is to pull from GitHub.

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


## Training the Models
Model training can be launched via the python scripts in the main directory. For
example, to laucnh GPT-J training, the following command can be used:
``` shell
python -m EasyLM.models.gptj.gptj_train
```
For Cloud TPU Pods, the same training command needs to be invoked on each host
in the pod.

## Serving and Evaluating Pre-trained Models
Pretrained langauge models can be served as an HTTP server and evaluated using
(lm-eval-harness)[https://github.com/EleutherAI/lm-evaluation-harness]. Use the
following command to launch the HTTP server for GPT-J with pretrained weights
from Huggingface transformers:

```shell
python -m EasyLM.models.gptj.gptj_serve \
    --load_gptj_config='huggingface::EleutherAI/gpt-j-6B' \
    --load_checkpoint='huggingface::EleutherAI/gpt-j-6B' \
    --dtype='bf16' \
    --input_length=512 \
    --seq_length=2048 \
    --lm_server.host='127.0.0.1' \
    --lm_server.pre_compile='loglikelihood' \
    --lm_server.batch_size=2
```

Use the following command to evaluate the the langauge model served with the
HTTP server:
```shell
python -m EasyLM.scripts.lm_eval \
    --lm_server_url='http://localhost:5007/' \
    --tasks='wsc,piqa,winogrande,openbookqa,logiqa' \
    --shots=0
```

You can change the number of shots and the list of tasks.


## Credits
* The JAX/Flax GPT-J and RoBERTa implementation are from [transformers](https://huggingface.co/docs/transformers/main/en/index)
* Most of the utilities are from [m3ae_public](https://github.com/young-geng/m3ae_public)
* The codebase is heavily inspired by [JAXSeq](https://github.com/Sea-Snell/JAXSeq)
