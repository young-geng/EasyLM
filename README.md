# EasyLM
Large language models (LLMs) made easy, EasyLM is a one-stop solution for
pre-training, fine-tuning, evaluating, and serving LLMs in JAX/Flax. EasyLM can
scale up LLM training to hundreds of TPU/GPU accelerators by leveraging
JAX's PJIT functionality.


Building on top of Hugginface's [transformers](https://huggingface.co/docs/transformers/main/en/index)
and [datasets](https://huggingface.co/docs/datasets/index), this repo provides
an easy-to-use and easy-to-customize codebase for training large language models
without the complexity seen in many other frameworks.


EasyLM is built with JAX/Flax. By leveraging JAX's PJIT utility, EasyLM is able
to train large models that don't fit on a single accelerator by sharding
the model weights and training data across multiple accelerators. Currently,
EasyLM supports multiple TPU/GPU training in a single host as well as multi-host
training on Google Cloud TPU Pods.

Currently, the following models are supported:
* [LLaMA](https://arxiv.org/abs/2302.13971)
* [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)
* [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)

## Discord Server
We are running an unofficial Discord community (unaffiliated with Google) for discussion related to training LLMs in JAX. [Follow this link to join the Discord server](https://discord.gg/Rf4drG3Bhp). We have dedicated channels for several JAX-based LLM frameworks, including EasyLM, [JaxSeq](https://github.com/Sea-Snell/JAXSeq), [Alpa](https://github.com/alpa-projects/alpa) and [Levanter](https://github.com/stanford-crfm/levanter).


## Models Trained with EasyLM
### OpenLLaMA
OpenLLaMA is our permissively licensed reproduction of LLaMA which can be used
for commercial purposes. Check out the [project main page here](https://github.com/openlm-research/open_llama).
OpenLLaMA can serve as a drop-in replacement for the LLaMA weights in EasyLM.
Please refer to the [LLaMA documentation](docs/llama.md) for more details.


### Koala
Koala is our new chatbot fine-tuned on top of LLaMA. If you are interested in
our Koala chatbot, you can check out the [blog post](https://bair.berkeley.edu/blog/2023/04/03/koala/)
and [documentation for running it locally](docs/koala.md).


## Installation
The installation method differs between GPU hosts and Cloud TPU hosts. The first
step is to pull this repository from GitHub.

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


## [Documentations](docs/README.md)
EasyLM's documentation can be found in the [docs](docs/) directory.


## Reference
If you found EasyLM useful in your research or applications, please cite it using the following BibTeX:
```
@software{geng2023easylm,
  author = {Geng, Xinyang},
  title = {EasyLM: A Simple And Scalable Training Framework for Large Language Models},
  month = March,
  year = 2023,
  url = {https://github.com/young-geng/EasyLM}
}
```



## Credits
* The LLaMA implementation is from [JAX_llama](https://github.com/Sea-Snell/JAX_llama)
* The JAX/Flax GPT-J and RoBERTa implementation is from [transformers](https://huggingface.co/docs/transformers/main/en/index)
* Most of the JAX utilities are from [mlxu](https://github.com/young-geng/mlxu)
* The codebase is heavily inspired by [JAXSeq](https://github.com/Sea-Snell/JAXSeq)
