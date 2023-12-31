# Optimizers
EasyLM provides several optimizers for training language models. The
optimizers are implemented in the [optimizer.py](/EasyLM/optimizer.py)

Currently, the following optimizers are supported:
* AdamW: The optimizer described in the [Decoupled Weight Decay Regularization paper](https://arxiv.org/abs/1711.05101)
* PaLM: the optimizer described in the [PaLM: Scaling Language Modeling with Pathways paper](https://arxiv.org/abs/2204.02311)

In addition to optimizer configurations, the optimizer module also provides
support for gradient accumulation.


## Selecting Optimizer and Gradient Accumulation
The optimizer type can be selected by setting the `type` field in the optimizer
configuration. For example, to use the AdamW optimizer, we can set the `type` to
`adamw` and configure the `adamw_optimizer` subfields. Here's an example:
```shell
python train.py --optimizer.type=adamw --optimizer.adamw_optimizer.lr=1e-4
```

To use gradient accumulation, we can set the `accumulate_gradient_steps` field
in the optimizer configuration. For example, to use gradient accumulation with
step size 2, we can set the `accumulate_gradient_steps` to 2:
```shell
python train.py --optimizer.accumulate_gradient_steps=2
```

The following options are supported for the optimizer module:
* `type`: The optimizer type. Currently, only `adamw` and `palm` are supported.
* `adamw_optimizer`: The configuration for the AdamW optimizer
* `palm_optimizer`: The configuration for the PaLM optimizer
* `accumulate_gradient_steps`: The number of steps for gradient accumulation


## AdamW Optimizer
The AdamW optimizer implements AdamW with a linear learning rate warmup and a cosine
learning rate decay. The following options are supported for the AdamW optimizer:
* `init_lr`: The initial learning rate
* `end_lr`: The final learning rate after decay
* `lr`: The peak learning rate
* `lr_warmup_steps`: The number of steps for linear learning rate warmup
* `lr_decay_steps`: The number of steps for cosine learning rate decay
* `b1`: The beta1 parameter for AdamW
* `b2`: The beta2 parameter for AdamW
* `clip_gradient`: The gradient clipping threshold
* `weight_decay`: The weight decay parameter for AdamW
* `bf16_momentum`: Whether to use bf16 for momentum to save memory and speed up training
* `multiply_by_parameter_scale`: Whether to multiply the gradient by the parameters scale (as in Adafactor)


## PaLM Optimizer
The PaLM optimizer implements the optimizer described in the [PaLM: Scaling Language Modeling with Pathways paper](https://arxiv.org/abs/2204.02311). The optimizer
is essentially Adafactor with no factoring and an inverse square root learning rate
decay and weight decay schedule. The following options are supported for the PaLM optimizer:
* `lr`: The initial learning rate
* `lr_warmup_steps`: The number of steps for constant learning rate warmup
* `b1`: The beta1 parameter for Adafactor
* `b2`: The beta2 parameter for Adafactor
* `clip_gradient`: The gradient clipping threshold
* `weight_decay`: The weight decay parameter
* `bf16_momentum`: Whether to use bf16 for momentum to save memory and speed up training





