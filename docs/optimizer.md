# Optimizers
EasyLM provides a number of optimizers for training neural language models. The
optimizers are implemented in the [optimizer.py](/EasyLM/optimizer.py)

Currently, the following optimizers are supported:
* AdamW
* PaLM: the optimizer described in the PaLM paper

In addition to optimizer configurations, the optimizer module also provides
support for gradient accumulation.


## Selecting Optimizer and Gradient Accumulation
Optimizer type can be selected by setting the `type` field in the optimizer
configuration. For example, to use the AdamW optimizer, we can set the `type` to
`adamw` and configuring the `adamw_optimizer` subfields:
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
* `type`: the optimizer type. Currently, `adamw` and `palm` are supported.
* `adamw_optimizer`: the configuration for the AdamW optimizer
* `palm_optimizer`: the configuration for the PaLM optimizer
* `accumulate_gradient_steps`: the number of steps for gradient accumulation


## AdamW Optimizer
The AdamW optimizer implements AdamW with liear learning rate warmup and cosine
learning rate decay. The following options are supported for the AdamW optimizer:
* `init_lr`: the initial learning rate
* `end_lr`: the final learning rate after decay
* `lr`: the peak learning rate
* `lr_warmup_steps`: the number of steps for linear learning rate warmup
* `lr_decay_steps`: the number of steps for cosine learning rate decay
* `b1`: the beta1 parameter for AdamW
* `b2`: the beta2 parameter for AdamW
* `clip_gradient`: the gradient clipping threshold
* `weight_decay`: the weight decay parameter for AdamW
* `bf16_momentum`: whether to use bf16 for momentum to save memory
* `multiply_by_parameter_scale`: whether to multiply the gradient by parameter scale (as in adafactor)


## PaLM Optimizer
The PaLM optimizer implements the optimizer described in the PaLM paper. The optimizer
is essential adafactor with no-factoring and a inverse square root learning rate
decay and weight decay schedule. The following options are supported for the PaLM optimizer:
* `lr`: the initial learning rate
* `lr_warmup_steps`: the number of steps for constant learning rate warmup
* `b1`: beta1 parameter for adafactor
* `b2`: beta2 parameter for adafactor
* `clip_gradient`: the gradient clipping threshold
* `weight_decay`: the weight decay parameter
* `bf16_momentum`: whether to use bf16 for momentum to save memory





