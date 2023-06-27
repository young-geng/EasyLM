# Model, Data and Fully Sharded Data Parallelism
EasyLM supports flexible model and data parallelism for training and serving
large language models. Specifically, EasyLM uses the PJIT feature of JAX
to parallelize the computation across multiple of accelerators or multiple hosts.
To do so, all the accelerators are first grouped into a multi-dimensional mesh,
where each axis represents a different type of parallelism. Currently, EasyLM
uses 3D meshes for most of the models. Typically, the first axis of the mesh is
used for data parallelism, the second axis used for fully sharded data
parallelism (FSDP), and the third axis is used for model parallelism.
For more information about FSDP, please refer
to [this FSDP tutorial](https://engineering.fb.com/2021/07/15/open-source/fsdp/).

For example, if we have 8 accelerators for each host and 32 hosts in total,
this gives us a total of 256 accelerators. We can use a 3D mesh of shape
(1, 8, 32) to specify that one model is partitioned across 32 accelerators for
model parallelism, and each parition has 8 replicas for fully sharded data parallelism.

## Specifying the Mesh Axis Dimensions
While the multi-dimensional mesh parallelism is not very intuitive, EasyLM hides
most of the complexity from the user. For most use cases, the user only needs
to specify the parallelism axis dimensions based on the memory capacity and the
compute performance of the accelerators used. Typically, this is done by passing
the `mesh_dim` command line argument to the training or serving script. The
`mesh_dim` is a comma separated list of integers representing the parallelism
mesh axis dimensions. One of the axis dimensions can be `-1`, which means that
the axis dimension will be inferred based on the total number of accelerators.

For example, if we want to train a LLaMA model on 8 accelerators,
we can pass in the following option for `mesh_dim`:
``` shell
python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,8,1' \
    ...
```

This specifies that the model is paritioned across 8 accelerators for FSDP. Note that
we can use `-1` for one of the axis dimensions, which means that the axis dimension
will be inferred based on the total number of accelerators. For example, on a 8
accelerator machine, specifying `1,1,-1` for `mesh_dim` is equivalent to
specifying `1,1,8`.


## Tuning the Parallelism Axis Dimensions
The parallelism axis dimensions can be tuned to achieve the best performance.
Generally, it is recommended to use larger FSDP axis and a small model parallelism
axis to achieve the best throughput.