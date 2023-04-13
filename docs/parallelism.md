# Model and Data Parallelism of EasyLM
EasyLM supports flexible model and data parallelism for training and serving
large language models. Specifically, EasyLM uses the PJIT feature of JAX to
to parallelize the computation across multiple of accelerators or multiple hosts.
To do so, all the accelerators are first grouped into a multi-dimensional mesh,
where each axis represents a different type of parallelism. Currently, EasyLM
uses 2D or 3D meshes. The first axis of the mesh is used for data parallelism,
and the rest of the axes are used for model parallelism.

For example, if we have 8 accelerators for each host and 32 hosts in total,
this gives us a total of 256 accelerators. We can use a 2D mesh of shape
(8, 32) to specify that one model is partitioned across 32 accelerators for
model parallelism, and each parition has 8 replicas for data parallelism.

## Specifying the Mesh Axis Dimensions
While the multi-dimensional mesh parallelism is not very intuitive, EasyLM hides
most of the complexity from the user. For most use cases, the user only needs
to specify the parallelism axis dimensions based on the memory capacity and the
compute performance of the accelerators used. Typically, this is done by passing
the `mp_mesh_dim` command line argument to the training or serving script. The
`mp_mesh_dim` is a comma separated list of integers representing only the model
parallelism axis dimensions. The data parallelism axis dimensions are automatically
inferred based on the number of accelerators.

For example, if we want to train a LLaMA model, which uses 3D mesh, on 8 accelerators,
we can pass in the following option for `mp_mesh_dim`:
``` shell
python -m EasyLM.models.llama.llama_train \
    --mp_mesh_dim='8,1' \
    ...
```

This specifies that the model is partitioned across 8 accelerators along the first
model parallelism axis, and we only have 1 replica for data parallelism. Note that
we can use `-1` for one of the axis dimensions, which means that the axis dimension
will be the total number of accelerators. For example, on a 8 accelerator machine,
specifying `-1,1` for `mp_mesh_dim` is equivalent to specifying `8,1`.


## Fully Sharded Data Parallelism
Some models in EasyLM support fully sharded data parallelism, which can further
reduce the memory footprint by sharding the model parameters also along the
data parallelism axis. This is done by setting the `fsdp` option to `True` in
the training or serving script. For more information about FSDP, please refer
to [this FSDP tutorial](https://engineering.fb.com/2021/07/15/open-source/fsdp/).