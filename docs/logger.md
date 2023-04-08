# Logger
EasyLM uses the [MLXU](https://github.com/young-geng/mlxu) library for logging.
Specifically, EasyLM uses the `mlxu.WandBLogger` module for logging. The `WandBLogger`
module is a wrapper of the [Weights & Biases](https://wandb.ai/site) library. The
following options are available for configuring the `WandBLogger` module:
* `online`: Whether to log online. If `False`, the logger will not upload to W&B service.
* `prefix`: The prefix of the W&B project name.
* `project`: The W&B project name.
* `output_dir`: The output directory for checkpointing, can be a local directory or a
  Google Cloud Storage directory.
* `wandb_dir`: The output directory for W&B logs. Must be a local directory.
* `random_delay`: Whether to add a random delay to the logging process.
* `experiment_id`: The experiment ID. If not specified, a random ID will be generated.
* `anonymous`: Whether to log anonymously.
* `notes`: The notes for the experiment.
* `entity`: The W&B entity name.
* `prefix_to_id`: Whether to add a prefix to the experiment ID.

For more information about the logger configuration, please refer to the
[MLXU](https://github.com/young-geng/mlxu) library.
