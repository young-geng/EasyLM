#! /bin/bash

# This is the example script to pretrain a 7B LLaMA model on a TPU v4-512 pod.
# These hyperparameters are the ones we used to train the OpenLLaMA 7B model on
# the RedPajama dataset. To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.

# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY='<your wandb api key here>'

# TPU specific flags to improve training throughput
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'


python -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,64,1' \
    --dtype='fp32' \
    --total_steps=250000 \
    --log_freq=50 \
    --save_model_freq=0 \
    --save_milestone_freq=2500 \
    --load_llama_config='7b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --tokenizer.vocab_file='/path/to/llama/tokenizer/file' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
    --optimizer.adamw_optimizer.lr_decay_steps=250000 \
    --train_dataset.type='json' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.json_dataset.path='/path/to/shuffled/redpajama/dataset' \
    --train_dataset.json_dataset.seq_length=2048 \
    --train_dataset.json_dataset.batch_size=2048 \
    --train_dataset.json_dataset.tokenizer_processes=16 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='EasyLM' \
    --logger.project="open_llama_7b" \
    --logger.output_dir="/path/to/checkpoint/dir" \
    --logger.wandb_dir="$HOME/experiment_output/open_llama_7b" \
|& tee $HOME/output.txt

