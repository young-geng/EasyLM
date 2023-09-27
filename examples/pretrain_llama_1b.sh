#! /bin/bash

# This is the example script to pretrain a 7B LLaMA model on a TPU v4-512 pod.
# These hyperparameters are the ones we used to train the OpenLLaMA 7B model on
# the RedPajama dataset. To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.

# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY='377cc158068c657cbce73df22cbf0965f94f4a5e'

# TPU specific flags to improve training throughput
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'
# --jax_enable_async_collective_offload=true
# --xla_enable_async_all_gather=true


python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --total_steps=500 \
    --log_freq=50 \
    --save_model_freq=0 \
    --save_milestone_freq=250 \
    --load_llama_config='1b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --tokenizer.vocab_file='/mnt/disks/persist/pphuc/EasyLM/model_pretrained/llama/spiece.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=20 \
    --optimizer.adamw_optimizer.lr_decay_steps=500 \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.huggingface_dataset.path='uonlp/CulturaX' \
    --train_dataset.huggingface_dataset.name='vi' \
    --train_dataset.huggingface_dataset.num_proc=60 \
    --train_dataset.huggingface_dataset.batch_size=16 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='EasyLM' \
    --logger.project="open_llama_1b" \
    --logger.output_dir="/mnt/disks/persist/pphuc/EasyLM/model_pretrained/llama/output_dir" \
    --logger.wandb_dir="$HOME/experiment_output/open_llama_1b" \
|& tee $HOME/output.txt

    # --train_dataset.huggingface_dataset.seq_length=2048 \
    # --train_dataset.json_dataset.tokenizer_processes=16 \
