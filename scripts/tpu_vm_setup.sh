#! /bin/bash

sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python-is-python3 \
    tmux \
    htop \
    git \
    nodejs \
    bmon \
    p7zip-full \
    nfs-common


# Python dependencies
cat > $HOME/tpu_requirements.txt <<- EndOfFile
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]==0.4.13
tensorflow==2.11.0
flax==0.7.0
optax==0.1.7
distrax==0.1.3
chex==0.1.7
einops
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.0.1
transformers==4.31.0
datasets==2.14.2
huggingface_hub==0.16.4
tqdm
h5py
ml_collections
wandb==0.13.5
gcsfs==2022.11.0
requests
typing-extensions
lm-eval==0.3.0
mlxu==0.1.11
sentencepiece
pydantic
fastapi
uvicorn
gradio
EndOfFile

pip install --upgrade -r $HOME/tpu_requirements.txt


# vim configurations
cat > $HOME/.vimrc <<- EndOfFile
set tabstop=4
set shiftwidth=4
set softtabstop=4
set expandtab
set backspace=indent,eol,start
syntax on
EndOfFile

# tmux configurations
cat > $HOME/.tmux.conf <<- EndOfFile
bind r source-file ~/.tmux.conf

set -g prefix C-a

set -g set-titles on
set -g set-titles-string '#(whoami)::#h::#(curl ipecho.net/plain;echo)'

set -g default-terminal "screen-256color"

# Status bar customization
#set -g status-utf8 on
set -g status-bg white
set -g status-fg black
set -g status-interval 5
set -g status-left-length 90
set -g status-right-length 60

set -g status-justify left

unbind-key C-o
bind -n C-o prev
unbind-key C-p
bind -n C-p next
unbind-key C-w
bind -n C-w new-window

unbind-key C-j
bind -n C-j select-pane -D
unbind-key C-k
bind -n C-k select-pane -U
unbind-key C-h
bind -n C-h select-pane -L
unbind-key C-l
bind -n C-l select-pane -R

unbind-key C-e
bind -n C-e split-window -h
unbind-key C-q
bind -n C-q split-window -v
unbind '"'
unbind %

unbind-key u
bind-key u split-window -h
unbind-key i
bind-key i split-window -v
EndOfFile


# htop Configurations
mkdir -p $HOME/.config/htop
cat > $HOME/.config/htop/htoprc <<- EndOfFile
# Beware! This file is rewritten by htop when settings are changed in the interface.
# The parser is also very primitive, and not human-friendly.
fields=0 48 17 18 38 39 40 2 46 47 49 1
sort_key=46
sort_direction=1
hide_threads=0
hide_kernel_threads=1
hide_userland_threads=1
shadow_other_users=0
show_thread_names=0
show_program_path=1
highlight_base_name=0
highlight_megabytes=1
highlight_threads=1
tree_view=0
header_margin=1
detailed_cpu_time=0
cpu_count_from_zero=0
update_process_names=0
account_guest_in_cpu_meter=0
color_scheme=0
delay=15
left_meters=CPU Memory Swap
left_meter_modes=1 1 1
right_meters=Tasks LoadAverage Uptime
right_meter_modes=2 2 2
EndOfFile
