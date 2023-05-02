#!/bin/env bash
#SBATCH --job-name="train_clap_prior_config"
#SBATCH --ntasks=1              # Количество MPI процессов
#SBATCH --ntasks-per-node=1     # Количество процессов на один узел
#SBATCH --cpus-per-task=4      # Количество ядер CPU, выделенных для одного процесса
#SBATCH --gpus-per-node=1       # Количество GPU на один узел
#SBATCH -C type_a|type_b|type_c
#SBATCH --time-min=30

module purge
module load Python

source deactivate

set -eo pipefail
set -o xtrace

# conda create -n audio_dalle2
# source activate audio_dalle2
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
source activate audio_dalle2

nvidia-smi
which python
python -V
python -c 'import torch; print(torch.cuda.is_available())'