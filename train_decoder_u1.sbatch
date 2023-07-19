#!/bin/env bash
#SBATCH --job-name="dec u1"
#SBATCH --ntasks=1              # Количество MPI процессов
#SBATCH --ntasks-per-node=1     # Количество процессов на один узел
#SBATCH --cpus-per-task=8       # Количество ядер CPU, выделенных для одного процесса
#SBATCH --gpus-per-node=2       # Количество GPU на один узел
#SBATCH -C type_c
#SBATCH --time-min=1400

module purge
module load Python

source deactivate
source activate audio_dalle2
nvidia-smi

set -eo pipefail
set -o xtrace

#Executable

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO

export PYTHONPATH=/home/dtarasov/workspace/hse-audio-dalle2/DALLE2-pytorch

srun /home/dtarasov/.conda/envs/audio_dalle2/bin/accelerate launch train_decoder.py --config_file configs/train_decoder_config.audio.u1.json