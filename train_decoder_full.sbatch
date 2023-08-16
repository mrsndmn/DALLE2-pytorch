#!/bin/env bash
#SBATCH --job-name="dec_f"
#SBATCH --ntasks=1              # Количество MPI процессов
#SBATCH --ntasks-per-node=1     # Количество процессов на один узел
#SBATCH --cpus-per-task=13       # Количество ядер CPU, выделенных для одного процесса
#SBATCH --gpus-per-node=2       # Количество GPU на один узел
#SBATCH -C type_e
#SBATCH --time-min=1400

module purge
module load Python

source deactivate
source activate audio_dalle2
nvidia-smi

set -eo pipefail
set -o xtrace

#Executable

export PYTHONPATH=/home/dtarasov/workspace/hse-audio-dalle2/DALLE2-pytorch

srun /home/dtarasov/.conda/envs/audio_dalle2/bin/accelerate launch --config_file /home/dtarasov/.cache/huggingface/accelerate/default_config.yaml --main_process_port 29501 train_decoder.py --config_file configs/train_decoder_config.audio.full.json