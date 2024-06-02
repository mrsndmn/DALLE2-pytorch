#!/bin/env bash
#SBATCH --job-name="eval"
#SBATCH --ntasks=1              # Количество MPI процессов
#SBATCH --ntasks-per-node=1     # Количество процессов на один узел
#SBATCH --cpus-per-task=2       # Количество ядер CPU, выделенных для одного процесса
#SBATCH --gpus-per-node=1       # Количество GPU на один узел
#SBATCH -C type_a|type_b|type_c|type_e
#SBATCH --time-min=600

module purge
module load Python

source deactivate
source activate audio_dalle2
nvidia-smi

set -eo pipefail
set -o xtrace

#Executable

export PYTHONPATH=/home/dtarasov/workspace/hse-audio-dalle2/DALLE2-pytorch

srun /home/dtarasov/.conda/envs/audio_dalle2/bin/python3 scripts/eval/test_evaluate_trainer.py
