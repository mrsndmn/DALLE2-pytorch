#!/bin/env bash
#SBATCH --job-name="gen_mels"
#SBATCH --ntasks=1              # Количество MPI процессов
#SBATCH --ntasks-per-node=1     # Количество процессов на один узел
#SBATCH --gpus-per-task=1       # Количество ядер GPU, выделенных для одного процесса
#SBATCH --cpus-per-task=1       # Количество ядер CPU, выделенных для одного процесса
#SBATCH -C type_a|type_b|type_c|type_e
#SBATCH --time-min=1000

module purge
module load Python

conda deactivate
conda activate audio_dalle2

set -eo pipefail
set -o xtrace

#Executable

conda run -n audio_dalle2 which python3

srun /home/dtarasov/.conda/envs/audio_dalle2/bin/python3 generate_melspectrograms.py