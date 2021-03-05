#!/bin/bash
#SBATCH --job-name=DiMP50C
#SBATCH --output=/home/yans/pytracking-models/ltr/logs/log-dimp50-depthaware-colormap-output.txt
#SBATCH --error=/home/yans/pytracking-models/ltr/logs/log-dimp50-depthaware-colormap-error.txt
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000
#SBATCH --partition=gpu --gres=gpu:teslav100:1

module load CUDA/10.0
module load fgci-common
module load ninja/1.9.0
module load all/libjpeg-turbo/2.0.0-GCCcore-7.3.0

source activate pytracking

python run_training.py dimp dimp50D_colormap

conda deactivate
