#!/bin/bash
#SBATCH --job-name=TT_S4
#SBATCH --output=/home/yans/pytracking-rgbd/ltr/logs/log-simple-layer4-output.txt
#SBATCH --error=/home/yans/pytracking-rgbd/ltr/logs/log-simple-layer4-error.txt
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

python run_training.py transformer transformer50_simple_layer4

conda deactivate
