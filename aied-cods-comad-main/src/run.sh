#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=100000M
#SBATCH --time=04:00:00
#SBATCH --account=def-fard
#SBATCH --mail-user=ksjoe30@gmail.com

source /home/ksjoe30/projects/def-fard/ksjoe30/ubc/bin/activate

pip install torch --no-index

python training_softmaxLoss.py --dataset without_agumentation --sentence_selection extracted