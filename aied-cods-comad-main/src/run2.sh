#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=100000M
#SBATCH --time=04:00:00
#SBATCH --account=def-fard
#SBATCH --mail-user=ksjoe30@gmail.com

source /home/ksjoe30/projects/def-fard/ksjoe30/ubc/bin/activate

module load gcc/9.3.0 arrow/8 python/3.8
pip list | grep pyarrow

pip install --no-index datasets
pip list | grep datasets

pip install --no-index transformers
pip list | grep transformers

pip install torch --no-index

python training_hf_softmaxLoss.py