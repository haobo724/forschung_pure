#!/bin/bash
#SBATCH --job-name=2_dice
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=25000
#SBATCH --gres=gpu:2
#SBATCH --exclude=lme170,lme171,lme221,lme223
#SBATCH -o /cluster/pa61wiqo/%x.out
#SBATCH -e /cluster/pa61wiqo/%x.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h

# %x-%j-on-%N

export XDG_DATA_HOME="/cluster/pa61wiqo/.local"
export XDG_CACHE_HOME="/cluster/pa61wiqo/.cache"
export PATH=/cluster/pa61wiqo/.python_packages/bin:$PATH
export PYTHONUSERBASE=/cluster/pa61wiqo/.python_packages
export PYTHONPATH=/cluster/pa61wiqo/forschung/mostoolkit:$PYTHONPATH
export PYTHONPATH=/cluster/pa61wiqo/forschung:$PYTHONPATH
export PYTHONPATH=/cluster/pa61wiqo/forschung/pure:$PYTHONPATH
export PYTHONPATH=/cluster/pa61wiqo/forschung/pure/data_module:$PYTHONPATH



echo "Your job is running on" $(hostname)


python3 /cluster/pa61wiqo/forschung/pure/basetrain_song.py --gpus -1 \
--max_epochs 40 --accelerator ddp \
--data_folder /cluster/liu/data/shuqing_2D \
--worker 16 \
--batch_size 8 \
--lr 5e-5 \
--datasetmode 2 \
--opt Adam \
--loss Dice