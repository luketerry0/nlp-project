#!/bin/bash
# the name of the partition you are submitting to
# the number of nodes you will use, usually only 1 is required.  If your code is not designed to use more then more will not be better.
#SBATCH --nodes=1
# the number of processes you will launch.  This should be equal to the number of nodes
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --container=el9-devel
#SBATCH --partition=sooner_gpu_test
# Thread count, or the number of hypercores you plan to use.  This is not enforced.
#SBATCH --cpus-per-task=16
# memory (RAM) you will use on each machine.  Provide an upper bound, this is not enforced
#SBATCH --mem=2G
# Where you would like your stdout and stderr to appear
#SBATCH --output=/home/cs529321/nlp-project/slurm_logs/R-%x.%j.out
#SBATCH --error=/home/cs529321/nlp-project/slurm_logs/R-%x.%j.err
# The maximum time your job can take (most partitions limit this)
#SBATCH --time=12:00:00
# job name which will appear in queue
#SBATCH --job-name=deberta_ex_train
# if you fill this out slurm will email you job status updates, consider sending them to a folder.
#SBATCH --mail-user=luke.h.terry-1@ou.edu
#SBATCH --mail-type=ALL
# the working directory for your job.  This must exist.
#SBATCH --chdir=/home/cs529321/nlp-project
#################################################

# using Dr. Fagg's conda setup script
. /home/fagg/tf_setup.sh
# activating a version of my environment
conda activate /home/cs529321/miniforge3/envs/nlp

# logging in to weights and biases
wandb login 0d4cc1d2edeff4340f22d03bf7bb009297d1af61

cd /home/cs529321/nlp-project

# launching a run 
srun python ./deberta_experimental.py \
    --learning_rate_bert=0.00002 \
    --learning_rate_head=0.001 \
    --batch_size=16 \
    --epochs=20

sleep 10