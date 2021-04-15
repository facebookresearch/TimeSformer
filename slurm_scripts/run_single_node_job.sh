# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# A script with a list of commands for submitting SLURM jobs

#SBATCH --job-name=timesformer
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=name@domain.com

## %j is the job id, %u is the user id
#SBATCH --output=/path/to/output/logs/slog-%A-%a.out

## filename for job standard error output (stderr)
#SBATCH --error=/path/to/error/logs/slog-%A-%a.err

#SBATCH --array=1
#SBATCH --partition=partition_of_your_choice
#SBATCH --nodes=1 -C volta32gb
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=480GB
#SBATCH --signal=USR1@600
#SBATCH --time=72:00:00
#SBATCH --open-mode=append

module purge
module load cuda/10.0
module load NCCL/2.4.7-1-cuda.10.0
module load cudnn/v7.4-cuda.10.0
source activate timesformer

WORKINGDIR=/path/to/TimeSformer
CURPYTHON=/path/to/python

srun --label ${CURPYTHON} ${WORKINGDIR}/tools/run_net.py --cfg ${WORKINGDIR}/configs/Kinetics/TimeSformer_divST_8x32_224.yaml NUM_GPUS 8 TRAIN.BATCH_SIZE 8

