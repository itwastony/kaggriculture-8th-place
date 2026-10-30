#!/bin/bash
# Step 3 on SLURM: behaviour cloning on one node. Choose the number of GPUs at submission (it must divide the global
# batch of 2,048 states), e.g.
#   mkdir -p runs/logs && sbatch --gres=gpu:G scripts/slurm/bc.sbatch
# (SLURM opens the log file before the script starts, hence the mkdir). Add partition/account/time for your cluster.
#SBATCH --job-name=kag-bc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 entry_points/train_bc.py --run d7 --nproc "${SLURM_GPUS_ON_NODE:?submit with --gres=gpu:G}"
