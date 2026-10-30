#!/bin/bash
# Step 2 on SLURM: BC demonstrations on one CPU node (224 shards: the shard count fixes the train/validation split,
# not the number of processes).
#   mkdir -p runs/logs && sbatch scripts/slurm/bcgen.sbatch      (from the repository root; logs go to runs/logs/)
# SLURM opens the log file before the script starts, hence the mkdir. Add partition/account/time for your cluster.
#SBATCH --job-name=kag-bcgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 entry_points/make_bc_data.py --run d7 --jobs "${SLURM_CPUS_ON_NODE:-32}"
