#!/bin/bash
# Step 4 on SLURM: APPO run d7b. Choose the number of nodes and GPUs per node at submission, e.g.
#   mkdir -p runs/logs && sbatch --nodes=N --gres=gpu:G scripts/slurm/appo_d7b.sbatch --no-pool-src
# (SLURM opens the log file before the script starts, hence the mkdir). Add partition/account/time for your cluster.
# Arguments after the script name go to entry_points/train_appo.py: --no-pool-src is needed because the snapshot pools
# of earlier runs are not published (ASSETS.md); add --league-available-only if not all league bots are installed.
# One task per node; node 0 hosts the torch.distributed rendezvous; every GPU is one rank (1 GPU + 26 CPU worker
# processes). The submitted weights are runs/d7/appo/ema_00110.pt, written after 110 iterations (0-109). --iters is 700
# in the config, so the job runs until its time limit; stop it once ema_00110.pt (or the iteration you want) exists.
#SBATCH --job-name=kag-appo
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
GPUS="${SLURM_GPUS_ON_NODE:?submit with --gres=gpu:G}"
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT=$((29000 + SLURM_JOB_ID % 1000))
EXTRA=""
if [ $# -gt 0 ]; then EXTRA=$(printf ' %q' "$@"); fi
srun --cpu-bind=none bash -c "python3 entry_points/train_appo.py --run d7 --nnodes $SLURM_NNODES --nproc $GPUS \
  --node-rank \$SLURM_NODEID --rdzv $MASTER:$PORT$EXTRA"
