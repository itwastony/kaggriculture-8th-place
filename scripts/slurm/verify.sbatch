#!/bin/bash
# Verification on one CPU node (no GPU): engines' self-tests, data integrity + tape replay, submission checks with games,
# inference games in the official engine, and the reference head-to-head d7b_110 vs d6_250 (expected P = 0.621).
# Prerequisites: the release assets (ASSETS.md) in submission/ and artifacts/checkpoints/. The tape replay check needs
# rebuilt tape worlds: run `python tools/rebuild_worlds.py` first (it needs a Kaggle API token and network access;
# data/README.md). Without them, or if no pool is complete, the replay check is skipped with a note and the other
# steps still run; a replay that does not match its record still stops the script.
# verify_submission --games 4 and predict play their games one after another; the head-to-head (1,024 games, one
# process per CPU) is by far the longest step (cost: README.md, "Reproduce the selection number").
#   mkdir -p runs/logs && sbatch scripts/slurm/verify.sbatch     (from the repository root)
# SLURM opens the log file before the script starts, hence the mkdir. Add partition/account/time for your cluster.
#SBATCH --job-name=kag-verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
NCPU="${SLURM_CPUS_ON_NODE:-16}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
bash scripts/build_engines.sh --check
# prepare_data.py without --repro only lists the world files of each pool; its last line is "data OK" or
# "K of 6 pools complete". The replay (--repro) needs at least one complete pool.
DATA_STATUS=$(python3 entry_points/prepare_data.py)
echo "$DATA_STATUS"
case "$(tail -n 1 <<<"$DATA_STATUS")" in
  "0 of "*) echo "NOTE: no complete tape-world pool; skipping the tape replay check (python tools/rebuild_worlds.py)" ;;
  *) python3 entry_points/prepare_data.py --repro 50 ;;
esac
python3 entry_points/verify_submission.py --games 4
python3 entry_points/predict.py --opp starter random pass agents/caretaker/main.py --seeds 2
python3 entry_points/evaluate.py --a submission/rl_d7b_110.tar.gz --b submission/rl_d6_250.tar.gz \
  --seeds 512 --seed0 98300000 --jobs "$NCPU"
