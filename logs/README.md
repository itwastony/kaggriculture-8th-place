# Training logs of the final line (d7)

Machine-readable logs of the run that produced the submitted weights, for comparison with a rerun
([docs/training.md](../docs/training.md)). The APPO statistics come from **training games**: the learning side samples
its actions and PFSP shifts the opponent mix, so they are not an evaluation. Three figures in `docs/images/` are built
from these files by `docs/make_figures.py`: `d7b_training.png` (`appo_d7b_curves.csv`), `league_mix.png`
(`appo_d7b_league_mix.json`) and `bc_dip.png` (`appo_d7b_vs_fixed_past.csv`). `live_ratings.csv` (hourly Kaggle API
snapshots of the two final submissions after the deadline) gives `live_ratings.png`. The other figures are diagrams or
evaluation numbers written in that script, or plain images.

The logs are **aggregated for publication**: blocks of iterations or steps instead of single records, values rounded
to a fixed step, and a few diagnostic columns left out. Each value is an unweighted mean over its block, then rounded;
the rounding step of each column is given below. A win rate rounded to 0.05 shows 1.0 when it is at least 0.975.

| File | Step | Content |
|---|---|---|
| `bcgen_d7_totals.json` | BC demonstrations (`make_bc_data.py`) | totals: 224 shards, 2,049 worlds, 809 trajectories, skip counters, trajectories per teacher team |
| `bc_d7_train.jsonl` | BC (`train_bc.py`) | one record per block of 1,000 steps (3 records): training losses and accuracies averaged over the block, validation accuracies at the end of the block |
| `appo_d7b_curves.csv` | APPO d7b (`train_appo.py`) | one row per block of 10 iterations, 0–117 (12 rows; the last block is 110–117). The submitted `ema_00110.pt` was written after 110 iterations (0–109); only the last block (110–117) comes after it |
| `appo_d7b_league_mix.json` | APPO d7b | share of learner trajectories by opponent group, iterations 0–109 |
| `appo_d7b_vs_fixed_past.csv` | APPO d7b | learner win rate against three fixed past checkpoints, blocks of 20 iterations (the last block is 100–109) |
| `live_ratings.csv` | live leaderboard | `public_score` of `d7b_110` and `d6_250` from hourly Kaggle API snapshots (UTC), Oct 1–15; not training data |

## Columns of `appo_d7b_curves.csv`

| Column | Meaning | Rounding |
|---|---|---|
| `iter_from`, `iter_to` | first and last iteration of the block (0-based, inclusive) | — |
| `P_field` | win rate against the field: the 101 code bots and the 6 tape pools (self/past/best excluded) | 0.05 |
| `P_code_bots` | win rate against the 101 code bots | 0.05 |
| `P_tapes_<pool>` | win rate against a tape pool: `sep30`, `sep29`, `sep28`, `sep27`, `fast`, `weak` ([data/README.md](../data/README.md)) | 0.05 |
| `P_past` | win rate against past versions (argmax play) | 0.05 |
| `margin_tapes_sep30` | our mean final bank minus the opponent's, against the Sep 30 tapes | 5,000 |
| `bank_code_bots` | our mean final bank against the code bots | 5,000 |
| `rec_P` | in tape games: share of games where our bank beat the recorded bank of the seat we replaced | 0.05 |
| `R_mean`, `v_mean`, `ev`, `ev_early`, `vl` | mean return, mean value, value explained variance (all / early steps), value loss | 2 significant digits |
| `ent` | mean policy entropy per component | 2 significant digits |
| `kl_coef` | coefficient of the KL term to the anchor (a fixed schedule) | 3 significant digits |
| `q4_end_pct` | % of learner seats (rank 0) that ended the game with 4 land quadrants | 0.1 |

## `bc_d7_train.jsonl`

| Field | Meaning | Rounding |
|---|---|---|
| `step_from`, `step_to` | first and last optimizer step of the block | — |
| `l_unit`, `l_mkt`, `l_val` | training losses: unit (target, operation) pairs, market heads, value; block means | 0.01 |
| `acc_unit`, `acc_op`, `acc_mkt`, `acc_mkt_nz` | training accuracies: unit pair, operation, market heads, market heads with a non-zero teacher decision; block means | 0.01 |
| `val_acc_*` | the same accuracies on the validation shards, measured at `step_to` | 0.01 |

To compare a rerun, average the training metrics of `runs/d7/bc/train.jsonl` over steps 1–1,000, 1,001–2,000 and
2,001–3,000, take the validation accuracies logged at steps 1,000, 2,000 and 3,000, and round to 0.01.

## `appo_d7b_vs_fixed_past.csv`

`opponent`, `iter_from`, `iter_to`, `learner_P`: the learner's win rate against one of three fixed past checkpoints
(`d6` EMA 140, `d4a` EMA 370, BC d7) in a block of 20 iterations, rounded to 0.1. The past opponents play argmax; the
learner samples.
