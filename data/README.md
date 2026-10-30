# Data: tape worlds

All training data are recorded Kaggriculture games played on Kaggle, stored as **tape worlds**. Nothing else is used:
no external datasets, no hand labels. This folder holds the **pool manifests** of the final run (`pools/`); the world
files themselves contain other teams' recorded actions and are rebuilt from public Kaggle replays with
`tools/rebuild_worlds.py`.

## 1. What a tape world is

One recorded episode in a form our engines can replay:

```
data/worlds/<folder>/ep<EPISODE_ID>.json.gz
{
  "ep": 113908167,                      # Kaggle episode id
  "seed": 946316727,                    # the episode's environment seed
  "names": ["<team seat 0>", "<team seat 1>"],
  "rewards": [107038.0, 100807.0],      # recorded final banks
  "statuses": ["DONE", "DONE"],
  "actions": [[...719 actions of seat 0...], [...seat 1...]],   # {"farmer": [...], "hands": [[...], ...], "market": [[...], ...]}
  "shops": ["FARMERS_MARKET", "BRUNCH_SPOT", ...],              # shop types in opening order (8 openings, repeats allowed)
  "lb": [...],                          # leaderboard entry of each seat at a snapshot (or null)
  "repro": true                         # see below
}
```

**Acceptance rule.** `harness/tape_worlds.py` replays both recorded action streams against each other in kagsim
(bit-exact with `kaggle-environments` 1.32.7) on the recorded seed. A world gets `repro: true` only if both final banks
equal the recorded rewards **to the dollar**; only such worlds are used. `entry_points/prepare_data.py --repro` repeats
the check on the training path (kagcore).

**Three uses of one format.**

| Use | How | Caveat |
|---|---|---|
| BC demonstrations (`rl/gen_bc_tapes.py`) | replay the world; for a teacher seat record the observation before every turn and the hindsight label of its actions | only seats of the selected teams; worlds involving our team are excluded |
| League opponents (`rl/appo.py`, league entry `tapes:<pool>`) | our policy takes one seat; the other seat replays its recorded actions (recorded seed and shop order) | a tape does not react to our market behaviour |
| Evaluation | paired differences between two of our agents on the same worlds | absolute P(win) on tapes is inflated: when we play differently, some recorded orders no longer fill (the tape's bank drops to 0.94–0.98 of the record) |

## 2. Pools of the final run (`pools/`)

A **pool** is a list of seats of tape worlds that the league samples as opponents. Entry fields:

| Field | Meaning |
|---|---|
| `world` | path of the world file, relative to the repository root |
| `opp_seat` | the seat that replays its record (the opponent); our policy plays seat `1 - opp_seat` |
| `seed`, `shops` | copied from the world; the game is run with them |
| `team`, `lb_rank`, `lb_score` | the opponent seat's team and its leaderboard rank/score at the snapshot used to build the pool |
| `me_team` | the team that played the seat our policy takes (not in every pool) |
| `ref`, `ref_opp` | recorded final banks of our seat and of the opponent seat (not used by d7b) |
| `ep`, `sub_src`, `src_pool` | episode id; the submission or pool the entry was collected from (informational) |

| Pool (league weight) | Seats | Episodes | Opponent teams | Opponent rank ≤ 10 / ≤ 30 | Source and selection |
|---|---:|---:|---:|---|---|
| `pool_d0930_train` (150) | 3,931 | 2,049 | 30 | 1,550 / 3,603 | Sep 30 games, replay API: episodes found through the submission lists of the top-50 teams in which both agents were ranked ≤ 30 when the list was taken (the rank columns use a later snapshot). Both seats are opponent candidates except seats of our team. Same worlds as the BC data (`worlds/d0930b`) |
| `pool_fast_train` (150) | 1,763 | 1,234 | 22 | 1,429 / 1,721 | "Fast opening": seats whose team opened its 3rd land quadrant before turn 216, from the Sep 24–25 top-vs-top daily datasets (`topvtop`), the Sep 27 daily dataset and our live games of Sep 27 (`live*`). 74 rows repeat an (episode, seat) stored in several live-game folders, so those seats are drawn more often, as in the original run |
| `pool_d0929_train` (150) | 874 | 437 | 19 | 717 / 850 | Kaggle daily dataset of Sep 29, both seats; episodes involving our team dropped; episodes with `sha256(ep) mod 5 == 0` held out |
| `pool_d0928_train` (150) | 952 | 476 | 21 | 638 / 803 | same procedure, daily dataset of Sep 28 |
| `pool_d0927_nohold` (50) | 1,190 | 595 | 20 | 1,047 / 1,181 | daily dataset of Sep 27, both seats, minus the episodes held out for the checkpoint gates of an earlier line; 19 opponent seats are our own agents of that day |
| `pool_weaktrain` (20) | 1,244 | 1,244 | 997 | — (median score 1,953 over the 1,184 seats with a score) | early-ladder games of 13 of our submissions against low-rated opponents (one opponent seat per game) |
| **Total** | **9,954** | 5,539 worlds | | | |

League weights are the `tapes:` entries of `configs/league_d7.json`. Within a pool, seats are drawn uniformly; PFSP
scales the pool's league weight by max(0.1, (1 − P)²), where P is our running win rate against that pool.

**BC data of the final line** come from `worlds/d0930b` only: the seats of the 10 teams in `configs/bc_teams_d7.txt`
(the leaderboard top-10 on Sep 30) that had at least 4 land quadrants by turn 300, in worlds without our team:
809 trajectories.

**Overlap, stated plainly.** The Sep 30 worlds are used both for BC and as league opponents (by design: the policy
learns the newest meta and is trained to beat it). Some Sep 30 games were later also part of our offline
checkpoint-selection sets, one of the biases listed in [docs/evaluation.md](../docs/evaluation.md).

## 3. World folders

| Folder | Worlds | Episode ids | Origin of the record |
|---|---:|---|---|
| `d0927`, `d0928`, `d0929` | 595, 476, 437 | 113908167–115474317 | Kaggle daily datasets `kaggle/kaggriculture-episodes-2026-09-27/28/29` |
| `d0930b` | 2,049 | 115480155–115777990 | replay API, Sep 30 |
| `topvtop` | 631 | 112648570–113480864 | Kaggle daily datasets of Sep 24–25 |
| `live0927`, `live0927b`, `live0927c`, `live1735`, `live2010` | 19, 29, 43, 5, 11 | 113746723–114007994 | replay API, live games of two of our submissions, Sep 26–27 |
| `weaktrain` | 1,244 | 93530071–113242739 | replay API, early games of 13 of our earlier submissions (Aug 16 – Sep 25) |

## 4. Rebuilding the worlds

```bash
pip install kaggle==2.2.4                       # and a Kaggle API token (see the Kaggle docs)
python tools/rebuild_worlds.py --list           # folders and episode counts
python tools/rebuild_worlds.py                  # download + replay check + copy to data/worlds/<folder>/
python entry_points/prepare_data.py --repro 50  # re-check on the training path
```

- Downloads go to `replays/compact/<folder>/` (compact tapes: the ~30 MB replay JSON is reduced to seed, names,
  rewards and actions), built worlds to `replays/worlds/<folder>/`, accepted ones are copied to `data/worlds/<folder>/`.
- Kaggle rate-limits replay downloads: expect hours for all 5,539 world files (5,465 distinct episodes; an episode
  used by two folders is downloaded for each). The script resumes where it stopped: tapes already in
  `replays/compact/<folder>/` are not downloaded again, and a folder is rebuilt when new tapes arrived after its last
  build (the previous index is kept as `<folder>.index.jsonl.old`).
- For the Sep 27–29 folders the Kaggle daily episode datasets are an alternative source:
  `python harness/tape_worlds.py zip --src kaggriculture-episodes-2026-09-27.zip --pool d0927 --lb configs/empty_lb.csv`.
- If an episode is no longer available, it is reported and skipped, and the pools that use it stay incomplete. The
  league then needs `entry_points/train_appo.py --league-available-only`, which keeps only the seats whose world files
  exist; without it the training script stops. `entry_points/prepare_data.py --repro` replays only complete pools and
  stops with "no complete pool to replay" if no pool is complete.
- `configs/empty_lb.csv` is a header-only leaderboard file: rebuilt worlds then have no leaderboard field (ranks depend
  on the snapshot, not on the episode). `tools/compare_worlds.py` compares two sets of worlds field by field.

## 5. Licenses of the data

- Kaggle daily episode datasets (`kaggle/kaggriculture-episodes-YYYY-MM-DD`): CC0-1.0.
- Episodes downloaded through the replay API: public competition replays, under the competition rules.
- The manifests in `pools/` contain episode ids, seeds, shop orders, team names and leaderboard ranks exactly as they
  appear in public replays and on the public leaderboard; nothing else about the players.
