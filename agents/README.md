# League bots

The APPO league of the final line (`configs/league_d7.json`) plays against **101 code bots**. None of them is part of
the submitted agent, which contains only our network and its NumPy runtime. Each bot is a folder `agents/<name>/` with
a `main.py` that defines `agent(observation, configuration)`, exactly as on Kaggle.

| Status | Bots | League weight | How to get them |
|---|---:|---:|---|
| shipped | 2 | 1.5 | in this folder: `caretaker` (our hand-written agent) and `pass_agent` (always passes; an empty-farm opponent) |
| public | 77 | 77 | public Kaggle notebooks (and one public dataset): `python tools/fetch_league_bots.py` |
| not released | 19 | 23 | tape clones (they embed another team's recorded game) and our modifications of other authors' notebooks |
| unavailable | 3 | 3 | copies whose notebook slug we did not record |
| **Total** | **101** | **104.5** | |

League weight is the weight of the bot in `configs/league_d7.json`; within the league, the code bots as a group get
the share shown in `logs/appo_d7b_league_mix.json`, and individual bots are then drawn by PFSP.

`caretaker` is our early hand-written agent, kept byte for byte (`agents/sources.json`) because the engine parity
checks play it. Its table `OPP_SCHED` is an opponent's per-step market flow (units of each product sold or bought),
measured from two public Kaggle replays of top teams (episodes 89831498 and 89855909, August 2026). Its comments
refer to our internal notes and a helper script (`research/...`, `harness/bake_opp_sched.py`) that are not part of
this repository.

## Getting the public bots

```bash
pip install kaggle==2.2.4                     # and a Kaggle API token
python tools/fetch_league_bots.py             # downloads each notebook's output, installs agents/<name>/
python tools/fetch_league_bots.py --status    # which bots are installed and whether main.py matches our record
```

- For every bot, `sources.json` records the first 8 hex digits of the sha256 of the `main.py` we trained against. The
  script installs a bot only if the notebook output (or an archive inside it) contains a file with that hash. With
  `--accept-any` it installs the newest `main.py` instead: a different version of the opponent.
- Notebooks change after the fact.
  On Oct 15, 2026 (UTC) a full run of the script installed **52 of the 77** public bots with the recorded `main.py`
  (47 folders byte-identical to ours, 5 with an extra `_source_bundle.tar.gz`). For 20 bots the current notebook
  output has a different `main.py`; 3 notebooks, and the dataset that held 2 more bots, were no longer accessible.
  With the 2 shipped bots, the restorable bots carry 53.5 of the 104.5 code-bot weight. In our test, requesting an
  older notebook version (`owner/slug/<version>`) returned the latest output, so a changed bot cannot be restored
  this way.
- Checked offline on our own Sep 23 copies of the notebook outputs: 63 bots matched, 61 of their folders are
  byte-identical to the folders we trained with, and the other 2 differ only by an extra `_source_bundle.tar.gz`.
- Train without the missing bots with `python entry_points/train_appo.py ... --league-available-only`. The CPU game
  workers then draw their opponents from the remaining bots and tape pools by weight: a different opponent mix.

`_source_bundle.tar.gz` is the notebook's own submission archive (some bots import helper modules from it at run time).
`LICENSE*` / `NOTICE*` / `THIRD_PARTY_NOTICES*` files of a notebook are kept in its folder. Kaggle shows public notebooks
as released under the Apache 2.0 license; the notebook pages are the authoritative source of each bot's license.

## Public notebooks collected on Sep 23, 2026

Downloaded with the Kaggle API on Sep 23, 2026 (UTC). "Last run" is the notebook's last run time shown by Kaggle
at that moment. Notebook refs are `owner/slug` on kaggle.com/code. "Oct 15" is the result of
`tools/fetch_league_bots.py` on Oct 15, 2026: `match` (installed, same `main.py`), `changed` (the current output
has a different `main.py`), `not accessible` (the notebook or dataset could not be downloaded).

| Folder | Notebook | Author | Title | Last run (UTC) | Approach | main.py sha256 | Oct 15 |
|---|---|---|---|---|---|---|---|
| `pub0923_ahmed_v51` | ahmedberatozer/kaggriculture-v51-lean-flock (our copy came from dataset `destbreso/kaggriculture-donor-agents`) | Ahmed Berat Özer | Kaggriculture V51 — Lean Flock | 2026-09-20 09:21 | hybrid route-portfolio + market overlays | `c1e3590d` | match |
| `pub0923_ahmed_v52` | ahmedberatozer/kaggriculture-v52-lean-flock-yarn-route | Ahmed Berat Özer | Kaggriculture V52 — Lean Flock, Yarn Route | 2026-09-20 10:38 | hybrid route-portfolio + market overlays | `8c0f7d90` | match |
| `pub0923_ahmed_v53` | ahmedberatozer/kaggriculture-v53-opening-signature | Ahmed Berat Özer | Kaggriculture V53 — Opening Signature | 2026-09-20 14:00 | hybrid route-portfolio + market overlays | `20fe549d` | match |
| `pub0923_ahmed_v54` | ahmedberatozer/kaggriculture-v54-productive-wheat-and-patient | Ahmed Berat Özer | Kaggriculture V54 — Productive Wheat and Patient | 2026-09-21 08:26 | hybrid route-portfolio + market overlays | `5fbb75c9` | match |
| `pub0923_ahmed_v55` | ahmedberatozer/kaggriculture-v55-one-turn-market-race-edge | Ahmed Berat Özer | Kaggriculture V55 — One-Turn Market Race Edge | 2026-09-21 11:27 | hybrid route-portfolio + market overlays | `f0903462` | match |
| `pub0923_ahmed_v56` | ahmedberatozer/kaggriculture-v56-smarter-seeds-and-fertilizer | Ahmed Berat Özer | Kaggriculture V56 — Smarter Seeds and Fertilizer | 2026-09-21 15:28 | hybrid route-portfolio + market overlays | `a1ad0fd1` | match |
| `pub0923_ahmed_v57` | ahmedberatozer/kaggriculture-v57-funding-order-invariant | Ahmed Berat Özer | Kaggriculture V57 — Funding-Order Invariant | 2026-09-22 12:50 | hybrid route-portfolio + market overlays | `2ee689fd` | changed |
| `pub0923_anhad_auto` | anhadmahajan06/kaggriculture-autonomous-ai-farming-agent | Anhad Mahajan | Kaggriculture: Autonomous AI Farming Agent | 2026-09-22 11:45 | hybrid route-portfolio + market overlays | `a701d59e` | changed |
| `pub0923_anhad_new` | anhadmahajan06/kaggriculture-autonomous-ai-farming-agent | Anhad Mahajan | Kaggriculture: Autonomous AI Farming Agent (later version, pulled Sep 23 evening) | 2026-09-23 | hybrid route-portfolio + market overlays | `2993cf92` | changed |
| `pub0923_arsg_herdsafe3` | arsgorynich/herd-safe-v3-experimental-risk-aware-feed | Arsgorynich | Herd Safe v3 Experimental Risk Aware Feed | 2026-09-23 10:15 | hybrid + herd-safe rules (risk-aware feed) | `4f8637a3` | match |
| `pub0923_arsg_obv3` | arsgorynich/order-book-v3-response-improvement | Arsgorynich | Order Book v3 Response Improvement | 2026-09-22 11:02 | hybrid route-portfolio + market overlays | `8045c887` | match |
| `pub0923_arsg_v40` | arsgorynich/kaggriculture-v40-challenger | Arsgorynich | Kaggriculture V40 Challenger | 2026-09-23 05:45 | hybrid route-portfolio + market overlays | `d5a192fa` | match |
| `pub0923_arsg_v43` | arsgorynich/kaggriculture-v43-local-confirmed | Arsgorynich | Kaggriculture V43 Local Confirmed | 2026-09-15 15:51 | hybrid route-portfolio + market overlays | `919fc1d6` | match |
| `pub0923_aurax7_v7` | aurax7/kaggriculture-shop-router-reactive-v7 | Yaxon | Kaggriculture_shop_router_reactive_v7 | 2026-09-17 04:33 | hybrid route-portfolio + market overlays | `e221f487` | match |
| `pub0923_busy_seedleak` | busyaprime/92-vs-the-best-public-farm-a-seed-leak-in-v54 | Akmal Xodarev (Busya PRIME) | 92% vs the best public farm: a seed leak in V54 | 2026-09-21 14:40 | hybrid route-portfolio + market overlays | `562f34ad` | not accessible |
| `pub0923_busy_v54fork` | busyaprime/v54-fork-wins-72-5-head-to-head-against-v54 | Akmal Xodarev (Busya PRIME) | V54 fork wins 72.5% head to head against V54 | 2026-09-21 12:41 | hybrid route-portfolio + market overlays | `a0060d10` | not accessible |
| `pub0923_datascikhan_farmcraft` | datascikhan/farmcraft-kaggriculture-grandmaster | Shair Khan | 🌾 FarmCraft: Kaggriculture Grandmaster | 2026-09-21 08:29 | not a working agent (submission.py) | `d87fae30` | match |
| `pub0923_dmitrii_herdsafe` | dmitriigluzdov/kaggriculture-herd-safe-sale-window-lb-2700 | Dmitrii Gluzdov | Kaggriculture: Herd-Safe Sale Window / LB 2700 | 2026-09-22 15:20 | hybrid + herd-safe rules + 7-turn terminal sim planner | `4889137f` | changed |
| `pub0923_dmitrii_morewheat` | dmitriigluzdov/kaggriculture-more-wheat-smarter-sales | Dmitrii Gluzdov | Kaggriculture: More Wheat, Smarter Sales | 2026-09-22 15:44 | hybrid route-portfolio + market overlays | `7ac1a254` | match |
| `pub0923_dmitrii_onemorewheat` | dmitriigluzdov/kaggriculture-one-more-wheat | Dmitrii Gluzdov | Kaggriculture: One More Wheat | 2026-09-20 17:25 | hybrid route-portfolio + market overlays | `10f58185` | match |
| `pub0923_dmitrii_rescue7` | dmitriigluzdov/kaggriculture-7-turn-rescue-historical-lb-2800 | Dmitrii Gluzdov | Kaggriculture: 7-Turn Rescue / Historical LB 2800+ | 2026-09-22 11:32 | hybrid + herd-safe rules + 7-turn terminal sim planner | `4889137f` | match |
| `pub0923_dmitrii_smallershock` | dmitriigluzdov/kaggriculture-a-smaller-market-shock | Dmitrii Gluzdov | Kaggriculture: A Smaller Market Shock | 2026-09-19 15:16 | hybrid route-portfolio + market overlays | `d5460fc2` | match |
| `pub0923_dmitrii_twocoins` | dmitriigluzdov/kaggriculture-two-coins-one-sheep-lb-2600 | Dmitrii Gluzdov | Kaggriculture: Two Coins, One Sheep / LB 2600 | 2026-09-10 12:09 | hybrid route-portfolio, multi-file | `495bfa48` | changed |
| `pub0923_flexonafft_multiroute` | flexonafft/kaggriculture-multi-route-farming-agent | Igor Zharov | Kaggriculture / Multi-Route Farming Agent | 2026-09-23 13:03 | hybrid route-portfolio (multi-route), multi-file | `495bfa48` | changed |
| `pub0923_flexonafft_v35` | flexonafft/kaggriculture-most-powerfull-route | Igor Zharov | 🌾 Kaggriculture / Most Powerfull Route | 2026-09-11 15:34 | hybrid route-portfolio + market overlays | `294e7e4d` | match |
| `pub0923_flyfarmer` | takamichitoda/flyfarmer-connectome-plays-kaggriculture | Takamichi Toda | 🪰FlyFarmer connectome plays Kaggriculture | 2026-09-19 21:02 | experimental (fly connectome) | `d33237f2` | match |
| `pub0923_goodpjw_melon` | goodpjw2008/kaggriculture-melon-threshold-squeeze-2749 | goodpjw2008 | Kaggriculture: Melon threshold squeeze [2749] | 2026-09-19 01:26 | hybrid route-portfolio + market overlays | `1f5c7a18` | match |
| `pub0923_guru_gtv2` | guruprasaathas111/game-theoretic-master-discrete-optimization | Guru Prasaath S | Game-Theoretic Master: Discrete Optimization | 2026-09-19 08:31 | hybrid route-portfolio + market overlays | `ed89be8c` | changed |
| `pub0923_guru_v3` | guruprasaathas111/kaggriculture-master-engine-v3 | Guru Prasaath S | Kaggriculture Master Engine V3 | 2026-09-23 17:05 | hybrid route-portfolio + market overlays | `ffbb800d` | changed |
| `pub0923_guru_v4` | guruprasaathas111/kaggriculture-top-2-master-engine-v4 | Guru Prasaath S | Kaggriculture (TOP 2 )Master Engine V4 | 2026-09-21 06:47 | hybrid route-portfolio + market overlays | `a5362d13` | changed |
| `pub0923_hak_fieldcraft` | hakdevelopment/kaggriculture-2887-score-fieldcraft-agent | Hak | Kaggriculture 2887 Score Fieldcraft Agent | 2026-09-21 13:30 | unclear (compact/obfuscated; main + mirror_plan) | `7fab02c8` | match |
| `pub0923_hanif_wonderful` | hanifnoerrofiq/a-wonderful-life | Hanif Noer Rofiq | A wonderful Life | 2026-09-20 02:27 | hybrid route-portfolio + market overlays | `caf490d3` | match |
| `pub0923_haodou_ledger` | haodou092/kaggriculture-harvest-ledger | haodou092 | Kaggriculture / Harvest Ledger | 2026-09-23 12:45 | hybrid route-portfolio + market overlays | `df06cc9f` | changed |
| `pub0923_jaxa_2780` | jaxa623/2780-beyond-48-0-128-128-worlds-with-95-cis | sdy623 | [2780] Beyond 48-0: 128/128 Worlds with 95% CIs | 2026-09-17 00:02 | hybrid route-portfolio + market overlays | `4757f3f5` | match |
| `pub0923_jaxa_2802` | jaxa623/2802-two-identical-agents-90-points-apart | sdy623 | [2802] Two Identical Agents, 90 Points Apart | 2026-09-17 02:43 | hybrid route-portfolio + market overlays | `944aa64c` | match |
| `pub0923_kaito_v43` | kaitofukami/103-128-fresh-public-v43-sparse-shop-hybrid | Kaito Fukami | 103/128 Fresh Public / v43 Sparse Shop Hybrid | 2026-08-23 15:43 | hybrid route-portfolio (sparse shop hybrid) | `69f06a80` | match |
| `pub0923_kaito_v48` | kaitofukami/40-40-early-floor-39-46-top-10-v48-fast-routes | Kaito Fukami | 40/40 Early Floor / 39/46 Top-10 / v48 Fast Routes | 2026-08-24 22:55 | hybrid route-portfolio (fast routes) | `dadee25a` | match |
| `pub0923_kaito_v58` | kaitofukami/238-238-known-streams-v58-minimax-closed-loop | Kaito Fukami | 238/238 Known Streams / v58 Minimax Closed Loop | 2026-09-01 07:15 | search-planner (minimax over known opp streams) on route portfolio | `b041058e` | match |
| `pub0923_koshinm_0920` | koshinm/kaggriculture-local-best-2026-09-20 (403; file from dataset `destbreso/kaggriculture-donor-agents`) | KoshinM | (kernel 403) | — | unknown (kernel now 403; payload from destbreso donors) | `5718b254` | not accessible |
| `pub0923_koshinm_0921` | koshinm/kaggriculture-local-best-2026-09-21 (403; file from dataset `destbreso/kaggriculture-donor-agents`) | KoshinM | (kernel 403) | — | unknown (kernel now 403; payload from destbreso donors) | `41dd60f7` | not accessible |
| `pub0923_municef1_2448` | municef1/lb-2448-single-file-kaggriculture-agent | 안녕하세요이세입니다잘부탁드립니다 | 🏆 LB 2448 Single File Kaggriculture Agent | 2026-09-12 11:58 | hybrid route-portfolio + market overlays | `cd0173d4` | match |
| `pub0923_mzcao_lgbm` | mzcao7/kaggriculture-2476-8-peak-lightgbm-xgboost | Mason Cao | Kaggriculture / 2476.8 Peak / LightGBM + XGBoost | 2026-09-18 05:21 | hybrid + ML (LightGBM/XGBoost), multi-file | `14c9354d` | match |
| `pub0923_nathan_pipe15` | nathanjacob/kaggriculture-pipe15-two-layers | Nathan Jacob | Kaggriculture Pipe15 Two Layers | 2026-09-20 08:48 | hybrid route-portfolio + market overlays | `ce9ec069` | match |
| `pub0923_nathan_pipe18` | nathanjacob/kaggriculture-pipe18-six-layers | Nathan Jacob | Kaggriculture Pipe18 Six Layers | 2026-09-21 21:32 | hybrid route-portfolio + market overlays | `a4a563c6` | match |
| `pub0923_orderbook` | shiiin9/your-market-list-is-an-order-book | shiiin9 | Your Market List Is an Order Book | 2026-09-21 23:16 | hybrid route-portfolio + market overlays | `a16e0e9b` | match |
| `pub0923_salemali_2900` | salemali7/kaggriculture-2900 | Salem Ali | Kaggriculture / 2900+ | 2026-09-17 22:19 | tape+overlay (V16-RC5 reconstruction) | `01f0cd47` | match |
| `pub0923_shiiin9_beatv48` | shiiin9/beat-v48-100-0-your-herd-is-decided-on-day-6 | shiiin9 | Beat V48 100-0: Your Herd Is Decided on Day 6 | 2026-09-19 07:50 | hybrid route-portfolio + market overlays | `ba612b6f` | not accessible |
| `pub0923_souvik_v5` | souvikdbiswas/kaggriculture-v5-hybrid-agent | Souvik_D_Biswas | Kaggriculture V5 Hybrid Agent | 2026-09-07 17:59 | hybrid route-portfolio + market overlays | `cb59afce` | match |
| `pub0923_statma_ca20` | statma/kaggriculture-herd-safe-sale-window-race-ca20 | statma | Kaggriculture: Herd-Safe Sale Window race ca20 | 2026-09-23 00:31 | hybrid + herd-safe sale-window race | `29454a13` | match |
| `pub0923_statma_ca25` | statma/kaggriculture-herd-safe-sale-window-race-ca25 | statma | Kaggriculture: Herd-Safe Sale Window race ca25 | 2026-09-23 11:59 | hybrid + herd-safe sale-window race | `c4b72e64` | match |
| `pub0923_statma_tdp` | statma/kaggriculture-tetsutani-demand-preserving | statma | kaggriculture-tetsutani-demand-preserving | 2026-09-23 01:55 | hybrid route-portfolio + market overlays | `939c4457` | match |
| `pub0923_statma_thomas2944` | statma/kaggriculture-thomas-2944-candidate | statma | Kaggriculture: Thomas 2944 candidate | 2026-09-23 09:58 | hybrid (Thomas <336, herd-safe after) | `2714b6eb` | match |
| `pub0923_tetsutani_dp` | tetsutani/demand-preserving-turn-sale-timing | tetsu2131 | 🌾 Demand-Preserving Turn Sale Timing | 2026-09-23 16:21 | hybrid route-portfolio + market overlays | `178ae0f7` | changed |
| `pub0923_tetsutani_mktsmart` | tetsutani/market-smart-farming-kaggriculture | tetsu2131 | 🌾 Market-Smart Farming / Kaggriculture | 2026-09-17 13:42 | hybrid route-portfolio + market overlays | `f6a756cf` | match |
| `pub0923_tetsutani_shape` | tetsutani/shape-the-shop-work-the-pasture-kaggriculture | tetsu2131 | 🌾 Shape the Shop Work the Pasture / Kaggriculture | 2026-09-06 17:07 | hybrid route-portfolio + market overlays | `2b97e2c6` | match |
| `pub0923_thomas_2945` | thomastschinkel/the-2945-farm-96-vs-the-top-10-public-bots | Thomas Tschinkel | The 2945 Farm: 96% vs the Top-10 Public Bots | 2026-09-19 00:15 | hybrid route-portfolio + market overlays | `bfee70e9` | match |
| `pub0923_thomas_938` | thomastschinkel/kaggriculture-93-8-win-rate-public-state-router | Thomas Tschinkel | Kaggriculture: 93.8% Win Rate Public State Router | 2026-09-07 10:22 | hybrid route-portfolio (public-state router) | `b87a27ed` | match |
| `pub0923_thomas_955` | thomastschinkel/kaggriculture-95-5-win-rate-via-replay-routing | Thomas Tschinkel | Kaggriculture: 95.5% Win Rate via Replay Routing | 2026-09-07 18:57 | hybrid route-portfolio (replay routing) | `82412467` | match |
| `pub0923_thomas_v13` | thomastschinkel/the-metav4-farm-submission-v13 | Thomas Tschinkel | The Metav4 Farm: Submission v13 | 2026-09-20 06:58 | hybrid route-portfolio + market overlays | `9d634946` | match |
| `pub0923_uninh_pipe7` | uninhibitedscholar/kaggriculture-pipe-7-market-microstructure | uninhibited-scholar | Kaggriculture Pipe-7: Market Microstructure | 2026-09-16 16:17 | hybrid route-portfolio + market overlays | `6150b7f9` | match |
| `pub0923_wzb_hybu` | wzhengbiao/kaggriculture-hybu-submit | WUZB | kaggriculture hybu submit | 2026-09-23 03:13 | hybrid route-portfolio + market overlays | `f6175fd1` | match |
| `pub0923_wzb_v15stack` | wzhengbiao/kaggriculture-v15stack-submit | WUZB | kaggriculture v15stack submit | 2026-09-22 19:00 | hybrid route-portfolio + market overlays | `d6565929` | match |
| `pub0923_yama_apsmr` | yamakawanin/kaggriculture-adaptive-public-state-multi-route | Roxy | Kaggriculture — Adaptive Public-State Multi-Route | 2026-08-25 06:06 | hybrid route-portfolio (adaptive public-state multi-route) | `9b5e1157` | match |
| `pub0923_yama_king4e` | yamakawanin/king-v4e-rc4 | Roxy | King-v4e RC4 | 2026-09-07 06:14 | hybrid (programme network + liquidity overlays) | `26ffba52` | match |
| `pub0923_yama_q30` | yamakawanin/kaggriculture-2312-9-q30-v2a | Roxy | Kaggriculture 2312.9 — Q30 v2a | 2026-09-06 11:16 | hybrid (programme network) | `6af46771` | match |

## Public notebooks collected in August

| Folder | Notebook | Author | Note | main.py sha256 | Oct 15 |
|---|---|---|---|---|---|
| `meta2_andrew` | andrewsokolovsky/kaggriculture-breaking-the-tie-2883-score | Andrey Naymushin; route credited to Nikita Lugovoy (submission 55440039) | — | `badf45ec` | changed |
| `meta2_prvsiyan_frontier` | prvsiyan/kaggriculture-frontier-the-soil-remembers-rain | prvsiyan | Aug 16 version | `34193f0c` | changed |
| `meta2_salemali` | salemali7/kaggriculture-3000-socre | Salem Ali (salemali7); indarkarhana credits boatlee/v17-r1-rc2-high-score-10c-4s-market-storage and a reconstruction of episode 92521336 seat 0 | same bytes also in indarkarhana/rank-top10-read-the-market-choose-the-farm and flexonafft/kaggriculture-multi-route-farming-agent | `d39dba50` | changed |
| `meta2_tetsutani` | tetsutani/adaptive-farming-strategy-for-kaggriculture | tetsu2131 | Aug 16 version; tag BL-V17-R1-RC2 | `191ed9fe` | changed |
| `meta3_boatlee_v20` | boatlee/v20-adaptive-r1-multi-route-agent | boatlee | — | `8ac34abc` | match |
| `meta_kaito_v27` | kaitofukami/25-27-strict-future-v27-midgame-meta-reset | Kaito Fukami | — | `f48c2116` | match |
| `meta_prvsiyan_moon` | prvsiyan/kaggriculture-frontier-the-moon-counts-melons | prvsiyan (by folder name) | probably prvsiyan/kaggriculture-frontier-the-moon-counts-melons (version not recorded) | `df4e899a` | changed |
| `meta_prvsiyan_soil` | prvsiyan/kaggriculture-frontier-the-soil-remembers-rain | prvsiyan | around Aug 13; version not recorded | `2282dfe2` | changed |
| `pub_flexonafft` | flexonafft/kaggriculture-adaptive-replay-agent | Igor Zharov (flexonafft) | probably flexonafft/kaggriculture-adaptive-replay-agent (version C68) | `e5cbb6ed` | changed |
| `pub_pilkwang` | pilkwang/kaggriculture-structured-economic-policy | Pilkwang Kim | a version newer than our Aug 4 snapshot | `9093eaeb` | changed |
| `pub_romanrozen` | romanrozen/strong-statr-baseline-agent-lb-950 | Roman Rozen | probably romanrozen/strong-statr-baseline-agent-lb-950 | `997e6bfc` | match |
| `soil_v7` | prvsiyan/kaggriculture-frontier-the-soil-remembers-rain | prvsiyan; tape Tran H Hoang | Aug 3 run; embedded tape: episode 89674601 seat 0 (Tran H Hoang) | `698cf0d9` | changed |

## Not released

Tape clones put the recorded actions of one seat of a public Kaggle episode into the machinery of the public
notebook prvsiyan/kaggriculture-frontier-the-soil-remembers-rain; some carry layers we added. Our modifications
change a public Apache-2.0 bot (the V54-family file with sha256 `178ae0f7…`, published in
tetsutani/demand-preserving-turn-sale-timing). They are listed for completeness and credit.

| Folder | League weight | Kind | Recorded game / base | Credit | main.py sha256 |
|---|---:|---|---|---|---|
| `bcr_tdp_v4` | 2 | our modification of public code | same base as c3bc; route layer appended at the end of main.py | authors of the upstream notebook (its NOTICE file); changes by this team | `8cddebba` |
| `c3bc` | 3 | our modification of public code | public V54-family file sha 178ae0f7 (tetsutani/demand-preserving-turn-sale-timing = prvsiyan moon/soil, Sep 23); our layers: per-product horizons, 58 constants tuned by CEM | authors of the upstream notebook (its NOTICE file); changes by this team | `5ca559fb` |
| `l12_combo3` | 1 | our modification of public code | same base as c3bc; constant overrides | authors of the upstream notebook (its NOTICE file); changes by this team | `0288b962` |
| `subin_pure` | 1 | pure tape replay (11-line wrapper of ours) | episode 90341664 seat 0 | Subin An (recorded actions) | `7dcc00ce` |
| `v10_lucaskna` | 1 | tape clone | episode 90154808 seat 1 | lucaskna; prvsiyan (machinery) | `476ba950` |
| `v12_faithful` | 1 | tape clone, small edit by us (order reordering removed) | episode 90154808 seat 1 | lucaskna; prvsiyan (machinery) | `26c663af` |
| `v13_ben` | 1 | tape clone | episode 90328411 seat 1 | Ben Hamilton; prvsiyan (machinery) | `bd17785d` |
| `v14b_noterm` | 1 | tape clone, small edit by us (terminal sell-off off) | episode 90341664 seat 0 | Subin An; prvsiyan (machinery) | `a00caa36` |
| `v16_dose` | 1 | tape clone + our sell-dosing layer | episode 90341664 seat 0 | Subin An; prvsiyan (machinery) | `4ab4f772` |
| `v17_wheat109` | 1 | tape clone | episode 91222822 seat 1 | Matthew Labrador; prvsiyan (machinery) | `4f3e33da` |
| `v18_adaptive` | 1 | tape clone + our sell-timing overlay | episode 91222822 seat 1 | Matthew Labrador; prvsiyan (machinery) | `dd522d51` |
| `v19_wheat148` | 1 | tape clone + our sell-timing overlay | episode 91533875 seat 1 | Valmorlee; prvsiyan (machinery) | `cf6f1419` |
| `v21_thunder` | 1 | tape clone + our sell-timing overlay | episode 91414845 seat 0 | THUNDER THUNDER; prvsiyan (machinery) | `e00ac850` |
| `v4_look12` | 2 | our modification of public code | same base as c3bc; _ADV_LOOK 4 -> 12 and a route layer | authors of the upstream notebook (its NOTICE file); changes by this team | `8a262d93` |
| `v8_tran_new` | 1 | tape clone (soil_v7 machinery + another recorded game) | episode 89729218 seat 1 | Tran H Hoang; prvsiyan (machinery) | `c5830207` |
| `v9_auto` | 1 | tape clone | episode 89831498 seat 1 | automatylicza; prvsiyan (machinery) | `bab8c4ba` |
| `v9_manual` | 1 | tape clone | episode 89855909 seat 1 | 'manual player'; prvsiyan (machinery) | `6114b15a` |
| `v9_thwait` | 1 | tape clone | episode 89839704 seat 0 | thwaitsusa2; prvsiyan (machinery) | `52aeb370` |
| `v9_ueddy` | 1 | tape clone | episode 89860562 seat 1 | Ueddy; prvsiyan (machinery) | `6a59ca60` |

## Unavailable

| Folder | Note | Author | main.py sha256 |
|---|---|---|---|
| `pub2_saiteja_v14_reactive` | unknown public origin; not recorded; embeds tapes of episodes 89418172, 89448781, 89474114, 89475210, 89511601, 89634061 | unknown (folder name says 'saiteja') | `f646f039` |
| `pub_kaito_break` | copy of a public notebook (version not recorded); Kaito Fukami 'v26 break-the-copy-cycle' notebook (slug not recorded) | Kaito Fukami | `7996bed5` |
| `pub_kaito_cond` | copy of a public notebook (version not recorded); Kaito Fukami 'v21.1 conditional-memory' notebook (slug not recorded) | Kaito Fukami | `d9dc24ce` |
