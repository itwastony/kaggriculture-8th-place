"""Loading of BC demonstrations for rl/bc_train.py: shards of rl/gen_bc_tapes.py (or rl/gen_bc.py) -> tensor batches.

Shards hold raw observations; the encoding (rl/encode.py) happens in the DataLoader workers.
"""

import gzip
import pickle
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import encode as E
from . import spec as S


def sample_from(obs, ulab, mlab, reward):
    """One training sample: encoded observation + unit labels (target, op), market bins, game outcome and margin."""
    e = E.encode(obs)
    tgt = np.full(S.MAX_UNITS, -1, np.int64)
    op = np.zeros(S.MAX_UNITS, np.int64)
    for i, (t, o) in enumerate(ulab[: S.MAX_UNITS]):
        if e["unit_mask"][i]:
            tgt[i], op[i] = t, o
            e["tmask"][i, t, o] = True  # the label is always legal
    tgt_c = np.where(tgt < 0, 0, tgt)
    margin = float(reward[0] - reward[1])
    win = 1.0 if margin > 0 else (-1.0 if margin < 0 else 0.0)
    return {
        "tiles": e["tiles"],
        "units": e["units"],
        "unit_pos": e["unit_pos"],
        "unit_mask": e["unit_mask"],
        "glob": e["glob"],
        "tmask": e["tmask"],
        "mmask": e["mmask"],
        "target": tgt_c,
        "op": op,
        "has_lab": tgt >= 0,
        "mchoice": np.asarray(mlab, np.int64),
        "win": np.float32(win),
        "margin": np.float32(margin / 1e5),
    }


class ShardSteps(IterableDataset):
    """Endless stream of samples: shards are split over DDP ranks and loader workers, shuffled every epoch; each game
    step is kept with probability `keep` (decorrelation)."""

    def __init__(self, files, seed=0, keep=1.0, rank=0, world=1):
        self.files, self.seed, self.keep, self.rank, self.world = list(files), seed, keep, rank, world

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(self.seed * 7919 + self.rank * 101 + wid)
        files = self.files[self.rank :: self.world][wid::nw]
        if not files:  # fewer shards than workers: take all of them (otherwise an endless empty loop)
            files = list(self.files)
        epoch = 0
        while True:
            rng.shuffle(files)
            for p in files:
                with gzip.open(p, "rb") as f:
                    recs = pickle.load(f)
                items = [(r, t) for r in recs for t in range(len(r["obs"])) if rng.random() < self.keep]
                rng.shuffle(items)
                for r, t in items:
                    yield sample_from(r["obs"][t], r["ulab"][t], r["mlab"][t], r["reward"])
            epoch += 1


def collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.as_tensor(np.stack([b[k] for b in batch]))
    return out
