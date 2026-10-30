"""Torch policy from a checkpoint: observation -> engine action (CPU or GPU). The reference for the NumPy agent.

    pol = Policy("ckpt.pt"); action = pol.act(obs)            # greedy
For games through harness/kagsim_play.py (an agent is a main.py file): write_agent_stub(ckpt, dir). rl/latency.py uses
Policy to check that the exported NumPy agent chooses the same actions (gate G2).
"""

import os

import numpy as np
import torch

from . import act as A
from . import encode as E
from . import spec as S
from .model import HEAD_SIZES, Net


def load_net(path, device="cpu"):
    """Checkpoint -> Net in eval mode (the architecture comes from the checkpoint's d/layers/vtower/pcritic)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net(d=ck.get("d", 256), layers=ck.get("layers", 6), vtower=ck.get("vtower", 0), pcritic=ck.get("pcritic", 0))
    net.load_state_dict(ck["model"])
    return net.to(device).eval()


def batch_of(encs, device="cpu"):
    keys = ("tiles", "units", "unit_pos", "unit_mask", "glob", "tmask", "mmask")
    return {k: torch.as_tensor(np.stack([e[k] for e in encs])).to(device) for k in keys}


def choose(lg, ml, b, greedy=True, gen=None):
    """-> (unit_choice [B,16] index target*NOPS+operation, market_choice [B,H])."""
    B = lg.shape[0]
    flat = lg.float().masked_fill(~b["tmask"], -1e9).view(B, S.MAX_UNITS, -1)
    if greedy:
        uc = flat.argmax(-1)
    else:
        uc = torch.distributions.Categorical(logits=flat).sample()
    mlm = ml.float().masked_fill(~b["mmask"], -1e9)
    mc, o = [], 0
    for n in HEAD_SIZES:
        x = mlm[:, o : o + n]
        mc.append(x.argmax(-1) if greedy else torch.distributions.Categorical(logits=x).sample())
        o += n
    return uc, torch.stack(mc, -1)


def to_action(obs, uc_row, mc_row):
    units = [(int(v) // S.NOPS, int(v) % S.NOPS) for v in uc_row]
    return A.execute(obs, units, [int(v) for v in mc_row])


class Policy:
    def __init__(self, path, device="cpu", threads=1):
        if threads:
            torch.set_num_threads(threads)
        self.dev = torch.device(device)
        self.net = load_net(path, self.dev)

    @torch.no_grad()
    def act(self, obs, greedy=True):
        e = E.encode(obs)
        b = batch_of([e], self.dev)
        lg, ml, _ = self.net(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"])
        uc, mc = choose(lg, ml, b, greedy)
        return to_action(obs, uc[0].tolist(), mc[0].tolist())


STUB = """# RL agent: policy from checkpoint {ckpt} (rl/policy.py). For local games, not for Kaggle.
import sys as _s
if {root!r} not in _s.path:
    _s.path.insert(0, {root!r})
from rl.policy import Policy as _P
_POL = None


def _plain(o):
    if isinstance(o, dict):
        return {{k: _plain(v) for k, v in o.items()}}
    if isinstance(o, list):
        return [_plain(v) for v in o]
    return o


def agent(observation, configuration=None):
    global _POL
    if _POL is None:
        _POL = _P({ckpt!r}, threads=1)
    return _POL.act(_plain(observation), greedy={greedy!r})
"""


def write_agent_stub(ckpt, out_dir, greedy=True):
    """Write out_dir/main.py that plays the checkpoint through Policy (local games only: needs this repository and
    torch). The file must not exist yet."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "main.py")
    with open(p, "x") as f:
        f.write(STUB.format(ckpt=os.path.abspath(ckpt), root=root, greedy=greedy))
    return p
