"""Build a self-contained agent from a checkpoint (NumPy, no torch): ROOT/<name>/{main.py, weights.npz, rl_<sha8>/}.

    python -m rl.export_agent --ckpt runs/d7/appo/ema_00110.pt --name my_agent --root runs/d7/export \\
        --dtype float16 --land-inject 248:72

entry_points/export_submission.py runs it with configs/export_d7b.json and packs the folder into the submission archive.

- rl_<sha8>/: copies of rl/{spec,encode,act,np_net}.py; the name is the start of the sha256 of their source, so two
  exported versions never mix in one process.
- weights.npz: the actor weights (np_net.export_weights; the value towers are not exported), --dtype float32 or float16.
- main.py: generated from the MAIN template (+ PREFIX, LATE or ROUTE for the multi-network options). It finds its folder
  through co_filename (Kaggle runs the file with exec, so there is no __file__) or /kaggle_simulations/agent. The choice
  is greedy (argmax), as in rl/policy.py, unless --temp T > 0. The inference-time options (--land-inject, --land-steps,
  --buy-cap, --term-sell, the opponent-view masks) become constants in main.py.

The output folder must not exist yet (nothing is overwritten). Prints "<folder> <package name>".
"""

import argparse
import hashlib
import os
import shutil

from . import np_net

# typical farm development of r2e220 at the end of days 0..28 (median over 329 top-team seats of our test set
# "topswap"); the reference of --self-gate
STD = [
    32,
    32,
    35,
    37,
    37,
    37,
    58,
    72,
    78,
    81,
    85,
    114,
    118,
    117,
    117,
    117,
    119,
    119,
    119,
    119,
    119,
    118,
    119,
    119,
    119,
    119,
    118,
    106,
    79,
]
# Template of the submitted main.py (str.format; {{ }} are literal braces). entry_points/verify_submission.py compares
# the generated program with the archives sent to Kaggle: do not edit the template text, its comments included.
MAIN = """# RL agent (transformer policy, numpy). Checkpoint: {ckpt}, weights {dtype}.
import os as _os
import sys as _sys


def _here():
    f = globals().get("__file__") or _here.__code__.co_filename
    return _os.path.dirname(_os.path.abspath(f))


_D = _here()
for _c in (_D, "/kaggle_simulations/agent", _os.getcwd()):
    if _os.path.exists(_os.path.join(_c, "{pkg}", "np_net.py")):
        _D = _c
        break
if _D not in _sys.path:
    _sys.path.insert(0, _D)
from {pkg} import act as _A, encode as _E, np_net as _N  # noqa: E402
from {pkg}.spec import MARKET_HEADS as _MH, ORDER_TEMPLATES as _OT  # noqa: E402

_HS = [len(b) for _, _, b in _MH]
_W = _N.load_weights(_os.path.join(_D, "weights.npz"))


def _plain(o):
    if isinstance(o, dict):
        return {{k: _plain(v) for k, v in o.items()}}
    if isinstance(o, list):
        return [_plain(v) for v in o]
    return o


_MASK = {mask!r}


_GATE = {gate!r}                    # None | (day_from, day_to, rho): mirror from the moment when, within the day window, the opponent's
_G = {{"on": False, "last": -1}}      # farm development < rho x ours (latched until the end of the game; reset on a new game)


def _dev(f):                        # farm development: plantings + 3 x animals + 5 x unlocked land quadrants beyond the first
    n = 0
    for row in f["tiles"]:
        for c in row:
            if isinstance(c, dict):
                if c.get("kind") == "PLANT":
                    n += 1
                elif c.get("animal"):
                    n += 3
    return n + 5 * (len(f.get("unlocked_quadrants") or []) - 1)


_OPEN = {open_steps!r}               # mirror for the first N turns of the game (opening), then the real view (0 = off)
_SELF = {self_gate!r}               # None | (day_from, day_to, fraction): mirror until the end of the game if at the start of a day d in the window
_STD = {std!r}                      # own development (end of day d-1) < fraction x _STD[d-1] (typical r2e220 trajectory in topswap)


def _mode(o):
    if not (_GATE or _OPEN or _SELF):
        return _MASK
    st, day, me = int(o.get("step", 0)), int(o.get("day", 0)), int(o["player"])
    if st <= _G["last"]:
        _G["on"] = False
    _G["last"] = st
    if not _G["on"] and _GATE:
        d0, d1, rho = _GATE
        if d0 <= day <= d1 and _dev(o["farms"][1 - me]) < rho * _dev(o["farms"][me]):
            _G["on"] = True
    if not _G["on"] and _SELF and st % 24 == 0 and st > 0:
        d0, d1, fr = _SELF
        if d0 <= day <= d1 and _dev(o["farms"][me]) < fr * _STD[min(day - 1, len(_STD) - 1)]:
            _G["on"] = True
    return "mirror" if (_G["on"] or st < _OPEN) else "none"


def _seen(o):                       # what the policy sees of the opponent: "mirror" — a copy of its own farm (as in self-play),
    mode = _mode(o)                 # "money" — only the opponent's money = own
    if mode == "none":
        return o
    me = int(o["player"])
    f = list(o["farms"])
    f[1 - me] = f[me] if mode == "mirror" else dict(f[1 - me], money=f[me]["money"])
    return dict(o, farms=f)


_LAND = {land!r}                    # [(step, unlocked land quadrants, price)]: from this step on, buy land if the policy did not
_LINJ = {linj!r}                    # None | (step, window): like rl/appo.py --land-inject — on turns [step, step+window), if 3 land quadrants and money >= 4000,
_HL = [k for k, _, _ in _MH].index("BUY_LAND")   # set BUY_LAND = 1 in the market choice and the order template SLPHDA
_HO = [k for k, _, _ in _MH].index("ORDER")
_OSL = _OT.index("SLPHDA")
_BCAP = {bcap!r}                    # {{item: C}}: BUY_PRODUCT of the item is capped at C - (in the shed) - (already bought this turn)
_TSELL = {tsell!r}                  # True: on the last executed turn (step >= episodeSteps - 2 = 718) append SELL of the shed leftovers
_TPROD = ("TOMATO", "STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "CARROT", "WHEAT", "FERTILIZER")   # after the policy's orders (those are left unchanged)


def _last_step(cfg):
    try:
        n = cfg.get("episodeSteps") if hasattr(cfg, "get") else getattr(cfg, "episodeSteps", None)
    except Exception:  # noqa: BLE001
        n = None
    return int(n or 720) - 2


_TEMP = {temp!r}                    # 0 = greedy (argmax); > 0 — choice by softmax(logits / _TEMP) (Gumbel-max, own RNG)
_RNG = []


def _choose(e):
    if not _TEMP:
        return _N.greedy(_W, e, _HS)
    import numpy as _np
    if not _RNG:
        _RNG.append(_np.random.default_rng())
    g = _RNG[0]
    lg, ml, _ = _N.forward(_W, e)
    U, NT, nops = lg.shape
    l = _np.where(e["tmask"], lg / _TEMP, -1e9).reshape(U, NT * nops)
    l = (l + g.gumbel(size=l.shape)).argmax(-1)
    units = [(int(v) // nops, int(v) % nops) for v in l]
    mlm = _np.where(e["mmask"], ml / _TEMP, -1e9)
    mc, o = [], 0
    for n in _HS:
        mc.append(int((mlm[o:o + n] + g.gumbel(size=n)).argmax()))
        o += n
    return units, mc


def agent(observation, configuration=None):
    o = _plain(observation)
    units, mc = _choose(_E.encode(_seen(o)))
    if _LINJ:
        st, f = int(o.get("step", 0)), o["farms"][int(o["player"])]
        if _LINJ[0] <= st < _LINJ[0] + _LINJ[1] and len(f.get("unlocked_quadrants") or []) == 3 and float(f["money"]) >= 4000 - 0.5:
            mc = list(mc)
            mc[_HL], mc[_HO] = 1, _OSL
    a = _A.execute(o, units, mc)
    if _LAND:
        st, f = int(o.get("step", 0)), o["farms"][int(o["player"])]
        n, mk = len(f.get("unlocked_quadrants") or []), list(a.get("market") or [])
        for t, k, c in _LAND:
            if st >= t and n == k and f["money"] >= c and not any(m and m[0] == "BUY_LAND" for m in mk):
                a = dict(a, market=[["BUY_LAND"]] + mk[:9])
                break
    if _BCAP:
        shed, got, mk = (o.get("private") or {{}}).get("shed") or {{}}, {{}}, []
        for m in list(a.get("market") or []):
            if m and m[0] == "BUY_PRODUCT" and m[1] in _BCAP:
                q = min(int(m[2]), max(0, _BCAP[m[1]] - int(shed.get(m[1], 0) or 0) - got.get(m[1], 0)))
                if q <= 0:
                    continue
                got[m[1]] = got.get(m[1], 0) + q
                m = [m[0], m[1], q]
            mk.append(m)
        a = dict(a, market=mk)
    if _TSELL and int(o.get("step", 0)) >= _last_step(configuration):
        shed, pr = (o.get("private") or {{}}).get("shed") or {{}}, (o.get("market") or {{}}).get("prices") or {{}}
        mk, sold = list(a.get("market") or []), {{}}
        for m in mk:
            if m and m[0] == "SELL" and len(m) >= 3:
                sold[m[1]] = sold.get(m[1], 0) + int(m[2])
        left = sorted(((float(pr.get(k, 1) or 1) * q, k, q) for k in _TPROD for q in [int(shed.get(k, 0) or 0) - sold.get(k, 0)] if q > 0),
                      reverse=True)
        mk += [["SELL", k, q] for _, k, q in left[:max(0, 10 - len(mk))]]
        a = dict(a, market=mk)
    return a
"""  # noqa: E501

# Hybrid: the first {steps} turns are played by the code prefix_agent.py (executed in full in its own namespace, agent =
# the last callable, as in kaggle_environments), then the RL policy from the farm state the code has built.
PREFIX = """

_PRE_FN = None
_PRE_STEPS = {steps}


def _pre():
    global _PRE_FN
    if _PRE_FN is None:
        _p = _os.path.join(_D, "prefix_agent.py")
        _env = {{"__name__": "prefix_agent"}}
        with open(_p) as _f:
            exec(compile(_f.read(), _p, "exec"), _env)
        _PRE_FN = [v for v in _env.values() if callable(v)][-1]
    return _PRE_FN


def hybrid_agent(observation, configuration=None):   # new name: the last callable of the module (redefining
    try:                                                # agent would keep the key's old position in globals)
        _st = int(observation["step"])
    except Exception:  # noqa: BLE001
        _st = int(getattr(observation, "step", 0))
    if _st < _PRE_STEPS:
        _fn = _pre()
        _n = getattr(getattr(_fn, "__code__", None), "co_argcount", 2)
        return _fn(observation, configuration) if _n >= 2 else _fn(observation)
    return agent(observation, configuration)
"""

# Two networks by turn: before turn {steps} the main checkpoint plays (weights.npz), from turn {steps} on --late-ckpt
# (weights_late.npz). One forward pass per turn (same turn time); the opening is bit-identical to the main checkpoint's.
LATE = """

_WL = _N.load_weights(_os.path.join(_D, "weights_late.npz"))
_W_EARLY = _W
_LATE_FROM = {steps}


def late_agent(observation, configuration=None):    # the last callable of the module (like hybrid_agent)
    global _W
    try:
        _st = int(observation["step"])
    except Exception:  # noqa: BLE001
        _st = int(getattr(observation, "step", 0))
    _W = _WL if _st >= _LATE_FROM else _W_EARLY
    return agent(observation, configuration)
"""

# Network switch (one forward pass per turn): the main network by default; --fast-ckpt (weights_fast.npz) from the
# turn on which the opponent first has >= {quads} unlocked land quadrants, if that happened before turn {by}, until
# turn {until} (latched per game, like the --fast-gate window in rl/appo.py); --late-ckpt (weights_late.npz, if
# present) from turn {late}.
ROUTE = """

_WF = _N.load_weights(_os.path.join(_D, "weights_fast.npz"))
_WLT = _N.load_weights(_os.path.join(_D, "weights_late.npz")) if {has_late} else None
_W_MAIN = _W
_FQ, _FBY, _FUNTIL, _LATE_FROM = {quads}, {by}, {until}, {late}
_FS = {{"on": False, "last": -1, "net": "main"}}


def route_agent(observation, configuration=None):   # the last callable of the module (like late_agent)
    global _W
    o = _plain(observation)
    st, me = int(o.get("step", 0)), int(o.get("player", 0))
    if st <= _FS["last"]:                           # new game
        _FS["on"] = False
    _FS["last"] = st
    if not _FS["on"] and st < _FBY and len(o["farms"][1 - me].get("unlocked_quadrants") or []) >= _FQ:
        _FS["on"] = True
    if _WLT is not None and st >= _LATE_FROM:
        _W, _FS["net"] = _WLT, "late"
    elif _FS["on"] and st < _FUNTIL:
        _W, _FS["net"] = _WF, "fast"
    else:
        _W, _FS["net"] = _W_MAIN, "main"
    return agent(o, configuration)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--root", default="agents")
    ap.add_argument("--prefix", default="")  # code that plays the first --prefix-steps turns
    ap.add_argument("--prefix-steps", type=int, default=0)
    # second network: plays from turn --late-from to the end of the game
    ap.add_argument("--late-ckpt", default="")
    ap.add_argument("--late-from", type=int, default=0)
    # network against a "fast" opponent (window as in --fast-gate of rl/appo.py)
    ap.add_argument("--fast-ckpt", default="")
    ap.add_argument("--fast-quads", type=int, default=3)
    ap.add_argument("--fast-by", type=int, default=216)
    ap.add_argument("--fast-until", type=int, default=432)
    ap.add_argument("--mask-opp", default="none", choices=["none", "mirror", "money"])
    # "2,6,0.6": mirror if on days 2..6 the opponent's development < 0.6 x ours
    ap.add_argument("--gate", default="")
    ap.add_argument("--open-mirror", type=int, default=0)  # mirror the opponent's view for the first N turns
    ap.add_argument("--self-gate", default="")  # "4,9,0.75": mirror if own development < 0.75 x typical (STD)
    ap.add_argument("--buy-cap", default="")  # "WHEAT:10,FERTILIZER:0": cap on the shed stock for BUY_PRODUCT
    # "150,265": safety-net purchase of the 2nd/3rd land quadrant from these turns (prices 1000/2000)
    ap.add_argument("--land-steps", default="")
    # "248:72": purchase of the 4th land quadrant as in training (rl/appo.py --land-inject/--land-window)
    ap.add_argument("--land-inject", default="")
    # 1: on the last executed turn (718) append SELL of the shed leftovers
    ap.add_argument("--term-sell", type=int, default=0)
    # 0 = greedy; > 0: choice with temperature (all of the agent's networks)
    ap.add_argument("--temp", type=float, default=0.0)
    x = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    srcs = ["spec.py", "encode.py", "act.py", "np_net.py"]
    h = hashlib.sha256()
    for s in srcs:
        h.update(open(os.path.join(here, s), "rb").read())
    pkg = "rl_" + h.hexdigest()[:8]
    out = os.path.join(x.root, x.name)
    os.makedirs(out)  # new folder: nothing is overwritten
    os.makedirs(os.path.join(out, pkg))
    open(os.path.join(out, pkg, "__init__.py"), "w").close()
    for s in srcs:
        shutil.copyfile(os.path.join(here, s), os.path.join(out, pkg, s))
    np_net.export_weights(x.ckpt, os.path.join(out, "weights.npz"), x.dtype)
    prices = [1000, 2000, 4000]  # LAND_PRICES of engine 1.32.7
    land = (
        [(int(t), i + 1, prices[i]) for i, t in enumerate(x.land_steps.split(",")) if t.strip()] if x.land_steps else []
    )
    linj = tuple(int(v) for v in x.land_inject.split(":")) if x.land_inject else None
    body = MAIN.format(
        ckpt=os.path.abspath(x.ckpt),
        dtype=x.dtype,
        pkg=pkg,
        mask=x.mask_opp,
        land=land,
        linj=linj,
        bcap={k: int(v) for k, v in (kv.split(":") for kv in x.buy_cap.split(",") if kv)},
        gate=tuple(float(v) if i == 2 else int(v) for i, v in enumerate(x.gate.split(","))) if x.gate else None,
        open_steps=x.open_mirror,
        std=STD,
        temp=x.temp,
        tsell=bool(x.term_sell),
        self_gate=tuple(float(v) if i == 2 else int(v) for i, v in enumerate(x.self_gate.split(",")))
        if x.self_gate
        else None,
    )
    if x.prefix:
        shutil.copyfile(x.prefix, os.path.join(out, "prefix_agent.py"))
        body += PREFIX.format(steps=x.prefix_steps)
    if x.late_ckpt:
        assert not x.prefix and x.late_from > 0
        np_net.export_weights(x.late_ckpt, os.path.join(out, "weights_late.npz"), x.dtype)
    if x.fast_ckpt:
        assert not x.prefix and not (x.gate or x.open_mirror or x.self_gate)
        np_net.export_weights(x.fast_ckpt, os.path.join(out, "weights_fast.npz"), x.dtype)
        body += ROUTE.format(
            has_late=bool(x.late_ckpt),
            quads=x.fast_quads,
            by=x.fast_by,
            until=x.fast_until,
            late=x.late_from if x.late_ckpt else 10**9,
        )
    elif x.late_ckpt:
        body += LATE.format(steps=x.late_from)
    with open(os.path.join(out, "main.py"), "x") as f:
        f.write(body)
    print(out, pkg)


if __name__ == "__main__":
    main()
