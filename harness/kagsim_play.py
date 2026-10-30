#!/usr/bin/env python3
"""Play two Python agents (`main.py` files) on kagsim instead of kaggle_environments.

kagsim (vendor/kagsim) is a C++ port of the Kaggriculture engine, bit-exact with kaggle-environments 1.32.7. Agents are
loaded and called as kaggle_environments does it (agent.build_agent / Agent.act):
  - the file's source is re-executed for every game, with the agent's folder on sys.path (get_last_callable(path=...));
  - the agent receives structify(observation) and structify(configuration), the latter with __raw_path__ = file path;
  - the agent's stdout/stderr are suppressed; an exception makes its status ERROR and ends the game (as in env.step).

    from kagsim_play import play_kagsim
    rewards, statuses = play_kagsim(seed, "agents/A/main.py", "agents/B/main.py")

    python harness/kagsim_play.py [SEED]    # caretaker vs pass_agent, prints the rewards (run from the repository root)

Parity with the official environment: vendor/kagsim/tests (test_golden.py, test_l1.py). Requires kagsim
(scripts/build_engines.sh) and kaggle-environments (for the configuration and Struct).
"""

import contextlib
import io
import os

_CFG = None  # kaggriculture environment configuration (as env.configuration), without the seed
_CODE = {}  # path -> compiled agent source (compiled once per process, executed for every game)


def fast_structify(o):
    """Same as kaggle_environments.utils.structify (Struct objects with dot access, the 'items' key
    is dropped), but without **kwargs and extra copies: an observation of ~1300 nodes is built several times faster."""
    from kaggle_environments.utils import Struct

    new = Struct.__new__

    def rec(x):
        if isinstance(x, list):
            return [rec(v) for v in x]
        if isinstance(x, dict):
            e = {k: rec(v) for k, v in x.items() if k != "items"}
            s = new(Struct)
            dict.update(s, e)
            s.__dict__.update(e)
            return s
        return x

    return rec(o)


def _base_config():
    global _CFG
    if _CFG is None:
        from kaggle_environments import make

        _CFG = dict(make("kaggriculture").configuration)
    return _CFG


def _load(path):
    """Like kaggle_environments.agent.get_last_callable(raw, path=path): fresh globals for every game,
    the agent's folder is in sys.path during execution, agent = the module's last callable."""
    import sys

    if path not in _CODE:
        with open(path) as f:
            _CODE[path] = compile(f.read(), path, "exec")
    env = {}
    sys.path.append(os.path.dirname(path))
    try:
        exec(_CODE[path], env)
    finally:
        sys.path.pop()
    return [v for v in env.values() if callable(v)][-1]


def play_kagsim(seed, a, b, steps=720, shops=None):
    """Play agent files a (seat 0) and b (seat 1) on seed; shops pins the shop opening order (up to 8 shop names or
    indices, see kagsim.Game).

    Returns (rewards, statuses): ([bank0, bank1], ["DONE", "DONE"]) for a finished game. If an agent raised, the game
    stops after that step; the failing seat gets reward None and status "ERROR", the other seat keeps its bank and
    status "ACTIVE"."""
    import kagsim

    structify = fast_structify
    cfg = dict(_base_config())
    cfg["seed"] = seed
    paths = [a, b]
    fns = [None, None]  # as in build_agent: the agent is loaded on the first call
    status = ["ACTIVE", "ACTIVE"]
    game = kagsim.Game(seed, steps, shops) if shops is not None else kagsim.Game(seed, steps)
    while not game.done:
        acts = [None, None]
        for p in (0, 1):
            obs = structify(game.observe(p))
            conf = structify(cfg)
            conf["__raw_path__"] = paths[p]
            with io.StringIO() as o, io.StringIO() as e, contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                try:
                    if fns[p] is None:
                        fns[p] = _load(paths[p])
                    fn = fns[p]
                    args = [obs, conf]
                    if hasattr(fn, "__code__"):
                        args = args[: fn.__code__.co_argcount]
                    acts[p] = fn(*args)
                except Exception:
                    status[p] = "ERROR"
        if "ERROR" in status:
            break
        game.step(acts[0] or {}, acts[1] or {})
    if "ERROR" in status:
        return [None if s == "ERROR" else game.reward(i) for i, s in enumerate(status)], status
    return [game.reward(0), game.reward(1)], ["DONE", "DONE"]


if __name__ == "__main__":
    import sys
    import time

    s = int(sys.argv[1]) if len(sys.argv) > 1 else 424242
    t = time.time()
    r = play_kagsim(s, "agents/caretaker/main.py", "agents/pass_agent/main.py")
    print(s, r, f"{time.time() - t:.2f}s")
