"""Trajectory recording: a game of two main.py agents on kagsim, saving (observation, action) pairs.

    from rl.record import play_record
    rewards, traj = play_record(seed, "agents/caretaker/main.py", "agents/pass_agent/main.py", seat=0)

traj = list of (obs_dict, action_dict) at every step for seat `seat` (obs is the raw dict from kagsim.observe); with
record_both=True, [traj of seat 0, traj of seat 1]. Agents are loaded and called as in harness/kagsim_play.py, with
their stdout/stderr suppressed. Used by entry_points/verify_submission.py --games and by rl/gen_bc.py; rl/appo.py and
rl/ppo.py take call_agent from here for the code-bot opponents.
"""

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from kagsim_play import _base_config, _load, fast_structify  # noqa: E402


def call_agent(fn, obs, cfg, path):
    """Call an agent as kaggle-environments does (structified observation and configuration, 1 or 2 arguments)."""
    o = fast_structify(obs)
    c = fast_structify(cfg)
    c["__raw_path__"] = path
    args = [o, c]
    if hasattr(fn, "__code__"):
        args = args[: fn.__code__.co_argcount]
    with io.StringIO() as so, io.StringIO() as se, contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        return fn(*args)


def play_record(seed, a, b, seat=0, steps=720, shops=None, record_both=False):
    """Play agent files a (seat 0) and b (seat 1); shops = a recorded shop order (default: the engine's own)."""
    import kagsim

    cfg = dict(_base_config())
    cfg["seed"] = seed
    paths = [a, b]
    fns = [_load(a), _load(b)]
    game = kagsim.Game(seed, steps, shops) if shops is not None else kagsim.Game(seed, steps)
    traj = [[], []]
    while not game.done:
        acts = [None, None]
        for p in (0, 1):
            obs = game.observe(p)
            acts[p] = call_agent(fns[p], obs, cfg, paths[p]) or {}
            if p == seat or record_both:
                traj[p].append((obs, acts[p]))
        game.step(acts[0], acts[1])
    rew = [game.reward(0), game.reward(1)]
    return rew, (traj if record_both else traj[seat])
