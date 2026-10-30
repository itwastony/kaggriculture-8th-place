#!/usr/bin/env python3
"""Read the engine's silent-loss ledger for any two agents. Stdlib plus kagsim.

    python tools/silent_leaks.py --agent my_agent.py --rival theirs.py
    python tools/silent_leaks.py --agent my_agent.py --rival theirs.py --seeds 1-32 --both-seats
    python tools/silent_leaks.py --agent my_agent.py --rival theirs.py --require-zero-escapes

An agent file is whatever Kaggle runs: a Python file whose last module-level
callable of one or two arguments takes the observation dict and returns the
action dict. Nothing about it has to change to be measured here.

WHY THIS EXISTS. The environment refuses, starves and rots in silence. An
animal two days unfed is gone in the morning and the pasture looks exactly like
one that never held a cow; a crop nobody watered becomes a weed; a FEED issued
by a hand carrying no wheat changes nothing at all. None of it appears in the
observation, so the author sees a bank that came out lower and no reason why.

THREE LEDGERS, NOT ONE. Losses the engine inflicts (an animal gone, a crop dried,
produce rotted, the shed cap); actions it accepted and ignored; and EFFICIENCY,
the value the farm had and did not use: fertilizer on offer and never collected,
milk held at the animal's cap because nobody came, hands told to PASS, land left
idle, and everything still in the shed when the season ended, which the bank
counts as nothing. An instrument that measures one channel invites tuning to
that channel, so this one measures them all.

THE USEFUL HALF IS THE PRECURSOR. `animals_escaped` is a post mortem.
`feed_no_wheat` fires while the animal is still alive, a median of about one
game day earlier, which is time enough to buy one unit of wheat.

THE CONTROL IS NOT CEREMONY. The escape count is computed twice: once by the
engine at the site where it removes the animal, and once from the outside by
reading the observation at hour 23 for an animal with one unfed day that was
not fed today. The two share no code. When they disagree this prints DISAGREE
and you should trust neither until you know why.

ONE COUNTER IS DELIBERATELY NOT IN THE LEDGER. `sell_dead_units`, the SELL
volume the shed could not cover, reads in the tens of thousands for a perfectly
healthy agent, because the ordinary way to say "sell everything I have" is to
ask for a thousand units and let the settle fill what exists. It measures the
idiom. `sell_zero_fill`, an order that delivered nothing at all, measures a
market slot burned, and there are only ten a turn.

Apache-2.0, same as the engine. Part of github.com/destbreso/kaggriculture-cppsim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import kagsim

PASS = {"farmer": ["PASS"], "hands": [], "market": []}

LEDGER = (
    ("LOSS", "animals_escaped", "animals gone at two consecutive unfed days"),
    ("LOSS", "escape_capital", "their purchase price, in coins"),
    ("LOSS", "escape_yield_units", "product held on the tile and lost with it"),
    ("LOSS", "plants_dry", "plants turned to weed by two dry days"),
    ("LOSS", "dry_seed_cost", "their seed price, in coins"),
    ("LOSS", "plants_decayed", "plants that rotted all the way to weed"),
    ("LOSS", "decay_units", "yield units rotted off a plant left standing"),
    ("LOSS", "shed_discarded_units", "production the hundred-item cap destroyed"),
    ("PRECURSOR", "feed_no_wheat", "FEED on a hungry animal, carrier held no wheat"),
    ("PRECURSOR", "animal_unfed_days", "animal-days ending unfed: the runway to a loss"),
    ("PRECURSOR", "plant_unwatered_days", "plant-days ending unwatered"),
    ("DEAD ACTION", "feed_no_animal", "FEED at a tile whose animal already left"),
    ("DEAD ACTION", "feed_redundant", "FEED on an animal already fed today"),
    ("DEAD ACTION", "water_dead", "WATER on a non-plant or an already-watered tile"),
    ("DEAD ACTION", "harvest_dead", "HARVEST that took nothing"),
    ("DEAD ACTION", "plant_dead", "PLANT dropped for want of seed, or a busy tile"),
    ("DEAD ACTION", "pickup_dead", "PICKUP away from the shed, or of what it lacks"),
    ("DEAD ACTION", "place_dead", "PLACE with nothing to place, or nowhere to put it"),
    ("DEAD ACTION", "build_dead", "BUILD on a tile that is not empty"),
    ("DEAD ACTION", "dig_dead", "DIG on an empty tile, or on an animal"),
    ("DEAD ACTION", "fertilize_dead", "FERTILIZE on a non-plant, or with none carried"),
    ("DEAD ACTION", "collect_dead", "COLLECT_FERTILIZER with none available"),
    ("DEAD ACTION", "care_dead", "CARE on no animal, or twice in a day"),
    ("DEAD ACTION", "move_dead", "a move off the board"),
    ("DEAD ACTION", "locked_dead", "a tile action on locked land"),
    ("DEAD ACTION", "phantom_hand_actions", "orders to hands the farm does not have"),
    ("EFFICIENCY", "fertilizer_forgone", "animal-days whose fertilizer was on offer and never collected"),
    ("EFFICIENCY", "tile_cap_units", "yield clipped at a tile's holding cap: not harvested in time"),
    ("EFFICIENCY", "hand_pass_turns", "hired hands that did nothing this turn"),
    ("EFFICIENCY", "farmer_pass_turns", "the farmer's PASS turns, apart: the farmer is free, a hand costs a day"),
    ("EFFICIENCY", "idle_tile_days", "unlocked tiles empty or weed at the end of a day"),
    ("EFFICIENCY", "weed_tile_days", "of those, weeds nobody dug"),
    ("EFFICIENCY", "animal_shed_days", "animals still in the shed at the end of a day"),
    ("EFFICIENCY", "care_unfed", "CARE without FEED the same day: the bonus never accrues"),
    ("EFFICIENCY", "sell_floor_units", "units sold at the price floor of 1"),
    ("MARKET", "hire_premium", "the same-day hire ladder above one coin a hand: the price of labour, hands leave at midnight"),
    ("STRANDED AT THE END", "shed_units_now", "units still in the shed when the game ended: worth nothing"),
    ("STRANDED AT THE END", "shed_value_now", "what they would have fetched at the final prices"),
    ("STRANDED AT THE END", "animals_in_shed_now", "animals bought and never placed"),
    ("STRANDED AT THE END", "seeds_unplanted_cost_now", "coins of seed never planted"),
    ("MARKET", "sell_zero_fill", "a SELL order that delivered nothing, one slot burned"),
    ("MARKET", "buy_zero_fill", "a BUY order that bought nothing: usually a broken financing chain"),
    ("MARKET", "refused_buy_product", "BUY_PRODUCT units refused on funds or capacity"),
    ("MARKET", "refused_buy_seed", "BUY_SEED units refused on funds"),
    ("MARKET", "refused_buy_animal", "BUY_ANIMAL units refused on funds or capacity"),
    ("MARKET", "refused_hire", "HIREs refused on funds or the unit cap"),
    ("MARKET", "refused_buy_land", "BUY_LANDs refused on funds, or none left"),
)


def load_agent(path):
    """Import a Kaggle agent file and return its entry point.

    Kaggle runs the LAST module-level callable taking one or two arguments,
    which is why several public agents end with `agent = globals().pop('agent')`:
    re-binding a name does not move it to the end of the module dict, so without
    that line an inner helper can win.
    """
    path = pathlib.Path(path)
    tag = f"_agent_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(tag, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    d = str(path.parent)
    added = d not in sys.path
    if added:
        sys.path.insert(0, d)
    try:
        spec.loader.exec_module(module)
    finally:
        if added and sys.path and sys.path[0] == d:
            sys.path.pop(0)
        sys.modules.pop(tag, None)
    fns = [v for v in vars(module).values()
           if callable(v) and getattr(v, "__module__", "") == module.__name__
           and getattr(v, "__code__", None) and v.__code__.co_argcount in (1, 2)]
    if not fns:
        raise SystemExit(f"{path} declares no entry point")
    return fns[-1]


def observed_escapes(observation, seat, step):
    """The CONTROL, computed from the observation and nothing else.

    An animal whose unfed counter already stands at one and that was not fed
    today reaches two at tonight's refresh and is gone. Read at hour 23.
    """
    if step % 24 != 23:
        return 0
    farms = observation.get("farms") or []
    if len(farms) <= seat:
        return 0
    n = 0
    for row in (farms[seat].get("tiles") or []):
        for tile in row or []:
            if (isinstance(tile, dict) and "animal" in tile
                    and int(tile.get("consecutive_unfed", 0)) >= 1
                    and not tile.get("fed_today")):
                n += 1
    return n


def play(agent, rival, seed, seat):
    """One episode. Returns the engine ledger and the control's escape count."""
    game = kagsim.Game(seed)
    control = 0
    for step in range(720):
        if game.done:
            break
        mine = dict(game.observe(seat) or {})
        mine["player"] = seat
        theirs = dict(game.observe(1 - seat) or {})
        theirs["player"] = 1 - seat
        control += observed_escapes(mine, seat, step)
        a, b = agent(mine), rival(theirs) if rival else PASS
        game.step(a if seat == 0 else b, b if seat == 0 else a)
    return (dict(game.telemetry(seat)), control,
            game.reward(seat), game.reward(seat) - game.reward(1 - seat))


def _name(path):
    """Kaggle agents are all called main.py, so the directory is the name."""
    p = pathlib.Path(path)
    return p.parent.name if p.stem == "main" else p.stem


def parse_seeds(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", required=True, help="the agent under test")
    ap.add_argument("--rival", default=None, help="a second agent file, or omit for an idle rival")
    ap.add_argument("--seeds", default="1-16", help="e.g. 1-16 or 3,7,11")
    ap.add_argument("--both-seats", action="store_true",
                    help="play each seed from both seats; the market settles player 0 first")
    ap.add_argument("--require-zero-escapes", action="store_true", help="exit 1 on any loss")
    ap.add_argument("--json", default=None, help="write the per-game rows here")
    a = ap.parse_args()

    agent = load_agent(a.agent)
    rival = load_agent(a.rival) if a.rival else None
    seeds = parse_seeds(a.seeds)
    seats = (0, 1) if a.both_seats else (0,)
    label = f"{_name(a.agent)} against {_name(a.rival) if a.rival else 'idle'}"
    print(f"{label}: {len(seeds) * len(seats)} games, playing...", flush=True)

    rows, total, control = [], {}, 0
    for seed in seeds:
        for seat in seats:
            tel, ctl, bank, margin = play(agent, rival, seed, seat)
            control += ctl
            rows.append({"seed": seed, "seat": seat, "bank": bank, "margin": margin,
                         "observed_escapes": ctl,
                         "telemetry": {k: (float(v) if isinstance(v, float) else int(v))
                                       for k, v in tel.items()}})
            for k, v in tel.items():
                if isinstance(v, (int, float)) and k not in ("escape_first_day", "dry_first_day"):
                    total[k] = total.get(k, 0) + v

    n = len(rows)
    engine = total.get("animals_escaped", 0)
    print(f"\n{label}: {n} games\n")
    print(f"escapes: {engine} (engine), {control} (observation control) "
          f"{'AGREE' if engine == control else 'DISAGREE, trust neither until you know why'}")
    hit = [r for r in rows if r["telemetry"]["animals_escaped"]]
    first = [r["telemetry"]["escape_first_day"] for r in hit
             if r["telemetry"]["escape_first_day"] >= 0]
    print(f"games losing an animal: {len(hit)} of {n}"
          + (f", first on day {min(first)}" if first else ""))
    for r in sorted(hit, key=lambda r: -r["telemetry"]["escape_capital"])[:8]:
        t = r["telemetry"]
        print(f"  seed {r['seed']:>5} seat {r['seat']}  {t['animals_escaped']} animal(s) "
              f"from day {t['escape_first_day']}, {t['escape_capital']:,.0f} coins, "
              f"feed_no_wheat {t['feed_no_wheat']}, bank {r['bank']:,.0f} "
              f"margin {r['margin']:+,.0f}")

    print("\nPER GAME, mean:")
    kind = None
    for k, key, what in LEDGER:
        if k != kind:
            print(f"\n  {k}")
            kind = k
        print(f"    {key:<22} {total.get(key, 0) / n:>10,.2f}   {what}")
    print(f"\n  {'SILENT LOSS, coins':<22} {total.get('silent_loss_coins', 0) / n:>10,.0f}"
          f"   {'units':<8} {total.get('silent_loss_units', 0) / n:>8,.1f}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {a.json}")
    if a.require_zero_escapes and engine:
        print(f"\nFAIL: {engine:,} animal(s) escaped, "
              f"{total.get('escape_capital', 0):,.0f} coins")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
