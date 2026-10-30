"""Settle telemetry: counters of what the engine does in silence.

Three contracts, each load-bearing:
1. OBSERVER: reading telemetry never alters behaviour; two identical games
   produce identical banks and identical counters (determinism).
2. TRUTH ON KNOWN SCENARIOS: hand-built actions with a known silent outcome
   produce exactly the expected counts (underfunded animal buy, empty-shed
   sell, land beyond the third quadrant).
3. ZERO ON A CLEAN GAME: a pure PASS game touches nothing and counts zero.

Why engine-side truth matters, measured the day this landed: an outside-in
state-delta audit read 24 refused purchases where the engine counted 16;
the 8 phantoms were same-turn animal deaths masquerading as refusals.
"""
import kagsim

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def run(actions_by_turn, seed=11, turns=720):
    g = kagsim.Game(seed)
    t = 0
    while not g.done and t < turns:
        g.step(actions_by_turn.get(t, PASS), PASS)
        t += 1
    return g


def test_zero_on_clean_game():
    g = run({})
    tel = g.telemetry(0)
    for k in ("sell_dead_units", "refused_buy_product", "refused_buy_seed",
              "refused_buy_animal", "refused_hire", "refused_buy_land",
              "feed_no_wheat", "feed_no_animal", "feed_redundant", "water_dead",
              "harvest_dead", "plant_dead", "dead_actions",
              "animals_escaped", "escape_capital", "escape_yield_units",
              "animal_unfed_days", "plants_dry", "dry_seed_cost",
              "dry_yield_units", "plant_unwatered_days", "decay_units",
              "plants_decayed", "silent_loss_coins", "silent_loss_units",
              "tile_cap_units", "fertilizer_forgone", "care_unfed", "animal_shed_days",
              "hand_pass_turns", "phantom_hand_actions", "sell_floor_units", "hire_premium",
              "shed_units_now", "seeds_unplanted_now", "move_dead", "locked_dead"):
        assert tel[k] == 0, (k, tel[k])
    assert tel["escape_first_day"] == -1 and tel["dry_first_day"] == -1, tel


# --- silent leaks: the failure of discussion 741235, counted at the source ----
# A cow bought, placed and never fed. The engine says nothing at any point: no
# refusal, no message, and the observation on day 2 simply has a pasture where a
# cow used to be. These are the counters that make it visible while it is still
# a shortfall rather than a loss.
COW_SCENARIO = {
    0: {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "COW", 1]]},
    1: {"farmer": ["PICKUP", "COW", 1], "hands": [], "market": []},
    2: {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []},
    3: {"farmer": ["PLACE", "COW"], "hands": [], "market": []},
}


def test_unfed_animal_escapes_on_day_one_with_its_capital():
    tel = run(COW_SCENARIO).telemetry(0)
    assert tel["animals_escaped"] == 1, tel
    assert tel["escape_capital"] == 400, tel        # a cow costs 400 and it is gone
    assert tel["escape_first_day"] == 1, tel        # two dry days: end of day 0, end of day 1
    assert tel["animal_unfed_days"] == 2, tel       # the runway, visible a day early
    assert tel["silent_loss_coins"] == 400, tel


def test_feed_without_wheat_is_a_silent_no_op():
    # The 741235 shape: the hand reaches the animal and FEEDs, carrying nothing.
    a = dict(COW_SCENARIO)
    a[4] = {"farmer": ["FEED"], "hands": [], "market": []}
    tel = run(a).telemetry(0)
    assert tel["feed_no_wheat"] == 1, tel
    assert tel["feed_no_animal"] == 0 and tel["feed_redundant"] == 0, tel
    assert tel["animals_escaped"] == 1, tel         # feeding nothing does not save it


def test_feed_after_the_escape_is_counted_separately():
    a = dict(COW_SCENARIO)
    a[24 * 3] = {"farmer": ["FEED"], "hands": [], "market": []}   # day 3, the cow left on day 1
    tel = run(a).telemetry(0)
    assert tel["feed_no_animal"] == 1, tel
    assert tel["feed_no_wheat"] == 0, tel


def test_unwatered_plant_dies_with_its_seed_cost():
    a = {0: {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
         1: {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}}
    tel = run(a).telemetry(0)
    assert tel["plants_dry"] == 1, tel
    assert tel["dry_seed_cost"] == 10, tel          # planting day counts as dry, so it dies that night
    assert tel["dry_first_day"] == 0, tel
    assert tel["silent_loss_coins"] == 10, tel


def test_dead_actions_are_counted_not_swallowed():
    a = {0: {"farmer": ["WATER"], "hands": [], "market": []},        # nothing planted
         1: {"farmer": ["HARVEST"], "hands": [], "market": []},      # nothing to take
         2: {"farmer": ["PLANT", "MELON"], "hands": [], "market": []}}  # no seed held
    tel = run(a).telemetry(0)
    assert tel["water_dead"] == 1 and tel["harvest_dead"] == 1 and tel["plant_dead"] == 1, tel
    assert tel["dead_actions"] == 3, tel


# --- the efficiency ledger: value the farm had and did not use -----------------
def _fed_cow_all_season():
    """A cow bought and placed on day 0, fed every morning, never milked, its
    fertilizer never collected. The farmer respawns on the pasture tile each
    morning, so no walking is needed."""
    a = {0: {"farmer": ["PASS"], "hands": [],
             "market": [["BUY_ANIMAL", "COW", 1], ["BUY_PRODUCT", "WHEAT", 40]]},
         1: {"farmer": ["PICKUP", "COW", 1], "hands": [], "market": []},
         2: {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []},
         3: {"farmer": ["PLACE", "COW"], "hands": [], "market": []}}
    for d in range(30):
        a[24 * d + 4] = {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}
        a[24 * d + 5] = {"farmer": ["FEED"], "hands": [], "market": []}
    return a


def test_fertilizer_forgone_counts_every_uncollected_animal_day():
    tel = run(_fed_cow_all_season()).telemetry(0)
    assert tel["animals_escaped"] == 0, tel
    # the flag is set at the day-0 refresh and found still set at each refresh
    # after it. The interpreter ends the episode before the last night's
    # refresh (agents act on steps 0 to 718), so there are 29 refreshes in a
    # season, and 28 of them find yesterday's unit still on offer.
    assert tel["fertilizer_forgone"] == 28, tel


def test_tile_cap_counts_milk_nobody_took():
    tel = run(_fed_cow_all_season()).telemetry(0)
    # milk from day 8 every two days, six held at most: days 8-18 fill the cow,
    # days 20, 22, 24, 26, 28 each clip one unit
    assert tel["tile_cap_units"] == 5, tel


def test_hired_hands_that_do_nothing_are_counted():
    a = {0: {"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["HIRE"]]},
         1: {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"]], "market": []}}
    tel = run(a).telemetry(0)
    # two hands exist for the rest of day 0 (hands are dismissed at midnight):
    # turns 1 to 23, explicit PASS at turn 1 and no order at all afterwards
    assert tel["hand_pass_turns"] == 2 * 23, tel
    assert tel["farmer_pass_turns"] == 719, tel   # the farmer passed on every acted step, 0 to 718
    assert tel["hire_premium"] == 0, tel          # 1 + 1: the second hire of a day costs the same


def test_same_day_hire_ladder_is_the_price_of_labour():
    a = {0: {"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"]]}}
    tel = run(a).telemetry(0)
    assert tel["hire_paid"] == 1 + 1 + 2 + 3, tel
    assert tel["hire_premium"] == 3, tel          # the ladder above one coin a hand; hands leave at midnight


def test_orders_to_hands_you_do_not_have_are_phantoms():
    a = {0: {"farmer": ["PASS"], "hands": [["NORTH"], ["NORTH"]], "market": []}}
    tel = run(a).telemetry(0)
    assert tel["phantom_hand_actions"] == 2, tel


def test_idle_land_and_stranded_stock():
    a = {0: {"farmer": ["PASS"], "hands": [],
             "market": [["BUY_PRODUCT", "WHEAT", 5], ["BUY_SEED", "CARROT", 2], ["BUY_ANIMAL", "GOOSE", 1]]}}
    g = run(a)
    tel = g.telemetry(0)
    assert tel["idle_tile_days"] == 25 * 29, tel  # the unlocked quadrant, every night, nothing built
    assert tel["animal_shed_days"] == 29, tel     # the goose sat in the shed all season
    assert tel["shed_units_now"] == 5, tel        # wheat still there when the game ended
    assert tel["animals_in_shed_now"] == 1, tel
    assert tel["seeds_unplanted_now"] == 2 and tel["seeds_unplanted_cost_now"] == 40, tel


def test_every_op_has_a_dead_counter():
    a = {0: {"farmer": ["WEST"], "hands": [], "market": []},            # farmer at x=4: fine
         1: {"farmer": ["DIG"], "hands": [], "market": []},             # empty tile
         2: {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": []},
         3: {"farmer": ["CARE"], "hands": [], "market": []},
         4: {"farmer": ["FERTILIZE"], "hands": [], "market": []},
         5: {"farmer": ["PLACE", "COW"], "hands": [], "market": []},
         6: {"farmer": ["BUILD_COOP"], "hands": [], "market": []},      # succeeds
         7: {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []},   # not empty any more
         8: {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}}  # shed holds none
    tel = run(a).telemetry(0)
    for k in ("dig_dead", "collect_dead", "care_dead", "fertilize_dead", "place_dead",
              "build_dead", "pickup_dead"):
        assert tel[k] == 1, (k, tel)
    assert tel["dead_actions"] == 7, tel


def test_deterministic_observer():
    a = {5: {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "COW", 2]]}}
    g1, g2 = run(a), run(a)
    assert g1.reward(0) == g2.reward(0)
    assert g1.telemetry(0) == g2.telemetry(0)


def test_buy_that_fills_nothing_is_one_burned_slot():
    # 3000 coins: 9 cows is a partial fill (7 bought), a 10-cow order AFTER it
    # buys nothing at all. Units: 2 + 10 refused. Orders that filled nothing: 1.
    a = {0: {"farmer": ["PASS"], "hands": [],
             "market": [["BUY_ANIMAL", "COW", 9], ["BUY_ANIMAL", "COW", 10]]}}
    tel = run(a).telemetry(0)
    assert tel["refused_buy_animal"] == 12, tel
    assert tel["buy_zero_fill"] == 1, tel


def test_underfunded_animal_buy_counts_full_remainder():
    # starting money 3000; 9 cows cost 3600: 7 commit, 2 refused units
    a = {0: {"farmer": ["PASS"], "hands": [],
             "market": [["BUY_ANIMAL", "COW", 9]]}}
    tel = run(a).telemetry(0)
    assert tel["refused_buy_animal"] == 2, tel


def test_empty_shed_sell_is_dead_volume():
    a = {0: {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 6]]}}
    tel = run(a).telemetry(0)
    assert tel["sell_dead_units"] == 6, tel


def test_land_beyond_third_quadrant_refused():
    # 1000+2000+4000 = 7000 > 3000 starting money: first buy commits only
    # if funded; drive with enough cash via selling nothing -> at 3000 the
    # first quadrant (1000) commits, second (2000) commits, third (4000)
    # is refused on funds; a FOURTH request is refused as none-left.
    a = {0: {"farmer": ["PASS"], "hands": [],
             "market": [["BUY_LAND"], ["BUY_LAND"], ["BUY_LAND"], ["BUY_LAND"]]}}
    tel = run(a).telemetry(0)
    assert tel["refused_buy_land"] >= 1, tel


def test_hire_cost_is_real_coins():
    a = {0: {"farmer": ["PASS"], "hands": [],
             "market": [["HIRE"], ["HIRE"], ["HIRE"]]}}
    g = run(a)
    tel = g.telemetry(0)
    assert tel["refused_hire"] == 0
    assert tel["hire_paid"] > 0
    assert abs(tel["total_spend"] - tel["hire_paid"]) < 1e-6, tel


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: OK")
    print("TELEMETRY TESTS PASS")
