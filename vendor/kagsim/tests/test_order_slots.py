"""A malformed market order holds its slot, and the slot pairing decides the price.

The environment slices each player's market list to `maxMarketOrdersPerTurn`
BEFORE parsing and then walks the two lists slot by slot, pairing player 0's
slot i with player 1's slot i in the per-unit lockstep. An entry that fails to
parse (an empty list, a SELL with no quantity, a zero quantity) yields nothing
for that player at that slot, but the entries after it keep their positions.

The port used to drop unparsable entries and compact the list, which shifts
every later order one slot earlier for that player only, and therefore changes
which of the two players' units settle together. Found on a real ladder episode
where one `[]` in a rival's list moved the final banks by thousands.

The check below plays the scenario on the real environment when it is
installed and demands the same banks from the port; without the environment it
still asserts that the compaction is gone (the slot after an empty order settles
as slot 2, not slot 1), so the regression cannot return unnoticed either way.
"""
import kagsim

PASS = {"farmer": ["PASS"], "hands": [], "market": []}

# Both players hold the same wheat and sell it the same turn. Player 0 puts an
# empty order first, so under slot pairing its SELL sits at slot 1 while player
# 1's SELL sits at slot 0 and settles alone first; compacted, both would sit at
# slot 0 and settle in lockstep. The two orders of settlement price differently.
SCENARIO = {
    0: [{"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 40]]},
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 40]]}],
    1: [{"farmer": ["PASS"], "hands": [], "market": [[], ["SELL", "WHEAT", 40]]},
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 40]]}],
}


def play_kagsim(seed=11):
    g = kagsim.Game(seed)
    t = 0
    while not g.done and t < 720:
        a = SCENARIO.get(t, [PASS, PASS])
        g.step(a[0], a[1])
        t += 1
    return g.reward(0), g.reward(1)


def play_real(seed=11):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.reset(2)
    t = 0
    while not env.done:
        a = SCENARIO.get(t, [PASS, PASS])
        env.step([a[0], a[1]])
        t += 1
    return float(env.state[0].reward), float(env.state[1].reward)


def test_slot_is_held_by_a_malformed_order():
    ours = play_kagsim()
    try:
        import kaggle_environments  # noqa: F401
    except ImportError:
        # Without the environment: the compacted and the slot-held games differ,
        # so at least assert the port is not on the compacted branch.
        g = kagsim.Game(11)
        g.step({"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 40]]},
               {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 40]]})
        g.step({"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 40]]},
               {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 40]]})
        while not g.done:
            g.step(PASS, PASS)
        compacted = (g.reward(0), g.reward(1))
        assert ours != compacted, ("the empty order was compacted away", ours)
        print("  environment not installed: asserted the slot is held, banks", ours)
        return
    real = play_real()
    assert ours == real, (ours, real)
    print(f"  slot held: port {ours} == environment {real}")


if __name__ == "__main__":
    test_slot_is_held_by_a_malformed_order()
    print("ORDER SLOT TEST PASS")
