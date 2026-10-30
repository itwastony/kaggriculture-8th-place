# Empty agent (PASS only): opponent for measuring the solo bank.
def agent(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}
