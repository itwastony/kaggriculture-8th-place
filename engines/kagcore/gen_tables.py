"""Generator of the sin/cos(2*pi*hour/24) tables of kagcore.hpp, computed exactly as in rl/encode.py (np.sin/np.cos in
double, then cast to float32).

Prints the C++ hex-float literals and the margin (in double ulp) to the nearest float32 rounding boundary: when the
margin is large, differences between libm/NumPy/CUDA in the last bits of the double do not change the float32 value.
The tables are printed on import (the script has no main guard):

    python engines/kagcore/gen_tables.py
"""

import numpy as np


def margin_ulps(d):
    """Distance (in double ulp) from d to the nearest midpoint between adjacent float32 values."""
    f = np.float32(d)
    lo = np.nextafter(f, np.float32(-np.inf))
    hi = np.nextafter(f, np.float32(np.inf))
    mids = [(float(f) + float(lo)) / 2, (float(f) + float(hi)) / 2]
    u = np.spacing(abs(d)) if d != 0 else np.spacing(0.0)
    return min(abs(d - m) for m in mids) / u


for name, fn in (("SIN", np.sin), ("COS", np.cos)):
    vals, marg = [], []
    for h in range(24):
        d = fn(2 * np.pi * h / 24)
        vals.append(float(np.float32(d)).hex())
        marg.append(margin_ulps(float(d)))
    print(f"// {name}: min margin = {min(marg):.3g} double ulp")
    print(", ".join(v + "f" for v in vals))
