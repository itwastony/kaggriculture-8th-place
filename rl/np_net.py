"""Forward pass of rl/model.Net in numpy (for the Kaggle submission without torch).

    w = load_weights("weights.npz"); lg, ml, v = forward(w, enc)     # enc = rl.encode.encode(obs), a single example
Weights: export_weights(ckpt, path, dtype) — state_dict -> npz (fp16 or fp32), plus [d, layers, 8 heads, 64 dims per head].
"""  # noqa: E501

# This file is copied verbatim into the submitted agent (rl/export_agent.py), and entry_points/verify_submission.py
# compares its code with the archives sent to Kaggle: change only comments and formatting.
import numpy as np

_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _erf(x):
    # Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((_A[4] * t + _A[3]) * t) + _A[2]) * t + _A[1]) * t + _A[0]) * t * np.exp(-a * a)
    return s * y


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def lin(x, w, p):
    y = x @ w[p + ".weight"].T
    bk = p + ".bias"
    return y + w[bk] if bk in w else y


def export_weights(ckpt, path, dtype="float16"):
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # the value is not needed in deployment: the value tower is not exported
    arr = {
        k: v.float().numpy().astype(dtype)
        for k, v in ck["model"].items()
        if not k.startswith(("venc.", "vnorm.", "vopp."))
    }
    arr["__meta__"] = np.array([ck.get("d", 256), ck.get("layers", 6), 8, 64], np.int64)
    arr["__vtower__"] = np.array(ck.get("vtower", 0), np.int64)
    np.savez(path, **arr)


def load_weights(path):
    z = np.load(path)
    w = {k: z[k].astype(np.float32) for k in z.files if k not in ("__meta__", "__vtower__")}
    w["__vtower__"] = int(z["__vtower__"]) if "__vtower__" in z.files else 0
    d, layers, heads, dk = (int(v) for v in z["__meta__"])
    w["__d__"], w["__layers__"], w["__heads__"], w["__dk__"] = d, layers, heads, dk
    return w


def forward(w, e):
    d, L, nh, dk = w["__d__"], w["__layers__"], w["__heads__"], w["__dk__"]
    NT = e["tiles"].shape[0]
    U = e["units"].shape[0]
    te = w["type_emb"]
    t = lin(e["tiles"], w, "tile_in") + w["tile_pos"] + te[1]
    u = lin(e["units"], w, "unit_in") + w["tile_pos"][e["unit_pos"]] + te[2]
    g = lin(gelu(lin(e["glob"], w, "glob_in.0")), w, "glob_in.2") + te[0]
    x = np.concatenate([g[None], t, u], 0)  # [S, d]
    keep = np.concatenate([np.ones(1 + NT, bool), e["unit_mask"]])
    neg = np.where(keep, 0.0, -np.inf).astype(np.float32)[None, None, :]
    hd = d // nh
    for i in range(L):
        p = f"enc.layers.{i}."
        h = layer_norm(x, w[p + "norm1.weight"], w[p + "norm1.bias"])
        qkv = h @ w[p + "self_attn.in_proj_weight"].T + w[p + "self_attn.in_proj_bias"]
        q, k, v = np.split(qkv, 3, -1)
        S_ = x.shape[0]
        q = q.reshape(S_, nh, hd).transpose(1, 0, 2)
        k = k.reshape(S_, nh, hd).transpose(1, 0, 2)
        v = v.reshape(S_, nh, hd).transpose(1, 0, 2)
        a = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + neg
        a = np.exp(a - a.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(S_, d)
        x = x + lin(o, w, p + "self_attn.out_proj")
        h = layer_norm(x, w[p + "norm2.weight"], w[p + "norm2.bias"])
        x = x + lin(gelu(lin(h, w, p + "linear1")), w, p + "linear2")
    x = layer_norm(x, w["norm.weight"], w["norm.bias"])
    g, t, u = x[0], x[1 : 1 + NT], x[1 + NT :]
    nops = w["op_bias"].shape[0]
    q = lin(u, w, "q").reshape(U, nops, dk)
    k = lin(t, w, "k")  # [NT, dk]
    logits = np.einsum("uod,td->uto", q, k) / np.sqrt(dk) + w["op_bias"]
    logits[np.arange(U), e["unit_pos"]] += w["here"]
    pooled = np.concatenate([g, t.mean(0)])
    ml = lin(gelu(lin(pooled, w, "market.0")), w, "market.2")
    val = None if w.get("__vtower__") else lin(gelu(lin(pooled, w, "value.0")), w, "value.2")[0]
    return logits, ml, val


def greedy(w, e, head_sizes):
    """-> (unit_choices [(target, operation)] for 16 units, market_choice [bin indices])."""
    lg, ml, _ = forward(w, e)
    U, NT, nops = lg.shape
    l = np.where(e["tmask"], lg, -1e9).reshape(U, NT * nops).argmax(-1)  # noqa: E741  name kept as submitted
    units = [(int(v) // nops, int(v) % nops) for v in l]
    mlm = np.where(e["mmask"], ml, -1e9)
    mc, o = [], 0
    for n in head_sizes:
        mc.append(int(mlm[o : o + n].argmax()))
        o += n
    return units, mc
