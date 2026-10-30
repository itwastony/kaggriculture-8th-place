"""Policy/value network: a transformer over tokens [global, 100 tiles, 16 units].

Outputs:
  unit_logits [B, 16, 100, NOPS] — (target, operation) for each unit (masked by tmask)
  market_logits [B, MB] — market bins (per head of S.MARKET_HEADS, softmax within a head)
  value [B] — on detached trunk features, optionally through its own tower (vtower layers)

Privileged critic (pcritic = number of layers, 0 = off; as in AlphaStar, the critic sees the opponent's observation):
value += joint([own pools, pools of the opponent's full observation through its own small encoder vopp, opponent-type
embedding]). The joint output is initialised to zero, so at the start the value equals the ordinary one. Not used in
deployment (the weight export skips vopp.*): the policy sees only its own observation.

rl/np_net.py re-implements the actor part of this network in NumPy for the submission; tests/test_rl_model_parity.py
checks that both give the same outputs.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encode as E
from . import spec as S

HEAD_SIZES = [len(b) for _, _, b in S.MARKET_HEADS]
NOID = 256  # vocabulary of opponent types (rl/appo.opp_id)


class Net(nn.Module):
    """d: width; layers: trunk layers; heads: attention heads; dk: key size of the unit pointer head; vtower: layers
    of the separate value tower (0 = value MLP on the trunk); pcritic: layers of the privileged critic (0 = off)."""

    def __init__(self, d=256, layers=6, heads=8, dk=64, vtower=0, pcritic=0):
        super().__init__()
        self.d, self.dk, self.vtower, self.pcritic = d, dk, vtower, pcritic
        self.tile_in = nn.Linear(E.FT, d)
        self.tile_pos = nn.Parameter(torch.zeros(S.NT, d))
        self.unit_in = nn.Linear(E.FU, d)
        self.glob_in = nn.Sequential(nn.Linear(E.FG, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        layer = nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, S.NOPS * dk)
        self.k = nn.Linear(d, dk)
        self.op_bias = nn.Parameter(torch.zeros(S.NOPS))
        self.here = nn.Parameter(torch.zeros(S.NOPS))  # added when the target = the unit's current tile
        self.market = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, sum(HEAD_SIZES)))
        self.value = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        if vtower:  # separate value tower on top of the detached trunk features
            vl = nn.TransformerEncoderLayer(
                d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
            )
            self.venc = nn.TransformerEncoder(vl, vtower)
            self.vnorm = nn.LayerNorm(d)
        if pcritic:  # privileged critic: the opponent's full observation
            dv = d // 2
            V = nn.Module()
            V.tile, V.unit, V.glob = nn.Linear(E.FT, dv), nn.Linear(E.FU, dv), nn.Linear(E.FG, dv)
            V.pos = nn.Parameter(torch.zeros(S.NT, dv))
            V.typ = nn.Parameter(torch.zeros(3, dv))
            ol = nn.TransformerEncoderLayer(
                dv, 4, 4 * dv, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
            )
            V.enc = nn.TransformerEncoder(ol, pcritic)
            V.norm = nn.LayerNorm(dv)
            V.id = nn.Embedding(NOID, dv)
            V.joint = nn.Sequential(nn.Linear(2 * d + 3 * dv, d), nn.GELU(), nn.Linear(d, 1))
            nn.init.normal_(V.pos, std=0.02)
            nn.init.normal_(V.typ, std=0.02)
            nn.init.zeros_(V.joint[2].weight)
            nn.init.zeros_(V.joint[2].bias)
            self.vopp = V
        nn.init.normal_(self.tile_pos, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)

    def trunk(self, tiles, units, unit_pos, unit_mask, glob):
        """Shared trunk: token features [B, 1+100+16, d] (after LayerNorm) and the padding mask."""
        B = tiles.shape[0]
        t = self.tile_in(tiles) + self.tile_pos + self.type_emb[1]
        u = self.unit_in(units) + self.tile_pos[unit_pos] + self.type_emb[2]
        g = (self.glob_in(glob) + self.type_emb[0]).unsqueeze(1)
        x = torch.cat([g, t, u], 1)
        pad = torch.cat([torch.zeros(B, 1 + S.NT, dtype=torch.bool, device=tiles.device), ~unit_mask], 1)
        return self.norm(self.enc(x, src_key_padding_mask=pad)), pad

    def policy_heads(self, x, unit_pos):
        return self._heads(
            x[:, 1 : 1 + S.NT], x[:, 1 + S.NT :], torch.cat([x[:, 0], x[:, 1 : 1 + S.NT].mean(1)], -1), unit_pos
        )

    def head_cache(self, x):
        """Head inputs in bf16, exactly what their Linear layers receive under autocast: the PPO anchor cache (the
        KL stays exact). Tiles and units are separate contiguous tensors: a non-contiguous slice sends the GEMM down
        another path (different rounding)."""
        pooled = torch.cat([x[:, 0], x[:, 1 : 1 + S.NT].mean(1)], -1)
        bf = torch.bfloat16
        return x[:, 1 : 1 + S.NT].to(bf).contiguous(), x[:, 1 + S.NT :].to(bf).contiguous(), pooled.to(bf)

    def heads_from_cache(self, t, u, pooled, unit_pos):
        return self._heads(t, u, pooled, unit_pos)

    def _heads(self, t, u, pooled, unit_pos):
        B = t.shape[0]
        q = self.q(u).view(B, S.MAX_UNITS, S.NOPS, self.dk)
        k = self.k(t)  # [B, 100, dk]
        logits = torch.einsum("buod,btd->buto", q, k) / math.sqrt(self.dk) + self.op_bias
        here = F.one_hot(unit_pos, S.NT).to(logits.dtype)  # [B, 16, 100]
        logits = logits + here.unsqueeze(-1) * self.here
        return logits, self.market(pooled)

    def opp_features(self, o):
        """The opponent's full observation (its encode from its own seat) -> pools [B, 2*dv] (critic only)."""
        V = self.vopp
        B = o["tiles"].shape[0]
        t = V.tile(o["tiles"]) + V.pos + V.typ[1]
        u = V.unit(o["units"]) + V.pos[o["unit_pos"]] + V.typ[2]
        g = (V.glob(o["glob"]) + V.typ[0]).unsqueeze(1)
        z = torch.cat([g, t, u], 1)
        pad = torch.cat([torch.zeros(B, 1 + S.NT, dtype=torch.bool, device=z.device), ~o["unit_mask"]], 1)
        z = V.norm(V.enc(z, src_key_padding_mask=pad))
        return torch.cat([z[:, 0], z[:, 1 : 1 + S.NT].mean(1)], -1)

    def value_head(self, x, pad, opp=None, oid=None):
        # the value does not pass gradient into the shared trunk: otherwise the value loss moves the policy (an early
        # PPO run collapsed this way)
        xv = x.detach()
        if self.vtower:
            xv = self.vnorm(self.venc(xv, src_key_padding_mask=pad))
        pv = torch.cat([xv[:, 0], xv[:, 1 : 1 + S.NT].mean(1)], -1)
        v = self.value(pv).squeeze(-1)
        if self.pcritic and opp is not None:
            j = torch.cat([pv, self.opp_features(opp), self.vopp.id(oid)], -1)
            v = v + self.vopp.joint(j).squeeze(-1)
        return v

    def forward(self, tiles, units, unit_pos, unit_mask, glob, value=True, opp=None, oid=None):
        x, pad = self.trunk(tiles, units, unit_pos, unit_mask, glob)
        logits, ml = self.policy_heads(x, unit_pos)
        return logits, ml, (self.value_head(x, pad, opp, oid) if value else None)


def unit_logp(logits, tmask, unit_mask, target, op):
    """log p of the chosen (target, operation) per unit: [B, 16] (0 for absent units)."""
    B = logits.shape[0]
    flat = logits.masked_fill(~tmask, -1e9).view(B, S.MAX_UNITS, -1)
    lp = F.log_softmax(flat.float(), -1)
    idx = (target * S.NOPS + op).clamp(min=0)
    out = lp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    return out * unit_mask


def market_logp(mlogits, mmask, choice):
    """log p of the chosen market bins: [B, n_heads]."""
    outs, o = [], 0
    ml = mlogits.float().masked_fill(~mmask, -1e9)
    for h, n in enumerate(HEAD_SIZES):
        lp = F.log_softmax(ml[:, o : o + n], -1)
        outs.append(lp.gather(-1, choice[:, h : h + 1]))
        o += n
    return torch.cat(outs, -1)


def entropy(logits, tmask, unit_mask, mlogits, mmask):
    """Policy entropy: (sum over present units [B], sum over market heads [B])."""
    B = logits.shape[0]
    flat = logits.float().masked_fill(~tmask, -1e9).view(B, S.MAX_UNITS, -1)
    p = F.softmax(flat, -1)
    eu = -(p * F.log_softmax(flat, -1)).sum(-1) * unit_mask
    ml = mlogits.float().masked_fill(~mmask, -1e9)
    em, o = 0.0, 0
    for n in HEAD_SIZES:
        lp = F.log_softmax(ml[:, o : o + n], -1)
        em = em - (lp.exp() * lp).sum(-1)
        o += n
    return eu.sum(-1), em
