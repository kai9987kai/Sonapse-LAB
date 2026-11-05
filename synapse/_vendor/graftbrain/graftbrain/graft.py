"""Multi-site grafting of several connectomes into one cross-wired graph.

For every *pair* of connectomes we add several independent graft sites, so the result is
structurally connected in multiple places rather than through one bottleneck:

1. **Anatomical interface sites** - wiring that mirrors real fly anatomy:
   brain descending neurons -> nerve-cord motor/premotor ("neck connective"),
   nerve-cord ascending neurons -> brain central, optic-lobe visual -> central-brain visual.
2. **Homolog sites** - neurons sharing a curated cell-type name are coupled bidirectionally
   (the way hemibrain / FlyWire / maleCNS share cell-type nomenclature).
3. **Bridge sites** - sparse central<->central links at several distinct neuropil-like
   locations, chosen far apart in each graph.

Connectome edges are frozen (the biological prior). Graft edges are the *plastic* part:
they are trained online while the system acts in the world.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .connectomes import Connectome

BRAINISH = {"brain", "cns", "larva"}
CORDISH = {"vnc", "cns"}


@dataclass
class GraftedGraph:
    names: list
    offsets: np.ndarray            # [C+1]
    role: np.ndarray               # [N]
    cell_type: np.ndarray          # [N]
    origin: np.ndarray             # [N] connectome index
    base_pre: np.ndarray
    base_post: np.ndarray
    base_w: np.ndarray             # signed, normalised
    graft_pre: np.ndarray
    graft_post: np.ndarray
    graft_w0: np.ndarray           # initial graft weights
    graft_site: np.ndarray         # site index per graft edge
    sites: list = field(default_factory=list)
    summaries: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.role)

    def describe(self) -> dict:
        return {"neurons": self.n, "connectome_edges": int(len(self.base_pre)),
                "graft_edges": int(len(self.graft_pre)), "graft_sites": len(self.sites),
                "connectomes": self.summaries, "sites": self.sites}


def _pick(rng, idx, k):
    if len(idx) == 0 or k <= 0:
        return np.empty(0, np.int64)
    return rng.choice(idx, size=min(k, len(idx)), replace=False)


def _wire(rng, src, dst, fan, sign_inhib=0.0):
    """Each src neuron sends `fan` edges to random dst neurons."""
    if len(src) == 0 or len(dst) == 0:
        return np.empty((0, 2), np.int64), np.empty(0, np.float32)
    pre = np.repeat(src, fan)
    post = rng.choice(dst, size=len(pre))
    sign = np.where(rng.random(len(pre)) < sign_inhib, -1.0, 1.0).astype(np.float32)
    return np.stack([pre, post], 1), sign


def build(connectomes: list[Connectome], gain: float = 1.1, graft_scale: float = 0.15,
          bridges_per_pair: int = 3, bridge_size: int = 48, homolog_max: int = 200, seed: int = 0) -> GraftedGraph:
    rng = np.random.default_rng(seed)
    offsets = np.cumsum([0] + [c.n for c in connectomes])
    role = np.concatenate([c.role for c in connectomes])
    ctype = np.concatenate([c.cell_type for c in connectomes])
    origin = np.concatenate([np.full(c.n, i) for i, c in enumerate(connectomes)])

    # ---- frozen connectome weights: signed log-synapse count, row-normalised per target ----
    bp = np.concatenate([c.pre + offsets[i] for i, c in enumerate(connectomes)])
    bq = np.concatenate([c.post + offsets[i] for i, c in enumerate(connectomes)])
    bw = np.concatenate([c.sign * np.log1p(c.weight) for c in connectomes]).astype(np.float32)
    row_abs = np.bincount(bq, weights=np.abs(bw), minlength=len(role))
    bw = (gain * bw / np.maximum(row_abs[bq], 1e-6)).astype(np.float32)

    # ---- graft sites ----
    edges, signs, site_of, sites = [], [], [], []

    def add_site(kind, a, b, e, s, detail):
        if len(e) == 0:
            return
        sid = len(sites)
        edges.append(e); signs.append(s); site_of.append(np.full(len(e), sid))
        sites.append({"id": sid, "kind": kind, "from": connectomes[a].name, "to": connectomes[b].name,
                      "edges": int(len(e)), "detail": detail})

    def members(i, roles):
        loc = np.flatnonzero(np.isin(connectomes[i].role, roles))
        return loc + offsets[i]

    for a, A in enumerate(connectomes):
        for b, B in enumerate(connectomes):
            if a == b:
                continue
            # 1. anatomical interfaces
            if A.kind in BRAINISH and B.kind in CORDISH and A.kind != B.kind or (A.kind == "brain" and B.kind == "cns"):
                e, s = _wire(rng, members(a, ["descending"]), members(b, ["motor", "central"]), fan=6)
                add_site("interface", a, b, e, s, "descending -> premotor/motor (neck connective)")
            if A.kind in CORDISH and B.kind in BRAINISH and A.kind != B.kind:
                e, s = _wire(rng, members(a, ["ascending"]), members(b, ["central"]), fan=6)
                add_site("interface", a, b, e, s, "ascending -> central brain")
            if A.kind == "optic" and B.kind in BRAINISH | {"cns"}:
                e, s = _wire(rng, members(a, ["visual"]), members(b, ["visual", "central"]), fan=4)
                add_site("interface", a, b, e, s, "optic-lobe visual projection -> central brain")

            # 2. homolog coupling (only one direction per loop; the reverse pair adds the other)
            shared = np.intersect1d(np.unique(A.cell_type), np.unique(B.cell_type))
            if len(shared):
                shared = shared[rng.permutation(len(shared))[:homolog_max]]
                pre_l, post_l = [], []
                for t in shared:
                    ia = np.flatnonzero(A.cell_type == t) + offsets[a]
                    ib = np.flatnonzero(B.cell_type == t) + offsets[b]
                    k = min(len(ia), len(ib), 4)
                    pre_l.append(rng.choice(ia, k, replace=False)); post_l.append(rng.choice(ib, k, replace=False))
                e = np.stack([np.concatenate(pre_l), np.concatenate(post_l)], 1)
                add_site("homolog", a, b, e, np.ones(len(e), np.float32), f"{len(shared)} shared cell types")

            # 3. multiple distinct central<->central bridges
            if a < b:
                ca, cb = members(a, ["central"]), members(b, ["central"])
                for k in range(bridges_per_pair):
                    # pick a contiguous neighbourhood in each graph -> spatially distinct sites
                    if len(ca) < bridge_size or len(cb) < bridge_size:
                        break
                    sa = ca[((k * len(ca)) // bridges_per_pair + np.arange(bridge_size)) % len(ca)]
                    sb = cb[((k * len(cb)) // bridges_per_pair + np.arange(bridge_size)) % len(cb)]
                    e1, s1 = _wire(rng, sa, sb, fan=3, sign_inhib=0.3)
                    e2, s2 = _wire(rng, sb, sa, fan=3, sign_inhib=0.3)
                    add_site("bridge", a, b, np.concatenate([e1, e2]), np.concatenate([s1, s2]),
                             f"reciprocal bridge #{k + 1}")

    if edges:
        E = np.concatenate(edges); S = np.concatenate(signs); site = np.concatenate(site_of)
    else:
        E = np.empty((0, 2), np.int64); S = np.empty(0, np.float32); site = np.empty(0, np.int64)
    fan_in = np.bincount(E[:, 1], minlength=len(role)) if len(E) else np.zeros(len(role))
    w0 = (graft_scale * S / np.sqrt(np.maximum(fan_in[E[:, 1]], 1))).astype(np.float32) if len(E) else S

    return GraftedGraph([c.name for c in connectomes], offsets, role, ctype, origin, bp, bq, bw,
                        E[:, 0].astype(np.int64), E[:, 1].astype(np.int64), w0, site.astype(np.int64), sites,
                        [c.summary() for c in connectomes])
