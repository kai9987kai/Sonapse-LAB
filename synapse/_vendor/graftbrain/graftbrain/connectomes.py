"""Connectome loading.

Every loader returns a :class:`Connectome` - a directed, signed, weighted graph with
per-neuron cell-type and role labels and full provenance back to the source dataset.

Supported sources
-----------------
* ``proxy:<profile>``      offline, statistically-shaped stand-in (default; no download)
* ``flywire:<path>``       FlyWire FAFB v783 ``proofread_connections_783.feather`` (Zenodo 10.5281/zenodo.10676866)
* ``neuprint:<dataset>``   any neuPrint dataset (hemibrain, MANC, optic-lobe, male CNS) - needs NEUPRINT_APPLICATION_CREDENTIALS
* ``csv:<edges.csv>``      generic edge list (pre, post, weight[, nt|sign]) - e.g. BANC / FANC / larval exports

Real datasets are large (FlyWire: ~139k neurons, ~3.8M neuropil-split edges). They are
subsampled to ``max_neurons`` by synaptic degree so the whole grafted system can run
closed-loop in real time on a CPU.
"""
from __future__ import annotations

import os
import zlib
from dataclasses import dataclass, field

import numpy as np

ROLES = ("sensory", "central", "motor", "descending", "ascending", "visual")
INHIBITORY_NT = {"gaba", "glut", "glutamate"}  # glutamate is largely inhibitory in fly CNS (GluCl)


@dataclass
class Connectome:
    name: str
    kind: str                      # brain | vnc | optic | larva | cns
    pre: np.ndarray                # int64 [E]
    post: np.ndarray               # int64 [E]
    weight: np.ndarray             # float32 [E] synapse counts
    sign: np.ndarray               # float32 [E] +1 excitatory / -1 inhibitory
    cell_type: np.ndarray          # object [N]
    role: np.ndarray               # object [N]
    source_ids: np.ndarray         # int64/object [N] ids in the original dataset
    provenance: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.cell_type)

    @property
    def e(self) -> int:
        return len(self.pre)

    def summary(self) -> dict:
        roles, counts = np.unique(self.role, return_counts=True)
        return {
            "name": self.name, "kind": self.kind, "neurons": self.n, "edges": self.e,
            "synapses": int(self.weight.sum()), "inhibitory_frac": float((self.sign < 0).mean()) if self.e else 0.0,
            "roles": dict(zip(roles.tolist(), counts.tolist())), "provenance": self.provenance,
        }


# --------------------------------------------------------------------------------------
# Proxy connectomes (offline). Shapes follow the real datasets' broad organisation:
# heavy-tailed degree, modular neuropils, ~30% inhibitory, sensory -> central -> motor flow.
# --------------------------------------------------------------------------------------
PROXY_PROFILES = {
    #               kind     n     roles fraction (sensory, central, motor, descending, ascending, visual)
    "flywire":     ("brain", 4000, (0.10, 0.62, 0.00, 0.06, 0.04, 0.18)),
    "hemibrain":   ("brain", 3000, (0.06, 0.78, 0.00, 0.08, 0.04, 0.04)),
    "malecns":     ("cns",   5000, (0.10, 0.50, 0.08, 0.06, 0.06, 0.20)),
    "banc":        ("cns",   4500, (0.12, 0.50, 0.10, 0.07, 0.07, 0.14)),
    "optic-lobe":  ("optic", 2500, (0.20, 0.30, 0.00, 0.00, 0.00, 0.50)),
    "manc":        ("vnc",   2500, (0.20, 0.45, 0.15, 0.10, 0.10, 0.00)),
    "fanc":        ("vnc",   2000, (0.20, 0.45, 0.15, 0.10, 0.10, 0.00)),
    "larval":      ("larva", 1500, (0.20, 0.60, 0.10, 0.05, 0.05, 0.00)),
}
# Shared "homologous" cell-type vocabulary so type-matched grafting has something to match,
# mirroring how hemibrain/FlyWire/maleCNS share curated cell-type names.
_TYPE_VOCAB = {r: [f"{r[:3].upper()}{i:03d}" for i in range(160)] for r in ROLES}


def make_proxy(profile: str, n: int | None = None, seed: int = 0) -> Connectome:
    if profile not in PROXY_PROFILES:
        raise ValueError(f"unknown proxy profile {profile!r}; choose from {sorted(PROXY_PROFILES)}")
    kind, n_default, fracs = PROXY_PROFILES[profile]
    n = int(n or n_default)
    rng = np.random.default_rng(zlib.crc32(f"{profile}:{seed}".encode()))  # deterministic across runs

    counts = np.floor(np.array(fracs) * n).astype(int)
    counts[1] += n - counts.sum()
    role = np.concatenate([[r] * c for r, c in zip(ROLES, counts)]).astype(object)
    n_mod = max(4, n // 250)                         # neuropil-like modules
    module = rng.integers(0, n_mod, n)
    cell_type = np.array([rng.choice(_TYPE_VOCAB[r][: max(8, c // 6)]) for r, c in
                          zip(role, [counts[ROLES.index(x)] for x in role])], dtype=object)

    # Allowed role->role flow (feed-forward backbone + recurrence inside central).
    flow = {
        "visual": {"visual": 1.0, "central": 1.0},
        "sensory": {"central": 1.0, "sensory": 0.2, "motor": 0.15},
        "central": {"central": 1.0, "descending": 0.6, "motor": 0.3, "visual": 0.1},
        "descending": {"motor": 1.0, "central": 0.2},
        "ascending": {"central": 1.0},
        "motor": {"motor": 0.1, "ascending": 0.4},
    }
    out_strength = rng.lognormal(0.0, 1.0, n)        # heavy-tailed hubs
    in_strength = rng.lognormal(0.0, 1.0, n)
    mean_out = 25
    pre_l, post_l = [], []
    idx_by_role = {r: np.flatnonzero(role == r) for r in ROLES}
    for i in range(n):
        targets = flow.get(role[i], {})
        k = int(rng.poisson(mean_out * min(out_strength[i], 6.0) / 1.6))
        if not targets or k == 0:
            continue
        cand = np.concatenate([idx_by_role[r] for r in targets if len(idx_by_role[r])])
        if len(cand) == 0:
            continue
        p = np.concatenate([np.full(len(idx_by_role[r]), w) for r, w in targets.items() if len(idx_by_role[r])])
        p = p * in_strength[cand] * np.where(module[cand] == module[i], 6.0, 1.0)
        p /= p.sum()
        tgt = np.unique(rng.choice(cand, size=min(k, len(cand)), p=p))
        tgt = tgt[tgt != i]
        pre_l.append(np.full(len(tgt), i)); post_l.append(tgt)
    pre = np.concatenate(pre_l).astype(np.int64)
    post = np.concatenate(post_l).astype(np.int64)
    weight = np.maximum(5, np.round(rng.lognormal(2.0, 0.9, len(pre)))).astype(np.float32)  # >=5 synapse threshold
    inhib_neuron = rng.random(n) < 0.3
    inhib_neuron[role == "sensory"] = False
    sign = np.where(inhib_neuron[pre], -1.0, 1.0).astype(np.float32)
    return Connectome(profile, kind, pre, post, weight, sign, cell_type, role, np.arange(n),
                      {"source": "proxy", "profile": profile, "seed": seed,
                       "note": "synthetic stand-in shaped like the real dataset; swap for real data with flywire:/neuprint:/csv:"})


# --------------------------------------------------------------------------------------
# Real-data loaders
# --------------------------------------------------------------------------------------
def _infer_roles(n: int, pre: np.ndarray, post: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Topological role inference when no annotation table is supplied.

    Sources (high out/in ratio) -> sensory, sinks -> motor, the rest -> central.
    """
    out_w = np.bincount(pre, weights=w, minlength=n)
    in_w = np.bincount(post, weights=w, minlength=n)
    ratio = (out_w + 1.0) / (in_w + 1.0)
    role = np.full(n, "central", dtype=object)
    role[ratio >= np.quantile(ratio, 0.92)] = "sensory"
    role[ratio <= np.quantile(ratio, 0.06)] = "motor"
    return role


def _subsample(ids, pre, post, w, sign, max_neurons):
    deg = np.bincount(pre, weights=w, minlength=len(ids)) + np.bincount(post, weights=w, minlength=len(ids))
    if len(ids) <= max_neurons:
        keep = np.arange(len(ids))
    else:
        keep = np.sort(np.argsort(-deg)[:max_neurons])
    remap = -np.ones(len(ids), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    m = (remap[pre] >= 0) & (remap[post] >= 0) & (pre != post)
    return keep, remap[pre[m]], remap[post[m]], w[m], sign[m]


def _finish(name, kind, ids, pre, post, w, sign, max_neurons, annotations, provenance):
    keep, pre, post, w, sign = _subsample(ids, pre, post, w, sign, max_neurons)
    kept_ids = ids[keep]
    n = len(kept_ids)
    cell_type = np.array([f"{name}_{i}" for i in kept_ids], dtype=object)
    role = _infer_roles(n, pre, post, w)
    if annotations:
        import pandas as pd
        ann = pd.read_csv(annotations)
        idcol = next(c for c in ann.columns if c.lower() in ("root_id", "bodyid", "id", "skid", "pt_root_id"))
        ann = ann.set_index(idcol)
        lookup = {k: v for k, v in zip(ann.index.astype(str), range(len(ann)))}
        tcol = next((c for c in ann.columns if c.lower() in ("cell_type", "type", "primary_type")), None)
        rcol = next((c for c in ann.columns if c.lower() in ("super_class", "superclass", "class", "role")), None)
        rmap = {"sensory": "sensory", "descending": "descending", "ascending": "ascending", "motor": "motor",
                "efferent": "motor", "optic": "visual", "visual_projection": "visual", "visual_centrifugal": "visual",
                "central": "central", "intrinsic": "central"}
        for j, sid in enumerate(kept_ids.astype(str)):
            r = lookup.get(sid)
            if r is None:
                continue
            row = ann.iloc[r]
            if tcol and isinstance(row[tcol], str):
                cell_type[j] = row[tcol]
            if rcol and isinstance(row[rcol], str):
                role[j] = rmap.get(row[rcol].lower(), role[j])
    provenance = dict(provenance, subsampled_to=n, selection="top synaptic degree")
    return Connectome(name, kind, pre, post, w.astype(np.float32), sign.astype(np.float32),
                      cell_type, role, kept_ids, provenance)


def load_flywire(path: str, max_neurons: int, annotations: str | None = None, min_syn: int = 5) -> Connectome:
    import pandas as pd
    df = pd.read_feather(path) if path.endswith(".feather") else pd.read_csv(path)
    pre_c = next(c for c in df.columns if c.startswith("pre"))
    post_c = next(c for c in df.columns if c.startswith("post"))
    syn_c = next(c for c in df.columns if "syn" in c or c == "weight")
    nt_cols = [c for c in df.columns if c.endswith("_avg")]
    agg = {syn_c: "sum", **{c: "mean" for c in nt_cols}}
    df = df.groupby([pre_c, post_c], as_index=False).agg(agg)          # merge neuropil splits
    df = df[df[syn_c] >= min_syn]
    ids, inv = np.unique(np.concatenate([df[pre_c].values, df[post_c].values]), return_inverse=True)
    pre, post = inv[: len(df)], inv[len(df):]
    if nt_cols:
        nt = np.array([c.replace("_avg", "") for c in nt_cols])[df[nt_cols].values.argmax(1)]
        sign = np.where(np.isin(nt, list(INHIBITORY_NT)), -1.0, 1.0)
    else:
        sign = np.ones(len(df))
    return _finish("flywire", "brain", ids, pre, post, df[syn_c].values.astype(np.float32), sign, max_neurons,
                   annotations, {"source": "FlyWire FAFB v783", "file": os.path.basename(path),
                                 "license": "CC-BY-NC-4.0", "doi": "10.5281/zenodo.10676866"})


def load_neuprint(dataset: str, max_neurons: int, server: str = "neuprint.janelia.org", min_weight: int = 5) -> Connectome:
    try:
        from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install neuprint-python  (and set NEUPRINT_APPLICATION_CREDENTIALS)") from e
    client = Client(server, dataset=dataset)
    neurons, _ = fetch_neurons(NC(status="Traced"), client=client)
    neurons = neurons.sort_values("post", ascending=False).head(max_neurons)
    nl, conn = fetch_adjacencies(neurons["bodyId"], neurons["bodyId"], min_total_weight=min_weight, client=client)
    conn = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    ids = neurons["bodyId"].values
    pos = {b: i for i, b in enumerate(ids)}
    pre = conn["bodyId_pre"].map(pos).values.astype(np.int64)
    post = conn["bodyId_post"].map(pos).values.astype(np.int64)
    nt_col = next((c for c in neurons.columns if c.lower() in ("predictednt", "consensusnt")), None)
    if nt_col:
        nt = neurons[nt_col].fillna("acetylcholine").str.lower().values
        sign = np.where(np.isin(nt[pre], ["gaba", "glutamate"]), -1.0, 1.0)
    else:
        sign = np.ones(len(pre))
    kind = "vnc" if "manc" in dataset or "fanc" in dataset else "optic" if "optic" in dataset else \
        "cns" if "cns" in dataset else "brain"
    c = _finish(dataset.split(":")[0], kind, ids, pre, post, conn["weight"].values.astype(np.float32), sign,
                max_neurons, None, {"source": f"neuPrint {dataset}", "server": server, "license": "CC-BY-4.0"})
    tmap = dict(zip(neurons["bodyId"], neurons.get("type", neurons["bodyId"]).astype(str)))
    c.cell_type = np.array([tmap.get(b, str(b)) for b in c.source_ids], dtype=object)
    for col in ("superclass", "class"):
        if col in neurons.columns:
            sc = dict(zip(neurons["bodyId"], neurons[col].astype(str).str.lower()))
            for j, b in enumerate(c.source_ids):
                s = sc.get(b, "")
                for key, r in (("sensory", "sensory"), ("descending", "descending"), ("ascending", "ascending"),
                               ("motor", "motor"), ("efferent", "motor"), ("optic", "visual"), ("visual", "visual")):
                    if key in s:
                        c.role[j] = r
            break
    return c


def load_csv(path: str, max_neurons: int, name: str | None = None, kind: str = "brain",
             annotations: str | None = None) -> Connectome:
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    pre_c = next(cols[c] for c in cols if c.startswith("pre") or c in ("source", "from"))
    post_c = next(cols[c] for c in cols if c.startswith("post") or c in ("target", "to"))
    w_c = next((cols[c] for c in cols if c in ("weight", "syn_count", "synapses", "count")), None)
    w = df[w_c].values.astype(np.float32) if w_c else np.ones(len(df), np.float32)
    if "sign" in cols:
        sign = np.sign(df[cols["sign"]].values).astype(np.float32)
    elif "nt" in cols or "nt_type" in cols:
        nt = df[cols.get("nt", cols.get("nt_type"))].astype(str).str.lower().values
        sign = np.where(np.isin(nt, list(INHIBITORY_NT)), -1.0, 1.0)
    else:
        sign = np.ones(len(df), np.float32)
    ids, inv = np.unique(np.concatenate([df[pre_c].values, df[post_c].values]), return_inverse=True)
    name = name or os.path.splitext(os.path.basename(path))[0]
    return _finish(name, kind, ids, inv[: len(df)], inv[len(df):], w, sign, max_neurons, annotations,
                   {"source": "csv", "file": os.path.basename(path)})


def load(spec: str, max_neurons: int = 4000, seed: int = 0) -> Connectome:
    """Load from a spec string such as ``proxy:hemibrain`` or ``flywire:data/proofread_connections_783.feather``."""
    src, _, arg = spec.partition(":")
    extra = {}
    if "@" in arg:                                   # e.g. csv:edges.csv@kind=vnc,annotations=nodes.csv
        arg, opts = arg.split("@", 1)
        extra = dict(kv.split("=", 1) for kv in opts.split(","))
    if src == "proxy":
        return make_proxy(arg, n=min(max_neurons, PROXY_PROFILES[arg][1]) if max_neurons else None, seed=seed)
    if src == "flywire":
        return load_flywire(arg, max_neurons, extra.get("annotations"))
    if src == "neuprint":
        return load_neuprint(arg, max_neurons, extra.get("server", "neuprint.janelia.org"))
    if src == "csv":
        return load_csv(arg, max_neurons, extra.get("name"), extra.get("kind", "brain"), extra.get("annotations"))
    raise ValueError(f"unknown connectome source in {spec!r}")
