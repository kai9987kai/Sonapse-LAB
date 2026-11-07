# Connectome Source Assets, Verification, and Runtime Sampling

SYNAPSE separates **catalog specimen metadata**, **bundled cell-type aggregates**, **verified neuron-level local source caches**, **sampled runtime subgraphs**, and **synthetic proxies**. A catalog entry or homology annotation never implies that a full neuron-level connectome has been downloaded or executed.

## 1. Four Distinct Representations

| Representation | Location / Module | Scope & Evidence Boundary |
|---|---|---|
| **Bundled cell-type aggregate** | `synapse/_vendor/expanse/expanse/data/malecns_types.npz` | 11,751 MaleCNS cell-type nodes and 3,830,931 type-to-type edges. Homology flags (`has_flywire`, `has_manc`) record cross-dataset type correspondence; they are **not** independent FlyWire or MANC neuron graphs. |
| **Verified neuron-level caches** | `artifacts/connectomes/<dataset>/` (`synapse/connectome_assets.py`) | Imported from official public releases and verified against cryptographic MD5 / SHA-256 digests. Stored as memory-mappable `pre.i32`, `post.i32`, `synapses.i64`, `neurons.npz`, and `manifest.json`. |
| **Induced runtime subgraphs** | `load_programmed_graft()` / `load_authentic_connectome()` | Seeded role-stratified sampling from the highest-degree $4N$ candidate pool with induced directed edges. Used for live source-backed substrates (`connectome_mode='prefer_authentic'` or `'strict_authentic'`) and the 128-neuron fixed sparse prior (`programmed_connectome=True`). |
| **Calibrated / synthetic proxies** | `make_proxy()` (`connectome_mode='proxy'`) | Fast procedural rate graphs matching regional role and degree distributions. Explicitly labelled `"representation": "synthetic"` or `"synthetic_proxy"`. |

## 2. Installed & Verified Neuron-Level Sources

Run `python -m synapse.cli connectome-inventory` or `python -m synapse.connectome_assets inventory` to inspect local asset status:

| Dataset ID | Specimen / Region | Prepared Neurons | Directed Edges | Total Synapses | Verification | Full Graph Present |
|---|---|---:|---:|---:|---|---|
| `malecns` | MaleCNS v1.0 (Brain + VNC) | 166,700 | 25,582,938 | 124,177,617 | `official_checksums` | Yes (read-only external Feather source) |
| `flywire` | FlyWire FAFB v783 (Adult Female Brain) | 139,255 | 16,847,997 | 54,492,922 | `official_checksums` | Yes (Zenodo v783 + v2.1.0 annotations) |
| `manc` | MANC v1.0 (Male Adult Ventral Nerve Cord) | 23,188 | 5,243,574 | 30,698,527 | `official_checksums` | Yes (Janelia FlyEM v1.0 traced adjacencies) |
| `optic-lobe` | FlyEM Optic Lobe v1.1 (Right Optic Lobe) | 52,445 | 6,484,936 | 26,492,194 | `official_checksums` | Yes (Traced-status neurons within volume) |
| `banc` | BANC v888 (Brain And Nerve Cord) | 155,858 | 11,546,261 | 35,361,769 | `official_checksums` | Yes (Proofread/roughly proofread, glia excluded) |
| `hemibrain` | Hemibrain v1.2 (Central Brain Hemisphere) | 21,739 | 3,550,403 | 14,329,229 | `official_checksums` | Yes (Official v1.2 non-cropped Traced export) |
| `larval` | Winding et al. 2023 (Larval Brain) | 2,952 | 110,677 | 352,611 | `pinned_repository_sha256` | Partial (published 2,952-neuron supplement matrix, not full 3,016 reconstruction) |
| `fanc` | FANC (Female Adult Nerve Cord) | — | — | — | `not_yet_verified` | No (requires independent community source files; never substituted with MaleCNS homologs) |

## 3. Sign Convention and Modelling Assumptions

Anatomical synapse counts are measured electron-microscopy data, whereas physiological signs, normalization, and continuous rate dynamics are explicit modelling assumptions:
- **Neurotransmitter signs (`SIGN_CONVENTION`)**:
  - Acetylcholine (`acetylcholine`, `ach`): $+1$ (excitatory)
  - GABA (`gaba`): $-1$ (inhibitory)
  - Glutamate (`glutamate`, `glut`): $-1$ (assumes inhibitory GluCl channels)
  - Histamine (`histamine`): $-1$ (inhibitory photoreceptor/visual synapses)
  - Monoamines (dopamine, serotonin, octopamine) and unknown/unclear transmitters: $0$ (no fast signed synaptic efficacy inferred)
- **Weight conversion**:
  - For the fixed sparse conditioning prior (`load_programmed_graft`): $w_{ij} = s_i \log(1 + c_{ij})$, normalized so incoming absolute row sums per postsynaptic neuron do not exceed $0.9$.
  - For live runtime connectome substrates (`load_authentic_connectome`): raw synapse counts and presynaptic signs are passed to `graftbrain.build`, which applies signed $\log(1 + c_{ij})$ scaling, incoming row normalization, and configured recurrent gain.

## 4. Managing Local Connectome Assets

Raw exports and prepared binary caches live under `artifacts/connectomes/` (which is excluded by `.gitignore` so multi-gigabyte raw files are never pushed to Git remotes). The external MaleCNS directory (`~/Desktop/New folder (9)/external/malecns_data`) is accessed strictly read-only.

```powershell
# Inspect current verified cache status
python -m synapse.cli connectome-inventory

# Download and verify public source files (enforces >= 2 GB free disk space)
python -m synapse.connectome_assets download --dataset flywire

# Prepare binary memory-mapped cache from raw files
python -m synapse.connectome_assets prepare --dataset flywire
```

If `connectome_mode='strict_authentic'` is requested for a dataset without a verified local cache (such as `fanc`), `build_runtime_connectomes` fails closed with `FileNotFoundError` rather than substituting a synthetic proxy.
