"""Context routing and budgeted support changes within fixed graft candidates."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PlasticityConfig:
    mode: str = "legacy"
    density: float = 0.6
    interval: int = 10
    warmup: int = 2
    turnover: float = 0.1
    router_lr: float = 1e-4
    growth_rule: str = "coactivity"
    growth_init: str = "original"
    min_age: int = 0

    def __post_init__(self):
        if self.mode not in {"legacy", "static", "routing", "structural", "adaptive"}:
            raise ValueError("graft mode must be legacy, static, routing, structural, or adaptive")
        if self.growth_rule not in {"coactivity", "gradient", "random"}:
            raise ValueError("growth_rule must be coactivity, gradient, or random")
        if self.growth_init not in {"original", "zero"}:
            raise ValueError("growth_init must be original or zero")
        for name, lower, upper, exclusive in (
            ("density", 0.0, 1.0, True), ("turnover", 0.0, 1.0, False),
            ("router_lr", 0.0, math.inf, True),
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value > upper
                    or (value <= lower if exclusive else value < lower)):
                bound = "(0, 1]" if name == "density" else "[0, 1]" if name == "turnover" else "> 0"
                raise ValueError(f"{name} must be finite and {bound}")
        for name, minimum in (("interval", 1), ("warmup", 0), ("min_age", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.min_age > torch.iinfo(torch.int64).max:
            raise ValueError("min_age must fit in a signed 64-bit integer")


class GraftPlasticity(nn.Module):
    """Owns active edge records, router parameters, and post-update rewiring state.

    Call ``observe`` once per world action. Call ``after_update`` only after all
    retained action graphs have been backpropagated and both optimizers and any
    pending masked Hebbian updates have finished. Dormant weights remain stored.
    """

    def __init__(self, graph, obs_dim, config=None, seed=0):
        super().__init__()
        self.config = config if config is not None else PlasticityConfig()
        if not isinstance(self.config, PlasticityConfig):
            raise TypeError("config must be a PlasticityConfig")
        if isinstance(obs_dim, bool) or not isinstance(obs_dim, Integral) or obs_dim < 1:
            raise ValueError("obs_dim must be a positive integer")
        self.obs_dim = int(obs_dim)
        self.n_sites = len(graph.sites)
        pre = torch.tensor(np.asarray(graph.graft_pre).copy(), dtype=torch.long)
        post = torch.tensor(np.asarray(graph.graft_post).copy(), dtype=torch.long)
        site = torch.tensor(np.asarray(graph.graft_site).copy(), dtype=torch.long)
        original = torch.tensor(np.asarray(graph.graft_w0).copy(), dtype=torch.float32)
        if pre.ndim != 1 or post.shape != pre.shape or site.shape != pre.shape or original.shape != pre.shape:
            raise ValueError("candidate endpoint, site, and weight arrays must have equal vector shapes")
        if len(pre) and (int(pre.min()) < 0 or int(post.min()) < 0
                         or int(pre.max()) >= graph.n or int(post.max()) >= graph.n
                         or int(site.min()) < 0 or int(site.max()) >= self.n_sites):
            raise ValueError("candidate endpoints or sites are out of graph bounds")
        origin = torch.tensor(np.asarray(graph.origin).copy(), dtype=torch.long)
        keys = sorted(set(zip(site.tolist(), origin[pre].tolist(), origin[post].tolist())))
        key_lookup = {key: i for i, key in enumerate(keys)}
        bucket_ids = torch.tensor([
            key_lookup[key] for key in zip(site.tolist(), origin[pre].tolist(), origin[post].tolist())
        ], dtype=torch.long)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        mask = torch.zeros(len(pre), dtype=torch.bool)
        budgets = []
        for bucket in range(len(keys)):
            indices = torch.nonzero(bucket_ids == bucket, as_tuple=True)[0]
            count = len(indices)
            budget = count if self.config.mode == "legacy" else max(1, min(count, round(self.config.density * count)))
            budgets.append(budget)
            selected = indices if budget == count else indices[torch.randperm(count, generator=generator)[:budget]]
            mask[selected] = True
        for name, value in (
            ("candidate_pre", pre), ("candidate_post", post), ("candidate_site", site),
            ("original_weights", original), ("bucket_ids", bucket_ids),
            ("bucket_keys", torch.tensor(keys, dtype=torch.long).reshape(-1, 3)),
            ("bucket_budgets", torch.tensor(budgets, dtype=torch.long)),
            ("active_mask", mask), ("coactivity_ema", torch.zeros(len(pre))),
            ("edge_age", mask.long() * max(1, self.config.min_age)),
            ("gradient_ema", torch.zeros(len(pre))),
            ("gradient_evidence_count", torch.zeros(len(pre), dtype=torch.long)),
            ("gradient_update_count", torch.tensor(0, dtype=torch.long)),
            ("gradient_nonfinite_count", torch.tensor(0, dtype=torch.long)),
            ("site_newborn_gradient_sum", torch.zeros(self.n_sites, dtype=torch.float64)),
            ("site_incumbent_gradient_sum", torch.zeros(self.n_sites, dtype=torch.float64)),
            ("site_newborn_gradient_count", torch.zeros(self.n_sites, dtype=torch.long)),
            ("site_incumbent_gradient_count", torch.zeros(self.n_sites, dtype=torch.long)),
            ("intervention_count", torch.tensor(0, dtype=torch.long)),
            ("intervention_swap_count", torch.tensor(0, dtype=torch.long)),
            ("site_intervention_swaps", torch.zeros(self.n_sites, dtype=torch.long)),
            ("last_gates", torch.ones(self.n_sites)), ("site_swaps", torch.zeros(self.n_sites, dtype=torch.long)),
            ("update_count", torch.tensor(0, dtype=torch.long)),
            ("last_rewire_update", torch.tensor(0, dtype=torch.long)),
            ("rewire_count", torch.tensor(0, dtype=torch.long)),
            ("swap_count", torch.tensor(0, dtype=torch.long)), ("rng_state", generator.get_state()),
        ):
            self.register_buffer(name, value)
        # This is the sole rollout-sized score accumulator. It is detached and
        # deliberately excluded from checkpoints: mid-rollout graphs are not saved.
        self.register_buffer("_pending_gradient", torch.zeros(len(pre)), persistent=False)
        self._gradient_collecting = False
        self._gradient_captures = 0
        identity = hashlib.sha256()
        identity.update(json.dumps({"n": graph.n, "sites": self.n_sites, "names": graph.names}, sort_keys=True).encode())
        for value in (pre, post, site, origin[pre], origin[post], original):
            identity.update(value.numpy().tobytes())
        self._candidate_digest = identity.hexdigest()
        self._last_event = None
        self.router = None
        if self.config.mode in {"routing", "adaptive"} and self.n_sites:
            # Linear initializes randomly before zeroing; preserve the caller's
            # network initialization and action sampling stream nevertheless.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                self.router = nn.Linear(self.obs_dim, self.n_sites)
                nn.init.zeros_(self.router.weight)
                nn.init.zeros_(self.router.bias)

    def gates(self, obs):
        observation = torch.as_tensor(obs, dtype=self.last_gates.dtype, device=self.last_gates.device)
        if observation.shape != (self.obs_dim,):
            raise ValueError(f"router observation must have shape ({self.obs_dim},)")
        gains = self.last_gates.new_ones(self.n_sites) if self.router is None else 2 * torch.sigmoid(self.router(observation))
        with torch.no_grad():
            self.last_gates.copy_(gains.detach())
        return gains

    def effective_weights(self, weights, gates=None):
        if weights.shape != self.active_mask.shape:
            raise ValueError("weights must have one value per candidate edge")
        gains = self.last_gates.clone() if gates is None else gates
        if gains.shape != (self.n_sites,):
            raise ValueError("gates must have one value per graft site")
        return weights * self.active_mask.to(weights.dtype) * gains[self.candidate_site]

    def capture_gradient(self, effective, gates):
        """Observe the latent raw-edge derivative without unmasking the signal.

        Hooks may be registered while assembling a rollout, before ``begin``.
        The callback only sums signed, detached upstream derivatives; it neither
        edits the actual gradient nor applies an EMA in backward traversal order.
        """
        if self.config.mode == "legacy" or not torch.is_grad_enabled() or not effective.requires_grad:
            return
        if effective.shape != self.active_mask.shape or gates.shape != (self.n_sites,):
            raise ValueError("gradient capture must match candidate edges and graft sites")
        if not effective.numel():
            return
        # Capture only the small site vector for each action, rather than an
        # additional edge-sized copy for every retained rollout graph.
        site_gains = gates.detach().clone()

        def accumulate(upstream):
            if not self._gradient_collecting:
                return
            with torch.no_grad():
                # No active mask: a dormant coordinate's upstream derivative is
                # sensitivity to inserting its latent raw coefficient.
                credit = upstream.detach() * site_gains[self.candidate_site]
                self._pending_gradient.add_(credit.to(self._pending_gradient))
                self._gradient_captures += 1

        effective.register_hook(accumulate)

    @torch.no_grad()
    def begin_gradient_update(self):
        """Begin one backward pass over all retained rollout action graphs."""
        self._pending_gradient.zero_()
        self._gradient_captures = 0
        self._gradient_collecting = True

    @torch.no_grad()
    def finish_gradient_update(self):
        """Apply one absolute-gradient EMA after signed rollout accumulation."""
        if not self._gradient_collecting:
            return
        self._gradient_collecting = False
        if not self._gradient_captures:
            return
        finite = torch.isfinite(self._pending_gradient)
        # Count invalid candidate/update entries, including accumulation overflow,
        # once after aggregation rather than once per action hook.
        self.gradient_nonfinite_count.add_((~finite).sum())
        magnitude = self._pending_gradient.abs()
        self.gradient_ema[finite] = 0.9 * self.gradient_ema[finite] + 0.1 * magnitude[finite]
        self.gradient_evidence_count[finite] += 1
        self.gradient_update_count.add_(1)
        newborn = self.active_mask & (self.edge_age < max(1, self.config.min_age))
        for cohort, selected in (("newborn", newborn), ("incumbent", self.active_mask & ~newborn)):
            chosen = selected & finite
            total = getattr(self, f"site_{cohort}_gradient_sum")
            count = getattr(self, f"site_{cohort}_gradient_count")
            total.zero_()
            count.zero_()
            total.index_add_(0, self.candidate_site[chosen], magnitude[chosen].to(total))
            count.index_add_(0, self.candidate_site[chosen], torch.ones_like(self.candidate_site[chosen]))
        self._pending_gradient.zero_()

    @torch.no_grad()
    def observe(self, rates):
        rates = torch.as_tensor(rates, dtype=self.coactivity_ema.dtype, device=self.coactivity_ema.device).detach()
        if rates.ndim != 1 or (len(self.candidate_pre) and len(rates) <= int(torch.maximum(self.candidate_pre.max(), self.candidate_post.max()))):
            raise ValueError("rates must cover every candidate endpoint")
        # Deliberately unmasked: currently dormant candidates need growth evidence.
        coactivity = rates[self.candidate_pre] * rates[self.candidate_post]
        self.coactivity_ema.mul_(0.95).add_(coactivity, alpha=0.05)

    def regularization(self, gates):
        return gates.sum() * 0.0 if gates.numel() == 0 else 1e-3 * (gates.mean() - 1).square()

    @torch.no_grad()
    def after_update(self, weights, optimizer, elig, heb, allow_rewire=True):
        """Advance the update clock and optionally swap equal numbers of records."""
        self._validate_update_inputs(weights, elig, heb)
        self.update_count.add_(1)
        # Mature ages stay representable even at the maximum configured age.
        aging = self.active_mask & (self.edge_age < torch.iinfo(self.edge_age.dtype).max)
        self.edge_age[aging] += 1
        self.edge_age[~self.active_mask] = 0
        update = int(self.update_count)
        if (not allow_rewire or self.config.mode not in {"structural", "adaptive"} or update <= self.config.warmup
                or update % self.config.interval):
            return None
        pruned, grown = [], []
        selected_scores = []
        protected = int((self.active_mask & (self.edge_age < self.config.min_age)).sum())
        eligible_count = int(self.active_mask.sum()) - protected
        generator = torch.Generator(device="cpu")
        if self.config.growth_rule == "random":
            generator.set_state(self.rng_state.cpu())
        for bucket, budget_tensor in enumerate(self.bucket_budgets):
            budget = int(budget_tensor)
            indices = torch.nonzero(self.bucket_ids == bucket, as_tuple=True)[0]
            active = indices[self.active_mask[indices]]
            dormant = indices[~self.active_mask[indices]]
            mature = active[self.edge_age[active] >= self.config.min_age]
            if self.config.growth_rule == "gradient":
                evidence = (self.gradient_evidence_count[dormant] > 0) & (self.gradient_ema[dormant] > 0)
                dormant = dormant[evidence & torch.isfinite(self.gradient_ema[dormant])]
            count = min(math.floor(self.config.turnover * budget), len(dormant), len(mature))
            if not count:
                continue
            # nonzero indices are ascending; stable sort makes candidate index
            # the tie breaker for both pruning and growth.
            utility = weights[mature].abs() * self.coactivity_ema[mature]
            drop = mature[torch.argsort(utility, stable=True)[:count]]
            if self.config.growth_rule == "random":
                order = torch.randperm(len(dormant), generator=generator).to(dormant.device)
                add = dormant[order[:count]]
            else:
                scores = self.gradient_ema if self.config.growth_rule == "gradient" else self.coactivity_ema
                add = dormant[torch.argsort(scores[dormant], descending=True, stable=True)[:count]]
                selected_scores.extend(scores[add].tolist())
            self._swap_records(weights, optimizer, elig, heb, drop, add)
            self.site_swaps[int(self.bucket_keys[bucket, 0])].add_(count)
            if int(self.active_mask[indices].sum()) != budget:
                raise RuntimeError("rewiring failed to preserve its directional bucket budget")
            pruned.extend(drop.tolist())
            grown.extend(add.tolist())
        if self.config.growth_rule == "random":
            self.rng_state.copy_(generator.get_state())
        if not grown:
            return None
        self.last_rewire_update.fill_(update)
        self.rewire_count.add_(1)
        self.swap_count.add_(len(grown))
        event = {"update": update, "pruned": pruned, "grown": grown, "swaps": len(grown),
                 "growth_rule": self.config.growth_rule, "growth_init": self.config.growth_init,
                 "min_age": self.config.min_age, "protected_active": protected,
                 "eligible_active": eligible_count,
                 "selection_score_mean": float(sum(selected_scores) / len(selected_scores)) if selected_scores else None}
        self._last_event = deepcopy(event)
        return event

    def _validate_update_inputs(self, weights, elig, heb):
        if weights.shape != self.active_mask.shape or elig.shape != weights.shape or heb.shape != weights.shape:
            raise ValueError("weights, eligibility, and Hebbian traces must match the candidate edges")
        for bucket, budget in enumerate(self.bucket_budgets):
            if int(self.active_mask[self.bucket_ids == bucket].sum()) != int(budget):
                raise RuntimeError("active directional bucket budget was changed outside plasticity")

    @torch.no_grad()
    def _swap_records(self, weights, optimizer, elig, heb, drop, add):
        self.active_mask[drop] = False
        self.active_mask[add] = True
        weights[add] = 0 if self.config.growth_init == "zero" else self.original_weights[add]
        changed = torch.cat((drop, add))
        self.edge_age[changed] = 0
        elig[changed] = 0
        heb[changed] = 0
        if optimizer is not None:
            for value in optimizer.state.get(weights, {}).values():
                if torch.is_tensor(value) and value.shape == weights.shape:
                    value[changed] = 0

    @torch.no_grad()
    def perturb_support(self, weights, optimizer, elig, heb, fraction, seed):
        """Disrupt support with an independent RNG and separate event counters.

        Interventions deliberately bypass age protection. They never advance the
        learning clock, private growth RNG, or learned-turnover counters.
        """
        if (isinstance(fraction, bool) or not isinstance(fraction, Real)
                or not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError("fraction must be finite and in [0, 1]")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        self._validate_update_inputs(weights, elig, heb)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        pruned, grown = [], []
        for bucket, budget in enumerate(self.bucket_budgets):
            indices = torch.nonzero(self.bucket_ids == bucket, as_tuple=True)[0]
            active = indices[self.active_mask[indices]]
            dormant = indices[~self.active_mask[indices]]
            count = min(math.floor(fraction * int(budget)), len(dormant))
            if not count:
                continue
            drop = active[torch.randperm(len(active), generator=generator).to(active.device)[:count]]
            add = dormant[torch.randperm(len(dormant), generator=generator).to(dormant.device)[:count]]
            self._swap_records(weights, optimizer, elig, heb, drop, add)
            self.site_intervention_swaps[int(self.bucket_keys[bucket, 0])].add_(count)
            pruned.extend(drop.tolist())
            grown.extend(add.tolist())
        if not grown:
            return None
        self.intervention_count.add_(1)
        self.intervention_swap_count.add_(len(grown))
        return {"update": int(self.update_count), "pruned": pruned, "grown": grown,
                "swaps": len(grown), "intervention": True, "fraction": float(fraction),
                "seed": int(seed), "growth_init": self.config.growth_init}

    def summary(self):
        gates = self.last_gates.detach()
        active = int(self.active_mask.sum())
        return {
            "mode": self.config.mode, "config": asdict(self.config), "density": self.config.density,
            "candidates": self.active_mask.numel(), "active": active,
            "dormant": self.active_mask.numel() - active, "updates": int(self.update_count),
            "last_rewire_update": int(self.last_rewire_update), "rewires": int(self.rewire_count),
            "swaps": int(self.swap_count), "last_event": deepcopy(self._last_event),
            "gate_mean": float(gates.mean()) if gates.numel() else 0.0,
            "gate_min": float(gates.min()) if gates.numel() else 0.0,
            "gate_max": float(gates.max()) if gates.numel() else 0.0,
            "gate_spread": float(gates.max() - gates.min()) if gates.numel() else 0.0,
            **self._credit_stats(torch.ones_like(self.active_mask)),
        }

    def _credit_stats(self, selected):
        active = self.active_mask & selected
        dormant = ~self.active_mask & selected
        evidence = self.gradient_evidence_count > 0
        protected = active & (self.edge_age < self.config.min_age)
        sites = torch.zeros(self.n_sites, dtype=torch.bool, device=self.candidate_site.device)
        sites[self.candidate_site[selected]] = True
        newborn_count = int(self.site_newborn_gradient_count[sites].sum())
        incumbent_count = int(self.site_incumbent_gradient_count[sites].sum())
        newborn_sum = float(self.site_newborn_gradient_sum[sites].sum())
        incumbent_sum = float(self.site_incumbent_gradient_sum[sites].sum())
        candidate_count = int(selected.sum())
        dormant_count = int(dormant.sum())
        evidence_count = int((evidence & selected).sum())
        dormant_evidence = int((evidence & dormant).sum())
        measured = evidence & selected
        return {
            "growth_rule": self.config.growth_rule, "growth_init": self.config.growth_init,
            "min_age": self.config.min_age, "protected_active": int(protected.sum()),
            "mature_active": int(active.sum()) - int(protected.sum()),
            "gradient_updates": int(self.gradient_update_count),
            "gradient_evidence": evidence_count,
            "gradient_coverage": evidence_count / candidate_count if candidate_count else 0.0,
            "dormant_gradient_evidence": dormant_evidence,
            "dormant_gradient_coverage": dormant_evidence / dormant_count if dormant_count else 0.0,
            "gradient_mean": float(self.gradient_ema[measured].double().mean()) if bool(measured.any()) else 0.0,
            "newborn_gradient_mean": newborn_sum / newborn_count if newborn_count else 0.0,
            "incumbent_gradient_mean": incumbent_sum / incumbent_count if incumbent_count else 0.0,
            "newborn_gradient_count": newborn_count, "incumbent_gradient_count": incumbent_count,
            "gradient_nonfinite": int(self.gradient_nonfinite_count),
            "interventions": int(self.intervention_count),
            "intervention_swaps": int(self.site_intervention_swaps[sites].sum()),
        }

    def site_stats(self):
        return [{"id": site, "candidates": int((self.candidate_site == site).sum()),
                 "active": int(self.active_mask[self.candidate_site == site].sum()),
                 "gate": float(self.last_gates[site]), "swaps": int(self.site_swaps[site]),
                 **self._credit_stats(self.candidate_site == site)}
                for site in range(self.n_sites)]

    def get_extra_state(self):
        return {"version": 2, "candidate_digest": self._candidate_digest,
                "config": asdict(self.config), "obs_dim": self.obs_dim,
                "last_event": deepcopy(self._last_event)}

    def set_extra_state(self, state):
        self._validate_metadata(state)
        self._last_event = deepcopy(state.get("last_event"))

    def _validate_metadata(self, state):
        if not isinstance(state, dict) or state.get("version") not in {1, 2}:
            raise RuntimeError("unsupported plasticity checkpoint metadata")
        if state.get("candidate_digest") != self._candidate_digest:
            raise RuntimeError("plasticity checkpoint candidate graph identity does not match")
        saved_config = state.get("config")
        if not isinstance(saved_config, dict):
            raise RuntimeError("plasticity checkpoint config is invalid")
        saved_config = deepcopy(saved_config)
        if state["version"] == 1:
            for name, default in (("growth_rule", "coactivity"), ("growth_init", "original"), ("min_age", 0)):
                if saved_config.get(name, default) != default:
                    raise RuntimeError("version 1 plasticity checkpoint config must use compatibility defaults")
                saved_config.setdefault(name, default)
        if saved_config != asdict(self.config):
            raise RuntimeError("plasticity checkpoint config differs; resume with its saved settings or start fresh")
        if state.get("obs_dim") != self.obs_dim:
            raise RuntimeError("plasticity checkpoint observation dimension differs")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Validate before copying any tensors, including when nested in a parent
        # nn.Module. Metadata alone must not permit overwriting fixed candidates.
        self._validate_metadata(state_dict.get(prefix + "_extra_state"))
        for name in ("candidate_pre", "candidate_post", "candidate_site", "original_weights",
                     "bucket_ids", "bucket_keys", "bucket_budgets"):
            saved = state_dict.get(prefix + name)
            current = getattr(self, name)
            if not torch.is_tensor(saved) or not torch.equal(saved.cpu(), current.detach().cpu()):
                raise RuntimeError(f"plasticity checkpoint candidate identity or budget differs: {name}")
        mask = state_dict.get(prefix + "active_mask")
        if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != self.active_mask.shape:
            raise RuntimeError("plasticity checkpoint active mask is invalid")
        for bucket, budget in enumerate(self.bucket_budgets):
            if int(mask.cpu()[self.bucket_ids.cpu() == bucket].sum()) != int(budget):
                raise RuntimeError("plasticity checkpoint violates a directional bucket budget")
        new_buffers = (
            "edge_age", "gradient_ema", "gradient_evidence_count", "gradient_update_count",
            "gradient_nonfinite_count", "site_newborn_gradient_sum", "site_incumbent_gradient_sum",
            "site_newborn_gradient_count", "site_incumbent_gradient_count", "intervention_count",
            "intervention_swap_count", "site_intervention_swaps",
        )
        if state_dict[prefix + "_extra_state"]["version"] == 1:
            for name in new_buffers:
                key = prefix + name
                if key not in state_dict:
                    state_dict[key] = (mask.long() * max(1, self.config.min_age) if name == "edge_age"
                                       else torch.zeros_like(getattr(self, name)))
        # Validate all new mutable tensors before super copies even the active
        # support. Invalid score/age data cannot partially overwrite the target.
        for name in (*new_buffers, "coactivity_ema"):
            value = state_dict.get(prefix + name)
            current = getattr(self, name)
            if (not torch.is_tensor(value) or value.shape != current.shape or value.dtype != current.dtype
                    or not bool(torch.isfinite(value).all())
                    or (name != "coactivity_ema" and bool((value < 0).any()))):
                raise RuntimeError(f"plasticity checkpoint {name} is invalid")
        age = state_dict[prefix + "edge_age"]
        if bool((age.cpu()[~mask.cpu()] != 0).any()):
            raise RuntimeError("plasticity checkpoint inactive edge_age is invalid")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self._gradient_collecting = False
        self._gradient_captures = 0
        self._pending_gradient.zero_()
