"""The grafted connectome as a running dynamical system, coupled to an artificial neural network
and biological sub-circuits (Central Complex ring attractor compass, Mushroom Body associative
learning with dopaminergic modulation, and intrinsic curiosity via RND).

Dynamics (rate model, one microstep):
    x   = W_conn r + W_graft r + b + P_s * I_sens
    r  <- (1 - a) r + a * relu(tanh(x)) + noise

Modeled neural mechanisms:
1. **Homeostatic Activity Regulation**: intrinsic excitability (b_i) continuously adapts
   to keep the 12,000-neuron recurrent graph within a stable dynamic range (target r* ~ 0.12).
2. **Central Complex (CX) Compass**: 16-wedge ring attractor path-integrates angular velocity
   and anchors to visual cues, providing a biological heading reference.
3. **Mushroom Body (MB) Circuit**: sparse Kenyon Cells (top-k WTA) with plastic KC->MBON
   synapses modulated by Dopaminergic Neurons (DANs) on TD error.
4. **One-pass actor-critic with Generalized Advantage Estimation (GAE)**: retained
   action likelihoods and multi-step lambda returns through recurrent microsteps.
5. **Intrinsic Curiosity (RND)**: exploration bonus from prediction error in sparse-reward tasks.
"""
from __future__ import annotations

import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graft import GraftedGraph
from .neuro_modules import CentralComplexCompass, CuriosityRND, MushroomBody
from .plasticity import GraftPlasticity, PlasticityConfig


class GraftedBrain(nn.Module):
    def __init__(self, g: GraftedGraph, n_sensory=256, n_motor=128, n_probe=256, leak=0.35,
                 noise=0.01, microsteps=4, target_rate=0.12, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        N = g.n
        self.n, self.leak, self.noise, self.microsteps = N, leak, noise, microsteps
        self.target_rate = target_rate

        self.register_buffer("base_pre", torch.tensor(g.base_pre, dtype=torch.long))
        self.register_buffer("base_post", torch.tensor(g.base_post, dtype=torch.long))
        self.register_buffer("base_w", torch.tensor(g.base_w, dtype=torch.float32))
        self.register_buffer("graft_pre", torch.tensor(g.graft_pre, dtype=torch.long))
        self.register_buffer("graft_post", torch.tensor(g.graft_post, dtype=torch.long))
        self.register_buffer("graft_site", torch.tensor(g.graft_site, dtype=torch.long))
        self.graft_w = nn.Parameter(torch.tensor(g.graft_w0))
        self.register_buffer("bias", torch.tensor(rng.normal(0.02, 0.02, N), dtype=torch.float32))
        self.register_buffer("r", torch.zeros(N))
        self.register_buffer("activity_ema", torch.full((N,), target_rate, dtype=torch.float32))
        self.register_buffer("elig", torch.zeros(len(g.graft_pre)))
        self.register_buffer("heb", torch.zeros(len(g.graft_pre)))

        def choose(roles, k):
            pool = np.flatnonzero(np.isin(g.role, roles))
            k = min(k, N)
            if len(pool) < k:
                remaining = np.setdiff1d(np.arange(N), pool)
                pool = np.concatenate([pool, rng.choice(remaining, k - len(pool), replace=False)])
            # spread the ports evenly across all grafted connectomes
            per = [pool[g.origin[pool] == c] for c in range(len(g.names))]
            out, i = [], 0
            while len(out) < k:
                p = per[i % len(per)]
                pos = i // len(per)
                if pos < len(p):
                    out.append(p[pos])
                i += 1
            return torch.tensor(sorted(out), dtype=torch.long)

        self.register_buffer("sensory_idx", choose(["sensory", "visual"], n_sensory))
        self.register_buffer("motor_idx", choose(["motor", "descending"], n_motor))
        self.register_buffer("probe_idx", torch.tensor(rng.choice(N, min(n_probe, N), replace=False), dtype=torch.long))

    def graft_current(self, r, graft_weights=None):
        weights = self.graft_w if graft_weights is None else graft_weights
        return torch.zeros_like(r).index_add(0, self.graft_post, weights * r[self.graft_pre])

    def forward(self, i_sens: torch.Tensor, graft_weights=None, adapt=True, advance=True):
        r = self.r.detach().clone()
        drive = torch.zeros(self.n).index_add(0, self.sensory_idx, i_sens)
        for _ in range(self.microsteps):
            x = (torch.zeros_like(r)
                 .index_add(0, self.base_post, self.base_w * r[self.base_pre])
                 + self.graft_current(r, graft_weights)) + self.bias + drive
            r = (1 - self.leak) * r + self.leak * F.relu(torch.tanh(x))
            if self.noise:
                r = r + self.noise * torch.randn_like(r)
            r = r.clamp(0.0, 1.0)
        with torch.no_grad():
            if adapt:
                self.elig.mul_(0.9).add_(r[self.graft_post].detach() * self.r[self.graft_pre])
                self.activity_ema.mul_(0.99).add_(0.01 * r.detach())
                self.bias.add_(0.0005 * (self.target_rate - self.activity_ema)).clamp_(-0.5, 0.5)
            if advance:
                self.r.copy_(r.detach())
        return r

    @torch.no_grad()
    def hebbian(self, td_error: float, eta=1e-3):
        """Accumulate a three-factor update; applied in :meth:`apply_hebbian` (after backprop)."""
        self.heb.add_(eta * float(np.clip(td_error, -2, 2)) * (self.elig - self.elig.mean()))

    @torch.no_grad()
    def apply_hebbian(self, clip=0.5):
        self.graft_w.add_(self.heb).clamp_(-clip, clip)
        self.heb.zero_()

    @torch.no_grad()
    def reset_state(self):
        self.r.zero_(); self.elig.zero_(); self.heb.zero_()


class NeuralInterface(nn.Module):
    """ANN <-> connectome coupling with Central Complex (CX) compass and Mushroom Body (MB) links."""

    def __init__(self, obs_dim, cam_shape, n_sensory, n_motor, n_probe, act_dim, hidden=256):
        super().__init__()
        # Camera visual pathway
        self.cam = nn.Sequential(nn.Conv2d(2, 8, 5, stride=2), nn.ReLU(), nn.Conv2d(8, 16, 5, stride=2), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Linear(16 * 12, 32), nn.ReLU())
        # Encoder: receives raw obs + cam (32) + CX compass bump (16) + MBON readout (2)
        in_dim = obs_dim + 32 + 16 + 2
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_sensory))
        # Decoder: reads motor neurons + probe neurons + MBON readout (2)
        dec_in = n_motor + n_probe + 2
        self.dec = nn.Sequential(nn.Linear(dec_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def encode(self, obs, cam, compass, mbon):
        cam_feat = self.cam(cam.unsqueeze(0)).squeeze(0)
        h_in = torch.cat([obs, cam_feat, compass, mbon])
        return 2.0 * torch.tanh(self.enc(h_in))

    def decode(self, motor_rates, probe_rates, mbon):
        h = self.dec(torch.cat([motor_rates, probe_rates, mbon]))
        return torch.tanh(self.mu(h)), self.v(h).squeeze(-1)


class Agent:
    """One-pass GAE actor-critic with optional adaptive graft support and routing."""

    def __init__(self, graph: GraftedGraph, obs_dim, cam_shape, act_dim, lr=3e-4, rollout=32,
                 gamma=0.98, gae_lambda=0.95, ppo_clip=0.2, microsteps=4, seed=0, plasticity_config=None):
        torch.manual_seed(seed)
        self.brain = GraftedBrain(graph, microsteps=microsteps, seed=seed)
        b = self.brain

        self.compass = CentralComplexCompass(n_wedges=16)
        self.mb = MushroomBody(in_dim=4, n_kc=64, k_sparse=4, n_mbon=2)
        self.curiosity = CuriosityRND(obs_dim=obs_dim)

        self.iface = NeuralInterface(obs_dim, cam_shape, len(b.sensory_idx), len(b.motor_idx), len(b.probe_idx), act_dim)
        self.opt = torch.optim.Adam([{"params": self.iface.parameters()},
                                     {"params": [b.graft_w], "lr": lr * 0.3}], lr=lr)
        self.plasticity = GraftPlasticity(graph, obs_dim, plasticity_config, seed=seed)
        self.router_opt = (torch.optim.Adam(self.plasticity.router.parameters(),
                                           lr=self.plasticity.config.router_lr)
                           if self.plasticity.router is not None else None)

        self.rollout = rollout
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.buf = []
        self.rewiring_enabled = True
        self.last_value = 0.0
        self.last_obs = np.zeros(obs_dim, dtype=np.float32)
        self.last_compass = torch.zeros(16)
        self.last_mbon = torch.zeros(2)

        self.stats = {
            "updates": 0, "loss": 0.0, "td": 0.0, "curiosity": 0.0,
            "graft_w_abs": float(b.graft_w.detach().abs().mean()) if b.graft_w.numel() else 0.0,
            "compass_bump": 0.0, "kc_sparsity": 0.0, "transitions": 0,
        }

    def act(self, obs: np.ndarray, cam: np.ndarray, training=True):
        if not training:
            with torch.no_grad():
                mu, v, _ = self._forward(obs, cam, adapt=False)
            return mu.clamp(-1, 1).numpy(), float(v)
        self.last_obs = obs
        mu, v, gates = self._forward(obs, cam, adapt=True)
        std = self.iface.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        self.buf.append([dist.log_prob(a).sum(), v, dist.entropy().sum(), None, None,
                         self.plasticity.regularization(gates)])
        return a.clamp(-1, 1).numpy(), float(v.detach())

    def _forward(self, obs, cam, adapt=True, advance=True):
        o = torch.tensor(obs, dtype=torch.float32)
        c = torch.tensor(cam, dtype=torch.float32)

        # 1. Update Central Complex ring attractor with angular velocity and retina contrast
        ang_vel = float(obs[33 * 4 + 5]) * 2.5
        compass_act = self.compass.step(ang_vel, retina_rays=obs[:33 * 4])

        # 2. Update Mushroom Body sparse Kenyon Cells with current task cue (color + active)
        cue_offset = 33 * 4 + 8 + 9
        cue_tensor = torch.tensor(obs[cue_offset:cue_offset + 4], dtype=torch.float32)
        mbon_act = self.mb(cue_tensor, adapt=adapt)

        if advance:
            self.last_compass = compass_act.detach()
            self.last_mbon = mbon_act.detach()
            self.stats["compass_bump"] = float(self.compass.bump_angle)
            self.stats["kc_sparsity"] = float((self.mb.kc_act > 0).float().mean())

        # 3. Step connectome dynamics
        i_sens = self.iface.encode(o, c, compass_act, mbon_act)
        gates = self.plasticity.gates(o)
        effective = self.plasticity.effective_weights(self.brain.graft_w, gates)
        if adapt:
            self.plasticity.capture_gradient(effective, gates)
        r = self.brain(i_sens, effective, adapt=adapt, advance=advance)
        if adapt:
            self.plasticity.observe(r.detach())
            self.brain.elig.mul_(self.plasticity.active_mask)

        # 4. Decode motor output and value
        mu, v = self.iface.decode(r[self.brain.motor_idx], r[self.brain.probe_idx], mbon_act)
        return mu, v, gates

    @torch.no_grad()
    def _value(self, obs, cam):
        angle = self.compass.bump_angle
        gates = self.plasticity.last_gates.clone()
        try:
            with torch.random.fork_rng():
                _, value, _ = self._forward(obs, cam, adapt=False, advance=False)
            return float(value)
        finally:
            self.compass.bump_angle = angle
            self.plasticity.last_gates.copy_(gates)

    def feedback(self, reward: float, done: bool, next_obs=None, next_cam=None):
        if not self.buf or self.buf[-1][3] is not None:
            raise RuntimeError("feedback requires a preceding training action")
        # 1. Add intrinsic curiosity bonus
        o_t = torch.tensor(self.last_obs, dtype=torch.float32)
        curiosity_bonus = self.curiosity.compute_bonus(o_t, train=True)
        r_total = float(reward) + 0.05 * curiosity_bonus
        self.stats["curiosity"] = 0.95 * self.stats["curiosity"] + 0.05 * curiosity_bonus

        self.buf[-1][3] = r_total
        self.buf[-1][4] = done

        current_v = float(self.buf[-1][1].detach())
        bootstrap = (0.0 if done else self._value(next_obs, next_cam)
                     if next_obs is not None and next_cam is not None else current_v)
        td = r_total + self.gamma * bootstrap - current_v
        self.brain.hebbian(td)
        self.brain.heb.mul_(self.plasticity.active_mask)
        self.mb.dopaminergic_update(td)
        self.stats["td"] = 0.95 * self.stats["td"] + 0.05 * td

        # 3. Multi-step GAE actor-critic update
        if done or len(self.buf) >= self.rollout:
            self._update(bootstrap=bootstrap)

    def flush(self, next_obs=None, next_cam=None):
        if self.buf:
            if self.buf[-1][3] is None:
                raise RuntimeError("cannot flush an action before feedback")
            bootstrap = (self._value(next_obs, next_cam) if next_obs is not None and next_cam is not None
                         else float(self.buf[-1][1].detach()))
            self._update(bootstrap)

    def reset_episode(self):
        self.brain.reset_state()
        self.compass.reset()
        self.mb.elig.zero_()
        self.mb.kc_act.zero_()
        self.last_compass.zero_()
        self.last_mbon.zero_()

    def _update(self, bootstrap: float):
        n_steps = len(self.buf)
        rewards = [b[3] for b in self.buf]
        dones = [b[4] for b in self.buf]
        values = [float(b[1].detach()) for b in self.buf] + [bootstrap]
        old_vs = torch.stack([b[1] for b in self.buf])
        ents = torch.stack([b[2] for b in self.buf])

        # Generalized Advantage Estimation (GAE)
        advantages = []
        gae = 0.0
        for t in reversed(range(n_steps)):
            next_val = values[t + 1]
            non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + old_vs.detach()

        if len(advantages) > 1 and advantages.std() > 1e-6:
            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        else:
            adv_norm = advantages

        # One-pass policy gradient: old-policy reevaluation/epochs are not performed.
        lps = torch.stack([b[0] for b in self.buf])
        policy_loss = -(lps * adv_norm).mean()

        value_loss = 0.5 * F.smooth_l1_loss(old_vs, returns)
        entropy_loss = -0.01 * ents.mean()

        route_loss = torch.stack([b[5] for b in self.buf]).mean()
        total_loss = policy_loss + value_loss + entropy_loss + route_loss

        self.opt.zero_grad()
        if self.router_opt is not None:
            self.router_opt.zero_grad()
        self.plasticity.begin_gradient_update()
        total_loss.backward()
        self.plasticity.finish_gradient_update()
        nn.utils.clip_grad_norm_(list(self.iface.parameters()) + [self.brain.graft_w], 1.0)
        self.opt.step()
        if self.router_opt is not None:
            nn.utils.clip_grad_norm_(self.plasticity.router.parameters(), 1.0)
            self.router_opt.step()
        self.brain.heb.mul_(self.plasticity.active_mask)
        self.brain.apply_hebbian()
        self.buf.clear()
        self.plasticity.after_update(self.brain.graft_w, self.opt, self.brain.elig, self.brain.heb,
                                     allow_rewire=self.rewiring_enabled)

        self.stats.update(
            updates=self.stats["updates"] + 1,
            loss=float(total_loss.detach()),
            transitions=self.stats["transitions"] + n_steps,
            graft_w_abs=(float(self.brain.graft_w.detach().abs().mean()) if self.brain.graft_w.numel() else 0.0)
        )

    # ---- persistence ----
    def state_dict(self):
        return {
            "brain": self.brain.state_dict(),
            "iface": self.iface.state_dict(),
            "mb": self.mb.state_dict(),
            "curiosity": self.curiosity.state_dict(),
            "opt": self.opt.state_dict(),
            "plasticity_config": dataclasses.asdict(self.plasticity.config),
            "plasticity": self.plasticity.state_dict(),
            "router_opt": self.router_opt.state_dict() if self.router_opt else None,
            "rewiring_enabled": self.rewiring_enabled,
            "compass_angle": self.compass.bump_angle,
            "curiosity_opt": self.curiosity.opt.state_dict(),
            "stats": self.stats,
        }

    def load_state_dict(self, sd):
        saved_cfg = PlasticityConfig(**sd.get("plasticity_config", {}))
        if saved_cfg != self.plasticity.config:
            raise ValueError("saved plasticity configuration differs; resume with saved settings or use --fresh")
        if "plasticity" in sd:
            self.plasticity.load_state_dict(sd["plasticity"])
        # Older port selection could return fewer unique ports than requested.
        # Rebuild only the interface shapes, then restore its saved parameters
        # and optimizer, rather than changing the saved brain's ports.
        port_names = ("sensory_idx", "motor_idx", "probe_idx")
        if any(sd["brain"][k].shape != getattr(self.brain, k).shape for k in port_names):
            obs_dim = self.iface.enc[0].in_features - 50
            act_dim = self.iface.log_std.numel()
            for key in port_names:
                setattr(self.brain, key, torch.empty_like(sd["brain"][key]))
            with torch.random.fork_rng(devices=[]):
                self.iface = NeuralInterface(obs_dim, (2, 48, 64), len(self.brain.sensory_idx),
                                             len(self.brain.motor_idx), len(self.brain.probe_idx), act_dim)
            self.opt = torch.optim.Adam([{"params": self.iface.parameters()},
                                         {"params": [self.brain.graft_w]}], lr=3e-4)
        self.brain.load_state_dict(sd["brain"], strict=False)
        self.iface.load_state_dict(sd["iface"], strict=False)
        if "mb" in sd:
            self.mb.load_state_dict(sd["mb"], strict=False)
        if "curiosity" in sd:
            self.curiosity.load_state_dict(sd["curiosity"], strict=False)
        self.opt.load_state_dict(sd["opt"])
        if self.router_opt is not None and sd.get("router_opt") is not None:
            self.router_opt.load_state_dict(sd["router_opt"])
        if "curiosity_opt" in sd:
            self.curiosity.opt.load_state_dict(sd["curiosity_opt"])
        self.compass.bump_angle = sd.get("compass_angle", 0.0)
        self.rewiring_enabled = sd.get("rewiring_enabled", True)
        self.stats.update(sd.get("stats", {}))
        self.buf.clear()
