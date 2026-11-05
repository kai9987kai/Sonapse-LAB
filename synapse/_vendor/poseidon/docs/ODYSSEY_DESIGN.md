# Odyssey: Spatial Cognitive Mapping & Navigational Memory Design

Odyssey is an episodic spatial cognitive mapping and navigational memory architecture for Poseidon's synthetic TidePool survival environment. It extends the frozen Tidal core policy and multi-horizon advantage memory with dynamic on-the-fly topological mapping, temporal resource replenishment decay tracking, and goal-directed waypoint pathfinding.

The learned Tidal core remains the frozen incumbent policy. Odyssey operates without simulator pointers, global seeds, or ground-truth coordinates, constructing its internal topological belief graph strictly from the sequential 16-dimensional observation stream.

The implementation is in [`poseidon/odyssey.py`](../poseidon/odyssey.py), [`poseidon/odyssey_experiments.py`](../poseidon/odyssey_experiments.py), [`poseidon/runtime.py`](../poseidon/runtime.py), [`poseidon/cli.py`](../poseidon/cli.py), and [`poseidon/web/`](../poseidon/web/).

---

## 1. Motivation and The Spatial Exploration Gap

In Poseidon v0.4 ([`Horizon Atlas`](HORIZON_DESIGN.md)), we introduced multi-horizon rollout advantage memory and depletion trap detection. When an agent's current tile is depleted ($food < 0.04$ or $water < 0.04$), Horizon Atlas successfully suppresses fruitless $forage$ or $drink$ actions.

However, Horizon Atlas suffered from a fundamental spatial limitation:
1. **Blind Greedy Macro Exploration**: When local resources are empty, TidePool's macro action $explore$ greedily steps to the immediate neighbor maximizing:
   \[
   \arg\max_{i \in \text{Neighbors}} (1 - \text{energy}) \cdot food(i) + (1 - \text{hydration}) \cdot water(i)
   \]
   In severe scarcity ($scarcity \ge 3.5$), if an entire $2 \times 2$ or $3 \times 3$ cluster of tiles is depleted, the agent oscillates back and forth blindly between empty neighbors.
2. **Amnesia Regarding Replenishment**: TidePool tiles replenish over time:
   \[
   food(t) = \min\left(food_{\text{cap}}, food_{\text{last}} + \Delta t \cdot \frac{0.007}{scarcity}\right)
   \]
   \[
   water(t) = \min\left(water_{\text{cap}}, water_{\text{last}} + \Delta t \cdot \frac{0.014}{scarcity}\right)
   \]
   An agent that completely depleted a rich patch at tick $t=5$ leaves it behind. By tick $t=45$ ($\Delta t = 40$), that patch has naturally recovered substantial reserves. Horizon Atlas has no memory of where that patch was or that it has replenished.
3. **Exhaustion in Transit**: Traveling across multiple hops expends metabolic energy ($0.038$/step), hydration ($0.037$/step), and stamina ($0.112$/step). Embarking on a multi-step transit when stamina $< 0.15$ causes severe exhaustion damage ($0.035$ health loss per tick). Without spatial lookahead, agents collapse mid-transit.

Odyssey resolves these challenges by building an on-the-fly topological cognitive map of visited patches, estimating dynamic replenishment, computing travel costs, and executing goal-directed navigation.

---

## 2. Spatial Fingerprinting & Topological Graph Construction

### Observation Constraints
TidePool provides only a 16-dimensional normalized observation vector:
- $obs[0..4]$: reserves ($health, energy, hydration, stamina, exposure$)
- $obs[5]$: local $threat$
- $obs[6..8]$: current patch yields ($food, water, shelter$)
- $obs[9..11]$: exogenous weather ($severity, temperature, daylight$)
- $obs[12]$: current patch $terrain\_difficulty$
- $obs[13]$: adjacent neighbor $resource\_scent$
- $obs[14]$: last action scaled
- $obs[15]$: episode progress ($tick / max\_steps$)

Critically, the agent is **never** provided its global coordinate $(x, y)$.

### Unique Spatial Fingerprinting
In TidePool, each tile's terrain difficulty $obs[12]$ and shelter quality $obs[8]$ are static and generated deterministically at episode spawn with 53-bit floating point precision:
\[
terrain_i = \text{event\_uniform}(seed, 0, i, \text{"terrain"})
\]
\[
shelter_i = 0.25 + 0.75 \cdot \text{event\_uniform}(seed, 0, i, \text{"shelter"})
\]
Across 36 patches in a $6 \times 6$ grid, collision probability is effectively zero ($< 10^{-13}$). Odyssey uses the tuple of rounded terrain and shelter as an immutable patch fingerprint:
\[
\text{id}(p) = \text{str}(\text{round}(terrain, 6)) + \text{"\_"} + \text{str}(\text{round}(shelter, 6))
\]

### Topological Graph Discovery
Odyssey maintains an undirected adjacency graph $G = (V, E)$:
- $V$: Set of discovered patches, each represented by a `CognitivePatch` record.
- $E$: Discovered topological transitions. Whenever action $a \in \{4, 5\}$ ($explore$ or $flee$) transitions from patch $u$ to patch $v \ne u$, an edge $(u, v)$ is registered in $G$.

Shortest path transit distance between any two discovered patches $d(u, v)$ is computed in microseconds using Breadth-First Search (BFS) over $G$.

---

## 3. Dynamic Replenishment Modeling

For each patch $p \in V$ registered in the cognitive map:
- $t_{\text{seen}}(p)$: Tick when patch $p$ was last observed.
- $F_{\text{obs}}(p), W_{\text{obs}}(p)$: Food and water yields observed at $t_{\text{seen}}(p)$.
- $F_{\text{cap}}(p), W_{\text{cap}}(p)$: Upper capacity limits observed at initial discovery.

At current step $t$, the expected available resources at patch $p$ are:
\[
\hat{F}_t(p) = \min\left(F_{\text{cap}}(p), F_{\text{obs}}(p) + (t - t_{\text{seen}}(p)) \cdot \frac{0.007}{scarcity}\right)
\]
\[
\hat{W}_t(p) = \min\left(W_{\text{cap}}(p), W_{\text{obs}}(p) + (t - t_{\text{seen}}(p)) \cdot \frac{0.014}{scarcity}\right)
\]
Because water replenishes at twice the velocity of food ($0.014$ vs $0.007$), the cognitive map accurately identifies when a previously visited water hole has fully recharged.

---

## 4. Goal-Directed Navigational Waypoint Engine

When local resources are depleted or the agent's vital reserves fall below comfort thresholds ($energy < 0.60$ or $hydration < 0.60$), Odyssey evaluates all remembered patches $p \in V \setminus \{current\}$:

1. **Topological Distance**: Shortest path hops $D(current, p) = d(current, p)$ from BFS.
2. **Transit Costs**:
   \[
   Cost_{\text{energy}} = D \cdot 0.038, \quad Cost_{\text{hyd}} = D \cdot 0.037, \quad Cost_{\text{stam}} = D \cdot 0.112
   \]
3. **Journey Viability Check**:
   \[
   energy - Cost_{\text{energy}} > 0.06 \quad \land \quad hydration - Cost_{\text{hyd}} > 0.06
   \]
   If the journey would lead to starvation or dehydration in transit, the candidate is marked unviable.
4. **Need Weights & Net Score**:
   \[
   w_e = \max(0.0, 0.70 - energy), \quad w_h = \max(0.0, 0.70 - hydration), \quad w_s = \max(0.0, exposure - 0.40)
   \]
   \[
   Gain(p) = w_e \min(0.60, \hat{F}_t(p)) + w_h \min(0.70, \hat{W}_t(p)) + w_s \cdot shelter(p)
   \]
   \[
   Score(p) = Gain(p) - (D \cdot 0.075) - (0.35 \cdot threat(p))
   \]
The candidate maximizing $Score(p)$ with $Score(p) > 0.05$ is designated the active **Navigational Waypoint** $p^*$.

---

## 5. Action Steering & Safety Guards

Once waypoint $p^*$ is established, Odyssey applies two cognitive safety guards:

### 1. Pre-Transit Rest Guard
If the agent must navigate to $p^*$ ($D \ge 1$), but current stamina is inadequate ($stamina < 0.20$), and the agent is not under immediate threat ($threat \le 0.68$), taking a movement step risks exhaustion collapse ($stamina < 0.04 \implies 0.035$ health loss). Odyssey boosts action 0 ($rest$) with reason `pretransit_rest_override`, allowing the agent to recover $+0.25$ stamina before starting transit.

### 2. Storm Shelter Guard
If weather severity is intense ($severity > 0.60$) and agent exposure is critical ($exposure > 0.70$), Odyssey checks current tile shelter quality. If current shelter $> 0.50$, action 3 ($shelter$) is prioritized over stepping out into a lethal storm (`storm_shelter_override`).

### 3. Directional Movement Bias
If stamina is sufficient:
- When exposure or storm threat is high, boost action 5 ($flee$, which selects the neighbor maximizing $shelter - 2 \cdot threat$).
- When seeking food or water, boost action 4 ($explore$, scaled by $Score(p^*)$).

---

## 6. Multi-Horizon Empirical Calibration

Odyssey integrates Horizon Atlas's multi-horizon rollout advantage memory ($H = 16$). Candidate advantages are modulated by cognitive navigational biases:
\[
A_{\text{odyssey}}(a) = A_{\text{horizon}}(a) + \Delta_{\text{nav}}(a)
\]
Where $\Delta_{\text{nav}}(a)$ incorporates depletion penalties ($-0.5$ on empty forage/drink), pre-transit rest boosts ($+0.35$ on rest), and waypoint navigation bonuses.

Overestimation error radii are calibrated on held-out episodes across three operational regimes:
- `memory`: Full cognitive map + multi-horizon advantage with empirical error radius $\varepsilon_{\text{adv}} = 0.1502$.
- `no_memory`: Cognitive map active without advantage memory ($\varepsilon_{\text{adv}} = 0.3291$).
- `uncalibrated`: Pure point advantage gate ($\varepsilon_{\text{adv}} = 0.0$).

Overrides are confirmed when the advantage margin clears the calibrated error radius plus safety threshold:
\[
A_{\text{odyssey}}(a^*) - \varepsilon_{\text{adv}} > \text{override\_margin} \quad (0.03)
\]
or when emergency depletion / pre-transit rest guards are triggered.
