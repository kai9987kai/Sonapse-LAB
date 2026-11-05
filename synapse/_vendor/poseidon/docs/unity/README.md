# Supermix Beyond: Unity WorldLab Integration Guide

Connect your Unity 3D environments to **Supermix Beyond's Adaptive Cognitive World Model** using local TCP IPC.

---

## 🚀 Quick Setup (3 Steps)

### Step 1: Start the Python Bridge Server
In your terminal, launch the Beyond socket bridge server (listens on `127.0.0.1:8085`):

```powershell
# From the repository root
python -c "from poseidon.unity_bridge import UnityBridgeServer; s = UnityBridgeServer(port=8085); s.start(); import time; print('Beyond Unity Bridge active on 127.0.0.1:8085'); time.sleep(86400)"
```

### Step 2: Import the C# Bridge Script into Unity
Copy [`SupermixAgentBridge.cs`](file:///c:/Users/kai99/Desktop/Poseidon/docs/unity/SupermixAgentBridge.cs) into your Unity project's `Assets/Scripts/` folder:

1. In Unity, create an **Agent GameObject** (e.g., a 3D Sphere or Capsule).
2. Attach the `SupermixAgentBridge` component in the Inspector.
3. Configure settings:
   - **Host**: `127.0.0.1`
   - **Port**: `8085`
   - **Decision Interval**: `0.2` (seconds per decision step)

### Step 3: Run the Scene
Press **Play** in Unity:
- The agent establishes a TCP handshake with the Beyond backend.
- Observations (vitals, raycast threat detection, resource scents) are packed into a 16-d vector every 200ms.
- Beyond queries its **3-head uncertainty ensemble** and **persistent memory**, returning risk-calibrated macro actions:
  - `0`: Rest (recovers stamina, decreases exposure)
  - `1`: Forage (consumes nearby food flora)
  - `2`: Drink (consumes water resource)
  - `3`: Shelter (hides in granite/cave terrain)
  - `4`: Explore (moves toward resource scents)
  - `5`: Flee (retreats from local threats)
- Live epistemic uncertainty $U(s,a)$ and failure risk $P(\text{failure})$ are displayed directly in the Unity Inspector!

---

## 🧊 Importing Procedural 3D Scene Graphs
To load procedurally generated environments created by Beyond's object-centric world model:

1. Run the procedural scene exporter:
   ```powershell
   python -m poseidon beyond-scene
   ```
2. Import the resulting [`outputs/media/beyond_world.obj`](file:///c:/Users/kai99/Desktop/Poseidon/outputs/media/beyond_world.obj) directly into Unity's `Assets/Models/` directory.
3. Drag the model into your hierarchy to immediately spawn the agent, water oasis, flora cache, granite shelter, and orbital hazard sentinel.
