import urllib.request
import json
import time

def post(url, data=None):
    if data is None:
        data = {}
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def get(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())

print("=== 1. SERVER STATE & LIVE EMBODIED SIMULATION ===")
state = get('http://127.0.0.1:8876/api/state')
print(f"Total Steps Completed: {state.get('steps')}")
world = state.get('world', {})
agent_pos = world.get('agent', {})
print(f"Agent Coordinates: x={agent_pos.get('x', 0):.3f}, y={agent_pos.get('y', 0):.3f}, th={agent_pos.get('th', 0):.3f}")
print(f"Episode Return:    {world.get('ep_return', 0.0):.4f}")
print(f"World Task:        {world.get('task')}")
print(f"Wind Streamlines:  {len(world.get('wind_streamlines', []))} active flow vectors")

print("\n=== 2. APICAL RESERVOIR HYPERBOLIC ASSAY ===")
t0 = time.perf_counter()
r1 = post('http://127.0.0.1:8876/api/apical_reservoir_hyperbolic', {'seed': 42})
apical = r1['apical_amplification']
res = r1['spectral_reservoir']
hyp = r1['hyperbolic_manifold']
print(f"Status: {r1['status']} (wallclock: {time.perf_counter()-t0:.2f}s)")
print(f"  - Apical Amplification Gain:    {apical['mean_amplification_gain']:.3f}x")
print(f"  - Calcium Plateaus / Bursts:    {apical['active_plateaus']} plateaus, {apical['calcium_spike_count']} spikes (burst ratio: {apical['coincidence_burst_ratio']:.2f})")
print(f"  - Spectral Reservoir Radius:    rho={res['spectral_radius']:.3f}, fading capacity={res['fading_memory_capacity']:.2f}")
print(f"  - Reservoir RLS Error:          {res['rls_prediction_error']:.4e} (entropy: {res['reservoir_entropy']:.3f})")
print(f"  - Hyperbolic Poincare Nodes:    {hyp['poincare_nodes']} nodes, radius={hyp['mean_hyperbolic_radius']:.3f}")
print(f"  - Hyperbolic Geodesic Distance: {hyp['mean_geodesic_distance']:.3f} ({hyp['active_geodesics']} active geodesics)")

print("\n=== 3. FRONTIER V4 ASSAY ===")
t0 = time.perf_counter()
r2 = post('http://127.0.0.1:8876/api/frontier_v4_assay', {'seed': 42})
spk = r2['spiking_neuromorphic']
gly = r2['glymphatic_clearance']
kur = r2['quantum_kuramoto']
print(f"Status: {r2['status']} (wallclock: {time.perf_counter()-t0:.2f}s)")
print(f"  - Spiking Mean Firing Rate:     {spk['mean_firing_rate_hz']:.2f} Hz across {spk['active_spiking_neurons']} active neurons")
print(f"  - Spiking Synchrony Index:      {spk['synchrony_index']:.4f} (delay: {spk['mean_conduction_delay_ms']:.1f}ms)")
print(f"  - Glymphatic Waste Level:       {gly['metabolic_waste_concentration']:.4f} mol/m^3 (cleared: {gly['cumulative_waste_cleared']:.4f})")
print(f"  - Interstitial Space Fraction:  {gly['interstitial_space_fraction']:.3f} (clearance rate: {gly['glymphatic_clearance_rate']:.4f})")
print(f"  - Kuramoto Order Parameter R:   {kur['order_parameter_r']:.4f} (coupling K={kur['coupling_strength_k']:.2f})")
print(f"  - Metastable Chimera Index:     {kur['metastable_chimera_index']:.4f}")

print("\n=== 4. SUPERMIX ARCHIMEDES NEXUS ASSAY ===")
t0 = time.perf_counter()
r3 = post('http://127.0.0.1:8876/api/supermix_archimedes_nexus_assay', {'seed': 42})
arch = r3['archimedes_buoyancy']
mem = r3['memory_carrier_nexus']
grf = r3['cross_connectome_graft']
print(f"Status: {r3['status']} (wallclock: {time.perf_counter()-t0:.2f}s)")
print(f"  - Archimedes Buoyant Force:     {arch['mean_buoyancy_force']:.4f} N (fluid density: {arch['fluid_density_at_depth']:.3f} kg/m^3)")
print(f"  - Fluid Vorticity & Drag:       vorticity={arch['vorticity_magnitude']:.4f}, Reynolds={arch['reynolds_number']:.1f}")
print(f"  - Memory Carrier SNR:           {mem['carrier_snr_db']:.2f} dB across {mem['num_stored_traces']} traces")
print(f"  - Holographic Cosine Fidelity:  {mem['retrieval_cosine_fidelity']:.4f} ({mem['active_carrier_modes']} carrier modes)")
print(f"  - Quad Connectome Synapses:     {grf['total_grafted_synapses']} synapses across {grf['num_active_bridges']} bridges")
print(f"  - Dale's Law Compliance:        {grf['dales_law_compliance_pct']:.1f}% compliant")

print("\n=== 5. CAPABILITIES REGISTRY ===")
caps = get('http://127.0.0.1:8876/api/capabilities')
print(f"Registered System Capabilities:  {len(caps.get('capabilities', []))} verified endpoints")

print("\n=======================================================")
print(">>> ALL 9 ADVANCED ARCHITECTURAL SYSTEMS ARE LIVE <<<")
print("=======================================================")
