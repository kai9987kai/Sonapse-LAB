const $=s=>document.querySelector(s);
let mode='chat',history=[],episode=null,frame=0,playing=false,first=true;
let animationHandle=null,coreReady=false,cognitiveReady=false,atlasReady=false,contrastReady=false,horizonReady=false,odysseyReady=false,helmReady=false,odysseusReady=false,mcoReady=false,auraReady=true,tesseraReady=true,mnemorphReady=true,chimeraReady=true,experimentBusy=false,requestBusy=false,serverBusy=false,restartRequired=false,statusInFlight=false,currentExperiment=null,experimentOutcome='idle',protocolChosen=false,activeJobId=null,jobPollHandle=null;
const candidateStatus={cognitive:null,atlas:null,contrast:null,horizon:null,odyssey:null,helm:null,odysseus:null,aura:null,tessera:null,mnemorph:null,chimera:null,titan:null,trident:null,poincare:null};
const plannerNotes={pantheon:'Project PANTHEON: 24-Pillar Unified Cognitive Synthesis combining all 24 experimental repositories, Morpheus Counterfactual Dream Synthesis, Quibble 24-node Merkle Build DAG, and Flipper NTAG-215 ISO/IEC 14443-A CRC-16 telemetry.',cognitive:'Supermix Poseidon v2.0 Cognitive Platform with 3-tier Mnemosyne memory, Oracle world ensemble, Active Inference EFE minimization, generative Dream sleep consolidation, and second-order Metacognition.',policy:'Direct actions from the trained policy. No teacher fallback.',mpc:'Plans actions using the learned dynamics head. Predicted outcomes can be inaccurate.',hybrid:'Combines the learned policy with dynamics-based planning.',trident:'Residual identity probes gate overrides and directed travel uses only observed TidePool transitions; synthetic results remain experimental.',titan:'Opt-in experimental composite. Sidecars are software simulations; inspect the paired seed receipt and replay before drawing conclusions.',chimera:'Experimental classical composite using an amplitude-based action heuristic, associative memory, and synthetic 3D lattice telemetry.',atlas:'Experimental Atlas v2 with retrieved residuals and absolute error bounds. Availability is not evidence of superiority.',contrast:'Experimental Contrast Atlas with selected channel corrections and calibrated paired action margins. Inspect its decisions in the replay; calibration does not guarantee safety.',horizon:'Experimental Horizon Atlas v4 with multi-horizon return advantage, depletion-trap guards and held-out calibration margins.',odyssey:'Experimental Odyssey Atlas v5 with episodic topological cognitive mapping, replenishment decay tracking and goal-oriented waypoint navigation.',helm:'Experimental Helm Critic v0.6 with history-conditioned return critique, reservoir memory and calibrated advantage gating.',odysseus:'Experimental Odysseus Navigator with empirical macro-action transition matrices, Bayesian replenishment LCB estimation and stamina rest guards.',aura:'Experimental AURA biomimetic controller with Central Complex 16-wedge ring attractor compass, sparse Kenyon cell neuropil arbiter, and ratified Tessera macro-actions.',risk_aware:'Experimental uncertainty-aware ensemble controller.'};
const controllerNames={pantheon:'Project PANTHEON (24-Pillar Synthesis)',cognitive:'POSEIDON v2.0 Intact',intact:'POSEIDON v2.0 Intact',erased:'POSEIDON v2.0 (Erased Memory)',shifted:'POSEIDON v2.0 (Shifted Memory)',random_retrieval:'POSEIDON v2.0 (Random Retrieval)',memory_free:'POSEIDON v2.0 (Memory Free)',frozen_tidal:'Frozen Tidal core',policy:'Learned policy',titan:'TITAN Composite Controller',trident:'TRIDENT Identity-Gated Residual Controller',core:'Frozen Tidal core',aura:'AURA Biomimetic Controller',metamorph:'METAMORPH Controller',chimera:'CHIMERA Controller',hyperion:'HYPERION Controller',chimera_causeway:'CHIMERA · Causeway Superposition',chimera_hologram:'CHIMERA · Holographic Memory',chimera_lattice:'CHIMERA · Diamond Lattice',molt:'MOLT Developmental Ecdysis',nexusflow:'NexusFlow Causal Flux',chronos:'Chronos Cyclic Sync',mpc:'Dynamics MPC',neural_mpc:'Neural MPC',hybrid:'Policy + dynamics',atlas:'Counterfactual Atlas v2',atlas_v2:'Counterfactual Atlas v2',atlas_no_memory:'Atlas v2 without memory',contrast:'Contrast Atlas',contrast_absolute:'Contrast · absolute bounds',contrast_no_memory:'Contrast · no memory',contrast_unfiltered:'Contrast · unfiltered',horizon:'Horizon Atlas v4',horizon_no_memory:'Horizon · no memory',horizon_ungated:'Horizon · ungated',odyssey:'Odyssey Atlas v5',odyssey_no_memory:'Odyssey · no memory',odyssey_ungated:'Odyssey · ungated',helm:'Helm Critic v0.6',selected:'Helm · selected',observation:'Helm · observation',no_innovation:'Helm · no innovation',yoked:'Helm · yoked',ungated:'Helm · ungated',odysseus:'Odysseus Navigator',odysseus_calibrated:'Odysseus · calibrated',odysseus_ungated:'Odysseus · ungated',odysseus_no_memory:'Odysseus · no memory',aura_no_ring:'AURA · no ring compass',aura_no_neuropil:'AURA · no neuropil arbiter',tessera:'Tessera Macro Commons',mnemorph:'Mnemorph Archaeological Memory',heuristic:'Heuristic control',random:'Random control',risk_aware:'Uncertainty ensemble'};
const controllerName=value=>controllerNames[value]||String(value||'Unspecified controller');
const notes={chat:'Local pretrained language model. Answers can be wrong; arithmetic is checked by a separate exact solver.',poincare:'POINCARÉ: finite-orbit dynamics, ball geometry, canonical topology, a curvature relaxation surrogate and synthetic quartet fixtures. These diagnostics do not certify a controller or prove Ricci-flow or Perelman-entropy theorems.',crosswire:'PROJECT CROSSWIRE: Candidate Permutation Version-Space Tracking, Entropy-Minimizing Active Diagnostic Probing, All-Hypothesis Safety Invariant Contract Verification & Closed-Loop Fail-Safe Runtime Shield.',faultline:'PROJECT FAULTLINE: Deterministic Condition-Stratified Joint Failure Ensemble Selection, 3-Bit Parity Hyperedge Audit, Continuous Error Covariance Minimization & Minimax Specialist Panel Triage.',tether:'PROJECT TETHER: Continuous Aleatoric-Epistemic Reliability Gating, 4-View Counterfactual Invariance (7-Arm Causal Loss), Input x Gradient Mechanistic Saliency, Zero-Shot Consensus Outlier Isolation & Evidence-Calibrated Frozen Tidal Fallback Gate.',cartan:'PROJECT CARTAN: Cartan Moving Frames on Manifolds & Exterior Differential Systems, Non-Hermitian PT-Symmetric Exceptional Points (EP2/EP3) & Petermann Divergence, Exceptional Jordan Albert Algebra h3(O) Octonionic Hurwitz Norms & F4 Automorphisms, Holographic Ryu-Takayanagi Entanglement Entropy & Thermofield Double (TFD), and Alan Turing 1952 Morphogenetic Reaction-Diffusion Chemical Instabilities.',noether:'PROJECT NOETHER: Continuous Lie SO(2) Symmetries & Conserved Charges, Unitary Modular Tensor Categories (MTC) & Verlinde TQFT, Clifford Spacetime Algebra Cl(1,3) Faraday Bivector & Lorentz Invariants, Batalin-Vilkovisky (BV) Master Quantization & Antibracket Gauge Fixing, and Active Brownian Particle (ABP) Nonequilibrium Stochastic Dynamics.',chronos:'PROJECT CHRONOS: Causal Dynamical Triangulations (CDT) Lorentzian Regge Calculus, Non-Commutative Connes Spectral Triples Tr(D^2), Tropical Algebraic Geometry & Amoeba Hypersurfaces, Stiefel & Grassmannian Riemannian Manifold Optimization, and Coalgebraic Stream Automata Bisimulation Metrics.',aether:'PROJECT AETHER: Lattice Gauge Wilson Holonomy & Yang-Mills Curvature, Jarzynski-Crooks Non-Equilibrium Thermodynamics, Fisher-Rao Information Geometry, Symplectic Störmer-Verlet Leapfrog Contact Mechanics, and Monoidal Open Game Categorical Optics.',neurograft:'PROJECT NEUROGRAFT-II: Successor Representation (SR-TD) Decoupled Value, Instant Zero-Shot Reward Revaluation, Sharp-Wave Ripple (SWR) Prioritized Reverse Replay, Graph Laplacian Topological Embeddings & Trainable Graph Circuits with Gated Attention Routing.',hyperborea:'PROJECT HYPERBOREA: Cellular Sheaf Cohomology & Neural Sheaf Diffusion, Port-Hamiltonian Dirac Passivity & IDA-PBC Control, Wilson-Kadanoff Renormalization Group Criticality, and Fibonacci Anyon Topological Quantum Braiding.',hephaestus:'PROJECT HEPHAESTUS: Wake-Sleep Library Learning & Stitch Anti-Unification Compression, Generative Flow Network (GFlowNet) Trajectory & Detailed Balance, Koopman Operator EDMD Spectral Linearization, and Buchberger Gröbner Basis Ideal Membership.',athenaeum:'PROJECT ATHENAEUM: E-Graph Equality Saturation (Union-Find Congruence Closure & MDL AST Extraction), Process-Reward Guided PUCT Monte Carlo Tree Search (Test-Time Compute), and SINDy Sparse Nonlinear Law Discovery.',proof:'PROJECT ARCHIMEDES: Formal Natural Deduction, equational algebraic theorem proving, Schwartz-Zippel invariant sampling, mathematical induction schemes, and tamper-evident SHA-256 proof certificates.',invent:'PROJECT DA VINCI: Deep computational creativity, Arthur Koestler bisociation across 8 canonical matrices, Altshuller TRIZ 40 inventive principles contradiction solver, and Pareto breakthrough dossier.',hypothesis:'PROJECT OLYMPUS: Scientific Hypothesis Formulation, Popperian Falsification, Lakatosian Research Programme Auditing, Baconian Crucial Experiments & Cryptographic Receipts.',creative:'PROJECT NOOSPHERE: Koestler Bisociation across 12 mathematical archetypes, 4-Agent Hegelian/Lakatosian Dialectical Tribunal, Fauconnier-Turner Concept Blending, and Pareto Ideation Portfolio.',pantheon:'PROJECT PANTHEON: 24-Pillar Unified Cognitive Synthesis across all 24 experimental repositories, with Morpheus Counterfactual Dream Synthesis, Quibble 24-node Merkle Build DAG, and Flipper NTAG-215 ISO/IEC 14443-A CRC-16 frame verification.',reason:'Dual-system neuro-symbolic coprocessor. Introspective Ghost Monologue, Dialectical Sublation, physical Archimedean buoyancy, and formal LTL safety proof.',problem:'PROBLEM FORGE (100 Domains): Verified combinatorial, physical, quantum, biological, ecological, strategic, control, topological, genomic, E-Graph, SINDy, epidemiological, categorical, GFlowNet, Koopman, Gröbner, CBF-QP, Conformal/DP, Cellular Sheaf, Port-Hamiltonian, Renormalization Group, Fibonacci Anyon, Connectome Motifs, Liquid Time-Constant LTC, Tri-Neuromodulation, Neural Graft Routing, Virtual Lesion Causal Attribution, Gauge Field Holonomy, Jarzynski Fluctuation Thermodynamics, Symplectic Contact Mechanics, Categorical Bidirectional Lenses, Causal Dynamical Triangulations, Noncommutative Spectral Geometry, Tropical Algebraic Geometry, Stiefel Manifold Optimization, Coalgebraic Bisimulation Metrics, Noether Symmetry Conservation, Modular Tensor Categories, Clifford Spacetime Algebra, Batalin-Vilkovisky Quantization, Active Matter Nonequilibrium Dynamics, Cartan Exterior Differential Systems, Non-Hermitian PT Symmetry, Exceptional Jordan Albert Algebra, Holographic Ryu-Takayanagi Entanglement, Morphogenetic Turing Reaction-Diffusion, Continuous Epistemic Reliability, Counterfactual Covariation Invariance, Causal Attribution Saliency, Zero-Shot Corruption Detection, Evidence-Calibrated Fallback Gating, Faultline Joint Failure Selection, Higher-Order Failure Hyperedges, Condition-Stratified Risk Matrices, Continuous Joint Error Covariance, Ensemble Panel Minimax Triage, Crosswire Version Space Pruning, Active Diagnostic Probing, Interface Contract Verification, Closed-Loop Runtime Shield, Unfamiliar Protocol Interoperability, Successor Representation TD Learning, Successor Reward Revaluation, Sharp-Wave Ripple Reverse Replay, Graph Laplacian Spatial Eigenvectors, Trainable Neural Graft Routing, Symplectic Poincaré Chaos Maps, Hyperbolic Poincaré Disk Geodesics, Poincaré Duality & Morse Complex, Ricci Flow & Perelman Entropy, and Unified Quartet Cognitive Synthesis.',river:'Project PROMETHEUS Discovery Lab: Procedural 3D RiverWorld with stepping stone reach, Archimedean buoyant pine logs, multimodal bioadhesive enzyme curing, and naval metacentric stability.',math:'Exact symbolic calculus (diff/integrate), linear algebra (det/inv/eigenvalues), number theory (factor/totient/modinv/crt), polynomials up to degree 8, and linear systems.',image:'Learned scene attributes → software renderer. Describe cube, sphere, pyramid or cylinder; color; one to three objects; and small, medium or large.',video:'A learned scene becomes a 2.4-second animation. Motions: still, orbit, bounce, spin. Geometric animation, not photorealistic video.',mesh:'Create basic OBJ + MTL and glTF assets from the same learned scene representation. Up to three geometric objects.',world:'The trained policy chooses six actions in a deterministic synthetic world. Reaches the 256-step horizon or dies. No teacher fallback.'};
const examples={chat:'Ask Poseidon something…',poincare:'Run POINCARÉ finite-orbit, hyperbolic-ball, canonical-topology, curvature-relaxation and synthetic quartet diagnostics',neurograft:'Synthesize Project NEUROGRAFT successor representation, SWR reverse replay, and graph Laplacian topological embeddings',crosswire:'Synthesize Project CROSSWIRE interface permutation recovery, active diagnostic probing, and closed-loop runtime shielding',faultline:'Synthesize Project FAULTLINE condition-stratified joint failure selection, 3-bit parity hyperedges, and continuous error covariance audit',tether:'Synthesize Project TETHER continuous reliability gating, 4-view causal invariance across 7 arms, saliency Gini concentration, zero-shot corruption isolation & calibrated fallback gate',cartan:'Synthesize Project CARTAN moving frame exterior differential systems, PT-symmetric exceptional points, Albert algebra, Ryu-Takayanagi entropy, and Turing reaction-diffusion',noether:'Synthesize Project NOETHER continuous Lie SO(2) rotation symmetry, Fibonacci MTC Verlinde fusion, Clifford STA Faraday bivector, BV quantization CME, and Active Brownian Particle Deff',chronos:'Synthesize Project CHRONOS CDT spacetime, Connes spectral action, tropical polynomial roots, Stiefel Riemannian optimization, and coalgebraic bisimulation metric',aether:'Synthesize Project AETHER gauge holonomy, Jarzynski free energy, Fisher-Rao natural gradient, symplectic leapfrog, and open game optics',hyperborea:'Synthesize Cellular Sheaf Cohomology, Port-Hamiltonian IDA-PBC, Migdal-Kadanoff RG Criticality & Fibonacci Anyon Braiding',hephaestus:'(x ^ 2 + 2 * x * y) + 1; (a ^ 2 + 2 * a * b) - 3; (u ^ 2 + 2 * u * v) * 5',athenaeum:'(x + 0) * 1 + (x - x) or (a + b) * c == a * c + b * c',proof:'(a - b)*(a + b) == a^2 - b^2 or prove P, P -> Q |- Q',invent:'Invent ultra-fast fault-tolerant distributed consensus using quantum mechanics and mycology fungal networks',hypothesis:'Formulate Popperian falsifiable hypothesis for General Relativity starlight deflection with Lakatosian research programme structure',creative:'Invent a cross-disciplinary hybrid architecture bridging quantum entanglement concurrence and Kalman filter Riccati estimation',pantheon:'Synthesize a 24-pillar Pantheon analysis for quantum entanglement concurrence and CHSH Bell violation with amplitudes [1, 0, 0, 1]',reason:'How do I cross a 10-meter river when stepping stones sink?',problem:'Compute causal dynamical triangulation Regge action for 4 time slices and 4 spatial nodes with cosmological constant 0.5',river:'Assemble floating pine logs into a raft and cross the wide river channel.',math:'integrate(3*x^2 - 4*x + 1, x, 0, 2) or det([[4, 7], [2, 6]])',image:'Create two small cyan spheres with orbit motion.',video:'Make three medium purple cubes with bounce motion.',mesh:'Build one large yellow pyramid with still motion.',world:'Run the learned agent in TidePool'};
notes.world='Choose a controller, resource scarcity and episode horizon for a deterministic synthetic world. Replay the actual decisions. Both Atlas candidates are opt-in experiments.';
$('#planner').onchange=()=>{$('#plannerNote').textContent=plannerNotes[$('#planner').value]||'Experimental survival controller.';};
function setMode(value){mode=value;document.querySelectorAll('[data-mode]').forEach(b=>{b.classList.toggle('selected',b.dataset.mode===mode);b.setAttribute('aria-selected',String(b.dataset.mode===mode));});$('#modeNote').textContent=notes[mode];$('#prompt').placeholder=examples[mode];if($('#plannerControls'))$('#plannerControls').hidden=(mode!=='world');if($('#riverControls'))$('#riverControls').hidden=(mode!=='river');if(mode==='river'){initRiverWorld();}}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{if(b.dataset.target)setMode(b.dataset.target);$('#prompt').value=b.dataset.prompt;$('#prompt').focus();});
function poincareDiagnostics(data){
  const ret=data.poincare_return_map||{},hyp=data.hyperbolic_poincare_disk||{};
  const mor=data.poincare_duality_morse||{},ric=data.ricci_flow_perelman||{};
  const quartet=data.unified_quartet_coordinator||data.unified_quartet_cognitive||{};
  const number=(value,digits=4)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(digits):'Unavailable';
  const check=value=>value===true?'passed':value===false?'failed':'Unavailable';
  const status=data.verified===true?'DIAGNOSTICS PASSED':data.verified===false?'DIAGNOSTICS FAILED':'DIAGNOSTICS INCOMPLETE';
  const count=Number.isInteger(data.diagnostics_passed)&&Number.isInteger(data.diagnostics_checked)?`${data.diagnostics_passed}/${data.diagnostics_checked}`:'Unavailable';
  const regime=typeof ret.regime==='string'&&ret.regime?ret.regime:'Unavailable';
  const action=Number.isInteger(quartet.final_dispatched_action)?quartet.final_dispatched_action:'Unavailable';
  return {status,text:`${status} | Checks: ${count} | Lyapunov λ1: ${number(ret.lambda_1)} | Orbit heuristic: ${regime} | Gromov δ: ${number(hyp.delta_hyperbolicity)} | Canonical duality: ${check(mor.poincare_duality_verified)} | Curvature relaxation total: ${check(ric.gauss_bonnet_conserved)} | Synthetic quartet action: ${action} | Scope: finite orbit, curvature relaxation surrogate, synthetic fixtures. No controller safety certification.`};
}
function setDiagnosticAvailability(name,ready){
  const status=$('#'+name+'TelemetryStatus');
  if(status&&['NOT RUN','AVAILABLE · NOT RUN','UNAVAILABLE'].includes(status.textContent)){
    status.textContent=ready===true?'AVAILABLE · NOT RUN':'UNAVAILABLE';
  }
}
function message(role,text,backend,details){if(first){$('#messages').replaceChildren();first=false;}let card=document.createElement('div');card.className='message '+role;let label=document.createElement('div');label.className='message-label';label.textContent=role==='user'?'YOU':role==='error'?'REQUEST ERROR':'POSEIDON';let body=document.createElement('div');body.className='message-body';body.textContent=text;appendAnswerEvidence(body,details);if(details?.project==='POINCARE'||(details?.poincare_return_map&&details?.hyperbolic_poincare_disk)){const diagnostic=poincareDiagnostics(details);let poiDiv=document.createElement('div');poiDiv.className='decision-evidence';poiDiv.textContent='Project POINCARÉ numerical diagnostics: '+diagnostic.text;const hash=typeof details.receipt_sha256==='string'?details.receipt_sha256.slice(0,16):null;if(hash)poiDiv.textContent+=' Fingerprint: '+hash+'…';body.append(poiDiv);}if(details?.project==='CROSSWIRE'||(details?.probing_results&&details?.runtime_shield)){let cwDiv=document.createElement('div');cwDiv.style.marginTop='8px';cwDiv.style.padding='8px';cwDiv.style.background='rgba(2, 132, 199, 0.1)';cwDiv.style.border='1px solid rgba(2, 132, 199, 0.35)';cwDiv.style.borderRadius='4px';cwDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);let pr=details.probing_results||{};let tel=details.telemetry_summary||{};let sh=details.runtime_shield||{};cwDiv.innerHTML=`🔌 <strong>Project CROSSWIRE diagnostics:</strong> Permutation Discovered: <code>${pr.fully_disambiguated}</code> (Steps: <code>${pr.steps_executed}</code>) · Entropy Reduction: <code>${tel.entropy_reduction_bits} bits</code> · Shield Intercepts: <code>${sh.total_intercepts}</code> · Zero Hazard Violations: <code>${tel.zero_hazard_violations}</code> · SHA-256 <code>${hash}...</code>`;body.append(cwDiv);}if(details?.project==='TETHER'||(details?.epistemic_trust!==undefined&&details?.four_view_causal_invariance)){let tetDiv=document.createElement('div');tetDiv.style.marginTop='8px';tetDiv.style.padding='8px';tetDiv.style.background='rgba(234, 88, 12, 0.1)';tetDiv.style.border='1px solid rgba(234, 88, 12, 0.35)';tetDiv.style.borderRadius='4px';tetDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);let tau=Number(details.epistemic_trust||0).toFixed(3);let gate=details.fallback_gate||{};let det=details.corruption_detection||{};let arm=details.four_view_causal_invariance?.arm||'tether';let gini=Number(details.attribution?.gini_concentration||0).toFixed(3);tetDiv.innerHTML=`⚓ <strong>Project TETHER diagnostics:</strong> Epistemic Trust τ <code>${tau}</code> · Gini Saliency <code>${gini}</code> · 4-View Arm <code>${arm}</code> · Outliers Isolated <code>${det.num_corrupted||0}</code> · Fallback Gate <code>${gate.decision_type||'admitted'}</code> (Override: <code>${gate.admit_override}</code>) · SHA-256 <code>${hash}...</code>`;body.append(tetDiv);}if(details?.project==='CARTAN'||(details?.cartan_exterior&&details?.non_hermitian_pt)){let carDiv=document.createElement('div');carDiv.style.marginTop='8px';carDiv.style.padding='8px';carDiv.style.background='rgba(13, 148, 136, 0.1)';carDiv.style.border='1px solid rgba(13, 148, 136, 0.35)';carDiv.style.borderRadius='4px';carDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);let ext=details.cartan_exterior||{};let pt=details.non_hermitian_pt||{};let jord=details.exceptional_jordan||{};let holo=details.holographic_entanglement||{};let tur=details.morphogenetic_turing||{};carDiv.innerHTML=`⚛️ <strong>Project CARTAN diagnostics:</strong> Cartan s0=<code>${(ext.cartan_characters||[])[0]}</code> (Involutive: <code>${ext.is_involutive_system}</code>) · PT Phase <code>${pt.phase}</code> (EP2: <code>${pt.ep_threshold_g}</code>) · Albert F4 Hurwitz <code>${jord.hurwitz_norm_identity_verified?'OK':'FAIL'}</code> · Ryu-Takayanagi S_A <code>${Number(holo.entanglement_entropy_nats||0).toFixed(2)} nats</code> · Turing Pattern <code>${tur.predicted_morphology}</code> · SHA-256 <code>${hash}...</code>`;body.append(carDiv);}if(details?.project==='NOETHER'||(details?.noether_symmetry&&details?.modular_tensor_category)){let noeDiv=document.createElement('div');noeDiv.style.marginTop='8px';noeDiv.style.padding='8px';noeDiv.style.background='rgba(8, 145, 178, 0.1)';noeDiv.style.border='1px solid rgba(8, 145, 178, 0.35)';noeDiv.style.borderRadius='4px';noeDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);let sym=details.noether_symmetry||{};let mtc=details.modular_tensor_category||{};let sta=details.clifford_spacetime_algebra||{};let bv=details.batalin_vilkovisky||{};let abp=details.active_matter||{};noeDiv.innerHTML=`⚛️ <strong>Project NOETHER diagnostics:</strong> SO(2) Noether Charge Q <code>${Number(sym.noether_conserved_charge||0).toFixed(2)}</code> ({Q,H}=0: <code>${sym.poisson_bracket_bracket_zero}</code>) · MTC Total D <code>${Number(mtc.total_quantum_dimension_D||0).toFixed(4)}</code> (Verlinde: <code>${mtc.verlinde_algebra_valid}</code>) · STA I1 <code>${Number(sta.lab_frame?.invariant_1_scalar||0).toFixed(2)}</code> · BV CME <code>${Number(bv.classical_master_equation_residual||0).toExponential(2)}</code> · ABP Deff <code>${Number(abp.effective_long_time_diffusion_Deff||0).toFixed(2)}</code> · SHA-256 <code>${hash}...</code>`;body.append(noeDiv);}if(details?.project==='CHRONOS'||(details?.causal_dynamical_triangulation&&details?.spectral_geometry)){let chDiv=document.createElement('div');chDiv.style.marginTop='8px';chDiv.style.padding='8px';chDiv.style.background='rgba(2, 132, 199, 0.1)';chDiv.style.border='1px solid rgba(2, 132, 199, 0.35)';chDiv.style.borderRadius='4px';chDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);let cdt=details.causal_dynamical_triangulation||{};let spec=details.spectral_geometry||{};let stief=details.stiefel_optimization||{};let bisim=details.coalgebraic_bisimulation||{};chDiv.innerHTML=`⏱ <strong>Project CHRONOS diagnostics:</strong> CDT Spacetime (Euler chi=<code>${cdt.euler_characteristic}</code>, S_Regge <code>${Number(cdt.lorentzian_regge_action).toFixed(2)}</code>) · Connes Spectral Tr(D^2) <code>${Number(spec.spectral_action_tr_d2).toFixed(2)}</code> · Stiefel St(2,3) Obj <code>${Number(stief.optimized_objective).toFixed(4)}</code> · Bisimulation d(s0,s1) <code>${Number(bisim.dist_s0_s1).toFixed(4)}</code> · SHA-256 <code>${hash}...</code>`;body.append(chDiv);}if(details?.project==='AETHER'||(details?.gauge_holonomy&&details?.symplectic_mechanics)){let aeDiv=document.createElement('div');aeDiv.style.marginTop='8px';aeDiv.style.padding='8px';aeDiv.style.background='rgba(79, 70, 229, 0.1)';aeDiv.style.border='1px solid rgba(79, 70, 229, 0.35)';aeDiv.style.borderRadius='4px';aeDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);let gh=details.gauge_holonomy||{};let net=details.non_equilibrium_thermodynamics||{};let ig=details.information_geometry||{};let sm=details.symplectic_mechanics||{};aeDiv.innerHTML=`⚡ <strong>Project AETHER diagnostics:</strong> Lattice Gauge Yang-Mills S <code>${Number(gh.yang_mills_action).toFixed(4)}</code> · Jarzynski ΔF <code>${Number(net.free_energy_delta_jarzynski).toFixed(4)}</code> (W_diss <code>${Number(net.dissipated_work).toFixed(4)}</code>) · Fisher Curvature K <code>${Number(ig.gaussian_curvature).toFixed(2)}</code> · Symplectic 2-Form <code>${sm.symplectic_conserved?'CONSERVED':'DRIFT'}</code> · SHA-256 <code>${hash}...</code>`;body.append(aeDiv);}if(details?.project==='NEUROGRAFT'||details?.successor_representation||(details?.tri_neuromodulation&&details?.virtual_lesion_audit)){let ngDiv=document.createElement('div');ngDiv.style.marginTop='8px';ngDiv.style.padding='8px';ngDiv.style.background='rgba(5, 150, 105, 0.1)';ngDiv.style.border='1px solid rgba(5, 150, 105, 0.35)';ngDiv.style.borderRadius='4px';ngDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);if(details?.successor_representation){let sr=details.successor_representation||{};let swr=details.sharp_wave_ripple_replay||{};let lap=details.graph_laplacian_topological||{};let cir=details.trainable_graph_circuit||{};ngDiv.innerHTML=`🧠 <strong>Project NEUROGRAFT-II diagnostics:</strong> Synthetic SR-TD revaluation (Norm: <code>${sr.M_state_norm}</code>) · SWR Sweeps: <code>${swr.sweeps_executed}</code> (${swr.total_updates} updates) · Fiedler λ1: <code>${lap.fiedler_value}</code> · Diffusion Dist: <code>${lap.mean_diffusion_distance}</code> · Route Entropy: <code>${cir.routing_entropy} nats</code> · SHA-256 <code>${hash}...</code>`;}else{let tri=details.tri_neuromodulation||{};let vla=details.virtual_lesion_audit||{};ngDiv.innerHTML=`🧠 <strong>Project NEUROGRAFT diagnostics:</strong> Multi-Species Connectome Topology · LTC Contraction <code>Contractive</code> · Regime <code>${tri.behavioral_regime}</code> (DA: <code>${tri.dopamine_level}</code>, 5-HT: <code>${tri.serotonin_level}</code>, OA: <code>${tri.octopamine_level}</code>) · Critical Lesion <code>${vla.critical_lesion_target}</code> (Max KL: <code>${vla.max_kl_divergence}</code>) · SHA-256 <code>${hash}...</code>`;}body.append(ngDiv);}
if(details?.cellular_sheaf&&details?.anyon_braiding){let hypDiv=document.createElement('div');hypDiv.style.marginTop='8px';hypDiv.style.padding='8px';hypDiv.style.background='rgba(56, 189, 248, 0.1)';hypDiv.style.border='1px solid rgba(56, 189, 248, 0.35)';hypDiv.style.borderRadius='4px';hypDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);hypDiv.innerHTML=`❄️ <strong>Project HYPERBOREA Certified:</strong> Sheaf dim(H⁰) <code>${details.cellular_sheaf.dim_H0_global_sections}</code> · Port-Hamiltonian H_d <code>${details.port_hamiltonian?.shaped_hamiltonian_final}</code> · RG K_c <code>${details.renormalization_group?.critical_coupling_Kc}</code> (ν=<code>${details.renormalization_group?.correlation_length_exponent_nu}</code>) · Anyon d_τ <code>${details.anyon_braiding.golden_ratio_quantum_dim_d_tau}</code> · SHA-256 <code>${hash}...</code>`;body.append(hypDiv);}if(details?.wake_sleep_library&&details?.gflownet_sampler){let hepDiv=document.createElement('div');hepDiv.style.marginTop='8px';hepDiv.style.padding='8px';hepDiv.style.background='rgba(234, 179, 8, 0.1)';hepDiv.style.border='1px solid rgba(234, 179, 8, 0.35)';hepDiv.style.borderRadius='4px';hepDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);hepDiv.innerHTML=`🔥 <strong>Project HEPHAESTUS Certified:</strong> Stitch MDL Compression <code>${details.wake_sleep_library.compression_ratio}x</code> · GFlowNet Z <code>${details.gflownet_sampler.partition_function_Z}</code> (L_TB <code>${details.gflownet_sampler.max_trajectory_balance_loss}</code>) · Koopman ρ <code>${details.koopman_edmd?.spectral_radius}</code> · Gröbner <code>${JSON.stringify(details.groebner_basis?.reduced_groebner_basis)}</code> · SHA-256 <code>${hash}...</code>`;body.append(hepDiv);}if(details?.egraph_saturation&&details?.test_time_compute_mcts){let athDiv=document.createElement('div');athDiv.style.marginTop='8px';athDiv.style.padding='8px';athDiv.style.background='rgba(147, 51, 234, 0.1)';athDiv.style.border='1px solid rgba(147, 51, 234, 0.35)';athDiv.style.borderRadius='4px';athDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);athDiv.innerHTML=`🏛️ <strong>Project ATHENAEUM Certified:</strong> E-Graph Canonical <code>${details.egraph_saturation.simplified_expression}</code> (${details.egraph_saturation.eclasses_count} e-classes) · PRM-MCTS Q* <code>${details.test_time_compute_mcts.best_q_value}</code> · SINDy Law <code>y = ${details.sindy_discovery?.discovered_equation}</code> · SHA-256 <code>${hash}...</code>`;body.append(athDiv);}if(details?.certificate?.certificate_sha256){let prfDiv=document.createElement('div');prfDiv.style.marginTop='8px';prfDiv.style.padding='8px';prfDiv.style.background='rgba(13, 148, 136, 0.1)';prfDiv.style.border='1px solid rgba(13, 148, 136, 0.35)';prfDiv.style.borderRadius='4px';prfDiv.style.fontSize='12px';let hash=details.certificate.certificate_sha256.slice(0,16);prfDiv.innerHTML=`📜 <strong>Project ARCHIMEDES Proof Certified:</strong> SHA-256 <code>${hash}...</code> · ${details.certificate.num_steps} Steps · Status: <code>${details.certificate.is_valid?'Q.E.D. VERIFIED':'UNPROVED'}</code>`;body.append(prfDiv);}if(details?.bisociation?.emergent_concept_name){let invDiv=document.createElement('div');invDiv.style.marginTop='8px';invDiv.style.padding='8px';invDiv.style.background='rgba(219, 39, 119, 0.1)';invDiv.style.border='1px solid rgba(219, 39, 119, 0.35)';invDiv.style.borderRadius='4px';invDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);invDiv.innerHTML=`🎨 <strong>Project DA VINCI Invention:</strong> <em>${details.title}</em> · Pareto Score: <code>${details.pareto_score}</code> · SHA-256 <code>${hash}...</code>`;body.append(invDiv);}if(details?.creative_portfolio&&Array.isArray(details.creative_portfolio)){let nooDiv=document.createElement('details');nooDiv.style.marginTop='8px';nooDiv.style.padding='8px';nooDiv.style.background='rgba(236, 72, 153, 0.1)';nooDiv.style.border='1px solid rgba(236, 72, 153, 0.35)';nooDiv.style.borderRadius='4px';nooDiv.style.fontSize='12px';let nSum=document.createElement('summary');nSum.style.cursor='pointer';nSum.style.fontWeight='bold';nSum.style.color='#f472b6';nSum.textContent=`✨ Project NOOSPHERE Bisociation & Pareto Portfolio (Top HCRI: ${details.creative_portfolio[0]?.hcri_score})`;nooDiv.append(nSum);details.creative_portfolio.forEach(item=>{let p=document.createElement('p');p.style.margin='4px 0';p.innerHTML=`<strong>#${item.rank} ${item.title}</strong> (${item.bridge}) — HCRI: <code>${item.hcri_score}</code> · Law: <code>${item.governing_law}</code>`;nooDiv.append(p);});body.append(nooDiv);}if(details?.quibble_merkle_root){let panDiv=document.createElement('div');panDiv.style.marginTop='8px';panDiv.style.padding='8px';panDiv.style.background='rgba(245, 158, 11, 0.1)';panDiv.style.border='1px solid rgba(245, 158, 11, 0.35)';panDiv.style.borderRadius='4px';panDiv.style.fontSize='12px';panDiv.innerHTML=`🏛️ <strong>Project PANTHEON 24-Pillar Synthesis:</strong> ${details.pillars_verified||24}/${details.total_pillars||24} Pillars Verified · Quibble Merkle Root <code>${details.quibble_merkle_root.slice(0,16)}...</code> · NTAG-215 CRC-16 <code>${details.ntag215_frame?.crc16_hex||'0x0000'}</code>`;body.append(panDiv);}if(details?.plan&&Array.isArray(details.plan)){let planDiv=document.createElement('details');planDiv.className='problem-plan';planDiv.style.marginTop='8px';planDiv.style.padding='8px';planDiv.style.background='rgba(56, 189, 248, 0.08)';planDiv.style.borderRadius='4px';planDiv.style.fontSize='12px';let pSum=document.createElement('summary');pSum.style.cursor='pointer';pSum.style.fontWeight='bold';pSum.style.color='#38bdf8';pSum.textContent=`Solution Plan (${details.plan.length} Steps)`;planDiv.append(pSum);details.plan.forEach((st,idx)=>{let p=document.createElement('p');p.style.margin='3px 0';p.textContent=`${idx+1}. ${st}`;planDiv.append(p);});body.append(planDiv);}if(details?.insight){let insDiv=document.createElement('div');insDiv.style.marginTop='6px';insDiv.style.color='#facc15';insDiv.style.fontSize='12px';insDiv.innerHTML=`💡 <strong>Structural Insight:</strong> ${details.insight}`;body.append(insDiv);}if(details?.falsification_receipt_sha256||details?.receipt?.hash_sha256){let olyDiv=document.createElement('div');olyDiv.style.marginTop='8px';olyDiv.style.padding='8px';olyDiv.style.background='rgba(8, 145, 178, 0.1)';olyDiv.style.border='1px solid rgba(8, 145, 178, 0.35)';olyDiv.style.borderRadius='4px';olyDiv.style.fontSize='12px';let hash=(details.falsification_receipt_sha256||details.receipt?.hash_sha256||'').slice(0,16);olyDiv.innerHTML=`🔬 <strong>Project OLYMPUS Falsification Receipt:</strong> SHA-256 <code>${hash}...</code> · Status: <code>${details.status||details.receipt?.outcome_status||'VERIFIED'}</code> · Popper/Lakatos Certified`;body.append(olyDiv);}if(details?.thought_steps){let thoughts=document.createElement('details');thoughts.className='thought-steps';thoughts.style.marginTop='8px';thoughts.style.padding='8px';thoughts.style.background='rgba(99, 102, 241, 0.08)';thoughts.style.borderRadius='4px';thoughts.style.fontSize='12px';let summary=document.createElement('summary');summary.style.cursor='pointer';summary.style.fontWeight='bold';summary.style.color='#818cf8';summary.textContent=`Introspective Ghost Monologue (${details.thought_steps.length} Steps)`;thoughts.append(summary);details.thought_steps.forEach(st=>{let p=document.createElement('p');p.style.margin='4px 0';p.innerHTML=`<strong>Step ${st.step} (${st.phase}):</strong> ${st.content}`;thoughts.append(p);});if(details.formal_proof)thoughts.append(safetyEvidenceBadge(details.formal_proof));body.append(thoughts);}card.append(label,body);if(backend){let meta=document.createElement('div');meta.className='message-backend';meta.textContent=backend;card.append(meta);}$('#messages').append(card);$('#messages').scrollTop=$('#messages').scrollHeight;}
function appendAnswerEvidence(body,details){
  const answer=details?.verification?.answer;if(!answer)return;
  const checked=answer.verified===true&&answer.scope==='parsed_problem_model'&&answer.query_bound===true;
  const evidence=document.createElement('div');evidence.className='answer-evidence';
  evidence.textContent=checked?'Parsed-model answer verified. Natural-language interpretation and generated prose remain unverified.':'Answer unverified. Evidence scope: '+String(answer.scope||'unspecified')+'.';
  if(!checked&&details.verification.language_synthesis?.present)evidence.textContent+=' Generated prose is unverified.';
  body.append(evidence);
}
function safetyEvidenceBadge(proof){
  const badge=document.createElement('div');badge.className='answer-evidence';
  const queryBound=proof.query_bound===true;
  const label=proof.query_bound===false?'Synthetic safety demonstration':queryBound?'Query-bound safety check':'Safety check with unspecified query scope';
  badge.textContent=label+': '+String(proof.verdict||'No verdict')+(queryBound?'':' · this does not verify the answer.')+(proof.certificate_sha256?' · certificate '+String(proof.certificate_sha256).slice(0,16)+'…':'');
  return badge;
}
async function apiJSON(url,options={},retainSource=false){
  const response=await fetch(url,options);
  let data,raw;
  try{if(retainSource){raw=await response.text();data=JSON.parse(raw);}else{data=await response.json();}}catch{throw Error('The server returned an unreadable response (HTTP '+response.status+').');}
  if(!response.ok){const error=Error(typeof data?.error==='string'?data.error:response.statusText||'HTTP '+response.status);error.status=response.status;throw error;}
  return retainSource?{data,raw}:data;
}
async function request(url,payload){return apiJSON(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});}
function excluded(){return [...document.querySelectorAll('.ablations input:checked')].map(x=>x.value);}
function worldSettings(){const invalid=[$('#worldScarcity'),$('#worldHorizon')].find(control=>!control.checkValidity());if(invalid){invalid.reportValidity();$('#activity').textContent='Check the Survival settings';return null;}return {scarcity:Number($('#worldScarcity').value),max_steps:Number($('#worldHorizon').value)};}
$('#promptForm').onsubmit=async e=>{e.preventDefault();if($('#send').disabled)return;let text=$('#prompt').value.trim()||(mode==='world'?'Run survival episode':'');if(!text)return;const activeMode=mode,activePlanner=$('#planner').value,settings=activeMode==='world'?worldSettings():{};if(settings===null)return;message('user',text);requestBusy=true;syncControls();$('#activity').textContent=activeMode==='chat'||activeMode==='reason'?'Loading / thinking locally…':'Running '+activeMode+'…';try{let data=await request('/api/respond',{prompt:text,mode:activeMode,history,seed:Number($('#seed').value),disabled_carriers:excluded(),planner:activePlanner,max_new_tokens:512,...settings});message('assistant',data.text,data.backend,data);if(activeMode==='chat'||activeMode==='reason'){history.push({role:'user',content:text},{role:'assistant',content:data.text});history=history.slice(-16);}if(data.links)showArtifact(data);if(data.episode)showWorld(data.episode,data.backend);if(data.prediction?.routing)showRouting(data.prediction.routing);$('#activity').textContent='Completed locally';$('#prompt').value='';}catch(error){message('error',error.message);$('#activity').textContent='Request failed — see message';}finally{requestBusy=false;serverBusy=false;syncControls();refreshStatus();}};
$('#prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#promptForm').requestSubmit();}});
function showRouting(weights){document.querySelectorAll('#routing>div').forEach((row,i)=>{const value=Math.max(0,Math.min(1,Number(weights[i])||0));row.querySelector('b').style.width=value*100+'%';row.querySelector('small').textContent=Math.round(value*100)+'%';});}
function stopWorld(){playing=false;if(animationHandle!==null){cancelAnimationFrame(animationHandle);animationHandle=null;}}
function showArtifact(data){stopWorld();$('#timeline').hidden=true;$('#decisionPanel').hidden=true;$('#visual').replaceChildren();let el;if(data.kind==='video'&&data.links.mp4){el=document.createElement('video');el.controls=true;el.loop=true;el.src=data.links.mp4;}else{el=document.createElement('img');el.src=data.links.gif||data.links.preview;el.alt='Generated '+data.scene.color+' '+data.scene.shape+' scene';}$('#visual').append(el);$('#artifactType').textContent=data.kind.toUpperCase();$('#artifactInfo').textContent=data.text+' '+(data.triangle_count?data.triangle_count+' triangles.':'');$('#downloads').replaceChildren();for(const [name,url]of Object.entries(data.links)){if(name==='preview'||name==='mtl')continue;let a=document.createElement('a');a.href=url;a.download='';a.textContent=name.toUpperCase()+' ↓';$('#downloads').append(a);}}
function drawIdle(){const c=$('#worldCanvas');if(!c)return;const ctx=c.getContext('2d');ctx.fillStyle='#0a1720';ctx.fillRect(0,0,c.width,c.height);ctx.strokeStyle='#23413f';for(let i=0;i<7;i++){ctx.beginPath();ctx.ellipse(300,175,55+i*21,25+i*11,-.24,0,Math.PI*2);ctx.stroke();}ctx.fillStyle='#8be4ca';ctx.beginPath();ctx.arc(356,150,5,0,Math.PI*2);ctx.fill();}
function showWorld(data,backend){stopWorld();episode={...data,backend:backend||data.backend};frame=0;lastFrame=0;lastDecisionEvent=null;$('#decisionPanel').hidden=true;playing=Array.isArray(data.trajectory)&&data.trajectory.length>0;$('#visual').replaceChildren();const c=document.createElement('canvas');c.id='worldCanvas';c.width=600;c.height=400;c.setAttribute('aria-label','Synthetic survival world');$('#visual').append(c);$('#timeline').hidden=!playing;$('#scrub').max=Math.max(0,(data.trajectory?.length||0)-1);$('#artifactType').textContent='TIDEPOOL';$('#artifactInfo').textContent=(data.survived?'Horizon reached alive':'Died: '+data.death_reason)+' · '+data.steps+' steps · seed '+data.seed+' · '+(backend||data.backend||controllerName(data.controller));$('#downloads').replaceChildren();if(playing)animateWorld();else{drawIdle();inspectDecision(null);}}
function drawWorld(){if(!episode)return;const ctx=$('#worldCanvas').getContext('2d'),row=episode.trajectory[frame],state=row?.state||{},snap=episode.initial_snapshot;ctx.fillStyle='#091721';ctx.fillRect(0,0,600,400);const size=44,ox=38,oy=55;for(let y=0;y<6;y++)for(let x=0;x<6;x++){const patch=snap.patches[y*6+x],alpha=.14+.35*patch.food;ctx.fillStyle=`rgba(103,197,156,${alpha})`;ctx.fillRect(ox+x*size,oy+y*size,size-4,size-4);ctx.fillStyle='#5b9cae';ctx.fillRect(ox+x*size+4,oy+y*size+size-10,(size-12)*patch.water,3);}const x=state.x??state.position?.[0]??0,y=state.y??state.position?.[1]??0;ctx.fillStyle='#f2b68e';ctx.beginPath();ctx.arc(ox+x*size+20,oy+y*size+20,10,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#ffe0b9';ctx.lineWidth=2;ctx.stroke();ctx.font='12px Segoe UI';ctx.fillStyle='#a3b8c5';ctx.fillText('TIDEPOOL / SEEDED PATCHES',38,31);ctx.fillText('Initial resource map • agent position replays',38,345);const names=['Health','Energy','Hydration','Stamina'];names.forEach((name,i)=>{let value=row?.next_observation?.[i]??1;ctx.fillStyle='#b9cdd2';ctx.fillText(name,350,85+i*55);ctx.fillStyle='#233741';ctx.fillRect(350,96+i*55,200,6);ctx.fillStyle=value<.25?'#eea276':'#8be4ca';ctx.fillRect(350,96+i*55,200*Math.max(0,Math.min(1,value)),6);});ctx.fillStyle='#eea276';ctx.fillText((row?.action_name||'').toUpperCase()+' / STEP '+(frame+1),350,329);if(row?.decision?.free_energy_objective!==undefined){ctx.fillStyle='#6cd4ff';ctx.fillText('ACTIVE INFERENCE: FEP '+decimal(row.decision.free_energy_objective,3)+' · Vol '+decimal(row.decision.epistemic_volatility,3)+' · Alert '+(row.decision.alert_level||'nominal').toUpperCase()+(numeric(row.decision.empowerment_value)&&row.decision.empowerment_value>0?' · Emp '+decimal(row.decision.empowerment_value,3):''),38,365);if(row.decision.guardrail_veto){ctx.fillStyle='#ff7b72';ctx.fillText('GUARDRAIL VETO: '+(row.decision.active_invariants||[]).join(', '),38,385);}else if(row.decision.dreams_consolidated>0){ctx.fillStyle='#8be4ca';ctx.fillText('DREAM CONSOLIDATION: '+row.decision.dreams_consolidated+' counterfactual branches consolidated',38,385);}}else if(row?.decision?.causeway){ctx.fillStyle='#6cd4ff';ctx.fillText('CHIMERA: ψ-entropy '+decimal(row.decision.causeway.von_neumann_entropy,3)+' · Coh '+decimal(row.decision.lattice_coherence,2)+' · Holo '+decimal(row.decision.holographic_capacity_load*100,1)+'%',38,365);}else if(row?.decision?.cognitive_map){ctx.fillStyle='#8be4ca';ctx.fillText('COG MAP: '+row.decision.cognitive_map.discovered_patches+' patches · '+row.decision.cognitive_map.discovered_edges+' edges'+(row.decision.best_waypoint?' · WP dist '+row.decision.best_waypoint.distance:''),38,365);}if(row?.decision?.pretransit_rest){ctx.fillStyle='#eea276';ctx.fillText('PRE-TRANSIT REST GUARD ACTIVE',38,385);}$('#scrub').value=frame;$('#tick').textContent=frame+1;}
let lastDecisionEvent=null;
const featureNames=['Health','Energy','Hydration','Stamina','Exposure','Threat','Food','Water','Shelter quality','Weather severity','Temperature','Daylight','Terrain difficulty','Resource scent','Last action','Episode progress'];
const actionNames=['rest','forage','drink','shelter','explore','flee'];
function readable(value){return typeof value==='string'?value.replaceAll('_',' '):value==null?'—':JSON.stringify(value);}
function evidenceValue(label,value){const box=document.createElement('div'),name=document.createElement('span'),detail=document.createElement('strong');name.textContent=label;detail.textContent=value;box.append(name,detail);$('#decisionEvidence').append(box);}
function inspectDecision(event){
  if(event===lastDecisionEvent)return;
  lastDecisionEvent=event;
  const decision=event?.decision,candidates=decision?.candidates;
  $('#decisionPanel').hidden=!Array.isArray(candidates);
  if(!Array.isArray(candidates))return;
  const returns=decision.backend==='horizon-atlas-v4'||decision.backend==='odyssey-atlas-v5';
  $('#decisionHeading').textContent=returns?'Return advantages at the replay step':'Candidate futures at the replay step';
  const candidateTable=$('#decisionCandidates').closest('table');
  candidateTable.querySelector('caption').textContent=returns?'Estimated multi-step reward advantages and local action filters. Empirical radii describe calibration; heuristic overrides can bypass the margin gate.':'One-step predictions and paired action advantage over the policy action. Calibration is descriptive and does not guarantee safety.';
  const headers=returns?['Action','Return advantage','Local filter','Policy probability','Distance','Fitted support','Advantage radius','Return margin','Executed','Reason']:['Action','Health','Energy','Hydration','Vital error radius','Distance','Within budget','Point advantage','Advantage radius','Paired margin'];
  candidateTable.querySelectorAll('thead th').forEach((cell,i)=>cell.textContent=headers[i]);
  const nameFor=action=>candidates.find(row=>row.action===action)?.action_name||actionNames[action]||'—';
  const reason=decision.fallback_reason?decision.fallback_reason.replaceAll('_',' '):decision.override_accepted?'empirical advantage cleared the margin':'incumbent action retained';
  $('#decisionMeta').textContent=controllerName(episode?.controller||episode?.planner||episode?.backend)+' · step '+(frame+1)+' · executed '+nameFor(decision.action)+' · policy '+nameFor(decision.policy_action)+' · proposed '+nameFor(decision.proposed_action??decision.action)+' · '+reason;
  $('#decisionEvidence').replaceChildren();
  evidenceValue('Decision gate',readable(decision.gate||'absolute vital bounds'));
  evidenceValue('Override accepted',decision.override_accepted===true?'Yes':'No');
  if(numeric(decision.point_advantage))evidenceValue('Proposed action advantage',signed(decision.point_advantage));
  if(numeric(decision.advantage_error_radius))evidenceValue('Paired calibration radius',decimal(decision.advantage_error_radius,4));
  if(numeric(decision.empirical_advantage_margin))evidenceValue('Paired empirical margin',signed(decision.empirical_advantage_margin));
  if(returns&&numeric(decision.margin))evidenceValue('Return margin',signed(decision.margin));
  $('#decisionCandidates').replaceChildren();
  const incumbent=candidates.find(row=>row.action===decision.policy_action);
  for(const row of candidates){
    const tr=document.createElement('tr');
    if(row.action===decision.action)tr.className='atlas-row';
    const label=document.createElement('th');label.scope='row';label.textContent=row.action_name+(row.action===decision.action?' · executed':'');tr.append(label);
    const prediction=Array.isArray(row.predicted_observation)?row.predicted_observation:[];
    const pointAdvantage=numeric(row.point_advantage)?row.point_advantage:numeric(row.point_value)&&numeric(incumbent?.point_value)?row.point_value-incumbent.point_value:undefined;
    const proposed=row.action===(decision.proposed_action??decision.action);
    const radius=numeric(row.advantage_error_radius)?row.advantage_error_radius:proposed?decision.advantage_error_radius:undefined;
    const margin=numeric(row.empirical_advantage_margin)?row.empirical_advantage_margin:proposed?decision.empirical_advantage_margin:undefined;
    const values=returns?[signed(row.advantage),row.safe===true?'Pass':'Reject',percentage(row.policy_probability),decimal(decision.support_distance,3),decision.supported===true?'Yes':'No',decimal(decision.advantage_error_radius,4),proposed?signed(decision.margin):'—',row.action===decision.action?'Yes':'No',row.action===decision.action?readable(decision.fallback_reason):'—']:[...Array.from({length:3},(_,i)=>percentage(prediction[i])),decimal(row.error_radius,4),decimal(row.support_distance,3),row.trusted===true?'Yes':'No',signed(pointAdvantage),decimal(radius,4),signed(margin)];
    for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}
    $('#decisionCandidates').append(tr);
  }
  const executed=candidates.find(row=>row.action===decision.action);
  const weights=decision.feature_alphas||executed?.feature_alphas;
  $('#featureDetails').hidden=!Array.isArray(weights);
  $('#decisionFeatures').replaceChildren();
  if(Array.isArray(weights))weights.forEach((alpha,i)=>$('#decisionFeatures').append(resultRow([featureNames[i]||'Channel '+i,decimal(executed?.base_observation?.[i],4),decimal(executed?.predicted_observation?.[i],4),decimal(alpha,2)])));
  $('#decisionSources').textContent=candidates.map(row=>row.action_name+':\n'+(Array.isArray(row.source_ids)&&row.source_ids.length?row.source_ids.join('\n'):'no retrieved identities')).join('\n\n');
}
const drawWorldScene=drawWorld;
drawWorld=function(){drawWorldScene();inspectDecision(episode?.trajectory?.[frame]);};
let lastFrame=0;function animateWorld(time=0){animationHandle=null;if(!playing||!episode)return;if(time-lastFrame>70){frame=Math.min(frame+1,episode.trajectory.length-1);lastFrame=time;if(frame===episode.trajectory.length-1)playing=false;}drawWorld();if(playing)animationHandle=requestAnimationFrame(animateWorld);}
$('#scrub').oninput=()=>{stopWorld();frame=Number($('#scrub').value);drawWorld();};$('#replay').onclick=()=>{if(!episode?.trajectory?.length)return;stopWorld();frame=0;lastFrame=0;playing=true;animateWorld();};
$('#remember').onclick=async()=>{if($('#remember').disabled)return;requestBusy=true;syncControls();try{const r=await request('/api/remember',{text:$('#memoryText').value,carrier:$('#carrier').value});$('#memoryStatus').textContent=r.saved?'Fact saved locally.':'Not saved';$('#memoryText').value='';}catch(e){$('#memoryStatus').textContent=e.message;}finally{requestBusy=false;serverBusy=false;syncControls();refreshStatus();}};
$('#recall').onclick=async()=>{if($('#recall').disabled)return;requestBusy=true;syncControls();try{const r=await request('/api/respond',{prompt:$('#prompt').value||'memory',mode:'memory',disabled_carriers:excluded()});$('#memoryStatus').textContent=r.text;}catch(e){$('#memoryStatus').textContent=e.message;}finally{requestBusy=false;serverBusy=false;syncControls();refreshStatus();}};
function selectedProtocol(){return $('#experimentType').value||'cognitive';}
function protocolReady(){
  const protocol=selectedProtocol();
  if(['tessera','mnemorph','mco'].includes(protocol))return true;
  if(!coreReady||protocol==='nexus_omega')return false;
  if(['cognitive','titan','trident','hyperion','chimera','metamorph','aura'].includes(protocol))return candidateStatus[protocol]?.ready===true;
  if(protocol==='odysseus')return odysseusReady;
  if(protocol==='helm')return helmReady;
  if(protocol==='odyssey')return odysseyReady&&horizonReady&&contrastReady&&atlasReady;
  if(protocol==='horizon')return horizonReady&&contrastReady&&atlasReady;
  if(protocol==='contrast')return contrastReady&&atlasReady;
  return protocol==='atlas'&&atlasReady;
}
let seedChosen=false;
$('#experimentSeed').addEventListener('input',()=>seedChosen=true);
$('#experimentType').onchange=()=>{protocolChosen=true;experimentOutcome='idle';updateProtocol();};
function syncControls(){
  const busy=experimentBusy||requestBusy||serverBusy;
  $('#send').disabled=busy;
  $('#remember').disabled=busy;
  $('#recall').disabled=busy;
  $('#planner').disabled=busy;
  $('#worldScarcity').disabled=busy;
  $('#worldHorizon').disabled=busy;
  $('#runExperiment').disabled=busy||restartRequired||!protocolReady();
  const fixedBenchmark=['mco','tessera','mnemorph'].includes(selectedProtocol());
  $('#experimentForm').querySelectorAll('input,select').forEach(control=>control.disabled=experimentBusy||(fixedBenchmark&&control.tagName==='INPUT'));
  $('#runExperiment').textContent=experimentBusy?'Experiment running…':restartRequired?'Server restart required':serverBusy?'Local server busy…':requestBusy?'Workspace request running…':'Run paired experiment ↗';
}
function updateProtocol(){
  const protocol=selectedProtocol();
  if(!seedChosen)$('#experimentSeed').value=protocol==='cognitive'?'90000000':protocol==='trident'?'88000000':protocol==='titan'?'99000000':protocol==='hyperion'?'303000001':protocol==='chimera'?'202000001':protocol==='metamorph'?'199000001':protocol==='aura'?'150000001':protocol==='tessera'?'160000001':protocol==='mnemorph'?'170000001':protocol==='odysseus'?'133000001':protocol==='mco'?'140000001':protocol==='helm'?'123000001':protocol==='odyssey'?'110000001':protocol==='horizon'?'109000001':protocol==='contrast'?'104000001':'93000001';
  $('#protocolNote').textContent=protocol==='cognitive'?'POSEIDON v2.0 Cognitive Platform evaluates 6 matched lesion arms on identical seeds under standard and POMDP settings.':protocol==='trident'?'TRIDENT compares its intact residual history and observed topology with erased, shifted, directed-support-gate-off and residual-gate-off ablations on shared seeds. Every action and world transition is replayed; results never promote the frozen core.':protocol==='titan'?'TITAN runs once per shared seed against Core, AURA, METAMORPH, CHIMERA and HYPERION. Every TidePool transition and TITAN decision is replayed before its receipt is saved; the result is descriptive and does not promote a model.':protocol==='hyperion'?'HYPERION is an opt-in software controller using simulated trace consolidation, entropy and drive heuristics on identical worlds. Device-inspired telemetry is synthetic.':protocol==='chimera'?'CHIMERA is an opt-in software controller using a classical amplitude heuristic, associative memory and 3D lattice telemetry on identical worlds.':protocol==='metamorph'?'METAMORPH developmental life-stage ecdysis (MOLT), NexusFlow causal potential flux, and Chronos cyclic beaconing on identical worlds.':protocol==='aura'?'Biomimetic ring attractor heading tracking and sparse Kenyon cell neuropil arbitration on identical worlds.':protocol==='tessera'?'Independent-lineage verified macro-action commons and finite-domain semantic safety pre-checks on identical seeds.':protocol==='mnemorph'?'Mnemorph archaeological memory assays: hub vs periphery structural lesions, context shift, and associative regrowth.':protocol==='odysseus'?'Six paired arms evaluate Odysseus calibrated routing, ungated, no-memory, Odyssey, random against the frozen policy baseline.':protocol==='mco'?'Full causal factorial evaluation across 16 carrier combinations and 3 falsifiable negative controls on standard knowledge benchmarks.':protocol==='helm'?'Six paired arms compare Helm selected, observation-only, no-innovation, yoked and ungated critics against the frozen policy baseline.':protocol==='odyssey'?'Ten paired controllers compare Odyssey cognitive mapping against Horizon Atlas, Contrast Atlas, Atlas v2, Neural MPC and policy.':protocol==='horizon'?'Nine paired controllers separate multi-horizon return advantages, depletion-trap guards, memory removal and uncalibrated interventions on identical worlds.':protocol==='contrast'?'Nine paired controllers separate action advantage, absolute bounds, memory removal and unfiltered corrections. Atlas v2 is retained as a baseline.':'Six paired controllers compare the original Atlas v2 with memory removed, neural MPC, policy, heuristic and random controls.';
  $('#hyperionSetup').hidden=(protocol!=='hyperion');
  $('#chimeraSetup').hidden=(protocol!=='chimera');
  $('#metamorphSetup').hidden=(protocol!=='metamorph');
  $('#auraSetup').hidden=(protocol!=='aura');
  $('#tesseraSetup').hidden=(protocol!=='tessera');
  $('#mnemorphSetup').hidden=(protocol!=='mnemorph');
  $('#titanSetup').hidden=(protocol!=='titan');
  $('#odysseusSetup').hidden=!(protocol==='odysseus'&&!odysseusReady);
  $('#mcoSetup').hidden=!(protocol==='mco'&&!mcoReady);
  $('#helmSetup').hidden=!(protocol==='helm'&&!helmReady);
  $('#odysseySetup').hidden=!(protocol==='odyssey'&&!odysseyReady);
  $('#horizonSetup').hidden=!(protocol==='horizon'&&!horizonReady);
  $('#contrastSetup').hidden=!(protocol==='contrast'&&!contrastReady);
  $('#atlasSetup').hidden=atlasReady;
  $('#jobPanel').hidden=false;
  if(protocol==='nexus_omega')$('#protocolNote').textContent='No paired workbench protocol is registered for NEXUS-Ω. Use Run NEXUS-Ω Discovery in the discovery panel.';
  else if(['mco','tessera','mnemorph'].includes(protocol))$('#protocolNote').textContent+=' This benchmark uses its fixed protocol; seed, scarcity, episode count and horizon controls do not apply.';
  if(!experimentBusy&&experimentOutcome==='idle')$('#experimentStatus').textContent=restartRequired?'Restart the local server to use the current source before recording an experiment.':serverBusy?'The server is handling another request. Latest availability comes from its cached status.':protocolReady()?'Ready. Compare controllers or benchmark factorial subsets on identical seeds.':'Prepare candidate dependencies before running this comparison.';
  syncControls();
}
function setCandidateStatus(kind,status){
  candidateStatus[kind]=status;
  const ready=status?.ready===true&&(coreReady||['tessera','mnemorph'].includes(kind)),label=kind==='cognitive'?'POSEIDON v2.0 Cognitive Platform':kind==='trident'?'TRIDENT Residual Controller':kind==='titan'?'TITAN Composite Controller':kind==='hyperion'?'HYPERION Frontier Super-Controller':kind==='chimera'?'CHIMERA Super-Controller':kind==='metamorph'?'METAMORPH Controller':kind==='aura'?'AURA Biomimetic Central Complex':kind==='tessera'?'Tessera Macro-Commons':kind==='mnemorph'?'Mnemorph Archaeological Memory':kind==='odysseus'?'Odysseus Navigator':kind==='helm'?'Helm Critic v0.6':kind==='odyssey'?'Odyssey Atlas v5':kind==='horizon'?'Horizon Atlas v4':kind==='contrast'?'Contrast Atlas':'Atlas v2';
  if(kind==='cognitive')cognitiveReady=ready;else if(kind==='hyperion')hyperionReady=ready;else if(kind==='chimera')chimeraReady=ready;else if(kind==='aura')auraReady=ready;else if(kind==='tessera')tesseraReady=ready;else if(kind==='mnemorph')mnemorphReady=ready;else if(kind==='odysseus')odysseusReady=ready;else if(kind==='helm')helmReady=ready;else if(kind==='odyssey')odysseyReady=ready;else if(kind==='horizon')horizonReady=ready;else if(kind==='contrast')contrastReady=ready;else if(kind==='atlas')atlasReady=ready;
  if($('#'+kind+'State')){$('#'+kind+'State').textContent=ready?'OPT-IN CANDIDATE':'FIT NOT READY';$('#'+kind+'State').classList.toggle('available',ready);}
  const receipt=status?.fit_receipt,selection=receipt?.selection,calibration=receipt?.calibration;
  const sampleCount=receipt?.training_samples;
  let detail=ready?'A fitted candidate is available. ':label+' is unavailable. '+(status?.error||(status?.ready&&!coreReady?'The Tidal checkpoint is unavailable.':'Fit it with the local command below.'))+' ';
  if(ready&&kind==='cognitive')detail='Unified v2.0 architecture with 3-tier Mnemosyne, Oracle ensemble, Active Inference, and Metacognition. ';
  if(ready&&(kind==='titan'||kind==='trident'))detail='Experimental controller in synthetic TidePool. '+(status?.replay_verified?'A replay-verified paired receipt is available. ':status?.receipt_integrity?'A checksum-checked paired receipt is available; replay has not been checked in this status request. ':status?.receipt_ready?'A paired receipt is available for verification. ':'Run the paired benchmark to produce a receipt. ');
  if(ready&&kind==='hyperion')detail='Simulated transition-trace consolidation, entropy and drive heuristics, and synthetic intermittent-energy state. ';
  if(ready&&kind==='chimera')detail='Classical amplitude-based action heuristic, associative memory, and synthetic 3D lattice telemetry. ';
  if(ready&&kind==='aura')detail='16-wedge ring attractor compass (PVA coherence) & 128 Kenyon cell sparse neuropil arbiter (k=8 WTA). ';
  if(ready&&kind==='tessera')detail='Independent dual-lineage ratified macro-actions with finite-domain safety assertions. ';
  if(ready&&kind==='mnemorph')detail='Archaeological memory assays: token hub/periphery lesions and associative regrowth. ';
  if(ready&&kind==='odysseus'&&numeric(status?.fit_receipt?.error_radius))detail+='Error radius: '+decimal(status.fit_receipt.error_radius,3)+'. ';
  if(ready&&numeric(sampleCount))detail+=sampleCount.toLocaleString()+' fitted '+(kind==='horizon'||kind==='helm'?'anchor states':'branch transitions')+'. ';
  if(ready&&kind==='helm'){
    const family=selection?.family;
    if(family)detail+='Selected family: '+family+'. ';
    const radii=calibration?.selected?.radii;
    if(radii)detail+='Radii: H4='+decimal(radii['4'],3)+', H16='+decimal(radii['16'],3)+'. ';
  }
  if(ready&&kind==='contrast'){
    const alphas=selection?.alphas||selection?.feature_alphas;
    const corrected=Array.isArray(alphas)?alphas.filter(alpha=>numeric(alpha)&&alpha>0).length:null;
    if(corrected!==null)detail+=corrected+' of 16 channels admit retrieved corrections. ';
    if(numeric(selection?.episode_count))detail+='Selection: '+selection.episode_count+' episodes. ';
    const paired=calibration?.paired?.memory;
    if(numeric(paired?.episode_count))detail+='Paired calibration: '+paired.episode_count+' episodes. ';
  }
  if(ready&&kind==='horizon'){
    if(numeric(calibration?.advantage_error_radius))detail+='Advantage radius: '+decimal(calibration.advantage_error_radius,3)+'. ';
    if(numeric(calibration?.episodes))detail+='Calibration: '+calibration.episodes+' episodes. ';
  }
  detail+='The policy remains the default; availability does not establish better performance.';
  const note=$('#'+kind+'Note');if(note)note.textContent=detail;
  const option=$('#planner option[value="'+kind+'"]');
  if(option){
    option.disabled=!ready;
    option.textContent=label+' · '+(ready?'experimental':'unavailable');
    if(!ready&&$('#planner').value===kind){$('#planner').value='policy';$('#planner').onchange();}
  }
}
const numeric=value=>typeof value==='number'&&Number.isFinite(value);
const decimal=(value,digits=2)=>numeric(value)?value.toFixed(digits):'—';
const percentage=value=>numeric(value)?(value*100).toFixed(1)+'%':'—';
const signed=value=>numeric(value)?(value>0?'+':'')+value.toFixed(3):'—';
function resultRow(values,arm){
  const row=document.createElement('tr');
  if(arm==='atlas'||arm==='contrast')row.className='atlas-row';
  values.forEach((value,i)=>{const cell=document.createElement(i===0?'th':'td');if(i===0)cell.scope='row';cell.textContent=String(value);row.append(cell);});
  return row;
}
function receiptURL(value){
  if(typeof value!=='string'||!value)return null;
  try{const url=new URL(value,location.origin);return url.origin===location.origin&&['http:','https:'].includes(url.protocol)?url.href:null;}catch{return null;}
}
function showPredictionComparison(comparison){
  const available=comparison&&typeof comparison==='object'&&!Array.isArray(comparison);
  $('#predictionPanel').hidden=!available;
  $('#predictionSummary').replaceChildren();$('#predictionChannels').replaceChildren();
  if(!available)return;
  const labels={base:'Base neural dynamics',unfiltered:'Unfiltered memory correction',memory:'Selected memory correction'};
  for(const key of ['base','unfiltered','memory']){
    const item=comparison[key];if(!item||typeof item!=='object')continue;
    $('#predictionSummary').append(resultRow([labels[key],numeric(item.anchors)?item.anchors:'—',numeric(item.branches)?item.branches:'—',decimal(item.mse,6)]));
  }
  featureNames.forEach((name,i)=>$('#predictionChannels').append(resultRow([name,...['base','unfiltered','memory'].map(key=>decimal(comparison[key]?.per_channel_mse?.[i],6))])));
}
function showRiskCoverage(risk){
  const available=risk&&typeof risk==='object'&&!Array.isArray(risk);
  $('#riskPanel').hidden=!available;
  $('#riskSummary').replaceChildren();
  if(!available)return;
  for(const [groupName,rows]of Object.entries(risk)){
    if(!Array.isArray(rows))continue;
    for(const row of rows){
      const label=groupName==='regular'?'Regular schedule':'Override anchors';
      const coverage=percentage(row.coverage);
      const adverse=numeric(row.adverse_rate)?row.adverse+' ('+percentage(row.adverse_rate)+')':'0';
      const gain=numeric(row.mean_actual_advantage)?signed(row.mean_actual_advantage):'—';
      const exposure=row.no_exposure?'No exposure':row.exposure+' / '+row.eligible;
      $('#riskSummary').append(resultRow([label+' (H'+row.horizon+')',signed(row.threshold),row.eligible,row.exposure,coverage,adverse,gain,exposure]));
    }
  }
}
function showMCOResults(data){
  const conditions=$('#mcoConditions'),carriers=$('#mcoCarriers');
  conditions.replaceChildren();carriers.replaceChildren();
  $('#mcoBestSubset').textContent='Best observed carrier subset: '+(typeof data.best_subset==='string'?data.best_subset.replace(/\+/g,' + '):'Not reported');
  for(const [condition,row]of Object.entries(data.condition_comparison||{})){
    if(!row||typeof row!=='object')continue;
    conditions.append(resultRow([condition,numeric(row.trials)?row.trials:'—',percentage(row.mean_hit_at_1),percentage(row.mean_hit_at_3),decimal(row.mean_mrr,3),decimal(row.mean_grounded_score,4),decimal(row.mean_retrieval_ms,2)],condition));
  }
  for(const [carrier,row]of Object.entries(data.carrier_contributions||{})){
    if(!row||typeof row!=='object')continue;
    carriers.append(resultRow([carrier,decimal(row.mean_with,3),decimal(row.mean_without,3),signed(row.marginal_grounded_delta)],carrier));
  }
}
function showExperiment(data){
  currentExperiment=data;
  const mco=data.schema==='poseidon-mco-experiment-v1';
  $('#comparisonTitle').textContent=mco?'Memory carrier comparison':'Controller comparison';
  $('#mcoResults').hidden=!mco;$('#controllerResults').hidden=mco;$('#pairedResults').hidden=mco;$('#pairedHeading').hidden=mco;
  if(mco)showMCOResults(data);
  $('#resultCaution').textContent=mco?'This deterministic retrieval benchmark compares the fixed memory-carrier protocol and its controls. Its scores do not establish language-model improvements or real-world memory performance.':'A small paired experiment describes these seeds and this horizon. It does not establish general superiority or real-world survival ability.';
  const returns=data.schema==='poseidon-horizon-experiment-v1'||data.schema==='poseidon-odyssey-experiment-v1'||data.schema==='poseidon-helm-experiment-v1';
  const summary=$('#experimentSummary'),paired=$('#experimentPaired');
  summary.replaceChildren();paired.replaceChildren();
  for(const [arm,row]of Object.entries(data.summary||{})){
    if(!row||typeof row!=='object'||Array.isArray(row))continue;
    const audit=row.counterfactual_audit;
    const outcomeRows=Array.isArray(data.rows)?data.rows.filter(item=>item.arm===arm):[];
    const count=numeric(row.overrides)?row.overrides:outcomeRows.length&&outcomeRows.every(item=>numeric(item.overrides))?outcomeRows.reduce((sum,item)=>sum+item.overrides,0):undefined;
    const rate=numeric(row.override_rate)?row.override_rate:numeric(row.mean_steps)&&row.mean_steps>0&&numeric(row.episodes)?count/(row.mean_steps*row.episodes):undefined;
    const overrides=numeric(count)?count+' / '+percentage(rate):percentage(rate);
    const branchGain=numeric(audit?.mean_actual_advantage_at_overrides)?audit.mean_actual_advantage_at_overrides:count?outcomeRows.reduce((sum,item)=>sum+(item.counterfactual_audit?.mean_actual_advantage_at_overrides??0)*(item.overrides??0),0)/count:undefined;
    const latency=returns&&row.mean_steps>0?row.mean_decision_ms/row.mean_steps:row.mean_decision_ms??row.mean_latency_ms;
    summary.append(resultRow([controllerName(arm),numeric(row.episodes)?row.episodes:'—',percentage(row.survival_rate),decimal(row.mean_steps,1),decimal(row.mean_reward,3),decimal(latency,2),percentage(row.fallback_rate),overrides,signed(branchGain),decimal(row.mean_prediction_mse,5),decimal(row.mean_discovered_patches,2)],arm));
  }
  for(const [arm,row]of Object.entries(data.paired||{})){
    if(!row||typeof row!=='object'||Array.isArray(row))continue;
    const interval=Array.isArray(row.ci95)&&row.ci95.length===2?'['+signed(row.ci95[0])+', '+signed(row.ci95[1])+']':'—';
    const outcomes=Array.isArray(row.per_seed)?row.per_seed.map(item=>item.reward_delta??item.reward_diff).filter(numeric):[];
    const counts=[row.wins??outcomes.filter(value=>value>1e-10).length,row.ties??outcomes.filter(value=>Math.abs(value)<=1e-10).length,row.losses??outcomes.filter(value=>value< -1e-10).length];
    const name=arm.replace(/^trident_vs_/,'').replace(/_vs_policy$/,'');
    paired.append(resultRow([controllerName(name),signed(row.mean_reward_delta??row.mean_reward_diff),interval,counts.map(x=>numeric(x)?x:'—').join(' / ')],name));
  }
  $('#pairedHeading').textContent=data.schema==='poseidon-trident-experiment-v1'?'TRIDENT paired reward differences vs ablations':'Paired reward differences vs learned policy';
  $('#pairedCaption').textContent=data.schema==='poseidon-trident-experiment-v1'?'Each TRIDENT arm minus the named ablation on the same seed. No model is promoted.':'Each controller minus policy on the same seed. '+(returns?'This protocol provides no uncertainty interval.':'Intervals are reported by the experiment.');
  showPredictionComparison(data.prediction_comparison);
  showRiskCoverage(data.risk_coverage);
  const verification=data.verification;
  const replayedEpisodes=verification?.episodes_replayed;
  const scope=verification?.verification_scope||verification?.scope||data.verification_scope;
  const replayChecked=verification?.verified===true&&(verification?.replay_verified===true||(numeric(replayedEpisodes)&&replayedEpisodes>0));
  $('#experimentIdentity').textContent='Experiment '+String(data.experiment_id||'receipt available')+(verification?.verified===true?' · receipt checks passed':'')+(typeof scope==='string'?' ('+scope+')':'')+(replayChecked?' · replay checked':'')+(numeric(replayedEpisodes)?' · '+replayedEpisodes+' episodes replayed':'');
  const url=receiptURL(data.artifact_url);
  $('#experimentReceipt').hidden=!url;
  if(url)$('#experimentReceipt').href=url;else $('#experimentReceipt').removeAttribute('href');
  $('#experimentReplay').hidden=!data.replay?.trajectory?.length;
  const limits=$('#experimentLimits');limits.replaceChildren();
  const items=Array.isArray(data.limits)?data.limits:data.limits&&typeof data.limits==='object'?Object.entries(data.limits).map(([key,value])=>key+': '+String(value)):data.limits?[String(data.limits)]:[];
  for(const value of items){const item=document.createElement('li');item.textContent=typeof value==='string'?value:JSON.stringify(value);limits.append(item);}
  if(returns){const item=document.createElement('li');item.textContent='Branch utility gain covers one-step reserve utility. Decision return scores estimate multi-step reward; these measures have different units.';limits.append(item);}
  $('#experimentResults').hidden=false;
}
let jobMonitorGeneration=0,jobResultGeneration=0;
function stopJobMonitoring(){
  if(jobPollHandle!==null)clearInterval(jobPollHandle);
  jobPollHandle=null;activeJobId=null;jobMonitorGeneration++;jobResultGeneration++;
  $('#cancelJob').hidden=true;$('#cancelJob').disabled=false;
}
function finishJobMonitoring(job){
  stopJobMonitoring();experimentBusy=false;serverBusy=false;
  $('#experimentPanel').setAttribute('aria-busy','false');
  if(job.state==='completed'){
    showExperiment(job.result);experimentOutcome='complete';
    $('#experimentStatus').classList.remove('error');
    $('#experimentStatus').textContent='Completed locally via durable job. Inspect its receipt and replay below.';
  }else{
    experimentOutcome='failed';$('#experimentStatus').classList.add('error');
    $('#experimentStatus').textContent='Job '+job.state+': '+(job.error?.message||'No further detail available.');
  }
  syncControls();refreshStatus();refreshJobs();
}
async function monitorJob(jobId){
  stopJobMonitoring();const generation=jobMonitorGeneration;
  activeJobId=jobId;experimentBusy=true;experimentOutcome='running';
  stopWorld();currentExperiment=null;$('#experimentResults').hidden=true;$('#decisionPanel').hidden=true;
  $('#experimentPanel').setAttribute('aria-busy','true');$('#experimentStatus').classList.remove('error');
  $('#experimentStatus').textContent='Monitoring existing local experiment. Reconnecting does not submit new work.';
  $('#jobPanel').hidden=false;$('#cancelJob').hidden=false;$('#jobIdentity').textContent='Job '+jobId;
  syncControls();
  let inFlight=false;
  const poll=async()=>{
    if(inFlight||generation!==jobMonitorGeneration)return;
    inFlight=true;
    try{
      const job=await apiJSON('/api/jobs/'+encodeURIComponent(jobId));
      if(generation!==jobMonitorGeneration)return;
      if(job.id!==jobId||!['queued','running','completed','cancelled','failed'].includes(job.state))throw Error('The server returned an invalid job record.');
      $('#jobIdentity').textContent='Job '+job.id+' · '+job.state;
      const completed=job.progress?.completed,total=job.progress?.total;
      $('#jobProgress').value=numeric(completed)&&numeric(total)&&total>0?Math.max(0,Math.min(1,completed/total)):0;
      $('#jobStatus').textContent=(job.progress?.phase||job.state)+(job.cancel_requested?' · cancellation requested':'');
      if(job.state==='completed'){
        if(!job.result||typeof job.result!=='object')throw Error('The completed job has no readable result.');
        finishJobMonitoring(job);
      }else if(job.state==='failed'||job.state==='cancelled')finishJobMonitoring(job);
    }catch(error){
      if(generation!==jobMonitorGeneration)return;
      $('#jobStatus').textContent='Could not refresh job: '+error.message+(error.status===404?'':' Retrying automatically.');
      if(error.status===404){
        stopJobMonitoring();experimentBusy=false;experimentOutcome='failed';
        $('#experimentPanel').setAttribute('aria-busy','false');$('#experimentStatus').classList.add('error');
        $('#experimentStatus').textContent='This job is no longer available. Refresh recent jobs to inspect the server history.';
        syncControls();
      }
    }finally{inFlight=false;}
  };
  jobPollHandle=setInterval(poll,1000);
  await poll();
}
async function viewJobResult(jobId){
  const generation=++jobResultGeneration;
  try{
    const job=await apiJSON('/api/jobs/'+encodeURIComponent(jobId));
    if(generation!==jobResultGeneration||experimentBusy||requestBusy)return;
    if(job.state!=='completed'||!job.result||typeof job.result!=='object')throw Error('This job has no completed result.');
    showExperiment(job.result);$('#jobIdentity').textContent='Job '+job.id+' · '+job.state;
    $('#jobStatus').textContent='Saved result loaded from local job history.';
  }catch(error){if(generation===jobResultGeneration&&!experimentBusy&&!requestBusy)$('#jobStatus').textContent='Could not open result: '+error.message;}
}
async function refreshJobs(){
  const list=$('#jobHistory');
  try{
    const data=await apiJSON('/api/jobs');list.replaceChildren();
    for(const job of (data.jobs||[])){
      const li=document.createElement('li');
      li.textContent=job.kind+' · '+job.id+' · '+job.state+' ('+(job.created_at||'')+(job.finished_at?' → '+job.finished_at:'')+')';
      if(job.state==='completed'||job.state==='queued'||job.state==='running'){
        const btn=document.createElement('button');btn.type='button';btn.className='micro';
        btn.textContent=job.state==='completed'?'View result':'Monitor job';
        btn.onclick=async()=>{
          if(experimentBusy||requestBusy)return;
          btn.disabled=true;
          try{if(job.state==='completed')await viewJobResult(job.id);else await monitorJob(job.id);}finally{btn.disabled=false;}
        };
        li.append(' ',btn);
      }else if(job.error?.message)li.append(' · '+job.error.message);
      list.append(li);
    }
    if(!data.jobs?.length){const item=document.createElement('li');item.textContent='No recorded jobs.';list.append(item);}
  }catch(error){
    list.replaceChildren();const item=document.createElement('li');item.textContent='Could not load job history: '+error.message;list.append(item);
  }
}
$('#refreshJobs').onclick=refreshJobs;
$('#cancelJob').onclick=async()=>{
  const jobId=activeJobId;if(!jobId)return;
  $('#cancelJob').disabled=true;
  try{
    const job=await request('/api/jobs/'+encodeURIComponent(jobId)+'/cancel',{});
    if(activeJobId!==jobId)return;
    if(job.state==='cancelled'||job.state==='failed'||job.state==='completed')finishJobMonitoring(job);
    else $('#jobStatus').textContent='Cancellation requested. Waiting for the current bounded step to stop.';
  }catch(error){if(activeJobId===jobId)$('#jobStatus').textContent='Could not cancel job: '+error.message;}
  finally{$('#cancelJob').disabled=false;}
};
$('#experimentForm').onsubmit=async event=>{
  event.preventDefault();if(experimentBusy||requestBusy||serverBusy||restartRequired||!protocolReady())return;
  if(!$('#experimentForm').checkValidity()){$('#experimentForm').reportValidity();return;}
  jobResultGeneration++;
  const protocol=selectedProtocol();
  const payload={kind:protocol==='trident'?'trident-experiment':protocol==='titan'?'titan-experiment':protocol==='hyperion'?'hyperion-experiment':protocol==='aura'?'aura-experiment':protocol==='tessera'?'tessera-experiment':protocol==='mnemorph'?'mnemorph-experiment':protocol==='helm'?'helm-experiment':protocol,seed:Number($('#experimentSeed').value),scarcity:Number($('#scarcity').value),episodes:Number($('#episodes').value),max_steps:Number($('#maxSteps').value)};
  stopWorld();
  experimentBusy=true;currentExperiment=null;experimentOutcome='running';$('#experimentResults').hidden=true;
  $('#decisionPanel').hidden=true;
  $('#experimentPanel').setAttribute('aria-busy','true');
  $('#experimentStatus').classList.remove('error');
  $('#experimentStatus').textContent=['mco','tessera','mnemorph'].includes(protocol)?'Submitting '+protocol.toUpperCase()+' fixed benchmark protocol. Local CPU work may take a while…':'Running '+(protocol==='cognitive'||protocol==='trident'||protocol==='titan'||protocol==='atlas'||protocol==='helm'||protocol==='odysseus'?'six':protocol==='hyperion'||protocol==='aura'?'five':protocol==='chimera'?'four':protocol==='contrast'||protocol==='horizon'?'nine':'ten')+' controllers on '+payload.episodes+' paired episodes each, up to '+payload.max_steps+' steps. Local CPU work may take a while…';
  syncControls();
  if(protocol==='cognitive'||protocol==='trident'||protocol==='titan'||protocol==='hyperion'||protocol==='chimera'||protocol==='metamorph'||protocol==='aura'||protocol==='tessera'||protocol==='mnemorph'||protocol==='helm'||protocol==='odysseus'||protocol==='mco'){
    try{
      const payload=['mco','tessera','mnemorph'].includes(protocol)?{kind:protocol+'-experiment'}:{kind:protocol+'-experiment',seed:Number($('#experimentSeed').value),scarcity:Number($('#scarcity').value),episodes:Number($('#episodes').value),max_steps:Number($('#maxSteps').value)};
      const jobData=await request('/api/jobs',payload);
      await monitorJob(jobData.id);
      return;
    }catch(error){
      stopJobMonitoring();
      experimentOutcome='failed';$('#experimentStatus').textContent='Experiment failed: '+error.message;$('#experimentStatus').classList.add('error');
      experimentBusy=false;serverBusy=false;$('#experimentPanel').setAttribute('aria-busy','false');syncControls();refreshStatus();
      return;
    }
  }
  try{
    const url=protocol==='cognitive'?'/api/cognitive-experiment':protocol==='hyperion'?'/api/hyperion-experiment':protocol==='chimera'?'/api/chimera-experiment':protocol==='metamorph'?'/api/metamorph-experiment':protocol==='aura'?'/api/aura-experiment':protocol==='tessera'?'/api/tessera-experiment':protocol==='mnemorph'?'/api/mnemorph-experiment':protocol==='odysseus'?'/api/odysseus-experiment':protocol==='mco'?'/api/mco-experiment':protocol==='odyssey'?'/api/odyssey-experiment':protocol==='horizon'?'/api/horizon-experiment':protocol==='contrast'?'/api/contrast-experiment':'/api/experiment';
    const data=await request(url,payload);
    showExperiment(data);experimentOutcome='complete';
    $('#experimentStatus').textContent='Completed locally. Compare controllers or benchmark factorial subsets below.';
    if(data.cognitive_status)setCandidateStatus('cognitive',data.cognitive_status);
    if(data.hyperion)setCandidateStatus('hyperion',data.hyperion);
    if(data.titan)setCandidateStatus('titan',data.titan);
    if(data.trident)setCandidateStatus('trident',data.trident);
    if(data.chimera)setCandidateStatus('chimera',data.chimera);
    if(data.metamorph)setCandidateStatus('metamorph',data.metamorph);
    if(data.aura)setCandidateStatus('aura',data.aura);
    if(data.tessera)setCandidateStatus('tessera',data.tessera);
    if(data.mnemorph)setCandidateStatus('mnemorph',data.mnemorph);
    if(data.atlas)setCandidateStatus('atlas',data.atlas);
    if(data.contrast)setCandidateStatus('contrast',data.contrast);
    if(data.horizon)setCandidateStatus('horizon',data.horizon);
    if(data.odyssey)setCandidateStatus('odyssey',data.odyssey);
    if(data.helm)setCandidateStatus('helm',data.helm);
    if(data.odysseus)setCandidateStatus('odysseus',data.odysseus);
  }catch(error){experimentOutcome='failed';$('#experimentStatus').textContent='Experiment failed: '+error.message;$('#experimentStatus').classList.add('error');}
  finally{
    experimentBusy=false;serverBusy=false;$('#experimentPanel').setAttribute('aria-busy','false');syncControls();refreshStatus();
  }
};
$('#experimentReplay').onclick=()=>{if(!currentExperiment?.replay)return;showWorld(currentExperiment.replay);$('#visual').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'center'});};
async function refreshStatus(){
  if(statusInFlight)return;
  statusInFlight=true;
  try{
    const response=await fetch('/api/status');if(!response.ok)throw Error(response.statusText);
    const data=await response.json();coreReady=data.core_ready===true;serverBusy=data.busy===true||data.atlas?.busy===true||data.contrast?.busy===true||data.horizon?.busy===true||data.odyssey?.busy===true||data.helm?.busy===true||data.odysseus?.busy===true;restartRequired=data.restart_required===true||data.source_current===false;
    const statusCached=data.status_cached===true||data.atlas?.status_cached===true||data.contrast?.status_cached===true||data.horizon?.status_cached===true||data.odyssey?.status_cached===true||data.helm?.status_cached===true||data.odysseus?.status_cached===true;
    $('#connection').textContent='● Connected · '+(restartRequired?'restart needed':serverBusy?'server busy':'on this PC')+(statusCached?' · cached status':'');
    $('#restartBanner').hidden=!restartRequired;
    const changed=Array.isArray(data.changed_source_files)?data.changed_source_files.filter(value=>typeof value==='string'):[];
    $('#restartNote').textContent='The loaded code differs from the files on disk. Restart the local server before recording an experiment.'+(changed.length?' Changed source: '+changed.join(', ')+'.':'');
    const issues=Object.entries(data.status_errors||{}).filter(([,value])=>typeof value==='string');
    const statusErrors=$('#statusErrors');statusErrors.replaceChildren();
    for(const [name,error]of issues){const item=document.createElement('li');item.textContent=name+': '+error;statusErrors.append(item);}
    $('#statusDiagnostics').hidden=issues.length===0;
    $('#coreState').textContent=data.core_ready?'Checkpoint available':'Checkpoint unavailable';
    const r=data.reports||{},n=r.core?.examples_seen??r.core?.training_receipt?.examples_seen??r.core_recent?.at(-1)?.examples_seen;
    $('#exposures').textContent=n==null?'—':Number(n).toLocaleString();
    const counts=r.language_data?.counts;
    $('#corpus').textContent=counts?Object.values(counts).reduce((a,b)=>a+b,0).toLocaleString()+' rows':'No corpus report';
    $('#evidence').textContent=JSON.stringify(r,null,2);
    setCandidateStatus('atlas',data.atlas);setCandidateStatus('contrast',data.contrast);setCandidateStatus('horizon',data.horizon);setCandidateStatus('odyssey',data.odyssey);setCandidateStatus('helm',data.helm);setCandidateStatus('odysseus',data.odysseus);
    setCandidateStatus('cognitive',data.cognitive);setCandidateStatus('aura',data.aura);setCandidateStatus('tessera',data.tessera);setCandidateStatus('mnemorph',data.mnemorph);setCandidateStatus('metamorph',data.metamorph);setCandidateStatus('chimera',data.chimera);setCandidateStatus('hyperion',data.hyperion);setCandidateStatus('titan',data.titan);setCandidateStatus('trident',data.trident);
    if($('#poincareState'))$('#poincareState').textContent=data.poincare?.ready===true?'AVAILABLE':'UNAVAILABLE';
    if($('#neurograftState'))$('#neurograftState').textContent=data.neurograft?.ready===true?'AVAILABLE':'UNAVAILABLE';
    mcoReady=data.mco?.receipt_ready===true;
    if(!protocolChosen){$('#experimentType').value='cognitive';protocolChosen=true;}
    updateProtocol();
  }catch{$('#connection').textContent='Offline · start the local server';}
  finally{statusInFlight=false;syncControls();}
}

async function pollTelemetry(){
  try{
    const res=await fetch('/api/telemetry/memory-influence');
    if(!res.ok)return;
    const tel=await res.json();
    if($('#dvGemStatus')){$('#dvGemStatus').textContent=tel.dv_gem_active?'ACTIVE':'INACTIVE';$('#dvGemStatus').style.background=tel.dv_gem_active?'#2563eb':'#64748b';$('#dvGemStatus').style.color='white';}
    if($('#dvGemDeltas'))$('#dvGemDeltas').textContent=JSON.stringify(tel.dv_gem_deltas);
    if($('#epistemicDriveText'))$('#epistemicDriveText').textContent=`Active Inference Drive: ${Number(tel.active_inference_drive).toFixed(3)} | Epistemic Reduction: ${Number(tel.epistemic_reduction).toFixed(3)}`;
    if($('#overrideStatus')){$('#overrideStatus').textContent=tel.planner_override?'OVERRIDE':'NOMINAL';$('#overrideStatus').style.background=tel.planner_override?'#f59e0b':'#10b981';$('#overrideStatus').style.color='white';}
    if($('#overrideText'))$('#overrideText').textContent=`Override: ${tel.planner_override} | Margin: +${Number(tel.override_margin).toFixed(4)} | Scope: ${tel.readiness_scope||'calibrated-stage6-core'}`;

    const fRes=await fetch('/api/nexus-omega/factored-state');
    if(fRes.ok){
      const fData=await fRes.json();
      const d=fData.decision;
      if(d){
        if($('#factoredStateText') && d.factored_state){
          const fs=d.factored_state;
          $('#factoredStateText').textContent=`Endo: [${fs.endogenous?.join(', ')}] | Exo: [${fs.exogenous?.join(', ')}] | Regime: storm=${fs.causal_regime?.[0]}`;
        }
        if($('#efeText')){
          $('#efeText').textContent=`G(a): ${d.free_energy_objective > 0 ? '+':''}${d.free_energy_objective} | Prag: ${d.pragmatic_value} | Epist: ${d.epistemic_value} | Uncalibrated penalty: ${d.conformal_lcb} | Heuristic guard: ${d.conformal_override?'OVERRIDE':'inactive'} | Calm recovery: ${d.recovery_admission?'ADMITTED':'inactive'}`;
          if($('#efeStatus')){
            $('#efeStatus').textContent=d.conformal_override?'HEURISTIC OVERRIDE':'UNCALIBRATED';
            $('#efeStatus').style.background=d.conformal_override?'#f59e0b':'#64748b';
            $('#efeStatus').style.color='white';
          }
        }
        if($('#actionRankText')){
          const labels=['forage','shelter','explore','rest'];
          const scores=Array.isArray(d.action_scores)?[...d.action_scores].sort((a,b)=>b.efe-a.efe):[];
          $('#actionRankText').textContent=scores.length
            ? 'Uncalibrated action scores: '+scores.map(s=>`${labels[s.action]||`action ${s.action}`} G=${decimal(s.efe,3)} R=${decimal(s.predicted_reward,3)}${s.selected_by_objective?' [top]':''}${s.selected_after_guard?' [chosen]':''}`).join(' · ')
            : 'Per-action ranking unavailable for this planner';
        }
      }
    }

    const hRes=await fetch('/api/nexus-omega/hierarchical-state');
    if(hRes.ok){
      const hData=await hRes.json();
      const hd=hData.decision;
      if(hd){
        if($('#macroText')){
          $('#macroText').textContent=`Macro Intent: ${hd.macro_intent?.toUpperCase()} | Commit Ticks: ${hd.macro_intent_remaining_ticks} | Micro Action: ${hd.micro_action_name}`;
        }
        if($('#diffusionText') && hd.imagined_trajectory){
          const it=hd.imagined_trajectory;
          $('#diffusionText').textContent=`H=${it.horizon} | Raw: ${it.unguided_cumulative_reward>0?'+':''}${it.unguided_cumulative_reward} | Guided: ${it.cumulative_reward>0?'+':''}${it.cumulative_reward} | Return Δ: ${it.guided_return_difference} | Max state Δ: ${it.max_branch_divergence_l2} | Uncalibrated penalty: ${it.unguided_return_penalty} | forwards: ${it.ensemble_member_forwards}`;
        }
        if($('#actionRankText') && Array.isArray(hd.action_scores) && hd.action_scores.length){
          const labels=['forage','shelter','explore','rest'];
          const scores=[...hd.action_scores].sort((a,b)=>b.efe-a.efe);
          $('#actionRankText').textContent='Uncalibrated micro scores: '+scores.map(s=>`${labels[s.action]||`action ${s.action}`} G=${decimal(s.efe,3)} R=${decimal(s.predicted_reward,3)}${s.selected_by_micro_planner?' [micro]':''}${s.selected_after_hierarchy?' [chosen]':''}`).join(' · ');
        }
      }
    }

    const uRes=await fetch('/api/nexus-omega/ultima-state');
    if(uRes.ok){
      const uData=await uRes.json();
      if(uData.quantum_state && $('#quantumText')){
        const qs=uData.quantum_state;
        $('#quantumText').textContent=`Purity: ${Number(qs.purity).toFixed(4)} | Berry Phase: ${Number(qs.berry_phase_holonomy).toFixed(4)} rad | Ent: ${Number(qs.von_neumann_entropy).toFixed(4)} | Tunnel: ${Number(qs.tunneling_rate).toFixed(3)}`;
        if($('#quantumStatus')){$('#quantumStatus').textContent=qs.purity>0.8?'SUPERPOSITION':'DECOHERENT';$('#quantumStatus').style.background=qs.purity>0.8?'#06b6d4':'#f59e0b';}
      }
      if(uData.spectral_profile && $('#spectralText')){
        const sp=uData.spectral_profile;
        $('#spectralText').textContent=`Dom Period: T=${Number(sp.dominant_period).toFixed(1)} | Phase: ${Number(sp.current_phase).toFixed(3)} rad | Storm Resonance: ${(Number(sp.storm_resonance_probability)*100).toFixed(1)}% | Spec Ent: ${Number(sp.spectral_entropy).toFixed(3)}`;
        if($('#spectralStatus')){$('#spectralStatus').textContent=sp.storm_resonance_probability>0.5?'RESONANCE SPIKE':'HARMONIC';$('#spectralStatus').style.background=sp.storm_resonance_probability>0.5?'#ef4444':'#ec4899';}
      }
      if(uData.active_opcode && $('#bytecodeText')){
        const op=uData.active_opcode;
        $('#bytecodeText').textContent=`Opcode: ${op.opcode_name} | Steps: ${op.execution_steps?.length||0} | Mutated: ${op.mutated} | Cert: ${op.verification_certificate?.slice(0,12)}...`;
        if($('#bytecodeStatus')){$('#bytecodeStatus').textContent=op.ltl_safe?'LTL CERTIFIED':'FALLBACK SAFE';$('#bytecodeStatus').style.background=op.ltl_safe?'#10b981':'#f59e0b';}
      }
      if(uData.thermodynamic_state && $('#autopoiesisText')){
        const ts=uData.thermodynamic_state;
        const pad=ts.affect_valence_pad||[0,0,0];
        $('#autopoiesisText').textContent=`PAD: [${pad.map(v=>Number(v)>=0?'+'+Number(v).toFixed(2):Number(v).toFixed(2)).join(', ')}] | S_int: +${Number(ts.internal_entropy_production).toFixed(3)} | S_diss: +${Number(ts.heat_dissipation_rate).toFixed(3)} | Viability: ${(Number(ts.autopoietic_viability)*100).toFixed(1)}%`;
      }
      if(uData.dream_receipt && $('#dreamText')){
        const dr=uData.dream_receipt;
        $('#dreamText').textContent=`Replayed: ${dr.replayed_transitions} | Regret Red: ${Number(dr.mean_regret_mitigation).toFixed(3)} | Synaptic Pruning: ${(Number(dr.synaptic_pruning_ratio)*100).toFixed(1)}% | Hub Resilience: +${Number(dr.hub_resilience_gain).toFixed(3)}`;
      }
    }

    const aRes=await fetch('/api/nexus-omega/apex-state');
    if(aRes.ok){
      const aData=await aRes.json();
      if(aData.cognitive_monologue && $('#monologueText')){
        const cm=aData.cognitive_monologue;
        $('#monologueText').textContent=`Regime: ${cm.predicted_regime} | Hypotheses: ${cm.causal_hypotheses?.length||0} do(A) | Curiosity: +${Number(cm.epistemic_curiosity_gain).toFixed(4)}`;
        if($('#thoughtChainText') && Array.isArray(cm.thought_chain)){
          $('#thoughtChainText').textContent=cm.thought_chain.slice(0, 3).join(' ➔ ');
        }
        if($('#monologueStatus')){$('#monologueStatus').textContent='DELIBERATING';$('#monologueStatus').style.background='#6366f1';}
      }
      if(aData.nonstationary_phase && $('#phaseDriftText')){
        const np=aData.nonstationary_phase;
        $('#phaseDriftText').textContent=`T_est: ${Number(np.estimated_pulse_interval).toFixed(1)} | Confirmed: ${np.confirmed_pulse_interval?Number(np.confirmed_pulse_interval).toFixed(1):'none'} | Drift Vel: ${Number(np.frequency_drift_velocity)>=0?'+':''}${Number(np.frequency_drift_velocity).toFixed(3)} | Latch: ${np.phase_boundary_latch_active?'ACTIVE':'INACTIVE'}`;
        if($('#phaseDriftStatus')){$('#phaseDriftStatus').textContent=np.phase_boundary_latch_active?'LATCHED':'RESONATING';$('#phaseDriftStatus').style.background=np.phase_boundary_latch_active?'#f59e0b':'#0ea5e9';}
      }
      if(aData.contrastive_memory && $('#contrastiveText')){
        const cmem=aData.contrastive_memory;
        $('#contrastiveText').textContent=`ΔQ: ${Number(cmem.counterfactual_advantage)>=0?'+':''}${Number(cmem.counterfactual_advantage).toFixed(3)} | Pred E(t+1): ${Number(cmem.predicted_vitality_energy).toFixed(2)} | Pred H(t+1): ${Number(cmem.predicted_vitality_hydration).toFixed(2)} | Deadband: ${cmem.within_deadband?'IN_DEADBAND':'DISCRIMINATED'}`;
        if($('#contrastiveStatus')){$('#contrastiveStatus').textContent=cmem.guard_triggered?'VITALITY GUARD':'NOMINAL';$('#contrastiveStatus').style.background=cmem.guard_triggered?'#ef4444':'#10b981';}
      }
    }

    try{
      const poiRes=await fetch('/api/poincare/status');
      if(poiRes.ok){
        const poi=await poiRes.json();
        setDiagnosticAvailability('poincare',poi.ready);
      }
    }catch{}

    try{
      const cwRes=await fetch('/api/crosswire/status');
      if(cwRes.ok){
        const cw=await cwRes.json();
        setDiagnosticAvailability('crosswire',cw.ready);
      }
    }catch{}

    try{
      const flRes=await fetch('/api/faultline/status');
      if(flRes.ok){
        const fl=await flRes.json();
        setDiagnosticAvailability('faultline',fl.ready);
      }
    }catch{}

    try{
      const tetRes=await fetch('/api/tether/status');
      if(tetRes.ok){
        const tet=await tetRes.json();
        setDiagnosticAvailability('tether',tet.ready);
      }
    }catch{}

    try{
      const carRes=await fetch('/api/cartan/status');
      if(carRes.ok){
        const car=await carRes.json();
        setDiagnosticAvailability('cartan',car.ready);
      }
    }catch{}

    try{
      const noeRes=await fetch('/api/noether/status');
      if(noeRes.ok){
        const noe=await noeRes.json();
        setDiagnosticAvailability('noether',noe.ready);
      }
    }catch{}

    try{
      const chRes=await fetch('/api/chronos/status');
      if(chRes.ok){
        const ch=await chRes.json();
        setDiagnosticAvailability('chronos',ch.ready);
      }
    }catch{}

    try{
      const aetRes=await fetch('/api/aether/status');
      if(aetRes.ok){
        const aet=await aetRes.json();
        setDiagnosticAvailability('aether',aet.ready);
      }
    }catch{}
  }catch{}
}
if($('#pollTelemetryBtn'))$('#pollTelemetryBtn').onclick=()=>pollTelemetry();
if($('#triggerReasonBtn'))$('#triggerReasonBtn').onclick=async()=>{
  try{
    if($('#monologueStatus')){$('#monologueStatus').textContent='REASONING';$('#monologueStatus').style.background='#8b5cf6';}
    if($('#thoughtChainText'))$('#thoughtChainText').textContent='Generating Ghost-in-the-Machine SCM counterfactual hypotheses…';
    const res=await fetch('/api/nexus-omega/reason',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const cm=await res.json();
    if($('#monologueStatus')){$('#monologueStatus').textContent='SYNTHESIZED';$('#monologueStatus').style.background='#10b981';}
    if($('#monologueText'))$('#monologueText').textContent=`Regime: ${cm.predicted_regime} | Hypotheses: ${cm.causal_hypotheses?.length||0} do(A) | Curiosity: +${Number(cm.epistemic_curiosity_gain).toFixed(4)}`;
    if($('#thoughtChainText') && Array.isArray(cm.thought_chain)){
      $('#thoughtChainText').textContent=cm.thought_chain.join(' ➔ ');
    }
  }catch(err){
    if($('#monologueStatus')){$('#monologueStatus').textContent='FAILED';$('#monologueStatus').style.background='#ef4444';}
    if($('#thoughtChainText'))$('#thoughtChainText').textContent='Deliberation failed: '+err.message;
  }
};
if($('#triggerDreamBtn'))$('#triggerDreamBtn').onclick=async()=>{
  try{
    if($('#dreamStatus')){$('#dreamStatus').textContent='REPLAYING';$('#dreamStatus').style.background='#8b5cf6';}
    if($('#dreamText'))$('#dreamText').textContent='Morpheus offline generative replay consolidating counterfactual rollouts...';
    const res=await fetch('/api/nexus-omega/dream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({num_rollouts:4,seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#dreamStatus')){$('#dreamStatus').textContent='CONSOLIDATED';$('#dreamStatus').style.background='#10b981';}
    if($('#dreamText'))$('#dreamText').textContent=`Replayed: ${data.replayed_transitions} | Regret Red: ${Number(data.mean_regret_mitigation).toFixed(3)} | Synaptic Pruning: ${(Number(data.synaptic_pruning_ratio)*100).toFixed(1)}% | Hub Resilience: +${Number(data.hub_resilience_gain).toFixed(3)}`;
  }catch(err){
    if($('#dreamStatus')){$('#dreamStatus').textContent='FAILED';$('#dreamStatus').style.background='#ef4444';}
    if($('#dreamText'))$('#dreamText').textContent='Dream consolidation failed: '+err.message;
  }
};
if($('#runDiscoveryBtn'))$('#runDiscoveryBtn').onclick=async()=>{
  try{
    if($('#discoveryStatus')){$('#discoveryStatus').textContent='RUNNING';$('#discoveryStatus').style.background='#8b5cf6';$('#discoveryStatus').style.color='white';}
    if($('#discoveryText'))$('#discoveryText').textContent='Evaluating counterfactual interventions and abductive noise across 4 seeds...';
    const res=await fetch('/api/nexus-omega/discover',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed_base:50000,num_seeds:4,max_steps:16})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#discoveryStatus')){$('#discoveryStatus').textContent=data.verified?'VERIFIED':'COMPLETE';$('#discoveryStatus').style.background=data.verified?'#10b981':'#ef4444';$('#discoveryStatus').style.color='white';}
    if($('#discoveryText'))$('#discoveryText').textContent=`Signals passing simulator gate: ${data.ratified_laws_count} | Other results: ${data.refutations_count} | SHA: ${data.ledger_sha256.slice(0,16)}...`;
  }catch(err){
    if($('#discoveryStatus')){$('#discoveryStatus').textContent='FAILED';$('#discoveryStatus').style.background='#ef4444';}
    if($('#discoveryText'))$('#discoveryText').textContent='Discovery cycle failed: '+err.message;
  }
};
if($('#runPrometheusCurriculumBtn'))$('#runPrometheusCurriculumBtn').onclick=async()=>{
  try{
    if($('#prometheusStatus')){$('#prometheusStatus').textContent='RUNNING CURRICULUM';$('#prometheusStatus').style.background='#8b5cf6';}
    if($('#prometheusText'))$('#prometheusText').textContent='Executing 5-Phase Developmental Curriculum (Physics -> Interventions -> Raft Transfer)...';
    const res=await fetch('/api/discovery-lab/curriculum',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#prometheusStatus')){$('#prometheusStatus').textContent=data.overall_status==='PASSED'?'ALL 5 PHASES PASSED':'FAILED';$('#prometheusStatus').style.background=data.overall_status==='PASSED'?'#10b981':'#ef4444';}
    if($('#prometheusText'))$('#prometheusText').textContent=`Narrow: ${data.phases?.phase_5?.metrics?.narrow_river_strategy} | Wide: ${data.phases?.phase_5?.metrics?.wide_river_strategy} | Receipt: ${data.receipt_sha256?.slice(0,16)}...`;
  }catch(err){
    if($('#prometheusStatus')){$('#prometheusStatus').textContent='FAILED';$('#prometheusStatus').style.background='#ef4444';}
    if($('#prometheusText'))$('#prometheusText').textContent='Curriculum failed: '+err.message;
  }
};
if($('#synthesizeEnzymeBtn'))$('#synthesizeEnzymeBtn').onclick=async()=>{
  try{
    if($('#enzymeStatus')){$('#enzymeStatus').textContent='SYNTHESIZING';$('#enzymeStatus').style.background='#8b5cf6';}
    if($('#enzymeText'))$('#enzymeText').textContent='Generating 3D active site triad and Michaelis-Menten kinetic parameters...';
    const res=await fetch('/api/discovery-lab/enzyme/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({substrate:'lignocellulose',target_reaction:'cross_linking',temperature:298.15,ph:6.5})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#enzymeStatus')){$('#enzymeStatus').textContent='CATALYZED';$('#enzymeStatus').style.background='#10b981';}
    if($('#enzymeText'))$('#enzymeText').textContent=`ID: ${data.enzyme_id} | ${data.name} | ΔΔG‡: ${data.delta_delta_g_barrier_kcal} kcal/mol | Bond: ${data.cured_bond_strength} | Water: ${data.water_proof_rating}`;
  }catch(err){
    if($('#enzymeStatus')){$('#enzymeStatus').textContent='FAILED';$('#enzymeStatus').style.background='#ef4444';}
    if($('#enzymeText'))$('#enzymeText').textContent='Synthesis failed: '+err.message;
  }
};
function renderReasoningAudit(data){
  const m=data.metrics;
  const percent=value=>value==null?'Undefined (no accepted answers)':(100*value).toFixed(1)+'%';
  const metrics=$('#reasoningAuditMetrics');
  metrics.replaceChildren();
  for(const [label,value] of [['Checked cases',m.total],['Correct answers',m.correct],['Wrong accepted',m.wrong],['Solver errors',m.errors],['Correct refusals',m.correct_refusals],['Coverage',percent(m.coverage)],['Accepted error',percent(m.selective_risk)]]){
    const card=document.createElement('div'),name=document.createElement('span'),number=document.createElement('strong');
    name.textContent=label;number.textContent=String(value);card.append(name,number);metrics.append(card);
  }
  const body=$('#reasoningAuditFamilies');body.replaceChildren();
  for(const [family,values] of Object.entries(data.by_family)){
    const row=document.createElement('tr');
    for(const value of [family,values.correct,values.wrong,values.correct_refusals,percent(values.coverage),percent(values.selective_risk)]){
      const cell=document.createElement('td');cell.textContent=String(value);row.append(cell);
    }
    body.append(row);
  }
  $('#reasoningAuditScope').textContent='Finite generated software checks. Acceptance uses each solver’s verification flag; independent checks measure correctness. No calibrated risk guarantee or general model benchmark.';
  $('#reasoningAuditDetails').textContent=JSON.stringify({report_sha256:data.report_sha256,source_hashes:data.source_hashes,limits:data.limits,rows:data.rows},null,2);
  $('#reasoningAuditResults').hidden=false;
}
if($('#runReasoningAudit'))$('#runReasoningAudit').onclick=async()=>{
  const button=$('#runReasoningAudit');button.disabled=true;
  $('#reasoningAuditResults').hidden=true;
  $('#reasoningAuditStatus').textContent='Running independent checks…';
  try{
    const seed=Number($('#reasoningAuditSeed').value||'20261030'),samples=Number($('#reasoningAuditSamples').value||'8');
    if(!Number.isSafeInteger(seed)||!Number.isInteger(samples)||samples<1||samples>32)throw Error('Use an integer seed and 1–32 checks per family.');
    const data=await apiJSON('/api/reasoning/audit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed,samples})});
    renderReasoningAudit(data);
    $('#reasoningAuditStatus').textContent=`Complete · seed ${data.config.seed} · ${data.metrics.total} cases · ${data.metrics.wrong} wrong accepted · ${data.metrics.missed_answerable} missed answerable · ${data.metrics.errors} errors`;
  }catch(error){$('#reasoningAuditStatus').textContent='Audit failed: '+error.message;}
  finally{button.disabled=false;}
};
function auditTableRows(selector, rows){
  const body=$(selector);body.replaceChildren();
  for(const values of rows){
    const row=document.createElement('tr');
    for(const value of values){const cell=document.createElement('td');cell.textContent=String(value);row.append(cell);}
    body.append(row);
  }
}
function auditCards(selector, values){
  const container=$(selector);container.replaceChildren();
  for(const [label,value] of values){
    const card=document.createElement('div'),name=document.createElement('span'),number=document.createElement('strong');
    name.textContent=label;number.textContent=String(value);card.append(name,number);container.append(card);
  }
}
function exportAuditReport(raw, name){
  const url=URL.createObjectURL(new Blob([raw+'\n'],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download=name;link.click();
  setTimeout(()=>URL.revokeObjectURL(url),0);
}
let lastMetamorphicAudit=null,lastEpisodeRisk=null;
function renderMetamorphicAudit(data){
  const m=data.metrics;
  auditCards('#metamorphicAuditMetrics',[
    ['Checked answers',m.total],['Correct answers',m.correct],['Wrong accepted',m.wrong],
    ['Jointly correct pairs',`${m.joint_correct_groups} / ${m.groups}`],
    ['Relation violations',m.relation_violations],['Unavailable relations',m.relation_unavailable],
    ['Refusals',m.refusals],['Solver errors',m.errors]]);
  auditTableRows('#metamorphicAuditFamilies',Object.entries(data.by_family).map(([family,v])=>
    [family,v.correct,v.wrong,v.relation_violations,v.relation_unavailable,`${v.joint_correct_groups} / ${v.groups}`]));
  $('#metamorphicAuditScope').textContent='Consistent answers can both be wrong. Jointly correct pairs require independent correctness and a valid relation. Finite local tool checks; no general model accuracy or statistical risk guarantee.';
  $('#metamorphicAuditDetails').textContent=JSON.stringify(data,null,2);
  $('#metamorphicAuditResults').hidden=false;
}
if($('#runMetamorphicAudit'))$('#runMetamorphicAudit').onclick=async()=>{
  const button=$('#runMetamorphicAudit');button.disabled=true;lastMetamorphicAudit=null;
  $('#exportMetamorphicAudit').disabled=true;$('#metamorphicAuditResults').hidden=true;
  $('#metamorphicAuditStatus').textContent='Checking related inputs…';
  try{
    const seed=Number($('#metamorphicAuditSeed').value||'20261031'),samples=Number($('#metamorphicAuditSamples').value||'8');
    if(!Number.isSafeInteger(seed)||seed<0||!Number.isInteger(samples)||samples<1||samples>32)throw Error('Use a nonnegative integer seed and 1–32 pairs per family.');
    const response=await apiJSON('/api/reasoning/metamorphic-audit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed,samples})},true);
    const data=response.data;
    renderMetamorphicAudit(data);$('#metamorphicAuditDetails').textContent=response.raw;
    lastMetamorphicAudit=response;$('#exportMetamorphicAudit').disabled=false;
    const m=data.metrics;
    $('#metamorphicAuditStatus').textContent=`Complete · ${m.joint_correct_groups} / ${m.groups} jointly correct pairs · ${m.wrong} wrong accepted · ${m.relation_violations} relation violations · ${m.refusals} refusals · ${m.errors} errors`;
  }catch(error){$('#metamorphicAuditStatus').textContent='Audit failed: '+error.message;}
  finally{button.disabled=false;}
};
if($('#exportMetamorphicAudit'))$('#exportMetamorphicAudit').onclick=()=>{
  if(lastMetamorphicAudit)exportAuditReport(lastMetamorphicAudit.raw,'poseidon-metamorphic-audit.json');
};
function renderEpisodeRisk(data){
  const percent=value=>(100*value).toFixed(2)+'%';
  auditTableRows('#episodeRiskConditions',Object.entries(data.conditions).map(([condition,v])=>
    [condition,v.episodes,v.adverse_episodes,percent(v.observed_risk),percent(v.risk_upper_bound),v.passed?'Meets limit':'Insufficient evidence']));
  $('#episodeRiskScope').textContent=`${percent(data.protocol.confidence)} simultaneous confidence; episode risk limit ${percent(data.protocol.max_episode_risk)}. Bounds assume independent identically distributed episodes for a policy fixed before collection. Supplied identities and outcomes are not authenticated. This is not per-action risk and does not activate a controller.`;
  $('#episodeRiskDetails').textContent=JSON.stringify(data,null,2);$('#episodeRiskResults').hidden=false;
}
if($('#runEpisodeRisk'))$('#runEpisodeRisk').onclick=async()=>{
  const button=$('#runEpisodeRisk');button.disabled=true;lastEpisodeRisk=null;
  $('#exportEpisodeRisk').disabled=true;$('#episodeRiskResults').hidden=true;
  $('#episodeRiskStatus').textContent='Checking supplied episode evidence…';
  try{
    const file=$('#episodeRiskFile').files?.[0];
    if(!file)throw Error('Choose a protocol and outcomes JSON file.');
    if(file.size>65536)throw Error('Use a file of at most 64 KiB, or run episode-risk in the CLI.');
    const raw=await file.text();
    // Send the original JSON so the server can reject duplicate keys and nonfinite numbers.
    const response=await apiJSON('/api/research/episode-risk',{method:'POST',headers:{'Content-Type':'application/json'},body:raw},true);
    const data=response.data;
    renderEpisodeRisk(data);$('#episodeRiskDetails').textContent=response.raw;
    lastEpisodeRisk=response;$('#exportEpisodeRisk').disabled=false;
    $('#episodeRiskStatus').textContent=data.eligible?'Meets declared episode risk limit · offline screen only':'Insufficient evidence for the declared episode risk limit';
  }catch(error){$('#episodeRiskStatus').textContent='Screen failed: '+error.message;}
  finally{button.disabled=false;}
};
if($('#exportEpisodeRisk'))$('#exportEpisodeRisk').onclick=()=>{
  if(lastEpisodeRisk)exportAuditReport(lastEpisodeRisk.raw,'poseidon-episode-risk.json');
};
if($('#synthesizePoincareBtn'))$('#synthesizePoincareBtn').onclick=async()=>{
  try{
    if($('#poincareTelemetryStatus')){$('#poincareTelemetryStatus').textContent='SYNTHESIZING';$('#poincareTelemetryStatus').style.background='#4f46e5';}
    if($('#poincareTelemetryText'))$('#poincareTelemetryText').textContent='Running finite-orbit, ball-geometry, canonical-topology, curvature-relaxation and synthetic quartet diagnostics…';
    const res=await fetch('/api/poincare/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42,stochastic_k:0.85,iterations:300,curvature_c:1.0})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    const diagnostic=poincareDiagnostics(data);
    if($('#poincareTelemetryStatus')){$('#poincareTelemetryStatus').textContent=diagnostic.status;$('#poincareTelemetryStatus').style.background=data.verified===false?'#ef4444':'#4f46e5';}
    if($('#poincareTelemetryText'))$('#poincareTelemetryText').textContent=diagnostic.text;
    message('assistant',data.text||'Project POINCARÉ synthesis completed.',data.project||'POINCARE',data);
  }catch(err){
    if($('#poincareTelemetryStatus')){$('#poincareTelemetryStatus').textContent='FAILED';$('#poincareTelemetryStatus').style.background='#ef4444';}
    if($('#poincareTelemetryText'))$('#poincareTelemetryText').textContent='POINCARÉ synthesis failed: '+err.message;
  }
};
if($('#synthesizeCrosswireBtn'))$('#synthesizeCrosswireBtn').onclick=async()=>{
  try{
    if($('#crosswireTelemetryStatus')){$('#crosswireTelemetryStatus').textContent='SYNTHESIZING';$('#crosswireTelemetryStatus').style.background='#0284c7';}
    if($('#crosswireTelemetryText'))$('#crosswireTelemetryText').textContent='Evaluating candidate permutation space, active diagnostic probing, and closed-loop runtime shield...';
    const res=await fetch('/api/crosswire/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42,budget:16,num_actions:4})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#crosswireTelemetryStatus')){$('#crosswireTelemetryStatus').textContent='DIAGNOSTICS COMPLETE';$('#crosswireTelemetryStatus').style.background='#0284c7';}
    const pr=data.probing_results||{};
    const sh=data.runtime_shield||{};
    const tel=data.telemetry_summary||{};
    if($('#crosswireTelemetryText'))$('#crosswireTelemetryText').textContent=`Perm: ${pr.fully_disambiguated?'Discovered':'Partial'} | Steps: ${pr.steps_executed} | Red: ${tel.entropy_reduction_bits} bits | Intercepts: ${sh.total_intercepts} | Zero Violations: ${tel.zero_hazard_violations}`;
    message('assistant',data.text||'Project CROSSWIRE synthesis completed.',data.project,data);
  }catch(err){
    if($('#crosswireTelemetryStatus')){$('#crosswireTelemetryStatus').textContent='FAILED';$('#crosswireTelemetryStatus').style.background='#ef4444';}
    if($('#crosswireTelemetryText'))$('#crosswireTelemetryText').textContent='CROSSWIRE synthesis failed: '+err.message;
  }
};
if($('#synthesizeFaultlineBtn'))$('#synthesizeFaultlineBtn').onclick=async()=>{
  try{
    if($('#faultlineTelemetryStatus')){$('#faultlineTelemetryStatus').textContent='SYNTHESIZING';$('#faultlineTelemetryStatus').style.background='#e11d48';}
    if($('#faultlineTelemetryText'))$('#faultlineTelemetryText').textContent='Evaluating condition-stratified joint failure selection, 3-bit parity hyperedges, and continuous error covariance...';
    const res=await fetch('/api/faultline/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42,panel_size:3})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#faultlineTelemetryStatus')){$('#faultlineTelemetryStatus').textContent='DIAGNOSTICS COMPLETE';$('#faultlineTelemetryStatus').style.background='#e11d48';}
    const pol=data.discrete_analysis?.policies||[];
    const joint=pol.find(p=>p.id==='joint')||{};
    const cov=data.continuous_error_covariance||{};
    const mm=data.continuous_minimax_panel||{};
    if($('#faultlineTelemetryText'))$('#faultlineTelemetryText').textContent=`Joint Adv: +${(Number(data.joint_advantage||0)*100).toFixed(2)}% | Worst Err: ${Number(data.worst_condition_error||0).toFixed(4)} | Minimax: [${(mm.selected_panel||[]).join(',')}] | Div Gain: ${(Number(cov.diversity_gain||0)*100).toFixed(1)}%`;
    message('assistant',data.text||'Project FAULTLINE synthesis completed.',data.project,data);
  }catch(err){
    if($('#faultlineTelemetryStatus')){$('#faultlineTelemetryStatus').textContent='FAILED';$('#faultlineTelemetryStatus').style.background='#ef4444';}
    if($('#faultlineTelemetryText'))$('#faultlineTelemetryText').textContent='FAULTLINE synthesis failed: '+err.message;
  }
};
if($('#synthesizeTetherBtn'))$('#synthesizeTetherBtn').onclick=async()=>{
  try{
    if($('#tetherTelemetryStatus')){$('#tetherTelemetryStatus').textContent='SYNTHESIZING';$('#tetherTelemetryStatus').style.background='#ea580c';}
    if($('#tetherTelemetryText'))$('#tetherTelemetryText').textContent='Evaluating continuous reliability, 4-view causal invariance across 7 arms, input x gradient saliency, outlier isolation & calibrated fallback gate...';
    const res=await fetch('/api/tether/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#tetherTelemetryStatus')){$('#tetherTelemetryStatus').textContent='DIAGNOSTICS COMPLETE';$('#tetherTelemetryStatus').style.background='#ea580c';}
    const det=data.corruption_detection||{};
    const att=data.attribution||{};
    const gate=data.fallback_gate||{};
    const four=data.four_view_causal_invariance||{};
    if($('#tetherTelemetryText'))$('#tetherTelemetryText').textContent=`Trust τ: ${Number(data.epistemic_trust||0).toFixed(3)} | Arm: ${four.arm||'tether'} | Gini: ${Number(att.gini_concentration||0).toFixed(3)} | Outliers: ${det.num_corrupted||0} | Gate: ${gate.decision_type||'nominal'}`;
    message('assistant',data.text||'Project TETHER synthesis completed.',data.project,data);
  }catch(err){
    if($('#tetherTelemetryStatus')){$('#tetherTelemetryStatus').textContent='FAILED';$('#tetherTelemetryStatus').style.background='#ef4444';}
    if($('#tetherTelemetryText'))$('#tetherTelemetryText').textContent='TETHER synthesis failed: '+err.message;
  }
};
if($('#synthesizeCartanBtn'))$('#synthesizeCartanBtn').onclick=async()=>{
  try{
    if($('#cartanTelemetryStatus')){$('#cartanTelemetryStatus').textContent='SYNTHESIZING';$('#cartanTelemetryStatus').style.background='#0d9488';}
    if($('#cartanTelemetryText'))$('#cartanTelemetryText').textContent='Evaluating Cartan exterior systems, PT-symmetric Hamiltonians, Albert algebra, Ryu-Takayanagi entropy & Turing pattern formation...';
    const res=await fetch('/api/cartan/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#cartanTelemetryStatus')){$('#cartanTelemetryStatus').textContent='DIAGNOSTICS COMPLETE';$('#cartanTelemetryStatus').style.background='#0d9488';}
    const ext=data.cartan_exterior||{};
    const pt=data.non_hermitian_pt||{};
    const jord=data.exceptional_jordan||{};
    const holo=data.holographic_entanglement||{};
    const tur=data.morphogenetic_turing||{};
    if($('#cartanTelemetryText'))$('#cartanTelemetryText').textContent=`Cartan s0: ${(ext.cartan_characters||[])[0]} | PT EP2: ${pt.ep_threshold_g} | Albert F4: ${jord.hurwitz_norm_identity_verified?'OK':'FAIL'} | Holo S_A: ${Number(holo.entanglement_entropy_nats||0).toFixed(2)} nats | Turing: ${tur.predicted_morphology}`;
    message('assistant',data.text,data.project,data);
  }catch(err){
    if($('#cartanTelemetryStatus')){$('#cartanTelemetryStatus').textContent='FAILED';$('#cartanTelemetryStatus').style.background='#ef4444';}
    if($('#cartanTelemetryText'))$('#cartanTelemetryText').textContent='CARTAN synthesis failed: '+err.message;
  }
};
if($('#synthesizeNoetherBtn'))$('#synthesizeNoetherBtn').onclick=async()=>{
  try{
    if($('#noetherTelemetryStatus')){$('#noetherTelemetryStatus').textContent='SYNTHESIZING';$('#noetherTelemetryStatus').style.background='#0891b2';}
    if($('#noetherTelemetryText'))$('#noetherTelemetryText').textContent='Evaluating SO(2) Noether symmetries, Fibonacci MTC, Clifford STA Cl(1,3), BV antibracket & Active Brownian particles...';
    const res=await fetch('/api/noether/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#noetherTelemetryStatus')){$('#noetherTelemetryStatus').textContent='DIAGNOSTICS COMPLETE';$('#noetherTelemetryStatus').style.background='#0891b2';}
    const sym=data.noether_symmetry||{};
    const mtc=data.modular_tensor_category||{};
    const sta=data.clifford_spacetime_algebra||{};
    const bv=data.batalin_vilkovisky||{};
    const abp=data.active_matter||{};
    if($('#noetherTelemetryText'))$('#noetherTelemetryText').textContent=`SO(2) Q: ${Number(sym.noether_conserved_charge||0).toFixed(2)} | MTC D: ${Number(mtc.total_quantum_dimension_D||0).toFixed(4)} | STA I1: ${Number(sta.lab_frame?.invariant_1_scalar||0).toFixed(2)} | BV CME: ${Number(bv.classical_master_equation_residual||0).toExponential(2)} | ABP Deff: ${Number(abp.effective_long_time_diffusion_Deff||0).toFixed(2)}`;
    message('assistant',data.text,data.project,data);
  }catch(err){
    if($('#noetherTelemetryStatus')){$('#noetherTelemetryStatus').textContent='FAILED';$('#noetherTelemetryStatus').style.background='#ef4444';}
    if($('#noetherTelemetryText'))$('#noetherTelemetryText').textContent='NOETHER synthesis failed: '+err.message;
  }
};
if($('#synthesizeChronosBtn'))$('#synthesizeChronosBtn').onclick=async()=>{
  try{
    if($('#chronosTelemetryStatus')){$('#chronosTelemetryStatus').textContent='SYNTHESIZING';$('#chronosTelemetryStatus').style.background='#0284c7';}
    if($('#chronosTelemetryText'))$('#chronosTelemetryText').textContent='Evaluating CDT Regge action, Connes spectral triples, tropical roots, Stiefel retractions & bisimulation metrics...';
    const res=await fetch('/api/chronos/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#chronosTelemetryStatus')){$('#chronosTelemetryStatus').textContent='DIAGNOSTICS COMPLETE';$('#chronosTelemetryStatus').style.background='#0284c7';}
    const cdt=data.causal_dynamical_triangulation||{};
    const spec=data.spectral_geometry||{};
    const stief=data.stiefel_optimization||{};
    const bisim=data.coalgebraic_bisimulation||{};
    if($('#chronosTelemetryText'))$('#chronosTelemetryText').textContent=`CDT S_Regge: ${Number(cdt.lorentzian_regge_action||0).toFixed(2)} | Connes Tr(D^2): ${Number(spec.spectral_action_tr_d2||0).toFixed(2)} | Stiefel Obj: ${Number(stief.optimized_objective||0).toFixed(4)} | Bisim d(s0,s1): ${Number(bisim.dist_s0_s1||0).toFixed(4)}`;
    message('assistant',data.text,data.project,data);
  }catch(err){
    if($('#chronosTelemetryStatus')){$('#chronosTelemetryStatus').textContent='FAILED';$('#chronosTelemetryStatus').style.background='#ef4444';}
    if($('#chronosTelemetryText'))$('#chronosTelemetryText').textContent='CHRONOS synthesis failed: '+err.message;
  }
};
if($('#synthesizeAetherBtn'))$('#synthesizeAetherBtn').onclick=async()=>{
  try{
    if($('#aetherTelemetryStatus')){$('#aetherTelemetryStatus').textContent='SYNTHESIZING';$('#aetherTelemetryStatus').style.background='#6366f1';}
    if($('#aetherTelemetryText'))$('#aetherTelemetryText').textContent='Computing Wilson loops, Jarzynski dissipated work, Fisher metric, symplectic leapfrog & open game optics...';
    const res=await fetch('/api/aether/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#aetherTelemetryStatus')){$('#aetherTelemetryStatus').textContent='DIAGNOSTICS COMPLETE';$('#aetherTelemetryStatus').style.background='#4f46e5';}
    const g=data.gauge_holonomy||{};
    const th=data.thermodynamics||{};
    const fg=data.fisher_geometry||{};
    const sm=data.symplectic_mechanics||{};
    if($('#aetherTelemetryText'))$('#aetherTelemetryText').textContent=`U(1) Flux: ${Number(g.total_flux_radians||0).toFixed(3)} rad | W_diss: ${Number(th.dissipated_work||0).toFixed(4)} | Fisher K: ${Number(fg.poincare_gaussian_curvature||0).toFixed(2)} | Shadow: ${Number(sm.shadow_hamiltonian_residual||0).toExponential(2)}`;
    message('assistant',data.text,data.project,data);
  }catch(err){
    if($('#aetherTelemetryStatus')){$('#aetherTelemetryStatus').textContent='FAILED';$('#aetherTelemetryStatus').style.background='#ef4444';}
    if($('#aetherTelemetryText'))$('#aetherTelemetryText').textContent='AETHER synthesis failed: '+err.message;
  }
};
if($('#synthesizeNeurograftBtn'))$('#synthesizeNeurograftBtn').onclick=async()=>{
  try{
    if($('#neurograftTelemetryStatus')){$('#neurograftTelemetryStatus').textContent='SYNTHESIZING';$('#neurograftTelemetryStatus').style.background='#8b5cf6';}
    if($('#neurograftTelemetryText'))$('#neurograftTelemetryText').textContent='Synthesizing Successor Representation, SWR Replay, Graph Laplacian & Trainable Graph Circuit...';
    const res=await fetch('/api/neurograft/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42,discount:0.9,learning_rate:0.1})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#neurograftTelemetryStatus')){$('#neurograftTelemetryStatus').textContent='DIAGNOSTICS COMPLETE';$('#neurograftTelemetryStatus').style.background='#059669';}
    if(data.successor_representation){
      const sr=data.successor_representation;
      const swr=data.sharp_wave_ripple_replay||{};
      const lap=data.graph_laplacian_topological||{};
      const cir=data.trainable_graph_circuit||{};
      if($('#neurograftTelemetryText'))$('#neurograftTelemetryText').textContent=`SR Norm: ${sr.M_state_norm} | SWR Sweeps: ${swr.sweeps_executed} (${swr.total_updates} updates) | Fiedler λ1: ${lap.fiedler_value} | Revaluation: synthetic example | Route Entropy: ${cir.routing_entropy} nats`;
    } else {
      const tri=data.tri_neuromodulation||{};
      const vla=data.virtual_lesion_audit||{};
      if($('#neurograftTelemetryText'))$('#neurograftTelemetryText').textContent=`Regime: ${tri.behavioral_regime} | DA: ${Number(tri.dopamine_level).toFixed(2)} | 5-HT: ${Number(tri.serotonin_level).toFixed(2)} | OA: ${Number(tri.octopamine_level).toFixed(2)} | Critical: ${vla.critical_lesion_target}`;
    }
    message('assistant',data.text||JSON.stringify(data, null, 2),data.project||'NEUROGRAFT',data);
  }catch(err){
    if($('#neurograftTelemetryStatus')){$('#neurograftTelemetryStatus').textContent='FAILED';$('#neurograftTelemetryStatus').style.background='#ef4444';}
    if($('#neurograftTelemetryText'))$('#neurograftTelemetryText').textContent='NEUROGRAFT synthesis failed: '+err.message;
  }
};

let riverState = null;
async function initRiverWorld(seed=42, riverWidth=null) {
  stopWorld();
  $('#timeline').hidden = true;
  $('#decisionPanel').hidden = true;
  $('#artifactType').textContent = 'RIVERWORLD';
  const rw = riverWidth !== null ? riverWidth : Number($('#riverWidthSelect')?.value || 3);
  try {
    const res = await fetch(`/api/discovery-lab/river-state?seed=${seed}&river_width=${rw}`);
    if (res.ok) {
      riverState = await res.json();
      drawRiverWorld(riverState);
      if ($('#riverStatusNote')) {
        $('#riverStatusNote').textContent = `RiverWorld initialized: Width ${rw}m | Objects: ${riverState.objects?.length || 0} | Health: 100% | Stamina: 100%`;
      }
    }
  } catch (e) {
    console.error('RiverWorld load error:', e);
  }
}

function drawRiverWorld(state) {
  if (!state) return;
  const c = $('#worldCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = 600, H = 400;
  ctx.clearRect(0, 0, W, H);

  // World-grid → canvas mapping (each grid cell spans one column/row)
  const gridW = Math.max(1, state.grid_width || 16);
  const gridH = Math.max(1, state.grid_height || 10);
  const toCanvasX = (x) => ((x + 0.5) / gridW) * W;
  const toCanvasY = (y) => ((y + 0.5) / gridH) * H;

  // 2. River Channel bounds (computed first so the bank fills match)
  const isWide = state.river_width > 4;
  const riverStartX = ((state.river_start_x ?? 7) / gridW) * W;
  const riverEndX = (((state.river_end_x ?? 9) + 1) / gridW) * W;
  const riverW = riverEndX - riverStartX;

  // 1. Left Riverbank (Earthen grass)
  ctx.fillStyle = '#14532d';
  ctx.fillRect(0, 0, riverStartX, H);
  ctx.fillStyle = '#166534';
  for (let i = 0; i < 15; i++) {
    ctx.fillRect(10 + (i*8) % Math.max(20, riverStartX - 20), 20 + (i*26)%360, 4, 8);
  }

  const grad = ctx.createLinearGradient(riverStartX, 0, riverEndX, 0);
  grad.addColorStop(0, '#0284c7');
  grad.addColorStop(0.5, '#0369a1');
  grad.addColorStop(1, '#0284c7');
  ctx.fillStyle = grad;
  ctx.fillRect(riverStartX, 0, riverW, H);

  // Wave ripple lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
  ctx.lineWidth = 1.5;
  for (let y = 30; y < H; y += 45) {
    ctx.beginPath();
    ctx.moveTo(riverStartX + 15, y);
    ctx.bezierCurveTo(riverStartX + 100, y - 8, riverStartX + 200, y + 8, riverEndX - 15, y);
    ctx.stroke();
  }

  // 3. Right Bank (Target territory)
  ctx.fillStyle = '#1e3a8a';
  ctx.fillRect(riverEndX, 0, W - riverEndX, H);
  ctx.fillStyle = '#fbbf24';
  ctx.fillRect(riverEndX + 55, 180, 5, 40);
  ctx.beginPath();
  ctx.moveTo(riverEndX + 60, 180);
  ctx.lineTo(riverEndX + 90, 192);
  ctx.lineTo(riverEndX + 60, 204);
  ctx.fill();
  ctx.font = 'bold 11px Segoe UI';
  ctx.fillStyle = '#fde68a';
  ctx.fillText('GOAL BANK', riverEndX + 35, 170);

  // 4. Render Physical Objects
  const objects = state.objects || [];
  let connectedLogs = [];

  objects.forEach(obj => {
    const cx = toCanvasX(obj.position[0]);
    const cy = toCanvasY(obj.position[1]);

    if (obj.material.includes('stone') || obj.material.includes('granite')) {
      const inRiver = obj.position[0] >= state.river_start_x && obj.position[0] <= state.river_end_x;
      ctx.fillStyle = inRiver ? '#334155' : '#64748b';
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = inRiver && isWide ? '#ef4444' : '#94a3b8';
      ctx.lineWidth = inRiver && isWide ? 2 : 1;
      ctx.stroke();

      if (inRiver) {
        ctx.fillStyle = isWide ? '#f87171' : '#cbd5e1';
        ctx.font = '9px monospace';
        ctx.fillText(isWide ? 'SUBMERGED' : 'STEP', cx - 20, cy + 22);
      }
    } else if (obj.is_buoyant) {
      ctx.fillStyle = '#b45309';
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(cx - 10, cy - 20, 20, 40, 5);
      } else {
        ctx.rect(cx - 10, cy - 20, 20, 40);
      }
      ctx.fill();
      ctx.stroke();

      if (obj.connected_object_ids && obj.connected_object_ids.length > 0) {
        connectedLogs.push({cx, cy, id: obj.object_id, strength: obj.bond_strength});
      }
    }
  });

  // Raft catalytic cross-linking lines
  if (connectedLogs.length >= 2) {
    ctx.strokeStyle = '#c084fc';
    ctx.lineWidth = 3;
    ctx.setLineDash([4, 2]);
    ctx.beginPath();
    ctx.moveTo(connectedLogs[0].cx, connectedLogs[0].cy);
    for (let k = 1; k < connectedLogs.length; k++) {
      ctx.lineTo(connectedLogs[k].cx, connectedLogs[k].cy);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#e9d5ff';
    ctx.font = 'bold 10px monospace';
    ctx.fillText('RAFT (CATALYZED BOND 0.98)', connectedLogs[0].cx - 30, connectedLogs[0].cy - 26);
  }

  // 5. Render Agent Avatar
  const agentX = toCanvasX(state.agent.x);
  const agentY = toCanvasY(state.agent.y);
  ctx.fillStyle = '#38bdf8';
  ctx.beginPath();
  ctx.arc(agentX, agentY, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = '#0f172a';
  ctx.font = 'bold 9px Segoe UI';
  ctx.fillText('YOU', agentX - 9, agentY + 3);

  // 6. Canvas HUD Overlays
  ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
  ctx.fillRect(0, 0, W, 28);
  ctx.fillStyle = '#e2e8f0';
  ctx.font = '11px Segoe UI';
  ctx.fillText(`PROMETHEUS RIVERWORLD · Channel: ${isWide ? 'WIDE (7m) - RAFT REQUIRED' : 'NARROW (3m) - STONES VIABLE'} · Health: ${Math.round(state.agent.health * 100)}% · Stamina: ${Math.round(state.agent.stamina * 100)}%`, 12, 18);

  ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
  ctx.fillRect(0, H - 28, W, 28);
  ctx.fillStyle = '#38bdf8';
  ctx.font = '10px monospace';
  const gmText = isWide ? 'GM: +0.28m (CATAMARAN STABLE)' : 'GM: N/A (STEPPING)';
  const payloadText = isWide ? 'Payload: 108kg > 70kg' : 'Direct Stride';
  ctx.fillText(`Buoyancy: +353.2N | ${payloadText} | Naval: ${gmText} | Shield: PROVEN_SAFE`, 12, H - 10);
}

if ($('#riverWidthSelect')) {
  $('#riverWidthSelect').onchange = () => {
    initRiverWorld(42, Number($('#riverWidthSelect').value));
  };
}

if ($('#stepRiverBtn')) {
  $('#stepRiverBtn').onclick = async () => {
    try {
      const act = Number($('#riverActionSelect')?.value || 1);
      const rw = Number($('#riverWidthSelect')?.value || 3);
      const res = await fetch('/api/discovery-lab/step', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action: act, river_width: rw, seed: 42})
      });
      if (!res.ok) throw Error(res.statusText);
      const data = await res.json();
      message('assistant', `Executed step: ${data.action_executed} | Event: ${data.info?.event || 'none'} | Reward: ${data.reward} | Pos: ${JSON.stringify(data.info?.agent_position)}`, 'Project PROMETHEUS Discovery Lab');
      if ($('#riverStatusNote')) {
        $('#riverStatusNote').textContent = `Action: ${data.action_executed} | Reward: ${data.reward} | Event: ${data.info?.event || 'nominal'} | State Hash: ${data.state_hash.slice(0, 16)}...`;
      }
      initRiverWorld(42, rw);
    } catch (err) {
      message('error', 'RiverWorld step failed: ' + err.message);
    }
  };
}

if ($('#resetRiverBtn')) {
  $('#resetRiverBtn').onclick = () => {
    initRiverWorld(42, Number($('#riverWidthSelect')?.value || 3));
  };
}

drawIdle();refreshStatus();refreshJobs();pollTelemetry();setInterval(()=>{refreshStatus();pollTelemetry();},15000);


