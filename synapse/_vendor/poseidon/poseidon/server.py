"""Loopback-only workbench with bounded requests and explicit artifact paths."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import argparse
import json
import math
import mimetypes
import threading
from .runtime import Poseidon
from .provenance import StaleSourceError, assert_source_current, source_status

WEB = Path(__file__).parent/"web"

def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON numbers must be finite.")
    return number


def make_handler(runtime, port):
    busy = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, payload):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/jobs" or path.startswith("/api/jobs/"):
                try:
                    if path == "/api/jobs":
                        result = {"jobs": runtime.jobs().list()}
                    elif path.count("/") == 3:
                        result = runtime.jobs().get(path.rsplit("/", 1)[1])
                    else:
                        raise KeyError(path)
                    self.send_json(200, result)
                except KeyError:
                    self.send_json(404, {"error": "Unknown experiment job."})
                except StaleSourceError as error:
                    self.send_json(409, {"error": str(error)} | source_status())
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                return
            if path == "/api/status":
                try:
                    self.send_json(200, runtime.status())
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error), "type": type(error).__name__})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/ecosystem/telemetry":
                try:
                    eco = runtime.ecosystem()
                    eco.spawn_default_colony(n_grazers=2, n_predators=1, n_symbionts=1)
                    _, _, _, telemetry = eco.step({"grazer_0": 4, "predator_0": 4, "symbiont_0": 0})
                    self.send_json(200, telemetry)
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/memory/topology":
                try:
                    from .memory_observatory import MemoryCarrierObservatory
                    obs = MemoryCarrierObservatory()
                    sample = [[1.0, 0.8, 0.8, 0.9], [1.0, 0.7, 0.75, 0.85], [0.95, 0.85, 0.8, 0.8]]
                    audit = obs.analyze_trajectory_topology(sample)
                    frame = obs.synthesize_telemetry_frame(audit)
                    self.send_json(200, frame)
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/autopoiesis/affect":
                try:
                    from .autopoiesis import AutopoieticEngine
                    engine = AutopoieticEngine()
                    audit = engine.update([1.0, 0.85, 0.85, 0.9])
                    self.send_json(200, {
                        "affect": audit.affect.to_dict(),
                        "negentropy_flux": audit.negentropy_flux,
                        "curiosity_drive": audit.curiosity_drive,
                        "hypotheses": audit.hypotheses_generated,
                    })
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/hypergraph/theorems":
                try:
                    from .hypergraph_discovery import InductiveHypergraphEngine
                    engine = InductiveHypergraphEngine()
                    engine.ingest_trajectory([{"action": 0, "stamina": 0.5, "next_stamina": 0.55}])
                    kg = engine.export_jsonld_knowledge_graph()
                    self.send_json(200, kg)
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/adapter/status":
                try:
                    self.send_json(200, runtime.external_adapter_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/telemetry/memory-influence":
                try:
                    bridge = runtime.external_adapter_bridge()
                    self.send_json(200, bridge.latest_telemetry.to_dict())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/nexus-omega/status":
                try:
                    self.send_json(200, runtime.nexus_omega_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/nexus-omega/factored-state":
                try:
                    self.send_json(200, runtime.nexus_omega_factored_state())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/nexus-omega/hierarchical-state":
                try:
                    self.send_json(200, runtime.nexus_omega_hierarchical_state())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/nexus-omega/ultima-state":
                try:
                    self.send_json(200, runtime.nexus_omega_ultima_state())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/nexus-omega/apex-state":
                try:
                    self.send_json(200, runtime.nexus_omega_apex_state())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/discovery-lab/status":
                try:
                    self.send_json(200, runtime.discovery_lab_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/discovery-lab/river-state":
                try:
                    from urllib.parse import parse_qs
                    params = parse_qs(urlsplit(self.path).query)
                    seed = int(params.get("seed", [42])[0])
                    rw = int(params.get("river_width", [3])[0])
                    self.send_json(200, runtime.discovery_lab_river_state(seed=seed, river_width=rw))
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/reasoning-status":
                try:
                    self.send_json(200, runtime.omni_reasoner_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/problem/capabilities":
                try:
                    self.send_json(200, {"ready": True, "capabilities": runtime.problem_capabilities()})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/pantheon/status":
                try:
                    self.send_json(200, runtime.pantheon_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/creative/status":
                try:
                    self.send_json(200, runtime.creative_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/hypothesis/status":
                try:
                    self.send_json(200, runtime.hypothesis_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/proof/status":
                try:
                    self.send_json(200, runtime.proof_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/creativity/status":
                try:
                    self.send_json(200, runtime.creativity_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/creativity/matrices":
                try:
                    self.send_json(200, runtime.creativity_matrices())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/creativity/principles":
                try:
                    self.send_json(200, runtime.creativity_principles())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/athenaeum/status":
                try:
                    self.send_json(200, runtime.athenaeum_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/hephaestus/status":
                try:
                    self.send_json(200, runtime.hephaestus_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/hyperborea/status":
                try:
                    self.send_json(200, runtime.hyperborea_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/neurograft/status":
                try:
                    self.send_json(200, runtime.neurograft_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/poincare/status":
                try:
                    self.send_json(200, runtime.poincare_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/aether/status":
                try:
                    self.send_json(200, runtime.aether_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/chronos/status":
                try:
                    self.send_json(200, runtime.chronos_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/noether/status":
                try:
                    self.send_json(200, runtime.noether_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/cartan/status":
                try:
                    self.send_json(200, runtime.cartan_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/crosswire/status":
                try:
                    self.send_json(200, runtime.crosswire_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/faultline/status":
                try:
                    self.send_json(200, runtime.faultline_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/tether/status":
                try:
                    self.send_json(200, runtime.tether_status())
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path in ("/", "/index.html", "/style.css", "/app.js"):
                file = WEB/("index.html" if path in ("/", "/index.html") else path[1:])
            elif path.startswith("/artifacts/"):
                output_root = (runtime.root/"outputs").resolve()
                file = (output_root/path[len("/artifacts/"):]).resolve()
                if not file.is_relative_to(output_root) or file.suffix.lower() not in (".png", ".gif", ".mp4", ".obj", ".mtl", ".gltf", ".bin", ".json") or file.name == "memory.json":
                    self.send_json(403, {"error": "Artifact path is not public."}); return
            else:
                self.send_json(404, {"error": "Not found"}); return
            if not file.is_file():
                self.send_json(404, {"error": "Not found"}); return
            content = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
            if self.headers.get("Origin") not in (None, *allowed):
                self.send_json(403, {"error": "Cross-origin requests are disabled."}); return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_json(415, {"error": "Use application/json."}); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request body must be 1–65536 bytes.")
                payload = json.loads(self.rfile.read(length), object_pairs_hook=_json_object,
                                     parse_float=_json_float, parse_constant=_json_float)
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
            except (ValueError, UnicodeDecodeError) as error:
                self.send_json(400, {"error": str(error)}); return
            path = unquote(urlsplit(self.path).path)
            if path in ("/api/reasoning/audit", "/api/reasoning/metamorphic-audit"):
                try:
                    if set(payload) - {"seed", "samples"}:
                        raise ValueError("Unknown reasoning audit setting.")
                    assert_source_current()
                    metamorphic = path.endswith("metamorphic-audit")
                    run = runtime.metamorphic_audit if metamorphic else runtime.reasoning_audit
                    self.send_json(200, run(
                        seed=payload.get("seed", 20261031 if metamorphic else 20261030),
                        samples=payload.get("samples", 8)))
                except StaleSourceError as error:
                    self.send_json(409, {"error": str(error)} | source_status())
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                return
            if path == "/api/research/episode-risk":
                try:
                    if set(payload) != {"protocol", "outcomes_by_condition"}:
                        raise ValueError("Provide protocol and outcomes_by_condition only.")
                    assert_source_current()
                    self.send_json(200, runtime.episode_risk_audit(**payload))
                except StaleSourceError as error:
                    self.send_json(409, {"error": str(error)} | source_status())
                except (ValueError, TypeError, KeyError) as error:
                    self.send_json(400, {"error": str(error)})
                return
            if path == "/api/jobs" or (path.startswith("/api/jobs/") and path.endswith("/cancel") and path.count("/") == 4):
                try:
                    if path == "/api/jobs":
                        assert_source_current()
                        kind = payload.get("kind")
                        if kind == "hyperion-experiment":
                            self.send_json(202, runtime.submit_hyperion_experiment(payload))
                        elif kind == "titan-experiment":
                            self.send_json(202, runtime.submit_titan_experiment(payload))
                        elif kind == "trident-experiment":
                            self.send_json(202, runtime.submit_trident_experiment(payload))
                        elif kind == "chimera-experiment":
                            self.send_json(202, runtime.submit_chimera_experiment(payload))
                        elif kind == "metamorph-experiment":
                            self.send_json(202, runtime.submit_metamorph_experiment(payload))
                        elif kind == "aura-experiment":
                            self.send_json(202, runtime.submit_aura_experiment(payload))
                        elif kind == "tessera-experiment":
                            self.send_json(202, runtime.submit_tessera_experiment(payload))
                        elif kind == "mnemorph-experiment":
                            self.send_json(202, runtime.submit_mnemorph_experiment(payload))
                        elif kind == "odysseus-experiment":
                            self.send_json(202, runtime.submit_odysseus_experiment(payload))
                        elif kind == "mco-experiment":
                            self.send_json(202, runtime.submit_mco_experiment(payload))
                        elif kind == "helm-experiment":
                            self.send_json(202, runtime.submit_helm_experiment(payload))
                        elif kind == "cognitive-experiment":
                            self.send_json(202, runtime.submit_cognitive_experiment(payload))
                        else:
                            raise ValueError(f"Unknown experiment job kind: {kind}")
                    else:
                        if payload:
                            raise ValueError("Cancellation requires an empty object.")
                        self.send_json(200, runtime.jobs().cancel(path.split("/")[3]))
                except KeyError:
                    self.send_json(404, {"error": "Unknown experiment job."})
                except StaleSourceError as error:
                    self.send_json(409, {"error": str(error)} | source_status())
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except RuntimeError as error:
                    self.send_json(409, {"error": str(error)})
                return
            if path == "/api/adapter/handshake":
                try:
                    from .external_adapter import ExternalAdapterIdentity
                    ident_data = payload.get("identity", {})
                    identity = ExternalAdapterIdentity(
                        adapter_id=str(ident_data.get("adapter_id", "external-sim")),
                        units=str(ident_data.get("units", "SI_METRIC")),
                        ticks_per_second=float(ident_data.get("ticks_per_second", 20.0)),
                        coordinate_frame=str(ident_data.get("coordinate_frame", "FLU")),
                        spatial_bounds=tuple(float(x) for x in ident_data.get("spatial_bounds", (100.0, 100.0, 20.0))),
                        max_linear_velocity=float(ident_data.get("max_linear_velocity", 15.0)),
                        sensor_latency_ms=float(ident_data.get("sensor_latency_ms", 50.0)),
                    )
                    bridge = runtime.external_adapter_bridge()
                    handshake = bridge.negotiate_handshake(
                        client_id=str(payload.get("client_id", "anonymous-client")),
                        engine_name=str(payload.get("engine_name", "generic-engine")),
                        identity=identity,
                    )
                    self.send_json(200, handshake.to_dict())
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/adapter/cancel":
                try:
                    session_token = str(payload.get("session_token", ""))
                    if not session_token:
                        raise ValueError("session_token is required")
                    bridge = runtime.external_adapter_bridge()
                    cancelled = bridge.cancel_session(session_token)
                    self.send_json(200, {"cancelled": cancelled, "session_token": session_token})
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/adapter/verify-3d-replay":
                try:
                    from .external_adapter import Compact3DReplayVerifier
                    receipt = Compact3DReplayVerifier.verify_replay(payload)
                    self.send_json(200, receipt)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/nexus-omega/discover":
                try:
                    seed_base = int(payload.get("seed_base", 50000))
                    num_seeds = int(payload.get("num_seeds", 4))
                    max_steps = int(payload.get("max_steps", 16))
                    result = runtime.nexus_omega_discovery(seed_base=seed_base, num_seeds=num_seeds, max_steps=max_steps)
                    self.send_json(200, result)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/nexus-omega/imagine":
                try:
                    intent = str(payload.get("intent", "forage_expedition"))
                    horizon = int(payload.get("horizon", 8))
                    obs = payload.get("obs")
                    seed = int(payload.get("seed", 42))
                    result = runtime.nexus_omega_imagine(obs=obs, intent=intent, horizon=horizon, seed=seed)
                    self.send_json(200, result)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/nexus-omega/dream":
                try:
                    num_rollouts = int(payload.get("num_rollouts", 4))
                    seed = int(payload.get("seed", 42))
                    result = runtime.nexus_omega_dream(num_rollouts=num_rollouts, seed=seed)
                    self.send_json(200, result)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/nexus-omega/synthesize-opcode":
                try:
                    intent = str(payload.get("intent", "forage_expedition"))
                    obs = payload.get("obs")
                    seed = int(payload.get("seed", 42))
                    result = runtime.nexus_omega_synthesize_opcode(intent=intent, obs=obs, seed=seed)
                    self.send_json(200, result)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/nexus-omega/reason":
                try:
                    obs = payload.get("obs")
                    seed = int(payload.get("seed", 42))
                    result = runtime.nexus_omega_reason(obs=obs, seed=seed)
                    self.send_json(200, result)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/discovery-lab/step":
                try:
                    act = int(payload.get("action", 0))
                    seed = int(payload.get("seed", 42))
                    rw = int(payload.get("river_width", 3))
                    res = runtime.discovery_lab_step(action=act, seed=seed, river_width=rw)
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/discovery-lab/enzyme/generate":
                try:
                    substrate = str(payload.get("substrate", "lignocellulose"))
                    shear = float(payload.get("target_shear_force", 300.0))
                    seed = int(payload.get("seed", 42))
                    res = runtime.discovery_lab_enzyme(substrate=substrate, target_shear=shear, seed=seed)
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/discovery-lab/curriculum":
                try:
                    seed = int(payload.get("seed", 42))
                    res = runtime.discovery_lab_curriculum(seed=seed)
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/reason":
                try:
                    prompt = payload.get("prompt", "")
                    seed = int(payload.get("seed", 42))
                    max_new_tokens = int(payload.get("max_new_tokens", 512))
                    res = runtime.reason(prompt, seed=seed, max_new_tokens=max_new_tokens)
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/problem/solve":
                try:
                    prompt = payload.get("prompt", "")
                    domain = payload.get("domain")
                    res = runtime.solve_problem(prompt, domain=domain)
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/pantheon/synthesize":
                try:
                    prompt = payload.get("prompt", "")
                    seed = int(payload.get("seed", 42))
                    res = runtime.pantheon_synthesize(prompt, seed=seed)
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/creative/synthesize":
                try:
                    prompt = payload.get("prompt", "")
                    seed = int(payload.get("seed", 42))
                    res = runtime.creative_synthesize(prompt, seed=seed)
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/hypothesis/formulate":
                try:
                    conjecture = str(payload.get("conjecture", ""))
                    domain = str(payload.get("domain", "science"))
                    hard_core = payload.get("hard_core_assumptions", ["Primary Axiom"])
                    protective_belt = payload.get("protective_belt", [])
                    falsification_criteria = payload.get("falsification_criteria", ["Null divergence"])
                    novel_predictions = payload.get("novel_predictions", [])
                    prior = float(payload.get("prior", 0.5))
                    res = runtime.hypothesis_formulate(
                        conjecture=conjecture,
                        domain=domain,
                        hard_core_assumptions=hard_core,
                        protective_belt=protective_belt,
                        falsification_criteria=falsification_criteria,
                        novel_predictions=novel_predictions,
                        prior=prior,
                    )
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/proof/prove":
                try:
                    theorem = payload.get("theorem", "")
                    premises = payload.get("premises")
                    goal = payload.get("goal")
                    lhs = payload.get("lhs")
                    rhs = payload.get("rhs")
                    induction_key = payload.get("induction_key")
                    res = runtime.proof_prove(
                        theorem=theorem,
                        premises=premises,
                        goal=goal,
                        lhs=lhs,
                        rhs=rhs,
                        induction_key=induction_key,
                    )
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/creativity/invent":
                try:
                    problem = payload.get("problem", "")
                    domain_a = payload.get("domain_a", "quantum_mechanics")
                    domain_b = payload.get("domain_b", "distributed_consensus")
                    imp = payload.get("improving_param", "speed")
                    det = payload.get("deteriorating_param", "precision")
                    res = runtime.creativity_invent(
                        problem=problem,
                        domain_a=domain_a,
                        domain_b=domain_b,
                        improving_param=imp,
                        deteriorating_param=det,
                    )
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/athenaeum/synthesize":
                try:
                    expr = payload.get("expression", "(x + y) * (x - y) + y ^ 2")
                    target = payload.get("target_expression", "x ^ 2")
                    samples = payload.get("samples")
                    budget = payload.get("budget_rollouts", 20)
                    res = runtime.athenaeum_synthesize(
                        expression=expr,
                        target_expression=target,
                        samples=samples,
                        budget_rollouts=budget,
                    )
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/hephaestus/synthesize":
                try:
                    corpus = payload.get("corpus")
                    terminal_rewards = payload.get("terminal_rewards")
                    koopman_params = payload.get("koopman_params")
                    res = runtime.hephaestus_synthesize(
                        corpus=corpus,
                        terminal_rewards=terminal_rewards,
                        koopman_params=koopman_params,
                    )
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/hyperborea/synthesize":
                try:
                    sheaf_angles_rad = payload.get("sheaf_angles_rad")
                    rg_initial_K = payload.get("rg_initial_K")
                    braid_word = payload.get("braid_word")
                    res = runtime.hyperborea_synthesize(
                        sheaf_angles_rad=sheaf_angles_rad,
                        rg_initial_K=rg_initial_K,
                        braid_word=braid_word,
                    )
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/neurograft/synthesize":
                try:
                    seed = int(payload.get("seed", 42))
                    discount = float(payload["discount"]) if "discount" in payload else None
                    learning_rate = float(payload["learning_rate"]) if "learning_rate" in payload else None
                    kwargs = {}
                    if "horizon" in payload:
                        kwargs["horizon"] = int(payload["horizon"])
                    if "modulator_overrides" in payload:
                        kwargs["modulator_overrides"] = payload["modulator_overrides"]
                    if "lesion_source" in payload:
                        kwargs["lesion_source"] = payload["lesion_source"]
                    res = runtime.neurograft_synthesize(
                        seed=seed,
                        discount=discount,
                        learning_rate=learning_rate,
                        **kwargs,
                    )
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/aether/synthesize":
                try:
                    coords = payload.get("grid_coords", [0, 0])
                    coords_tuple = (int(coords[0]), int(coords[1])) if isinstance(coords, (list, tuple)) and len(coords) >= 2 else (0, 0)
                    thermo_end = float(payload.get("thermo_center_end", 2.0))
                    mu = float(payload.get("mu", 1.0))
                    sigma = float(payload.get("sigma", 1.5))
                    steps = int(payload.get("symplectic_steps", 60))
                    res = runtime.aether_synthesize(
                        grid_coords=coords_tuple,
                        thermo_center_end=thermo_end,
                        mu=mu,
                        sigma=sigma,
                        symplectic_steps=steps,
                    )
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/chronos/synthesize":
                try:
                    seed = int(payload.get("seed", 42))
                    res = runtime.chronos_synthesize(seed=seed)
                    self.send_json(200, res)
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/noether/synthesize":
                try:
                    seed = int(payload.get("seed", 42))
                    res = runtime.noether_synthesize(seed=seed)
                    self.send_json(200, res)
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/cartan/synthesize":
                try:
                    seed = int(payload.get("seed", 42))
                    res = runtime.cartan_synthesize(seed=seed)
                    self.send_json(200, res)
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/poincare/synthesize":
                try:
                    seed = payload.get("seed", 42)
                    stochastic_k = payload.get("stochastic_k", 0.85)
                    surface_type = str(payload.get("surface_type", "torus"))
                    res = runtime.poincare_synthesize(seed=seed, stochastic_k=stochastic_k, surface_type=surface_type,
                        iterations=payload.get("iterations", 400), curvature_c=payload.get("curvature_c", 1.0),
                        ricci_vertices=payload.get("ricci_vertices", 6), ricci_steps=payload.get("ricci_steps", 30))
                    self.send_json(200, res)
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error), "type": type(error).__name__})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/crosswire/synthesize":
                try:
                    seed = int(payload.get("seed", 42))
                    budget = int(payload.get("budget", 16))
                    num_actions = int(payload.get("num_actions", 4))
                    res = runtime.crosswire_synthesize(seed=seed, budget=budget, num_actions=num_actions)
                    self.send_json(200, res)
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/faultline/synthesize":
                try:
                    seed = int(payload.get("seed", 42))
                    panel_size = int(payload.get("panel_size", 3))
                    res = runtime.faultline_synthesize(seed=seed, panel_size=panel_size)
                    self.send_json(200, res)
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path == "/api/tether/synthesize":
                try:
                    seed = int(payload.get("seed", 42))
                    res = runtime.tether_synthesize(seed=seed)
                    self.send_json(200, res)
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if self.path not in ("/api/adapter/tick", "/api/respond", "/api/remember", "/api/experiment", "/api/contrast-experiment", "/api/horizon-experiment", "/api/odyssey-experiment", "/api/helm-experiment", "/api/odysseus-experiment", "/api/mco-experiment", "/api/aura-experiment", "/api/tessera-experiment", "/api/mnemorph-experiment", "/api/metamorph-experiment", "/api/chimera-experiment", "/api/hyperion-experiment", "/api/titan-experiment", "/api/trident-experiment", "/api/cognitive-experiment", "/api/ecosystem-experiment", "/api/tessera/shield", "/api/poseidon-x-experiment"):
                self.send_json(404, {"error": "Not found"}); return
            if not busy.acquire(blocking=False):
                self.send_json(409, {"error": "Poseidon is processing another request. Try again shortly."}); return
            try:
                if self.path == "/api/adapter/tick":
                    from .external_adapter import ExternalTickMessage
                    bridge = runtime.external_adapter_bridge()
                    token = str(payload.get("session_token", ""))
                    if token in bridge.cancelled_sessions:
                        self.send_json(409, {"error": f"Session {token} was cancelled."})
                        return
                    if token not in bridge.active_sessions:
                        self.send_json(404, {"error": f"Unknown session token {token}."})
                        return
                    msg = ExternalTickMessage(
                        session_token=token,
                        tick=int(payload.get("tick", 0)),
                        delta_time=float(payload.get("delta_time", 0.05)),
                        observation_16d=[float(x) for x in payload.get("observation_16d", [])],
                        position=tuple(float(x) for x in payload["position"]) if payload.get("position") is not None else None,
                        velocity=tuple(float(x) for x in payload["velocity"]) if payload.get("velocity") is not None else None,
                        reward=float(payload.get("reward", 0.0)),
                        terminal=bool(payload.get("terminal", False)),
                        vitals=dict(payload.get("vitals", {})),
                    )
                    controller_choice = payload.get("controller", "cognitive")
                    if controller_choice == "pantheon":
                        agent = runtime.pantheon()
                    elif controller_choice == "poseidon_x":
                        agent = runtime.poseidon_x()
                    elif controller_choice == "hyper":
                        agent = runtime.hyper_agent()
                    elif controller_choice == "nexus_omega":
                        agent = runtime.nexus_omega()
                    else:
                        agent = runtime.cognitive_agent()
                    resp = bridge.process_tick(msg, agent)
                    result = resp.to_dict()
                elif self.path == "/api/poseidon-x-experiment":
                    assert_source_current()
                    episodes = payload.get("episodes", 2)
                    max_steps = payload.get("max_steps", 32)
                    result = runtime.poseidon_x_experiment(seed=payload.get("seed", 101), episodes=episodes, max_steps=max_steps)
                elif self.path == "/api/ecosystem-experiment":
                    assert_source_current()
                    episodes = payload.get("episodes", 2)
                    max_steps = payload.get("max_steps", 32)
                    result = runtime.ecosystem_experiment(seed=payload.get("seed", 42), episodes=episodes, max_steps=max_steps)
                elif self.path == "/api/tessera/shield":
                    from .tessera_prover import TesseraProver
                    prover = TesseraProver()
                    obs_vec = payload.get("obs", [1.0, 0.8, 0.8, 0.8])
                    cand = payload.get("action", 4)
                    certified, receipt = prover.shield_action(obs_vec, cand)
                    result = {
                        "certified_action": certified,
                        "shield_triggered": receipt.shield_triggered,
                        "is_safe": receipt.is_safe,
                        "certificate_sha256": receipt.certificate_sha256,
                    }
                elif self.path == "/api/mco-experiment":
                    assert_source_current()
                    result = runtime.mco_experiment()
                elif self.path == "/api/tessera-experiment":
                    assert_source_current()
                    result = runtime.tessera_experiment()
                elif self.path == "/api/mnemorph-experiment":
                    assert_source_current()
                    result = runtime.mnemorph_experiment()
                elif self.path in ("/api/experiment", "/api/contrast-experiment", "/api/horizon-experiment", "/api/odyssey-experiment", "/api/helm-experiment", "/api/odysseus-experiment", "/api/aura-experiment", "/api/metamorph-experiment", "/api/chimera-experiment", "/api/hyperion-experiment", "/api/titan-experiment", "/api/trident-experiment", "/api/cognitive-experiment"):
                    if set(payload) - {"kind", "seed", "episodes", "max_steps", "scarcity", "use_pomdp"}:
                        raise ValueError("Unknown experiment setting.")
                    episodes, max_steps = payload.get("episodes", 4), payload.get("max_steps", 64)
                    if type(episodes) is not int or not 1 <= episodes <= 8 or type(max_steps) is not int or not 32 <= max_steps <= 256:
                        raise ValueError("Workbench experiments require 1–8 paired episodes and 32–256 steps.")
                    assert_source_current()
                    if self.path == "/api/cognitive-experiment":
                        result = runtime.cognitive_experiment(payload.get("seed", 90000000), episodes, max_steps, payload.get("scarcity", 2.5), use_pomdp=payload.get("use_pomdp", False))
                    elif self.path == "/api/hyperion-experiment":
                        result = runtime.hyperion_experiment(payload.get("seed", 303000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/trident-experiment":
                        result = runtime.trident_experiment(payload.get("seed", 88100000), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/titan-experiment":
                        result = runtime.titan_experiment(payload.get("seed", 99000000), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/chimera-experiment":
                        result = runtime.chimera_experiment(payload.get("seed", 202000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/metamorph-experiment":
                        result = runtime.metamorph_experiment(payload.get("seed", 199000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/aura-experiment":
                        result = runtime.aura_experiment(payload.get("seed", 144000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/odysseus-experiment":
                        result = runtime.odysseus_experiment(payload.get("seed", 133000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/helm-experiment":
                        result = runtime.helm_experiment(payload.get("seed", 112000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/odyssey-experiment":
                        result = runtime.odyssey_experiment(payload.get("seed", 110000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/horizon-experiment":
                        result = runtime.horizon_experiment(payload.get("seed", 108000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/contrast-experiment":
                        result = runtime.contrast_experiment(payload.get("seed", 104000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    else:
                        result = runtime.experiment(payload.get("seed", 93000001), episodes, max_steps, payload.get("scarcity", 2.5))
                elif self.path == "/api/remember":
                    result = runtime.remember(payload.get("text"), payload.get("carrier", "episodic"))
                else:
                    max_steps = payload.get("max_steps", 256)
                    if payload.get("mode", "chat") == "world" and (type(max_steps) is not int or not 1 <= max_steps <= 512):
                        raise ValueError("Workbench worlds require a 1–512 step horizon.")
                    max_new_tokens = payload.get("max_new_tokens")
                    result = runtime.respond(payload.get("prompt"), payload.get("mode", "chat"), payload.get("history"), payload.get("seed", 42), payload.get("disabled_carriers"), planner=payload.get("planner", "policy"), scarcity=payload.get("scarcity", 1.0), max_steps=max_steps, max_new_tokens=max_new_tokens)
                    if "paths" in result:
                        result["links"] = {key: "/artifacts/"+Path(value).resolve().relative_to(runtime.root/"outputs").as_posix() for key,value in result["paths"].items()}
                self.send_json(200, result)
            except StaleSourceError as error:
                self.send_json(409, {"error": str(error)} | source_status())
            except (ValueError, TypeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception as error:
                self.send_json(500, {"error": str(error), "type": type(error).__name__})
            finally:
                busy.release()
    return Handler

def serve(root=".", port=8787, adapter=None):
    runtime = Poseidon(root, adapter)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(runtime, port))
    server.daemon_threads = True
    print(f"Supermix Poseidon: http://127.0.0.1:{port}", flush=True)
    try:
        while True:
            try:
                server.serve_forever()
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                print(f"Server error recovered: {e}", flush=True)
    finally:
        server.server_close()
        runtime.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--adapter")
    args = p.parse_args()
    serve(args.root, args.port, args.adapter)
