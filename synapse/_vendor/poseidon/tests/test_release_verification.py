"""Model-free tests for the downloaded release's integrity boundary."""
import hashlib
import json
from pathlib import Path

import pytest

from tools import verify_huggingface_release as release


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _seal(root, manifest):
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    (root / "manifest.json").write_bytes(encoded)
    sums = [f'{row["sha256"]}  {row["path"]}' for row in manifest["files"]]
    sums.append(f"{_digest(encoded)}  manifest.json")
    (root / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")


def _package(root, version="0.5.1"):
    payload = b"bounded test payload\n"
    (root / "model.bin").write_bytes(payload)
    manifest = {
        "schema": release.SCHEMA, "version": version,
        "source_revision": "1" * 40, "repo_id": "Kai9987kai/Poseidon",
        "core_sha256": "2" * 64, "atlas_artifact_sha256": "3" * 64,
        "files": [{"path": "model.bin", "size": len(payload), "sha256": _digest(payload)}],
    }
    if version in ("0.4.0", "0.5.0", "0.5.1", "0.6.0", "0.7.0", "0.8.0", "0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0"):
        manifest.update(horizon_artifact_sha256="4" * 64, contrast_artifact_sha256="5" * 64)
    if version in ("0.5.0", "0.5.1", "0.6.0", "0.7.0", "0.8.0", "0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0"):
        manifest["odyssey_artifact_sha256"] = "6" * 64
    if version in ("0.6.0", "0.7.0", "0.8.0", "0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0"):
        manifest["helm_artifact_sha256"] = "7" * 64
        manifest["helm_experiment_verification"] = {"test.json": {"verified": True}}
    if version in ("0.7.0", "0.8.0", "0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0"):
        manifest["odysseus_artifact_sha256"] = "8" * 64
        manifest["odysseus_experiment_verification"] = {"test.json": {"verified": True}}
        manifest["mco_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
    if version in ("0.8.0", "0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0"):
        manifest["aura_experiment_verification"] = {"test.json": {"verified": True}}
        manifest["tessera_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
        manifest["mnemorph_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
    if version in ("0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0"):
        manifest["metamorph_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
    if version in ("1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0"):
        manifest["chimera_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
    if version in ("1.1.0", "1.2.0", "1.3.0", "2.0.0"):
        manifest["hyperion_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
    if version in ("1.2.0", "1.3.0", "2.0.0"):
        manifest["titan_receipt_sha256"] = "9" * 64
        manifest["titan_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
    if version in ("1.3.0", "2.0.0"):
        manifest["trident_receipt_sha256"] = "a" * 64
        manifest["trident_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
    _seal(root, manifest)
    return manifest


@pytest.mark.parametrize("version", ("0.2.0", "0.4.0", "0.5.0", "0.5.1", "0.6.0", "0.7.0", "0.8.0", "0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0"))
def test_hash_verification_preserves_supported_release_versions(tmp_path, version):
    manifest = _package(tmp_path, version)
    actual, receipt = release.verify_hashes(tmp_path)
    assert actual == manifest
    assert receipt["verified"] and receipt["version"] == version
    assert receipt["files_verified"] == 1
    assert receipt["payload_bytes"] == (tmp_path / "model.bin").stat().st_size


@pytest.mark.parametrize("key", ("helm_artifact_sha256", "helm_experiment_verification"))
def test_helm_release_requires_bound_critic_and_evidence(tmp_path, key):
    manifest = _package(tmp_path, "0.6.0")
    manifest.pop(key)
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("key", ("odysseus_artifact_sha256", "odysseus_experiment_verification", "mco_experiment_verification"))
def test_odysseus_and_mco_release_requires_bound_evidence(tmp_path, key):
    manifest = _package(tmp_path, "0.7.0")
    manifest.pop(key)
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("key", ("metamorph_experiment_verification",))
def test_metamorph_release_requires_bound_evidence(tmp_path, key):
    manifest = _package(tmp_path, "0.9.0")
    manifest.pop(key)
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("key", ("chimera_experiment_verification",))
def test_chimera_release_requires_bound_evidence(tmp_path, key):
    manifest = _package(tmp_path, "1.0.0")
    manifest.pop(key)
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("key", ("hyperion_experiment_verification",))
def test_hyperion_release_requires_bound_evidence(tmp_path, key):
    manifest = _package(tmp_path, "1.1.0")
    manifest.pop(key)
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError):
        release.verify_hashes(tmp_path)


def test_titan_release_requires_replay_evidence(tmp_path):
    manifest = _package(tmp_path, "1.2.0")
    manifest.pop("titan_experiment_verification")
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="TITAN replay verification"):
        release.verify_hashes(tmp_path)


def test_trident_release_requires_replay_evidence(tmp_path):
    manifest = _package(tmp_path, "1.3.0")
    manifest.pop("trident_experiment_verification")
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="TRIDENT replay verification"):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("digest_key", (
    "horizon_artifact_sha256", "contrast_artifact_sha256", "odyssey_artifact_sha256",
))
@pytest.mark.parametrize("invalid", (None, "not-a-digest"))
def test_patch_release_requires_all_planning_artifact_digests(tmp_path, digest_key, invalid):
    manifest = _package(tmp_path)
    if invalid is None:
        manifest.pop(digest_key)
    else:
        manifest[digest_key] = invalid
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="Invalid release digest: " + digest_key):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("payload", (b"different contents!!\n", b"short"))
def test_payload_tampering_rejected_before_model_loading(tmp_path, payload):
    _package(tmp_path)
    (tmp_path / "model.bin").write_bytes(payload)
    with pytest.raises(ValueError, match="size or checksum mismatch"):
        release.verify_hashes(tmp_path)


def test_missing_payload_is_rejected(tmp_path):
    _package(tmp_path)
    (tmp_path / "model.bin").unlink()
    with pytest.raises(ValueError, match="file is missing"):
        release.verify_hashes(tmp_path)


def test_unmanifested_source_is_rejected(tmp_path):
    _package(tmp_path)
    (tmp_path / "extra.py").write_text("print('unexpected source')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differs from manifest"):
        release.verify_hashes(tmp_path)


def test_generated_metadata_allowed_but_manifested_cache_payload_still_checked(tmp_path):
    manifest = _package(tmp_path)
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text("{}", encoding="utf-8")
    payload = b"manifested cached payload"
    (cache / "payload.bin").write_bytes(payload)
    manifest["files"].append({"path": ".cache/huggingface/payload.bin", "size": len(payload),
                              "sha256": _digest(payload)})
    _seal(tmp_path, manifest)
    _, receipt = release.verify_hashes(tmp_path)
    assert receipt["files_verified"] == 2
    assert receipt["ignored_local_metadata"]["directories"] == [".cache/huggingface"]
    (cache / "payload.bin").write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="size or checksum mismatch"):
        release.verify_hashes(tmp_path)


def test_sha256sums_must_bind_the_manifest_itself(tmp_path):
    manifest = _package(tmp_path)
    manifest["repo_id"] = "another-owner/Poseidon"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256SUMS.txt does not match"):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("path", ("../model.bin", "outputs\\model.bin"))
def test_noncanonical_or_escaping_inventory_paths_rejected(tmp_path, path):
    manifest = _package(tmp_path)
    manifest["files"][0]["path"] = path
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="Package paths"):
        release.verify_hashes(tmp_path)


def test_case_colliding_manifest_entries_rejected_on_all_platforms(tmp_path):
    manifest = _package(tmp_path)
    manifest["files"].append({**manifest["files"][0], "path": "MODEL.BIN"})
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="Duplicate, colliding, or reserved"):
        release.verify_hashes(tmp_path)


def test_isolated_smoke_script_is_valid_python():
    compile(release.SMOKE_SCRIPT, "packaged-release-smoke", "exec")


def test_v314_requires_its_new_reasoning_and_robustness_evidence(tmp_path):
    manifest = _package(tmp_path, "2.0.0")
    manifest["version"] = "3.14.0"
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="reasoning audit"):
        release.verify_hashes(tmp_path)


def test_v314_reasoning_paths_preserve_historical_evidence_names():
    paths = getattr(release, "reasoning_audit_paths", lambda version: ())
    assert paths("3.14.0") == (
        "outputs/research/robustness-upgrade-20261031/reasoning-20261031.json",
        "outputs/research/robustness-upgrade-20261031/reasoning-20261101.json",
    )
    assert paths("3.13.0") == release.REASONING_AUDIT_PATHS


def test_v314_model_card_identifies_new_tools_without_promoting_a_controller():
    from tools.package_huggingface import model_card
    card = model_card("1" * 40, *("2" * 64 for _ in range(5)), version="3.14.0")
    assert "31 October robustness upgrade" in card
    assert "metamorphic-audit" in card
    assert "episode-risk" in card
    assert "synthetic episode fixture" in card


@pytest.mark.parametrize("version", release.V3_RELEASE_VERSIONS)
def test_v3_releases_preserve_full_evidence_requirements(tmp_path, version):
    manifest = _package(tmp_path, "2.0.0")
    manifest["version"] = version
    _seal(tmp_path, manifest)
    if version == "3.14.0":
        with pytest.raises(ValueError, match="reasoning audit"):
            release.verify_hashes(tmp_path)
    else:
        _, receipt = release.verify_hashes(tmp_path)
        assert receipt["version"] == version
    manifest.pop("trident_experiment_verification")
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="TRIDENT replay verification"):
        release.verify_hashes(tmp_path)


def test_reasoning_metadata_cannot_claim_replay_of_missing_package_evidence(tmp_path):
    manifest = _package(tmp_path, "2.0.0")
    manifest["version"] = "3.13.0"
    manifest["reasoning_audit_verification"] = {
        "outputs/research/evidence-upgrade-20261030/reasoning-20261030.json": {
            "integrity_verified": True, "case_manifest_verified": True,
            "source_matches": True, "replay_verified": True,
            "scope": "finite_generated_suite", "report_sha256": "b" * 64,
            "file_sha256": "c" * 64}}
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="reasoning audit"):
        release.verify_hashes(tmp_path)


def _reasoning_evidence(root):
    from poseidon.research.reasoning_audit import run_audit
    for seed in (20261030, 20261031):
        relative = f"outputs/research/evidence-upgrade-20261030/reasoning-{seed}.json"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_audit(seed=seed, samples=1)), encoding="utf-8")


def test_reasoning_release_evidence_replays_and_changed_metadata_is_rejected(tmp_path):
    _reasoning_evidence(tmp_path)
    records = release.verify_reasoning_audits(tmp_path)
    assert len(records) == 2
    assert all(record["replay_verified"] is True for record in records.values())
    assert release.verify_reasoning_audits(tmp_path, recorded=records) == records
    first_path = next(iter(records))
    records[first_path]["replay_verified"] = False
    with pytest.raises(ValueError, match="reasoning audit"):
        release.verify_reasoning_audits(tmp_path, recorded=records)


def test_reasoning_release_metadata_binds_inventoried_bytes(tmp_path):
    manifest = _package(tmp_path, "2.0.0")
    manifest["version"] = "3.13.0"
    _reasoning_evidence(tmp_path)
    records = release.verify_reasoning_audits(tmp_path)
    for relative in records:
        payload = (tmp_path / relative).read_bytes()
        manifest["files"].append({"path": relative, "size": len(payload), "sha256": _digest(payload)})
    manifest["reasoning_audit_verification"] = records
    _seal(tmp_path, manifest)
    assert release.verify_hashes(tmp_path)[1]["verified"] is True
    first = next(iter(records))
    records[first]["file_sha256"] = "d" * 64
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="reasoning audit"):
        release.verify_hashes(tmp_path)


def test_reasoning_release_rejects_saved_report_that_does_not_replay(tmp_path):
    _reasoning_evidence(tmp_path)
    first = tmp_path / "outputs/research/evidence-upgrade-20261030/reasoning-20261030.json"
    report = json.loads(first.read_text(encoding="utf-8"))
    report["rows"][0]["answer"] = "999999"
    # A new fingerprint cannot make the changed answer agree with actual replay.
    encoded = json.dumps({key: value for key, value in report.items() if key != "report_sha256"},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    report["report_sha256"] = _digest(encoded)
    first.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="replay"):
        release.verify_reasoning_audits(tmp_path)


@pytest.fixture
def robustness_package(tmp_path):
    from poseidon.research.reasoning_audit import run_audit as reasoning
    from poseidon.research.metamorphic_audit import run_audit as metamorphic
    from poseidon.research.episode_risk import make_episode_risk_protocol, audit_episode_risk
    manifest = _package(tmp_path, "2.0.0")
    manifest["version"] = "3.14.0"
    for seed, relative in zip((20261031, 20261101), release.ROBUSTNESS_REASONING_AUDIT_PATHS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reasoning(seed=seed, samples=32)), encoding="utf-8")
    for seed, relative in zip((20261031, 20261101), release.METAMORPHIC_AUDIT_PATHS):
        (tmp_path / relative).write_text(json.dumps(metamorphic(seed=seed, samples=32)), encoding="utf-8")
    ids = list(range(100, 120))
    protocol = make_episode_risk_protocol(
        policy_sha256="a" * 64, predictor_sha256="b" * 64, panel_sha256="c" * 64,
        thresholds={"margin": 0.1}, required_conditions=["id"], fit_episode_ids=[1],
        selection_episode_ids=[2], calibration_episode_ids=ids, max_episode_risk=0.1)
    fixture = audit_episode_risk(protocol, {"id": [{"episode_id": number, "adverse_event": False} for number in ids]})
    (tmp_path / release.EPISODE_RISK_FIXTURE_PATH).write_text(json.dumps(fixture), encoding="utf-8")
    manifest["reasoning_audit_verification"] = release.verify_reasoning_audits(tmp_path, version="3.14.0")
    manifest["robustness_audit_verification"] = release.verify_robustness_audits(tmp_path)
    for relative in (*release.ROBUSTNESS_REASONING_AUDIT_PATHS, *release.ROBUSTNESS_AUDIT_PATHS):
        payload = (tmp_path / relative).read_bytes()
        manifest["files"].append({"path": relative, "size": len(payload), "sha256": _digest(payload)})
    _seal(tmp_path, manifest)
    return tmp_path, manifest


def test_v314_new_evidence_replays_and_is_bound_to_package_inventory(robustness_package):
    root, manifest = robustness_package
    records = manifest["robustness_audit_verification"]
    assert release.verify_robustness_audits(root, recorded=records) == records
    assert release.verify_hashes(root)[1]["version"] == "3.14.0"
    assert records[release.EPISODE_RISK_FIXTURE_PATH]["eligible"] is False
    assert records[release.EPISODE_RISK_FIXTURE_PATH]["evidence_role"] == "synthetic_episode_software_fixture"
    assert all(records[path]["replay_verified"] is True for path in release.METAMORPHIC_AUDIT_PATHS)


@pytest.mark.parametrize("mutation", ["missing", "missing_report", "file_hash", "false_replay", "promoted", "authenticity", "performance_claim"])
def test_v314_incomplete_or_misleading_robustness_metadata_fails_closed(robustness_package, mutation):
    root, manifest = robustness_package
    records = manifest["robustness_audit_verification"]
    if mutation == "missing":
        del manifest["robustness_audit_verification"]
    elif mutation == "missing_report":
        del records[release.METAMORPHIC_AUDIT_PATHS[0]]
    elif mutation == "file_hash":
        records[release.METAMORPHIC_AUDIT_PATHS[0]]["file_sha256"] = "d" * 64
    elif mutation == "false_replay":
        records[release.METAMORPHIC_AUDIT_PATHS[0]]["replay_verified"] = False
    else:
        key, value = {"promoted": ("promoted", True), "authenticity": ("outcome_authenticity_verified", True),
                      "performance_claim": ("evidence_role", "controller_performance")}[mutation]
        records[release.EPISODE_RISK_FIXTURE_PATH][key] = value
    _seal(root, manifest)
    with pytest.raises(ValueError, match="robustness|metamorphic|synthetic episode"):
        release.verify_hashes(root)


def test_v314_new_replay_metadata_cannot_be_changed_without_recomputation(robustness_package):
    root, manifest = robustness_package
    records = manifest["robustness_audit_verification"]
    records[release.EPISODE_RISK_FIXTURE_PATH]["eligible"] = True
    with pytest.raises(ValueError, match="metadata differs"):
        release.verify_robustness_audits(root, recorded=records)
