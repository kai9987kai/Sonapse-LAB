"""Media contract tests: geometry, temporal output, reproducibility and data splits."""
import base64
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import struct

from PIL import Image
import pytest


@pytest.fixture
def media():
    assert importlib.util.find_spec("poseidon.media") is not None, "Scene renderer is missing"
    return importlib.import_module("poseidon.media")


SCENE = {"shape": "cube", "color": "cyan", "motion": "bounce", "count": 2, "scale": "medium"}


def test_media_module_exists():
    assert importlib.util.find_spec("poseidon.media") is not None, "Scene renderer is missing"


def test_missing_or_invalid_scene_fields_are_rejected_before_writes(media, tmp_path):
    for scene in ({}, {**SCENE, "count": True}, {**SCENE, "count": 4}, {**SCENE, "shape": "../../file"}):
        with pytest.raises(ValueError):
            media.render_scene(scene, tmp_path, kind="image")
    assert list(tmp_path.iterdir()) == []


def test_seeded_image_is_exact_and_contains_colored_geometry(media, tmp_path):
    a = media.render_scene(SCENE, tmp_path / "a", kind="image", seed=17)
    b = media.render_scene(SCENE, tmp_path / "b", kind="image", seed=17)
    assert Path(a["paths"]["image"]).read_bytes() == Path(b["paths"]["image"]).read_bytes()
    with Image.open(a["paths"]["image"]) as image:
        assert image.format == "PNG"
        assert image.width >= 256 and image.height >= 192
        rgb = image.convert("RGB")
        assert sum(1 for r, g, b in (rgb.getpixel((x, y)) for y in range(rgb.height) for x in range(rgb.width)) if g > r * 1.4 and b > r * 1.4) > 500
    assert a["scene"] == SCENE
    assert a["generation_method"] == "procedural_scene_renderer"


def test_video_contains_distinct_frames_and_duration(media, tmp_path):
    result = media.render_scene(SCENE, tmp_path, kind="video", seed=3)
    with Image.open(result["paths"]["gif"]) as gif:
        assert gif.n_frames == result["frame_count"] == 24
        frames, duration = [], 0
        for i in range(gif.n_frames):
            gif.seek(i)
            frames.append(hashlib.sha256(gif.convert("RGB").tobytes()).hexdigest())
            duration += gif.info["duration"]
        assert len(set(frames)) > 12
        assert duration == 2400


@pytest.mark.parametrize("shape", ["cube", "sphere", "pyramid", "cylinder"])
def test_exported_obj_faces_and_normals_are_valid(media, tmp_path, shape):
    result = media.render_scene({**SCENE, "shape": shape, "count": 3}, tmp_path, kind="mesh")
    rows = Path(result["paths"]["obj"]).read_text().splitlines()
    vertices = [tuple(map(float, row.split()[1:])) for row in rows if row.startswith("v ")]
    normals = [tuple(map(float, row.split()[1:])) for row in rows if row.startswith("vn ")]
    faces = [row.split()[1:] for row in rows if row.startswith("f ")]
    assert len(vertices) == result["vertex_count"]
    assert len(faces) == result["triangle_count"] > 0
    assert all(math.isfinite(x) for point in vertices + normals for x in point)
    assert all(abs(sum(x*x for x in normal) - 1) < 1e-5 for normal in normals)
    for face in faces:
        assert len(face) == 3
        ids = [int(token.split("//")[0]) for token in face]
        assert len(set(ids)) == 3 and all(1 <= i <= len(vertices) for i in ids)
        normal_ids = [int(token.split("//")[1]) for token in face]
        assert all(1 <= i <= len(normals) for i in normal_ids)
    assert len([row for row in rows if row.startswith("o ")]) == 3
    assert Path(result["paths"]["mtl"]).is_file()


def test_gltf_self_contained_buffers_match_accessors(media, tmp_path):
    result = media.render_scene(SCENE, tmp_path, kind="mesh")
    gltf = json.loads(Path(result["paths"]["gltf"]).read_text())
    assert gltf["asset"]["version"] == "2.0"
    data = base64.b64decode(gltf["buffers"][0]["uri"].split(",", 1)[1])
    assert len(data) == gltf["buffers"][0]["byteLength"]
    for view in gltf["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= len(data)
    position_count = gltf["accessors"][0]["count"]
    index_accessor = gltf["accessors"][2]
    index_view = gltf["bufferViews"][index_accessor["bufferView"]]
    indices = struct.unpack_from("<" + "I" * index_accessor["count"], data, index_view["byteOffset"])
    assert all(0 <= i < position_count for i in indices)
    assert len(indices) == result["triangle_count"] * 3


def test_scene_dataset_is_deterministic_unique_and_semantically_disjoint(media):
    train = media.generate_scene_examples(1500, seed=41, split="train")
    assert train == media.generate_scene_examples(1500, seed=41, split="train")
    dev = media.generate_scene_examples(500, seed=42, split="dev")
    assert len({row["prompt"] for row in train}) == len(train)
    assert {row["group"] for row in train}.isdisjoint({row["group"] for row in dev})
    assert {row["prompt"] for row in train}.isdisjoint({row["prompt"] for row in dev})
    for row in train + dev:
        for name, value in row["scene"].items():
            assert media.SCENE_LABELS[name][row["labels"][name]] == value
    assert set(row["scene"]["shape"] for row in train) == {"cube", "sphere", "pyramid", "cylinder"}


def test_final_test_scene_groups_are_disjoint_from_training_and_development(media):
    splits = {name: media.generate_scene_examples(1000, seed=13, split=name) for name in ("train", "dev", "test")}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert {row["group"] for row in splits[left]}.isdisjoint({row["group"] for row in splits[right]})
        assert {row["prompt"] for row in splits[left]}.isdisjoint({row["prompt"] for row in splits[right]})

