"""CPU procedural media from mandatory learned scene fields.

This module does not parse prompts or generate pixels with a neural image model.
The scene planner learns a bounded description; this renderer executes it.
"""
from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
from io import BytesIO
import itertools
import json
import math
import os
from pathlib import Path
import random
import struct
import tempfile
from typing import Any

from PIL import Image, ImageDraw, ImageFilter


SCENE_LABELS = {
    "shape": ["cube", "sphere", "pyramid", "cylinder"],
    "color": ["red", "blue", "green", "yellow", "purple", "cyan"],
    "motion": ["still", "orbit", "bounce", "spin"],
    "count": [1, 2, 3],
    "scale": ["small", "medium", "large"],
}
COLORS = {
    "red": (242, 80, 99), "blue": (77, 136, 250), "green": (64, 211, 151),
    "yellow": (248, 206, 73), "purple": (175, 115, 250), "cyan": (59, 212, 232),
}
_SCALES = {"small": 0.62, "medium": 0.88, "large": 1.12}
_SHAPES = {
    "cube": (("cube", "cubes"), ("cubical block", "cubical blocks")),
    "sphere": (("sphere", "spheres"), ("ball", "balls")),
    "pyramid": (("pyramid", "pyramids"), ("square pyramid", "square pyramids")),
    "cylinder": (("cylinder", "cylinders"), ("round cylinder", "round cylinders")),
}
_COLOR_WORDS = {"red": ("red", "crimson"), "blue": ("blue", "azure"),
                "green": ("green", "emerald"), "yellow": ("yellow", "golden yellow"),
                "purple": ("purple", "violet"), "cyan": ("cyan", "aqua cyan")}
_SCALE_WORDS = {"small": ("small", "tiny"), "medium": ("medium", "medium-sized"),
                "large": ("large", "big")}
_MOTION_WORDS = {"still": ("remaining still", "stationary"),
                 "orbit": ("orbiting", "moving in an orbit"),
                 "bounce": ("bouncing", "moving up and down"),
                 "spin": ("spinning", "rotating in place")}
_TEMPLATES = (
    "Create a scene with {count} {scale} {color} {shape}, {motion}.",
    "Show {count} {scale} {color} {shape} {motion}.",
    "Generate {count} {scale} {color} {shape}; keep them {motion}.",
    "Make an image of {count} {scale} {color} {shape}, {motion}.",
    "Produce a video of {count} {scale} {color} {shape} {motion}.",
    "Build a 3D scene: {count} {scale} {color} {shape}, {motion}.",
    "Please draw {count} {scale} {color} {shape} {motion}.",
    "I want {count} {scale} {color} {shape}; motion: {motion}.",
    "Render {count} {color} {shape}, {scale} in size and {motion}.",
    "Design {count} {color} {shape}; size {scale}; movement {motion}.",
    "Visualize {count} {scale} {shape}, colored {color}, {motion}.",
    "My scene contains {count} {scale} {shape} in {color}, {motion}.",
)


def validate_scene(scene: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete shared scene grammar; do not infer missing fields."""
    if not isinstance(scene, dict):
        raise ValueError("scene must be a dictionary with all five scene fields")
    result = {}
    for name, values in SCENE_LABELS.items():
        value = scene.get(name)
        if name == "count":
            valid = type(value) is int and value in values
        else:
            valid = isinstance(value, str) and value in values
        if not valid:
            raise ValueError(f"scene.{name} must be one of {values!r}")
        result[name] = value
    return result


def _group(scene: dict[str, Any]) -> str:
    return "scene-v1:" + ":".join(str(scene[name]) for name in SCENE_LABELS)


def _split(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:16], 16) % 10
    return "dev" if bucket == 0 else "test" if bucket == 1 else "train"


def generate_scene_examples(n: int, seed: int = 42, split: str = "train") -> list[dict[str, Any]]:
    """Unique controlled-language examples, grouped by complete scene semantics.

    Split membership is fixed by the semantic group, independently of seed.
    Every prompt explicitly describes all five target fields. The training set
    is synthetic; it contains no copied repository source, images or videos.
    """
    if type(n) is not int or n < 0:
        raise ValueError("n must be a non-negative integer")
    if split not in {"train", "dev", "test", "all"}:
        raise ValueError("split must be train, dev, test or all")
    rng = random.Random(seed)
    scenes = [dict(zip(SCENE_LABELS, values)) for values in itertools.product(*SCENE_LABELS.values())]
    scenes = [scene for scene in scenes if split == "all" or _split(_group(scene)) == split]
    # Five binary wording choices and twelve templates = 384 unique prompts/scene.
    variants_per_scene = len(_TEMPLATES) * 32
    if n > len(scenes) * variants_per_scene:
        raise ValueError(f"n exceeds the {len(scenes) * variants_per_scene} unique {split} examples")
    rng.shuffle(scenes)
    variants = list(range(variants_per_scene))
    rng.shuffle(variants)
    rows = []
    for i in range(n):
        scene = scenes[i % len(scenes)].copy()
        variant = variants[i // len(scenes)]
        template, bits = divmod(variant, 32)
        options = [(bits >> bit) & 1 for bit in range(5)]
        quantity = scene["count"]
        shape = _SHAPES[scene["shape"]][options[0]][int(quantity > 1)]
        words = {
            "shape": shape, "color": _COLOR_WORDS[scene["color"]][options[1]],
            "motion": _MOTION_WORDS[scene["motion"]][options[2]],
            "scale": _SCALE_WORDS[scene["scale"]][options[3]],
            "count": str(quantity) if options[4] else ("one", "two", "three")[quantity - 1],
        }
        rows.append({"prompt": _TEMPLATES[template].format(**words), "scene": scene,
                     "labels": {name: values.index(scene[name]) for name, values in SCENE_LABELS.items()},
                     "group": _group(scene)})
    rng.shuffle(rows)
    return rows


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(v):
    length = math.sqrt(sum(x*x for x in v))
    if length <= 1e-12:
        raise ValueError("degenerate geometry")
    return tuple(x / length for x in v)


@lru_cache(maxsize=4)
def _primitive(shape):
    """Return outward oriented triangular geometry in a unit-sized local frame."""
    vertices, faces = [], []
    if shape == "cube":
        vertices = [(x, y, z) for x, y, z in itertools.product((-.5, .5), repeat=3)]
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        faces = [(a, b, c) for a, b, c, d in quads] + [(a, c, d) for a, b, c, d in quads]
    elif shape == "pyramid":
        vertices = [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5), (0, .6, 0)]
        faces = [(0, 1, 2), (0, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)]
    elif shape == "cylinder":
        segments = 24
        vertices = [(0, -.5, 0), (0, .5, 0)]
        for y in (-.5, .5):
            vertices += [(.5*math.cos(i*math.tau/segments), y, .5*math.sin(i*math.tau/segments)) for i in range(segments)]
        for i in range(segments):
            a, b = 2+i, 2+(i+1) % segments
            c, d = a+segments, b+segments
            faces += [(0, b, a), (1, c, d), (a, b, d), (a, d, c)]
    elif shape == "sphere":
        segments, rings = 24, 12
        vertices = [(0, .5, 0)]
        for ring in range(1, rings):
            latitude = math.pi * ring / rings
            vertices += [(.5*math.sin(latitude)*math.cos(i*math.tau/segments),
                          .5*math.cos(latitude), .5*math.sin(latitude)*math.sin(i*math.tau/segments))
                         for i in range(segments)]
        bottom = len(vertices)
        vertices.append((0, -.5, 0))
        for i in range(segments):
            faces.append((0, 1+i, 1+(i+1) % segments))
            faces.append((bottom, 1+(rings-2)*segments+(i+1) % segments, 1+(rings-2)*segments+i))
        for ring in range(rings-2):
            for i in range(segments):
                a, b = 1+ring*segments+i, 1+ring*segments+(i+1) % segments
                c, d = a+segments, b+segments
                faces += [(a, c, d), (a, d, b)]
    else:
        raise ValueError("unsupported primitive")
    oriented = []
    for a, b, c in faces:
        normal = _cross(_sub(vertices[b], vertices[a]), _sub(vertices[c], vertices[a]))
        centroid = tuple((vertices[a][j]+vertices[b][j]+vertices[c][j])/3 for j in range(3))
        if sum(x*y for x, y in zip(normal, centroid)) < 0:
            b, c = c, b
        oriented.append((a, b, c))
    return tuple(vertices), tuple(oriented)


def _rotate(v, angle):
    x, y, z = v
    return (x*math.cos(angle)+z*math.sin(angle), y, -x*math.sin(angle)+z*math.cos(angle))


def _geometry(scene, phase=0.0):
    base_vertices, base_faces = _primitive(scene["shape"])
    vertices, faces, objects = [], [], []
    scale = _SCALES[scene["scale"]]
    for obj in range(scene["count"]):
        offset = len(vertices)
        x = (obj-(scene["count"]-1)/2)*1.48
        y, z, angle = scale*.5, 0.0, 0.0
        if scene["motion"] == "orbit":
            x += .28 * math.cos(math.tau*phase + obj*.9)
            z += .28 * math.sin(math.tau*phase + obj*.9)
        elif scene["motion"] == "bounce":
            y += .33 * (1 + math.sin(math.tau*phase + obj*.72))
        elif scene["motion"] == "spin":
            angle = math.tau*phase
        vertices.extend((v[0]*scale+x, v[1]*scale+y, v[2]*scale+z)
                        for v in (_rotate(v, angle) for v in base_vertices))
        faces.extend(tuple(i+offset for i in face) for face in base_faces)
        objects.append((offset, len(vertices), len(faces)-len(base_faces), len(faces)))
    return vertices, faces, objects


def _frame(scene, seed, phase=0, size=(512, 384)):
    # Orthographic geometry, painter ordering and Lambert-style face shading.
    width, height = size
    factor = 2
    width, height = width*factor, height*factor
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        t = y/max(height-1, 1)
        draw.line((0, y, width, y), fill=(int(12+13*t), int(23+19*t), int(43+20*t)))
    yaw = .47 + random.Random(seed).uniform(-.06, .06)
    pitch = .37
    unit = min(width/(scene["count"]*1.7+1.3), height*.36)

    def camera(v):
        x, y, z = _rotate(v, yaw)
        return x, y*math.cos(pitch)-z*math.sin(pitch), y*math.sin(pitch)+z*math.cos(pitch)

    def project(v):
        x, y, z = camera(v)
        return width*.5+x*unit, height*.73-y*unit

    for grid in range(-10, 11):
        draw.line((*project((grid*.5, -.008, -4)), *project((grid*.5, -.008, 4))), fill=(30, 51, 72), width=1)
        draw.line((*project((-5, -.008, grid*.5)), *project((5, -.008, grid*.5))), fill=(30, 51, 72), width=1)
    vertices, faces, objects = _geometry(scene, phase)
    shadow = Image.new("RGBA", image.size)
    sd = ImageDraw.Draw(shadow)
    for start, end, _, _ in objects:
        center = tuple(sum(v[k] for v in vertices[start:end])/(end-start) for k in range(3))
        px, py = project((center[0], 0, center[2]))
        radius = unit*_SCALES[scene["scale"]]*.68
        sd.ellipse((px-radius, py-radius*.3, px+radius, py+radius*.3), fill=(0, 3, 9, 155))
    image = Image.alpha_composite(image.convert("RGBA"), shadow.filter(ImageFilter.GaussianBlur(9))).convert("RGB")
    draw = ImageDraw.Draw(image)
    light = _unit((-.5, 1.0, 1.2))
    color = COLORS[scene["color"]]
    polygons = []
    for index, face in enumerate(faces):
        points = [vertices[i] for i in face]
        normal = _unit(_cross(_sub(points[1], points[0]), _sub(points[2], points[0])))
        if camera(normal)[2] <= 0:
            continue
        strength = .35 + .65*max(0, sum(a*b for a, b in zip(normal, light)))
        # A modest permanent surface variation makes rotation visible on round solids.
        strength *= .88 + .12*((index*17 % 19)/18)
        fill = tuple(min(255, int(c*strength+12)) for c in color)
        depth = sum(camera(point)[2] for point in points)/3
        polygons.append((depth, [project(point) for point in points], fill))
    for _, points, fill in sorted(polygons, key=lambda item: item[0]):
        draw.polygon(points, fill=fill)
    return image.resize(size, Image.Resampling.LANCZOS)


def _write_bytes(path: Path, content: bytes):
    descriptor, temporary = tempfile.mkstemp(prefix=".poseidon-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def _mesh_files(scene, directory, stem):
    vertices, faces, objects = _geometry(scene)
    normals = [_unit(_cross(_sub(vertices[b], vertices[a]), _sub(vertices[c], vertices[a]))) for a, b, c in faces]
    obj_path, mtl_path, gltf_path = (directory/(stem+extension) for extension in (".obj", ".mtl", ".gltf"))
    rows = ["# Poseidon procedural geometry; scene plan supplied separately", f"mtllib {mtl_path.name}"]
    rows += ["v " + " ".join(f"{x:.8f}" for x in vertex) for vertex in vertices]
    rows += ["vn " + " ".join(f"{x:.8f}" for x in normal) for normal in normals]
    for obj, (_, _, start, end) in enumerate(objects):
        rows += [f"o {scene['shape']}_{obj+1}", "usemtl poseidon_surface"]
        rows += ["f " + " ".join(f"{i+1}//{face_index+1}" for i in faces[face_index]) for face_index in range(start, end)]
    _write_bytes(obj_path, ("\n".join(rows)+"\n").encode())
    rgb = [channel/255 for channel in COLORS[scene["color"]]]
    material = "newmtl poseidon_surface\nKd " + " ".join(f"{x:.6f}" for x in rgb) + "\nKa 0.1 0.1 0.1\nKs 0.12 0.12 0.12\nNs 24\nd 1.0\nillum 2\n"
    _write_bytes(mtl_path, material.encode())
    # glTF uses separate triangle vertices so flat normals are preserved exactly.
    positions = [vertices[i] for face in faces for i in face]
    flat_normals = [normal for normal in normals for _ in range(3)]
    pos_data = struct.pack("<"+"f"*len(positions)*3, *(x for p in positions for x in p))
    norm_data = struct.pack("<"+"f"*len(flat_normals)*3, *(x for p in flat_normals for x in p))
    index_data = struct.pack("<"+"I"*len(positions), *range(len(positions)))
    binary = pos_data+norm_data+index_data
    views, offset = [], 0
    for chunk, target in ((pos_data, 34962), (norm_data, 34962), (index_data, 34963)):
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(chunk), "target": target})
        offset += len(chunk)
    gltf = {"asset": {"version": "2.0", "generator": "Supermix Poseidon procedural scene renderer v1"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0, "mode": 4}]}],
            "materials": [{"name": "poseidon_surface", "pbrMetallicRoughness": {"baseColorFactor": rgb+[1], "metallicFactor": 0.05, "roughnessFactor": 0.55}}],
            "buffers": [{"byteLength": len(binary), "uri": "data:application/octet-stream;base64,"+base64.b64encode(binary).decode()}],
            "bufferViews": views, "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": [min(v[k] for v in positions) for k in range(3)], "max": [max(v[k] for v in positions) for k in range(3)]},
                {"bufferView": 1, "componentType": 5126, "count": len(positions), "type": "VEC3"},
                {"bufferView": 2, "componentType": 5125, "count": len(positions), "type": "SCALAR"}],
            "extras": {"scene_plan": scene, "motion_note": "Static mesh at animation phase zero; motion is represented in GIF/video output."}}
    _write_bytes(gltf_path, _json_bytes(gltf))
    return {"obj": str(obj_path), "mtl": str(mtl_path), "gltf": str(gltf_path)}, len(vertices), len(faces)


def render_scene(scene: dict[str, Any], output_dir, kind: str = "image", seed: int = 42) -> dict[str, Any]:
    """Render an already planned scene to PNG, animated GIF, or OBJ/MTL/glTF.

    Optional imageio and imageio-ffmpeg installations additionally enable MP4.
    The backend is never downloaded or installed here. Returned paths are absolute.
    """
    scene = validate_scene(scene)
    if kind not in {"image", "video", "mesh"}:
        raise ValueError("kind must be image, video or mesh")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps([scene, seed, kind], sort_keys=True).encode()).hexdigest()[:14]
    stem = f"poseidon-{kind}-{digest}"
    image_path = directory/(stem+".png")
    preview = _frame(scene, seed)
    stream = BytesIO()
    preview.save(stream, format="PNG")
    _write_bytes(image_path, stream.getvalue())
    result = {"kind": kind, "scene": scene, "seed": seed, "renderer_version": 1,
              "generation_method": "procedural_scene_renderer", "paths": {"image": str(image_path), "preview": str(image_path)},
              "width": preview.width, "height": preview.height,
              "limits": "Four geometric primitives, six colors, one uniform size, one motion and one to three objects. No photorealism or neural pixel synthesis."}
    if kind == "video":
        frames = [_frame(scene, seed, i/24, (320, 240)) for i in range(24)]
        stream = BytesIO()
        frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False, disposal=2)
        gif_path = directory/(stem+".gif")
        _write_bytes(gif_path, stream.getvalue())
        with Image.open(gif_path) as gif:
            frame_count = gif.n_frames
        result.update({"frame_count": frame_count, "requested_frame_count": 24, "duration_seconds": 2.4, "fps": 10, "video_width": 320, "video_height": 240})
        result["paths"]["gif"] = str(gif_path)
        result["mp4_status"] = "unavailable_optional_encoder"
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg
            import numpy as np
            encoder = Path(imageio_ffmpeg.get_ffmpeg_exe())
            if encoder.is_file():
                mp4_path = directory/(stem+".mp4")
                with imageio.get_writer(str(mp4_path), fps=10, codec="libx264", quality=7, macro_block_size=16) as writer:
                    for frame in frames:
                        writer.append_data(np.asarray(frame))
                result["paths"]["mp4"] = str(mp4_path)
                result["mp4_status"] = "written"
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            result["mp4_status"] = "unavailable_optional_encoder"
            result["mp4_detail"] = type(error).__name__
    elif kind == "mesh":
        paths, vertices, triangles = _mesh_files(scene, directory, stem)
        result["paths"].update(paths)
        result.update({"vertex_count": vertices, "triangle_count": triangles, "object_count": scene["count"]})
    primary = {"image": "image", "video": "gif", "mesh": "gltf"}[kind]
    result["path"] = result["paths"][primary]
    manifest_path = directory/(stem+".json")
    result["paths"]["manifest"] = str(manifest_path)
    result["sha256"] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in result["paths"].items() if name != "manifest"}
    _write_bytes(manifest_path, _json_bytes(result))
    return result
