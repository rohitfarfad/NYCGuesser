"""Audit extracted official inputs without training models or changing raw data.

Uses Pillow, NumPy and SciPy already available in the inspected environment.
Reads the extraction manifest and decodes at most two images concurrently.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import re
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


DIRECTIONS = ("north", "east", "south", "west")
BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
SUBMISSION_COLUMNS = ["sample_id", "borough", "latitude", "longitude"]
IMAGE_NAME = re.compile(r"img_(\d{6})_(north|east|south|west)\.jpg")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        require(len(columns) == len(set(columns)), f"Duplicate columns: {path}")
        rows = []
        for line, values in enumerate(reader, 2):
            require(len(values) == len(columns), f"Field count at {path}:{line}")
            require(all(value.strip() for value in values), f"Empty field at {path}:{line}")
            rows.append(dict(zip(columns, values)))
    return columns, rows


def audit_q1(root):
    records = []
    for name in ("bicycles", "street", "bird"):
        with Image.open(root / "Q1 Data" / f"{name}_masked.png") as image:
            require(image.size == (224, 224) and image.mode == "RGB", f"Q1 image: {name}")
            pixels = np.asarray(image)
        with Image.open(root / "Q1 Data" / f"{name}_mask.png") as mask:
            require(mask.size == (224, 224) and mask.mode == "L", f"Q1 mask: {name}")
            binary = np.asarray(mask)
        require(set(np.unique(binary)) == {0, 255}, f"Nonbinary Q1 mask: {name}")
        patches = binary.reshape(14, 16, 14, 16).transpose(0, 2, 1, 3)
        uniform = np.all(patches == patches[:, :, :1, :1], axis=(2, 3))
        require(bool(uniform.all()), f"Q1 mask crosses the 16-pixel patch grid: {name}")
        missing = binary == 255
        missing_patches = int((patches[:, :, 0, 0] == 255).sum())
        require(missing_patches == 98, f"Q1 mask is not 50% missing: {name}")
        records.append({"name": name, "size": [224, 224], "mask_values": [0, 255],
                        "missing_patches": missing_patches, "total_patches": 196,
                        "missing_fraction": float(missing.mean()),
                        "masked_pixel_colors": np.unique(pixels[missing], axis=0).tolist()})
    return records


def audit_q2(root):
    records = []
    for name in ("coffee", "cat", "astronaut"):
        path = root / "Q2 Data" / f"{name}_noisy.png"
        with Image.open(path) as image:
            pixels = np.asarray(image)
            require(image.mode == "RGB", f"Q2 must be RGB: {path}")
            row_mean = pixels.astype(np.float32).mean(axis=(1, 2))
            records.append({"path": str(path.relative_to(root)), "size": list(image.size),
                            "mode": image.mode, "minimum": int(pixels.min()),
                            "maximum": int(pixels.max()),
                            "channel_means": pixels.mean(axis=(0, 1)).tolist(),
                            "fraction_at_0_or_255": float(((pixels == 0) | (pixels == 255)).mean()),
                            "median_adjacent_row_mean_difference": float(np.median(np.abs(np.diff(row_mean))))})
    return {"images": records, "clean_reference_images_supplied": False,
            "interpretation": "Intensity summaries are descriptive; scene content confounds noise estimates."}


def audit_q3(root):
    pair_root = root / "Q3 Data" / "q3"
    folders = sorted(path for path in pair_root.iterdir() if path.is_dir())
    require({path.name for path in folders} == {f"eval_{i:03d}" for i in range(1, 33)},
            "Q3 must contain eval_001 through eval_032")
    records = []
    for folder in folders:
        require({path.name for path in folder.iterdir()} == {"reference.png", "scene.png"},
                f"Unexpected Q3 pair contents: {folder}")
        with Image.open(folder / "reference.png") as reference, Image.open(folder / "scene.png") as scene:
            require(scene.size == (800, 600), f"Q3 scene dimensions: {folder}")
            records.append({"pair": folder.name, "reference_size": list(reference.size),
                            "scene_size": list(scene.size), "reference_mode": reference.mode,
                            "scene_mode": scene.mode})
    return {"root": str(pair_root.relative_to(root)), "pairs": len(records), "records": records,
            "ground_truth_boxes_or_absence_labels_supplied": False}


def audit_kaggle(root):
    data = root / "Data"
    train_columns, train = read_csv(data / "train.csv")
    sample_columns, sample = read_csv(data / "sample_submission.csv")
    expected_train = ["sample_id", *(f"image_{d}" for d in DIRECTIONS),
                      "borough", "latitude", "longitude"]
    require(train_columns == expected_train, f"Unexpected train columns: {train_columns}")
    require(sample_columns == SUBMISSION_COLUMNS, f"Unexpected sample columns: {sample_columns}")
    split_info = {}
    for split, rows, count in (("train", train, 11864), ("test", sample, 7908)):
        ids = [int(row["sample_id"]) for row in rows]
        require(len(ids) == count and len(set(ids)) == count and set(ids) == set(range(count)),
                f"Incorrect {split} sample IDs/count")
        require(all(row["borough"] in BOROUGHS for row in rows), f"Invalid {split} borough")
        coordinates = np.array([[float(row["latitude"]), float(row["longitude"])] for row in rows])
        require(bool(np.isfinite(coordinates).all()), f"Nonfinite {split} coordinates")
        require(bool((np.abs(coordinates) <= [90, 180]).all()), f"Invalid {split} coordinates")
        files = list((data / split).iterdir())
        groups = defaultdict(set)
        for path in files:
            match = IMAGE_NAME.fullmatch(path.name)
            require(path.is_file() and match is not None, f"Unexpected {split} input: {path}")
            groups[int(match[1])].add(match[2])
        require(set(groups) == set(ids) and len(files) == count * 4,
                f"{split} image IDs/count do not match CSV")
        require(all(directions == set(DIRECTIONS) for directions in groups.values()),
                f"Incomplete four-view group: {split}")
        if split == "train":
            for row in rows:
                for direction in DIRECTIONS:
                    expected = f"img_{int(row['sample_id']):06d}_{direction}.jpg"
                    require(row[f"image_{direction}"] == expected,
                            f"CSV-to-image mapping mismatch: {row['sample_id']} {direction}")
        split_info[split] = {"samples": count, "images": len(files),
                             "all_four_view_groups_complete": True,
                             "borough_counts": dict(Counter(row["borough"] for row in rows)),
                             "unique_coordinate_pairs": len(set(map(tuple, coordinates)))}
    coordinates = np.array([[float(row["latitude"]), float(row["longitude"])] for row in train])
    duplicates = Counter(map(tuple, coordinates))
    radians = np.deg2rad(coordinates)
    lat, lon = radians.T
    xyz = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
    chord, _ = cKDTree(xyz).query(xyz, k=2)
    nearest_m = 2 * 6371008.8 * np.arcsin(np.clip(chord[:, 1] / 2, 0, 1))
    by_borough = {}
    for borough in BOROUGHS:
        points = coordinates[[row["borough"] == borough for row in train]]
        by_borough[borough] = {"samples": len(points), "latitude_range_deg": [float(points[:, 0].min()), float(points[:, 0].max())],
                               "longitude_range_deg": [float(points[:, 1].min()), float(points[:, 1].max())]}
    return {"train_csv": "Data/train.csv", "sample_submission_csv": "Data/sample_submission.csv",
            "train_columns": train_columns, "submission_columns": sample_columns,
            "splits": split_info, "training_coordinate_ranges": {
                "latitude_deg": [float(coordinates[:, 0].min()), float(coordinates[:, 0].max())],
                "longitude_deg": [float(coordinates[:, 1].min()), float(coordinates[:, 1].max())]},
            "borough_summaries": by_borough,
            "duplicate_training_coordinate_groups": sum(n > 1 for n in duplicates.values()),
            "training_rows_in_duplicate_coordinate_groups": sum(n for n in duplicates.values() if n > 1),
            "nearest_other_training_location_m": {
                "quantiles": dict(zip(["min", "p10", "median", "p90", "max"], np.quantile(nearest_m, [0, .1, .5, .9, 1]).tolist())),
                "within_10m": int((nearest_m <= 10).sum()), "within_50m": int((nearest_m <= 50).sum()),
                "within_100m": int((nearest_m <= 100).sum()),
                "earth_radius_m_local_descriptive_choice": 6371008.8},
            "sample_submission_unique_values": {column: sorted({row[column] for row in sample})[:10] for column in SUBMISSION_COLUMNS[1:]},
            "sample_submission_is_template_not_ground_truth": True,
            "numeric_ids_overlap_between_train_and_test": True,
            "split_identity_rule": "Use split plus sample_id; equal numeric IDs across splits identify different inputs.",
            "packaged_rules_readme_or_scorer": None}


def inspect_image(root, record):
    path = root / record["path"]
    try:
        raw = path.read_bytes()
        require(len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"],
                f"Input bytes differ from extraction manifest: {record['path']}")
        with Image.open(io.BytesIO(raw)) as image:
            image.load()  # Full decode; truncated JPEG loading is not enabled.
            digest = hashlib.sha256(f"{image.mode}:{image.size}".encode())
            digest.update(image.tobytes())
            return {"path": record["path"], "size": list(image.size), "mode": image.mode,
                    "format": image.format, "pixel_sha256": digest.hexdigest(), "file_sha256_matches": True}
    except Exception as exc:
        return {"path": record["path"], "error": f"{type(exc).__name__}: {exc}"}


def audit_images(root, manifest):
    records = [record for record in manifest if Path(record["path"]).suffix.lower() in (".jpg", ".png")]
    require(len(records) == 79161, f"Unexpected total image count: {len(records)}")
    counts = defaultdict(Counter)
    hashes = defaultdict(list)
    pixel_by_path = {}
    errors = []
    started = time.monotonic()
    with (root / "outputs/image_audit.jsonl").open("w", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0, len(records), 128):
            for result in pool.map(lambda record: inspect_image(root, record), records[start:start + 128]):
                stream.write(json.dumps(result) + "\n")
                if "error" in result:
                    errors.append(result)
                    continue
                path = result["path"]
                section = path.split("/")[0]
                if section == "Data":
                    section += "/" + path.split("/")[1]
                    if result["size"] != [640, 640] or result["mode"] != "RGB" or result["format"] != "JPEG":
                        errors.append({"path": path, "error": "Unexpected Kaggle image dimensions/mode/format"})
                counts[section][f"{result['size'][0]}x{result['size'][1]} {result['mode']} {result['format']}"] += 1
                hashes[result["pixel_sha256"]].append(path)
                pixel_by_path[path] = result["pixel_sha256"]
            done = min(start + 128, len(records))
            if done % 4096 == 0 or done == len(records):
                stream.flush()
                print(json.dumps({"event": "full_image_decode_progress", "images": done, "of": len(records),
                                  "elapsed_seconds": round(time.monotonic() - started, 1), "errors": len(errors)}), flush=True)
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split = [paths for paths in duplicates if any(path.startswith("Data/train/") for path in paths)
                   and any(path.startswith("Data/test/") for path in paths)]
    signatures = defaultdict(list)
    if not errors:
        for split, count in (("train", 11864), ("test", 7908)):
            for sample_id in range(count):
                signature = tuple(pixel_by_path[f"Data/{split}/img_{sample_id:06d}_{d}.jpg"] for d in DIRECTIONS)
                signatures[signature].append({"split": split, "sample_id": sample_id})
    repeated_locations = [values for values in signatures.values() if len(values) > 1]
    return {"decoded_images": len(records) - sum("error" in item for item in errors), "errors": errors,
            "size_mode_format_counts": {key: dict(value) for key, value in counts.items()},
            "decoded_pixel_duplicate_groups": len(duplicates),
            "images_in_decoded_pixel_duplicate_groups": sum(map(len, duplicates)),
            "cross_train_test_pixel_duplicate_groups": len(cross_split),
            "cross_train_test_duplicate_examples": cross_split[:10],
            "identical_ordered_four_view_location_groups": repeated_locations,
            "duplicate_scope": "Exact decoded pixels and identical ordered four-view signatures only; near duplicates remain unaudited.",
            "elapsed_seconds": round(time.monotonic() - started, 2)}


def main(root):
    started = time.monotonic()
    with (root / "outputs/input_manifest.jsonl").open() as stream:
        manifest = [json.loads(line) for line in stream]
    report = {"timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat(),
              "python_packages": {name: version(name) for name in ("Pillow", "numpy", "scipy")},
              "q1": audit_q1(root), "q2": audit_q2(root), "q3": audit_q3(root),
              "kaggle": audit_kaggle(root)}
    print(json.dumps({"event": "schema_and_small_input_checks_complete", "q1_pairs": len(report["q1"]),
                      "q2_images": len(report["q2"]["images"]), "q3_pairs": report["q3"]["pairs"],
                      "kaggle": report["kaggle"]["splits"]}), flush=True)
    report["image_integrity"] = audit_images(root, manifest)
    report["status"] = "passed" if not report["image_integrity"]["errors"] else "failed"
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["free_disk_bytes"] = shutil.disk_usage(root).free
    report["audit_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (root / "outputs/data_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"event": "audit_complete", "status": report["status"],
                      "elapsed_seconds": report["elapsed_seconds"], "image_integrity": report["image_integrity"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    raise SystemExit(main(parser.parse_args().root.resolve()))
