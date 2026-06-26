"""Convert PASCAL VOC XML annotations to PROBE JSONL manifests.

Class mapping (RDD2022 → 5-class PROBE paper setting):
  D00/D01 → 0 (longitudinal crack)
  D10/D11 → 1 (transverse crack)
  D20     → 2 (alligator crack)
  D40/D43/D44/D50 → 3 (pothole / manhole / deformation)
  Repair  → 4

Output per country: {country}_train.jsonl, {country}_val.jsonl, {country}_unlabeled.jsonl
"""

import os, json, random
from pathlib import Path
from xml.etree import ElementTree as ET

IMAGES_DIR = Path("/root/NEW-PROBE/images")
OUTPUT_DIR = Path("/root/NEW-PROBE/data")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_MAP = {
    "D00": 0, "D01": 0,                              # longitudinal crack
    "D10": 1, "D11": 1,                               # transverse crack
    "D20": 2,                                          # alligator crack
    "D40": 3, "D43": 3, "D44": 3, "D50": 3,          # pothole / others
    "Repair": 4,                                       # repair
    "D0w0": 0,                                        # typo → crack
}

# Collect per-country samples
countries = {}
for xml_path in sorted(IMAGES_DIR.glob("*.xml")):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except:
        continue

    filename = root.findtext("filename", xml_path.stem + ".jpg")
    prefix = filename.split("_")[0]
    if prefix == "China":
        prefix = "China_MotorBike"

    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        x1 = int(float(bbox.findtext("xmin", "0")))
        y1 = int(float(bbox.findtext("ymin", "0")))
        x2 = int(float(bbox.findtext("xmax", "0")))
        y2 = int(float(bbox.findtext("ymax", "0")))
        label = CLASS_MAP.get(name, 0)
        boxes.append([x1, y1, x2, y2])
        labels.append(label)

    img_path = IMAGES_DIR / filename
    if not img_path.exists():
        continue

    countries.setdefault(prefix, []).append({
        "image": filename,
        "boxes": boxes,
        "labels": labels,
    })

for country, samples in sorted(countries.items()):
    random.seed(42)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    unlabeled = samples[n_train + n_val:]

    with open(OUTPUT_DIR / f"{country}_train.jsonl", "w") as f:
        for s in train:
            f.write(json.dumps(s) + "\n")

    with open(OUTPUT_DIR / f"{country}_val.jsonl", "w") as f:
        for s in val:
            f.write(json.dumps(s) + "\n")

    with open(OUTPUT_DIR / f"{country}_unlabeled.jsonl", "w") as f:
        for s in unlabeled:
            f.write(json.dumps({"image": s["image"], "boxes": [], "labels": []}) + "\n")

    print(f"{country}: total={n} train={n_train} val={n_val} unlabeled={n - n_train - n_val}")

print("\nDone →", OUTPUT_DIR)
