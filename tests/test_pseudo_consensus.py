import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts" / "build_pseudo_dataset.py"
spec = importlib.util.spec_from_file_location("build_pseudo_dataset", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
box_iou = module.box_iou
consensus_rows = module.consensus_rows


def test_consensus_requires_class_and_localization_agreement():
    first = [
        (0, (0.5, 0.5, 0.2, 0.2), 0.9),
        (1, (0.2, 0.2, 0.1, 0.1), 0.8),
    ]
    second = [
        (0, (0.51, 0.5, 0.2, 0.2), 0.8),
        (2, (0.2, 0.2, 0.1, 0.1), 0.9),
    ]

    agreed = consensus_rows(first, second, min_confidence=0.5, min_iou=0.6)

    assert len(agreed) == 1
    assert agreed[0][0] == 0
    assert box_iou(first[0][1], second[0][1]) > 0.6
