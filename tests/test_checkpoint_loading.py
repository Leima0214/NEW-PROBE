import importlib.util
from pathlib import Path

import torch
from torch import nn

path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
spec = importlib.util.spec_from_file_location("probe_train", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_checkpoint_loading_skips_changed_position_shape():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    state = model.state_dict()
    state["0.weight"] = torch.zeros(3, 2)

    compatible, skipped = module._compatible_state_dict(model, state)

    assert "0.weight" in skipped
    assert "0.weight" not in compatible
    assert "1.weight" in compatible
