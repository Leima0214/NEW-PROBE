import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch

ckpt = torch.load("checkpoints/probe_det_best.pt", weights_only=False)
det_keys = [k for k in ckpt["model"] if "detection_head" in k]
print(f"Total detection_head keys: {len(det_keys)}")
for k in det_keys[:5]:
    print(k)
