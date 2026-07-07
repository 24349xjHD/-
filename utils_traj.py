# scripts/utils_traj.py
import re
import numpy as np

coord_pattern = r"\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\)"

def extract_coords(text: str):
    matches = re.findall(coord_pattern, text)
    return [(float(x), float(y)) for x, y in matches]

def compute_ade_fde(pred_traj, true_traj):
    pred = np.array(pred_traj, dtype=np.float32)
    true = np.array(true_traj, dtype=np.float32)
    ade = np.mean(np.linalg.norm(pred - true, axis=1))
    fde = np.linalg.norm(pred[-1] - true[-1])
    return float(ade), float(fde)

def coords_to_str(traj):
    return ";".join([f"({x:.4f},{y:.4f})" for x, y in traj])

def str2bool(s: str) -> bool:
    s = s.strip().lower()
    return s in ("1", "true", "yes", "y", "t")

def compute_recall_like_from_fde(fde_list, thresholds=(1.0, 2.0, 3.0)):
    """
    recall-like 指标：命中比例 P(FDE <= tau)
    """
    metrics = {}
    for t in thresholds:
        if len(fde_list) == 0:
            metrics[f"hit_rate_fde_le_{t}"] = None
        else:
            metrics[f"hit_rate_fde_le_{t}"] = float(np.mean(np.array(fde_list) <= t))
    return metrics