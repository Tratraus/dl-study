"""
week14/tools/checkpoint_io.py
------------------------------------------------------------
统一Checkpoint存取格式，杜绝多格式兼容债务。
全项目只允许通过 save_checkpoint / load_checkpoint 两个函数做存取。
格式固定为字典包裹，且必须包含 seed 和 config 字段。
"""

from datetime import datetime
from pathlib import Path

import torch

REQUIRED_FIELDS = ["seed", "config"]


def save_checkpoint(model, path, seed: int, config: dict, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "config": config,
        "timestamp": datetime.now().isoformat(),
        **extra,
    }
    torch.save(payload, path)
    print(f"[checkpoint_io] 已保存: {path}  (seed={seed})")


def load_checkpoint(model, path, device="cpu"):
    path = Path(path)
    ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"[checkpoint_io] 加载失败：{path} 不是本项目规范格式。"
            f"请确认该文件通过 save_checkpoint() 保存（禁止直接torch.save裸state_dict）。"
        )

    missing = [f for f in REQUIRED_FIELDS if f not in ckpt]
    if missing:
        raise ValueError(f"[checkpoint_io] 加载失败：{path} 缺少必填字段 {missing}。")

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[checkpoint_io] 已加载: {path}  (seed={ckpt['seed']}, "
          f"保存时间={ckpt.get('timestamp', '未知')})")
    return model, ckpt
