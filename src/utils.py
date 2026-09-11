from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    for relative in (
        "data/raw",
        "data/processed",
        "models",
        "results/tables",
        "results/figures",
        "results/metrics",
        "paper/figures",
        "paper/tables",
        "paper/references",
    ):
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

