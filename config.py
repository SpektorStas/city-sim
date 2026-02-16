from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class Scenario:
    raw: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        return cls(raw=load_yaml(path))
