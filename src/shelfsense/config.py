from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass(frozen=True)
class Costs:
    margin_rate: float
    holding_rate: float


@dataclass(frozen=True)
class Promo:
    discount_threshold: float
    baseline_weeks: int


@dataclass(frozen=True)
class Config:
    data_dir: str
    db_path: str
    stores: list[str]
    depts: list[str]
    horizon: int
    train_days: int
    backtest_origins: list[int]
    costs: Costs
    promo: Promo
    seed: int


def load_config(path: Path = CONFIG_PATH) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        data_dir=raw["data_dir"],
        db_path=raw["db_path"],
        stores=raw["stores"],
        depts=raw["depts"],
        horizon=raw["horizon"],
        train_days=raw["train_days"],
        backtest_origins=raw["backtest_origins"],
        costs=Costs(**raw["costs"]),
        promo=Promo(**raw["promo"]),
        seed=raw["seed"],
    )


if __name__ == "__main__":
    config = load_config()
    print(config)
