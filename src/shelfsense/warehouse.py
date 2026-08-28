from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from shelfsense.config import Config, load_config

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"

SQL_FILES = [
    "01_load_raw.sql",
    "02_sales_long.sql",
    "03_calendar_features.sql",
    "04_price_features.sql",
    "05_feature_matrix.sql",
]


def _quoted_list(values: list[str]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def build(config: Config | None = None) -> None:
    config = config or load_config()
    con = duckdb.connect(config.db_path)
    for fname in SQL_FILES:
        sql = (SQL_DIR / fname).read_text()
        sql = sql.format(
            stores=_quoted_list(config.stores),
            depts=_quoted_list(config.depts),
        )
        start = time.time()
        con.execute(sql)
        print(f"{fname}: {time.time() - start:.1f}s")
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    args = parser.parse_args()
    if args.command == "build":
        build()


if __name__ == "__main__":
    main()
