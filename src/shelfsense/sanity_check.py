from __future__ import annotations

import pandas as pd

from shelfsense.config import load_config


def main() -> None:
    config = load_config()
    files = [
        "calendar.csv",
        "sales_train_evaluation.csv",
        "sell_prices.csv",
        "sample_submission.csv",
    ]
    for fname in files:
        path = f"{config.data_dir}/{fname}"
        df = pd.read_csv(path)
        print(f"{fname}: {len(df):,} rows, {len(df.columns)} cols")
        print(f"  nulls: {df.isnull().sum().sum():,}")
        if "date" in df.columns:
            print(f"  date range: {df['date'].min()} -> {df['date'].max()}")


if __name__ == "__main__":
    main()
