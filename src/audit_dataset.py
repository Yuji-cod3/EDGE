import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

FILES = [
    RAW_DIR / "EPL_2015_16.csv",
    RAW_DIR / "EPL_2024_25.csv",
]


def audit_file(path):
    print("\n" + "=" * 70)
    print(f"FILE: {path.name}")
    print("=" * 70)

    df = pd.read_csv(path)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nDate range:")
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        print(f"  First: {dates.min()}")
        print(f"  Last:  {dates.max()}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing):
        for column, count in missing.items():
            percentage = count / len(df) * 100
            print(f"  {column}: {count:,} ({percentage:.1f}%)")
    else:
        print("  None")

    print("\nDuplicate rows:")
    print(f"  {df.duplicated().sum():,}")

    if "HomeTeam" in df.columns and "AwayTeam" in df.columns:
        print("\nUnique teams:")
        teams = sorted(
            set(df["HomeTeam"].dropna()) |
            set(df["AwayTeam"].dropna())
        )

        print(f"  Count: {len(teams)}")
        print("  " + ", ".join(teams))

    print("\nOver/Under 2.5 related columns:")
    ou_columns = [
        column for column in df.columns
        if ">2.5" in column or "<2.5" in column
    ]

    if ou_columns:
        for column in ou_columns:
            available = df[column].notna().sum()
            percentage = available / len(df) * 100
            print(
                f"  {column}: "
                f"{available:,}/{len(df):,} "
                f"({percentage:.1f}%)"
            )
    else:
        print("  None")

    print("\nBasic match validation:")

    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

    for column in required:
        if column not in df.columns:
            print(f"  MISSING COLUMN: {column}")

    if "FTHG" in df.columns:
        print(
            f"  Missing home goals: "
            f"{df['FTHG'].isna().sum():,}"
        )

    if "FTAG" in df.columns:
        print(
            f"  Missing away goals: "
            f"{df['FTAG'].isna().sum():,}"
        )


def main():
    print("EDGE V0.1 — DATASET AUDIT")

    for file in FILES:
        if not file.exists():
            print(f"\nERROR: File not found: {file}")
            continue

        audit_file(file)


if __name__ == "__main__":
    main()
