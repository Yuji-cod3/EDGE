import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")


def season_from_filename(path):
    return path.stem.replace("EPL_", "")


def coverage(df, columns):
    available = [c for c in columns if c in df.columns]

    if not available:
        return None

    mask = df[available].notna().any(axis=1)

    return {
        "columns": available,
        "matches": int(mask.sum()),
        "total": len(df),
        "percentage": mask.mean() * 100,
    }


def main():
    print("\nEDGE — ODDS COVERAGE AUDIT")
    print("=" * 85)

    for file in sorted(RAW_DIR.glob("EPL_*.csv")):

        df = pd.read_csv(file)
        season = season_from_filename(file)

        print(f"\n{season}")
        print("-" * 85)

        groups = {
            "Average O/U pre-closing": [
                "Avg>2.5",
                "Avg<2.5",
                "BbAv>2.5",
                "BbAv<2.5",
            ],

            "Maximum O/U pre-closing": [
                "Max>2.5",
                "Max<2.5",
                "BbMx>2.5",
                "BbMx<2.5",
            ],

            "Bet365 O/U pre-closing": [
                "B365>2.5",
                "B365<2.5",
            ],

            "Average O/U closing": [
                "AvgC>2.5",
                "AvgC<2.5",
            ],

            "Maximum O/U closing": [
                "MaxC>2.5",
                "MaxC<2.5",
            ],

            "Bet365 O/U closing": [
                "B365C>2.5",
                "B365C<2.5",
            ],
        }

        for name, columns in groups.items():

            result = coverage(df, columns)

            if result is None:
                print(f"{name:<32} NOT AVAILABLE")
            else:
                print(
                    f"{name:<32} "
                    f"{result['matches']:>3}/{result['total']} "
                    f"({result['percentage']:>5.1f}%) "
                    f"{result['columns']}"
                )


if __name__ == "__main__":
    main()
