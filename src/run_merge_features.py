import argparse, os, glob
import pandas as pd

def main():
    ap = argparse.ArgumentParser(
        description="Merge multiple feature CSVs (vertical concat, align by column names)."
    )
    ap.add_argument(
        "--inputs", nargs="*", default=None,
        help="Explicit list of CSV paths to merge (space-separated)."
    )
    ap.add_argument(
        "--glob", default=None,
        help="Glob pattern to find CSVs (e.g., 'outputs/*.csv' or 'outputs\\features*.csv')."
    )
    ap.add_argument(
        "--out", default="outputs/features_all.csv",
        help="Output CSV path (default: outputs/features_all.csv)."
    )
    ap.add_argument(
        "--drop-duplicates", action="store_true",
        help="Drop duplicate rows after merging."
    )
    args = ap.parse_args()

    files = []
    if args.inputs:
        files.extend(args.inputs)
    if args.glob:
        files.extend(glob.glob(args.glob))
    files = [f for f in files if os.path.isfile(f)]
    files = sorted(set(files))

    if not files:
        print("No input CSVs found. Use --inputs and/or --glob.")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"[+] Loaded {f} with shape {df.shape}")
        except Exception as e:
            print(f"[!] Skipped {f}: {e}")

    if not dfs:
        print("No readable CSVs.")
        return

    # Align columns by union; fill missing with NaN to avoid misaligned concat
    all_cols = sorted(set().union(*[set(d.columns) for d in dfs]))
    dfs = [d.reindex(columns=all_cols) for d in dfs]

    out_df = pd.concat(dfs, ignore_index=True)
    if args.drop_duplicates:
        before = len(out_df)
        out_df = out_df.drop_duplicates()
        print(f"[i] Dropped duplicates: {before - len(out_df)} rows")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[✓] Merged {len(files)} file(s) -> {args.out} with shape {out_df.shape}")

if __name__ == "__main__":
    main()
