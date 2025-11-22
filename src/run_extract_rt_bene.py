import os, argparse, numpy as np, pandas as pd
from utils_io import load_cfg

COMMON_LABEL_NAMES = [
    "label","blink","blink_label","is_blink","isBlink","blinkState",
    "state","y","gt","target","blink_gt","blink_gt_label"
]

def coerce_to_binary(series):
    s = series.dropna()

    # Already ints 0/1?
    if set(s.unique()).issubset({0,1}):
        return series.astype(int)

    # Strings open/closed (case-insensitive)
    lower = s.astype(str).str.lower().str.strip()
    if set(lower.unique()).issubset({"open","closed","blink","noblink","no_blink","yes","no","true","false"}):
        m = {"open":0,"closed":1,"blink":1,"noblink":0,"no_blink":0,"yes":1,"no":0,"true":1,"false":0}
        return lower.map(m).astype(int).reindex(series.index).fillna(0).astype(int)

    # Floats/probabilities -> threshold at 0.5
    try:
        f = pd.to_numeric(s, errors="coerce")
        if f.notna().mean() > 0.9:
            binarized = (f >= 0.5).astype(int)
            return binarized.reindex(series.index).fillna(0).astype(int)
    except Exception:
        pass

    # As a fallback, try anything with <= 3 distinct values (e.g., 0/1/2) -> map >0 to 1
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().all() and len(set(vals.unique())) <= 3:
        return (vals > 0).astype(int).reindex(series.index).fillna(0).astype(int)

    raise ValueError("Could not coerce labels to binary 0/1.")


def find_label_column(df, user_col=None, debug=False):
    # User override wins
    if user_col and user_col in df.columns:
        if debug: print(f"[debug] Using user-specified label column: {user_col}")
        return user_col

    # Try common names first (case-insensitive)
    lowmap = {c.lower(): c for c in df.columns}
    for name in COMMON_LABEL_NAMES:
        if name in lowmap:
            if debug: print(f"[debug] Found common label column: {lowmap[name]}")
            return lowmap[name]

    # Otherwise, search for a column that looks binary/prob
    candidates = []
    for c in df.columns:
        s = df[c]
        uniq = s.dropna().astype(str).str.lower().str.strip().unique()
        if set(uniq).issubset({"0","1","open","closed","blink","noblink","no_blink","yes","no","true","false"}):
            candidates.append(c)
            continue
        # small set of ints/floats also a candidate
        try:
            x = pd.to_numeric(s, errors="coerce").dropna()
            if len(x) and (len(set(x.unique())) <= 3 or (x.between(0,1).mean() > 0.8)):
                candidates.append(c)
        except Exception:
            pass

    if debug: print(f"[debug] Candidate label columns by heuristic: {candidates}")
    if candidates:
        return candidates[0]
    return None


def seq_to_blinks(labels):
    blinks = []
    start = None
    for i, v in enumerate(labels):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            blinks.append((start, i-1)); start = None
    if start is not None:
        blinks.append((start, len(labels)-1))
    return blinks


def blink_features_from_labels(labels, fps=30.0):
    if len(labels) == 0:
        return dict(rate_per_min=0.0, mean_dur_s=0.0, ibi_mean_s=np.nan, ibi_cv=np.nan)

    blinks = seq_to_blinks(labels)
    if not blinks:
        dur_s = len(labels)/max(fps,1e-6)
        return dict(rate_per_min=0.0, mean_dur_s=0.0, ibi_mean_s=np.nan, ibi_cv=np.nan)

    durs = [(e - s + 1)/fps for (s,e) in blinks]
    onsets = [s for (s,_) in blinks]

    if len(onsets) > 1:
        ibis = np.diff(onsets) / fps
        ibi_mean = float(np.mean(ibis))
        ibi_cv   = float(np.std(ibis)/ibi_mean) if ibi_mean>0 else np.nan
    else:
        ibi_mean, ibi_cv = np.nan, np.nan

    dur_s = len(labels)/max(fps,1e-6)
    rate = (len(blinks)/dur_s) * 60.0
    return dict(rate_per_min=rate, mean_dur_s=float(np.mean(durs)), ibi_mean_s=ibi_mean, ibi_cv=ibi_cv)


def load_rt_bene_csvs(root):
    # any *_blink_labels.csv or *.csv in that root
    all_csvs = []
    for f in os.listdir(root):
        if f.lower().endswith(".csv"):
            all_csvs.append(os.path.join(root, f))
    return sorted(all_csvs)


def parse_one_csv(path, label_col=None, fps=30.0, debug=False):
    # Flexible reader: handles commas/semicolons/spaces/tabs
    try:
        df = pd.read_csv(path, engine="python")
    except Exception:
        # try semicolon
        df = pd.read_csv(path, sep=";", engine="python")

    if debug:
        print(f"[debug] {path} columns: {list(df.columns)}")
        print(f"[debug] head:\n{df.head(3)}")

    col = find_label_column(df, user_col=label_col, debug=debug)
    if col is None:
        if debug: print(f"[warn] No label-like column found in: {path}")
        return None

    try:
        y = coerce_to_binary(df[col])
    except Exception as e:
        if debug: print(f"[warn] Could not coerce labels in {path}: {e}")
        return None

    return y.astype(int).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--rt_root", default=None, help="RT-BENE root (folder that has the sXXX_*.csv files)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--label_col", default=None, help="Force a specific column name if autodetect fails")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    rt_root = args.rt_root or cfg["paths"]["controlled_dir"]
    out = args.out or os.path.join(cfg["paths"]["outputs"], "features_rt_bene.csv")
    os.makedirs(cfg["paths"]["outputs"], exist_ok=True)

    csvs = load_rt_bene_csvs(rt_root)
    if args.debug:
        print(f"[debug] Searching CSVs under: {rt_root}")
        print(f"[debug] Found {len(csvs)} CSV(s).")

    rows = []
    for c in csvs:
        labels = parse_one_csv(c, label_col=args.label_col, fps=args.fps, debug=args.debug)
        if labels is None or len(labels) == 0:
            if args.debug:
                print(f"[warn] No usable binary labels parsed: {c}")
            continue

        short_n = int(cfg["baseline"]["short_sec"] * args.fps)
        ext_n   = int(cfg["baseline"]["extended_sec"] * args.fps)
        short_labels = labels[:min(short_n, len(labels))]
        ext_labels   = labels[:min(ext_n, len(labels))]

        f_short = blink_features_from_labels(short_labels, fps=args.fps)
        f_ext   = blink_features_from_labels(ext_labels,   fps=args.fps)

        rows.append({
            "video": os.path.basename(c),
            "env": "controlled",
            **{f"short_{k}": v for k,v in f_short.items()},
            **{f"ext_{k}": v for k,v in f_ext.items()},
        })

    if len(rows) == 0:
        print("Parsed zero files. Try again with --debug and/or --label_col <column_name>.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Saved {out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
