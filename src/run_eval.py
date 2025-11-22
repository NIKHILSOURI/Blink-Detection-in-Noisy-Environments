import os, argparse, pandas as pd, numpy as np
from utils_io import load_cfg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def stability_cv(df, prefix):
    # intra-video proxy: use CV of IBI where available
    col = f"{prefix}_ibi_cv"
    return df[col].replace([np.inf, -np.inf], np.nan).dropna().mean()

def robustness_drop(df, metric="rate_per_min"):
    # drop = controlled mean - noisy mean (absolute) / controlled mean
    c = df[df["env"]=="controlled"][f"short_{metric}"].mean()
    n = df[df["env"]=="noisy"][f"short_{metric}"].mean()
    c_ext = df[df["env"]=="controlled"][f"ext_{metric}"].mean()
    n_ext = df[df["env"]=="noisy"][f"ext_{metric}"].mean()
    drop_short = (c - n)/max(1e-6, abs(c))
    drop_ext   = (c_ext - n_ext)/max(1e-6, abs(c_ext))
    return drop_short, drop_ext

def simple_auc(df, use_ext=False):
    feats = ["rate_per_min","mean_dur_s","ibi_mean_s","ibi_cv"]
    X = df[[("ext_" if use_ext else "short_")+f for f in feats]].fillna(0.0).values
    y = (df["env"]=="noisy").astype(int).values
    if len(np.unique(y))<2: return np.nan
    clf = LogisticRegression(max_iter=1000).fit(X,y)
    probs = clf.predict_proba(X)[:,1]
    return roc_auc_score(y, probs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)
    csv = os.path.join(cfg["paths"]["outputs"], "features.csv")
    df = pd.read_csv(csv)
    st_short = stability_cv(df, "short")
    st_ext   = stability_cv(df, "ext")
    drop_s, drop_e = robustness_drop(df, metric="rate_per_min")
    auc_short = simple_auc(df, use_ext=False)
    auc_ext   = simple_auc(df, use_ext=True)
    print(f"Stability (mean IBI CV) — short: {st_short:.3f} | extended: {st_ext:.3f}")
    print(f"Robustness drop (rate) — short: {drop_s:.3f} | extended: {drop_e:.3f} (lower is better)")
    print(f"Controlled vs Noisy AUC — short: {auc_short:.3f} | extended: {auc_ext:.3f} (higher is better)")

if __name__ == "__main__":
    main()
