import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

FEATURES = ["rate_per_min","mean_dur_s","ibi_mean_s","ibi_cv"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_summary_csv(df, outdir):
    # Per-env means (short vs ext)
    parts = []
    for env in sorted(df["env"].dropna().unique()):
        row = {"env": env}
        for f in FEATURES:
            row[f"short_{f}_mean"] = df.loc[df["env"]==env, f"short_{f}"].mean()
            row[f"ext_{f}_mean"]   = df.loc[df["env"]==env, f"ext_{f}"].mean()
        parts.append(row)
    summary = pd.DataFrame(parts)
    summary.to_csv(os.path.join(outdir, "summary_stats.csv"), index=False)
    return summary

def plot_box(df, feat, outdir):
    plt.figure(figsize=(6,4))
    data = [
        df[f"short_{feat}"].dropna().values,
        df[f"ext_{feat}"].dropna().values
    ]
    plt.boxplot(data, labels=["short","extended"], showfliers=False)
    plt.title(f"{feat} — all env combined")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"box_{feat}.png"), dpi=150)
    plt.close()

    # By env
    envs = sorted(df["env"].dropna().unique())
    for env in envs:
        sub = df[df["env"]==env]
        if sub.empty: continue
        plt.figure(figsize=(6,4))
        data = [sub[f"short_{feat}"].dropna().values, sub[f"ext_{feat}"].dropna().values]
        plt.boxplot(data, labels=["short","extended"], showfliers=False)
        plt.title(f"{feat} — {env}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"box_{feat}_{env}.png"), dpi=150)
        plt.close()

def plot_stability(df, outdir):
    # Mean IBI CV (lower is better); compare short vs ext
    short_cv = df["short_ibi_cv"].replace([np.inf,-np.inf], np.nan).dropna().mean()
    ext_cv   = df["ext_ibi_cv"].replace([np.inf,-np.inf], np.nan).dropna().mean()
    plt.figure(figsize=(5,4))
    plt.bar(["short","extended"], [short_cv, ext_cv])
    plt.ylabel("Mean IBI CV")
    plt.title("Stability (lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "stability_bar.png"), dpi=150)
    plt.close()

def plot_robustness(df, outdir):
    # Robustness drop: (controlled mean - noisy mean)/controlled mean, using rate_per_min
    def mean_safe(s): return np.nanmean(s.replace([np.inf,-np.inf], np.nan).values)
    c_short = mean_safe(df[df["env"]=="controlled"]["short_rate_per_min"])
    n_short = mean_safe(df[df["env"]=="noisy"]["short_rate_per_min"])
    c_ext   = mean_safe(df[df["env"]=="controlled"]["ext_rate_per_min"])
    n_ext   = mean_safe(df[df["env"]=="noisy"]["ext_rate_per_min"])
    if np.isfinite(c_short) and abs(c_short)>1e-9:
        drop_s = (c_short - n_short)/abs(c_short)
    else:
        drop_s = np.nan
    if np.isfinite(c_ext) and abs(c_ext)>1e-9:
        drop_e = (c_ext - n_ext)/abs(c_ext)
    else:
        drop_e = np.nan
    plt.figure(figsize=(5,4))
    plt.bar(["short","extended"], [drop_s, drop_e])
    plt.ylabel("Robustness drop (rate)")
    plt.title("Controlled → Noisy (lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "robustness_drop.png"), dpi=150)
    plt.close()

def plot_roc(df, outdir, use_ext=False):
    feats = [("ext_" if use_ext else "short_")+f for f in FEATURES]
    X = df[feats].fillna(0.0).values
    y = (df["env"]=="noisy").astype(int).values
    if len(np.unique(y)) < 2:
        return
    clf = LogisticRegression(max_iter=1000).fit(X,y)
    prob = clf.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, prob)
    A = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={A:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {'extended' if use_ext else 'short'} features")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"roc_{'ext' if use_ext else 'short'}.png"), dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="outputs/features.csv", help="Input features CSV")
    ap.add_argument("--outdir", default="outputs", help="Where to save figures and summaries")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.in_csv)

    # 1) summary stats CSV
    summary = save_summary_csv(df, args.outdir)

    # 2) per-feature boxplots (all env + by env)
    for feat in FEATURES:
        if f"short_{feat}" in df.columns and f"ext_{feat}" in df.columns:
            plot_box(df, feat, args.outdir)

    # 3) stability and robustness
    if "short_ibi_cv" in df.columns and "ext_ibi_cv" in df.columns:
        plot_stability(df, args.outdir)
    if set(["env","short_rate_per_min","ext_rate_per_min"]).issubset(df.columns):
        if ("controlled" in set(df["env"])) and ("noisy" in set(df["env"])):
            plot_robustness(df, args.outdir)

    # 4) ROC curves (controlled vs noisy)
    if "env" in df.columns:
        plot_roc(df, args.outdir, use_ext=False)
        plot_roc(df, args.outdir, use_ext=True)

    print(f"Saved figures & CSVs to: {args.outdir}")

if __name__ == "__main__":
    main()
