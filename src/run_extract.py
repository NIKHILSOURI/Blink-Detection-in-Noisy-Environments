import os, json, argparse, numpy as np, pandas as pd
from utils_io import load_cfg, list_videos, choose_baseline_frames
from blink_ear import BlinkEAR, blink_features

def process_dir(root, cfg, label):
    be = BlinkEAR(cfg["ear_smooth_win"], cfg["blink_min_frames"], cfg["blink_merge_gap"])
    rows=[]
    vids = list_videos(root)
    for v in vids:
        ear, blinks = be.process_video(v, cfg["fps_target"])
        fps = cfg["fps_target"]  # assume resampled or near
        total = len(ear)
        (s1,e1),(s2,e2) = choose_baseline_frames(total, fps, cfg["baseline"]["short_sec"], cfg["baseline"]["extended_sec"])
        # segment blinks within ranges
        def within(seg): 
            return [(max(s,seg[0]), min(e,seg[1])) for (s,e) in blinks if e>=seg[0] and s<=seg[1]]
        f_short = blink_features(within((s1,e1)), fps)
        f_ext   = blink_features(within((s2,e2)), fps)
        rows.append({
            "video": os.path.relpath(v, root),
            "env": label,
            **{f"short_{k}":v for k,v in f_short.items()},
            **{f"ext_{k}":v for k,v in f_ext.items()}
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)
    os.makedirs(cfg["paths"]["outputs"], exist_ok=True)

    dfs=[]
    ctrl = cfg["paths"]["controlled_dir"]
    if os.path.isdir(ctrl):
        dfs.append(process_dir(ctrl, cfg, "controlled"))
    noisy = cfg["paths"]["noisy_dir"]
    if os.path.isdir(noisy):
        dfs.append(process_dir(noisy, cfg, "noisy"))
    out = os.path.join(cfg["paths"]["outputs"], "features.csv")
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(out, index=False)
        print(f"Saved {out} with {len(df)} rows.")
    else:
        print("No videos found.")

if __name__ == "__main__":
    main()
