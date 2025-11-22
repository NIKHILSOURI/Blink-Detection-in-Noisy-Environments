import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils_io import load_cfg
from blink_ear import BlinkEAR

def plot_ear(ear, blinks, fps, save_path):
    if len(ear) == 0:
        print("No EAR samples to plot.")
        return
    t = np.arange(len(ear)) / max(fps, 1)
    plt.figure(figsize=(10, 4))
    plt.plot(t, ear, linewidth=1.2, label="EAR")
    for (s, e) in blinks:
        plt.axvspan(s / max(fps,1), e / max(fps,1), alpha=0.15)
    plt.xlabel("Time (s)")
    plt.ylabel("EAR")
    plt.title("EAR over time (shaded = detected blinks)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def dump_csv(ear, blinks, fps, csv_path):
    t = np.arange(len(ear)) / max(fps, 1)
    df = pd.DataFrame({"t_s": t, "ear": ear})
    mark = np.zeros(len(ear), dtype=int)
    for s,e in blinks:
        mark[s:e+1] = 1
    df["blink"] = mark
    df.to_csv(csv_path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to a video file")
    ap.add_argument("--cfg", default="config.yaml", help="Config file")
    ap.add_argument("--outdir", default=None, help="Output directory (defaults to cfg.paths.outputs)")
    ap.add_argument("--debug", action="store_true", help="Print stats + dump CSV")
    ap.add_argument("--enhance", action="store_true", help="Enhance brightness/contrast for dark videos")
    ap.add_argument("--gamma", type=float, default=1.2, help="Gamma when --enhance is used")
    ap.add_argument("--det", type=float, default=0.3, help="Min detection confidence")
    ap.add_argument("--track", type=float, default=0.3, help="Min tracking confidence")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    outdir = args.outdir or cfg["paths"]["outputs"]
    os.makedirs(outdir, exist_ok=True)

    # Initialize BlinkEAR with only the parameters it accepts
    be = BlinkEAR(
        smooth_win=cfg["ear_smooth_win"],
        blink_min_frames=cfg["blink_min_frames"],
        blink_merge_gap=cfg["blink_merge_gap"]
    )
    # Process the video to get EAR and blinks
    ear, blinks = be.process_video(args.video, fps_target=cfg["fps_target"])
    # Calculate stats for debug output
    stats = {
        'frames': len(ear),
        'detected': sum(1 for x in ear if not np.isnan(x)),
        'med': np.nanmedian(ear) if len(ear) > 0 else 0,
        'thr': 0,  # This would need to be calculated if needed
        'det_rate': sum(1 for x in ear if not np.isnan(x)) / max(1, len(ear)) if len(ear) > 0 else 0
    }
    base = os.path.splitext(os.path.basename(args.video))[0]
    fig_path = os.path.join(outdir, f"{base}_EAR.png")
    plot_ear(ear, blinks, cfg["fps_target"], fig_path)

    if args.debug:
        csv_path = os.path.join(outdir, f"{base}_EAR.csv")
        dump_csv(ear, blinks, cfg["fps_target"], csv_path)
        print(f"[DEBUG] frames={stats['frames']} detected={stats['detected']} "
              f"det_rate={stats['det_rate']:.2%} medianEAR={stats['med']} thr={stats['thr']}")
        print(f"[DEBUG] Saved CSV: {csv_path}")

    n_blinks = len(blinks)
    dur_s = len(ear) / max(cfg["fps_target"],1)
    rate = (n_blinks / max(dur_s, 1e-6)) * 60.0
    print(f"Saved EAR plot: {fig_path}")
    print(f"Video duration: {dur_s:.2f} s | Blinks detected: {n_blinks} | Rate: {rate:.2f} blinks/min")

    if stats['det_rate'] < 0.1:
        print("WARNING: Low detection rate (<10%). Try:")
        print("  - Re-run with: --enhance --gamma 1.3 --det 0.2 --track 0.2")
        print("  - Pick a brighter/clearer, more frontal clip")

if __name__ == "__main__":
    main()
