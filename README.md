# Blink-Based Cognitive Load under Noise — Robustness via Extended Rest Baselines

This project evaluates whether **extending the neutral rest baseline** improves the **stability** and **robustness** of blink-based cognitive-load features when moving from **controlled** to **noisy** visual environments. We use existing datasets only, a lightweight pipeline (face/eye localization → EAR signal → adaptive blink segmentation), and simple statistics (blink rate, mean duration, IBI/IBI-CV).

## Repository structure

```
PROJECT/
├─ config.yaml
├─ requirements.txt
├─ data/
│  ├─ controlled/            # RT-BENE eye-patch data (CSV labels + images)
│  └─ noisy/                 # HUST-LEBW / NTHU-DDD videos
├─ notebooks/
│  └─ sanity_checks.ipynb    # quick, interactive checks
├─ outputs/                  # features.csv, plots, summaries
└─ src/
   ├─ blink_ear.py
   ├─ utils_io.py
   ├─ run_extract.py         # extract blink features from VIDEOS
   ├─ run_extract_rt_bene.py # extract features from RT-BENE CSV labels
   ├─ run_merge_features.py  # merge multiple features CSVs
   ├─ make_plots.py          # figures + summary CSVs
   └─ visualize_examples.py  # EAR vs time plot for a single video
```



---

## Datasets and download links

- **RT-BENE (Controlled eye-patch sequences + blink labels)** — Zenodo record (CSV labels `s000...s016_blink_labels.csv`).  
  https://zenodo.org/records/3685316

- **NTHU Driver Drowsiness Detection (Noisy: illumination/occlusion scenarios)** — Official page (BareFace, Glasses, Sunglasses, Night).  
  https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/

- **HUST-LEBW “Eyeblink in the Wild” (Noisy: movies, strong variability)** — Project page with dataset/code pointers.  
  https://thorhu.github.io/Eyeblink-in-the-wild/

**Method reference (EAR):** Soukupová & Čech, “Real-Time Eye Blink Detection using Facial Landmarks,” CVWW 2016 (PDF).  
https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

---

## Where to place the data

After downloading:

```
data/
├─ controlled/
│  ├─ s000_blink_labels.csv
│  ├─ s001_blink_labels.csv
│  ├─ ...
│  └─ (RT-BENE eye-patch image folders if you downloaded them)
└─ noisy/
   ├─ HUST_LEBW/
   │  ├─ <movie-derived>/*.mp4 (or .avi)
   └─ NTHU_DDD/
      ├─ BareFace/*.mp4
      ├─ Glasses/*.mp4
      ├─ Sunglasses/*.mp4
      ├─ Night_BareFace/*.mp4
      └─ Night_Glasses/*.mp4
```

### If HUST-LEBW comes as split archives (e.g., `blink.zip`, `blink.z01`, `blink.z02`)

1. Put **all parts** in the same directory: `data\noisy\HUST_LEBW\`.  
2. Extract **starting from** `blink.zip` using 7-Zip (GUI) or CLI:
   ```bat
   cd data\noisy\HUST_LEBW
   "C:\Program Files\7-Zip\7z.exe" x blink.zip
   ```
   You should then see real video files (`.mp4`/`.avi`).

---

## Installation (Windows, Python 3.12)

### 1) Install Python 3.12
- Install from python.org for “All Users.”
- On the installer, either check **“Add python.exe to PATH”** or rely on the **py launcher** (recommended).

### 2) Verify Python and set environment variables (optional)
Using the **py launcher**:
```bat
py -3.12 --version
```

If you must set PATH manually (not needed if using `py`):
```bat
setx PATH "%LocalAppData%\Programs\Python\Python312;%LocalAppData%\Programs\Python\Python312\Scripts;%PATH%"
```
Restart the terminal after `setx`.

### 3) Create and activate a virtual environment
From the project folder:
```bat
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate
```

### 4) Install dependencies
```bat
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Configuration

Open `config.yaml` and adjust if needed:
```yaml
fps_target: 30
ear_smooth_win: 5
blink_min_frames: 2
blink_merge_gap: 3
baseline:
  short_sec: 30
  extended_sec: 90
paths:
  controlled_dir: "data/controlled"
  noisy_dir: "data/noisy"
  outputs: "outputs"
```

---

## Usage

### A) Visualize a single video’s EAR curve (quick spot check)
```bat
python src\visualize_examples.py data\noisy\HUST_LEBW\some_video.mp4 --cfg config.yaml
```
Output: `outputs\some_video_EAR.png`

### B) Extract features from videos (NTHU-DDD / HUST-LEBW)
```bat
python src\run_extract.py --cfg config.yaml
```
Output: `outputs\features.csv`

### C) Extract features from RT-BENE CSVs (controlled labels)
RT-BENE is not videos; use the label reader:
```bat
python src\run_extract_rt_bene.py --cfg config.yaml --rt_root "data\controlled" --fps 30 --debug
```
If it prints your CSV headers but does not find labels, re-run with the label column name, e.g.:
```bat
python src\run_extract_rt_bene.py --cfg config.yaml --rt_root "data\controlled" --fps 30 --label_col label --debug
```
Output: `outputs\features_rt_bene.csv`

### D) Merge feature files (recommended)
```bat
python src\run_merge_features.py --glob "outputs\features*.csv" --out outputs\features_all.csv --drop-duplicates
copy outputs\features_all.csv outputs\features.csv
```

### E) Evaluate stability/robustness and AUC (text metrics)
```bat
python src\run_eval.py --cfg config.yaml
```

### F) Produce plots and summary tables
```bat
python src\make_plots.py --in_csv outputs\features.csv --outdir outputs
```
Outputs in `outputs\`:
- `summary_stats.csv`
- `box_*.png` (per-feature boxplots; overall + per-env)
- `stability_bar.png`
- `robustness_drop.png`
- `roc_short.png`, `roc_ext.png`

---

## Notes and troubleshooting

- **Split archives**: Always extract from the `.zip` file with all `.z0*` parts present.  
- **Paths with spaces**: Quote them in commands.  
- **No videos found**: Ensure `.mp4` / `.avi` files exist under `data\noisy\...`.  
- **RT-BENE parsing**: Use `--debug` once to see columns; if needed, pass `--label_col <name>`.  
- **Install issues**: Upgrade pip/wheels; some environments require recent Microsoft C++ Build Tools for certain packages.

---

## Citation

If you use this repository or parts of it, please cite the following where appropriate:
- Soukupová & Čech (CVWW 2016) — EAR baseline.
- RT-BENE (ICCVW 2019) — controlled eye-patch blink labels.
- HUST-LEBW (TIFS 2020) — in-the-wild blink dataset.
- NTHU Driver Drowsiness — driver scenarios with illumination/occlusion.
