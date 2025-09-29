#Blink-Based Cognitive Load under Noise — Robustness via Extended Rest Baselines

This project evaluates whether extending the neutral rest baseline improves the stability and robustness of blink-based cognitive-load features when moving from controlled to noisy visual environments. We use existing datasets only, a lightweight pipeline (face/eye localization → EAR signal → adaptive blink segmentation), and simple statistics (blink rate, mean duration, IBI/IBI-CV).

#Repository structure
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
   ├─ run_extract.py         # extract blink features from VIDEOS (noisy or controlled-if-video)
   ├─ run_extract_rt_bene.py # extract blink features from RT-BENE CSV labels
   ├─ run_merge_features.py  # merge multiple features CSVs
   ├─ make_plots.py          # figures + summary CSVs
   └─ visualize_examples.py  # EAR vs time plot for a single video


#Datasets and download links

RT-BENE (Controlled eye-patch sequences + blink labels) — Zenodo record (CSV labels s000...s016_blink_labels.csv).
https://zenodo.org/records/3685316
 
Zenodo

NTHU Driver Drowsiness Detection (Noisy: illumination/occlusion scenarios) — Official page (BareFace, Glasses, Sunglasses, Night).
https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
 
NTHU CV

HUST-LEBW “Eyeblink in the Wild” (Noisy: movies, strong variability) — Project page with dataset/code pointers.
https://thorhu.github.io/Eyeblink-in-the-wild/
 
thorhu.github.io
+1

Method reference (for EAR): Soukupová & Čech, “Real-Time Eye Blink Detection using Facial Landmarks,” CVWW 2016 (PDF).
https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
 
Machine Vision Laboratory

#Where to place the data

After downloading:

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

If HUST-LEBW comes as split archives (e.g., blink.zip, blink.z01, blink.z02)

Put all parts in the same directory: data\noisy\HUST_LEBW\.

Extract starting from blink.zip using 7-Zip (GUI) or CLI:

cd data\noisy\HUST_LEBW
"C:\Program Files\7-Zip\7z.exe" x blink.zip


You should then see real video files (.mp4/.avi).

#Windows installation (Python 3.12)
1) Install Python 3.12

Download Python 3.12 from python.org and install for “All Users”.

On the installer:

Check “Add python.exe to PATH.”

Or install without PATH and use the py launcher (recommended on Windows).

2) Verify Python and set environment variables (optional)

Using the py launcher (works even if PATH is not set):

py -3.12 --version


If you must set PATH manually (not recommended if using py):

setx PATH "%LocalAppData%\Programs\Python\Python312;%LocalAppData%\Programs\Python\Python312\Scripts;%PATH%"


Close and reopen the terminal after setx.

3) Create and activate a virtual environment

From your project folder:

py -3.12 -m venv .venv312
.\.venv312\Scripts\activate

4) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


If you have trouble with MediaPipe on some Windows builds, upgrade pip/wheels:
pip install --upgrade pip setuptools wheel

#Configuration

#Open config.yaml and adjust if needed:

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

Quick sanity check (optional, notebook)
jupyter notebook notebooks\sanity_checks.ipynb


The notebook will try to locate the first available video in data/noisy or data/controlled and plot EAR with blink shading.

#End-to-end run (scripts)
#A) Visualize a single video’s EAR curve (manual spot check)
python src\visualize_examples.py data\noisy\HUST_LEBW\some_video.mp4 --cfg config.yaml


Output: outputs\some_video_EAR.png

#B) Extract features from videos (NTHU-DDD / HUST-LEBW)
python src\run_extract.py --cfg config.yaml


Output: outputs\features.csv (env column is set based on folder you processed)

#C) Extract features from RT-BENE CSVs (controlled labels)

RT-BENE does not ship as videos; use the label reader:

python src\run_extract_rt_bene.py --cfg config.yaml --rt_root "data\controlled" --fps 30 --debug


If it says “no usable labels parsed,” it will print your CSV headers; then run again with the detected column, e.g.:

python src\run_extract_rt_bene.py --cfg config.yaml --rt_root "data\controlled" --fps 30 --label_col label --debug


Output: outputs\features_rt_bene.csv

#D) Merge feature files (recommended)
python src\run_merge_features.py --glob "outputs\features*.csv" --out outputs\features_all.csv --drop-duplicates
copy outputs\features_all.csv outputs\features.csv

#E) Evaluate stability/robustness and AUC (text metrics)
python src\run_eval.py --cfg config.yaml


This prints:

Stability (mean IBI-CV): short vs extended

Controlled → Noisy robustness drop (blink rate): short vs extended

Simple separability AUC: short vs extended

#F) Produce plots and summary tables
python src\make_plots.py --in_csv outputs\features.csv --outdir outputs


Outputs in outputs\:

summary_stats.csv

box_*.png (per-feature boxplots; overall + per-env)

stability_bar.png

robustness_drop.png

roc_short.png, roc_ext.png

#Notes and troubleshooting

Split archives: Always extract from the .zip file with all .z0* parts present.

Paths with spaces: Quote them in commands.

No videos found: Make sure you actually have .mp4 / .avi files under data\noisy\....

RT-BENE parsing: Use --debug once; if autodetect misses the label column, pass --label_col <name>. The labels may be 0/1, open/closed, or probabilities; the reader handles all.

Python 3.12 wheels: If a package fails to install, update pip/wheel and try again.

#Citations

RT-BENE dataset: Zenodo record with blink labels. 
Zenodo

NTHU-DDD dataset: official NTHU CV lab page. 
NTHU CV

HUST-LEBW dataset: project page/paper pointers. 
thorhu.github.io
+1

EAR baseline: Soukupová & Čech, CVWW 2016. 
Machine Vision Laboratory
