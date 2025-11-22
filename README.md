# Blink-Based Cognitive Load Detection under Controlled and Noisy Environments

This project evaluates whether **extending the neutral rest baseline** improves the **stability** and **robustness** of blink-based cognitive-load features when moving from **controlled** to **noisy** visual environments. We use existing datasets only, a lightweight pipeline (face/eye localization → EAR signal → adaptive blink segmentation), and simple statistics (blink rate, mean duration, IBI/IBI-CV).

## 📋 Table of Contents

- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Details](#project-details)
- [Citation](#citation)
- [License](#license)

## 📁 Repository Structure

```
PROJECT/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── README.md                # This file
├── data/
│   ├── controlled/          # RT-BENE eye-patch data (CSV labels + images)
│   └── noisy/               # HUST-LEBW / NTHU-DDD videos
├── notebooks/
│   └── sanity_checks.ipynb  # Quick, interactive checks
├── outputs/                 # Generated features.csv, plots, summaries
└── src/
    ├── blink_ear.py         # EAR computation and blink detection
    ├── utils_io.py          # I/O utilities
    ├── run_extract.py       # Extract blink features from videos
    ├── run_extract_rt_bene.py  # Extract features from RT-BENE CSV labels
    ├── run_merge_features.py   # Merge multiple features CSVs
    ├── run_eval.py          # Evaluate stability/robustness and AUC
    ├── make_plots.py        # Generate figures + summary CSVs
    └── visualize_examples.py # EAR vs time plot for a single video
```

## 📊 Datasets

### RT-BENE (Controlled Environment)
- **Description**: Eye-patch sequences with blink labels
- **Download**: [Zenodo record](https://zenodo.org/records/3685316)
- **Files**: CSV labels (`s000...s016_blink_labels.csv`) and optional eye-patch image folders

### HUST-LEBW "Eyeblink in the Wild" (Noisy Environment)
- **Description**: Movies with strong variability in illumination and occlusion
- **Download**: [Project page](https://thorhu.github.io/Eyeblink-in-the-wild/)
- **Files**: Video files (`.mp4` or `.avi`)

### NTHU Driver Drowsiness Dataset
- **Description**: Driver scenarios with illumination/occlusion variations
- **Files**: Videos organized by condition (BareFace, Glasses, Sunglasses, Night_BareFace, Night_Glasses)

### Data Organization

After downloading, organize your data as follows:

```
data/
├── controlled/
│   ├── s000_blink_labels.csv
│   ├── s001_blink_labels.csv
│   ├── ...
│   └── (RT-BENE eye-patch image folders if downloaded)
└── noisy/
    ├── HUST-LEBW/
    │   └── *.mp4 (or .avi)
    └── NTHU-DDD/
        ├── BareFace/*.mp4
        ├── Glasses/*.mp4
        ├── Sunglasses/*.mp4
        ├── Night_BareFace/*.mp4
        └── Night_Glasses/*.mp4
```

**Note**: Large data files (videos, tar archives) are excluded from this repository due to GitHub's file size limits. Please download datasets separately from the sources above.

### Extracting Split Archives (HUST-LEBW)

If HUST-LEBW comes as split archives (e.g., `blink.zip`, `blink.z01`, `blink.z02`):

1. Put **all parts** in the same directory: `data/noisy/HUST-LEBW/`
2. Extract starting from `blink.zip` using 7-Zip:
   ```powershell
   cd data\noisy\HUST-LEBW
   "C:\Program Files\7-Zip\7z.exe" x blink.zip
   ```

## 🚀 Installation

### Prerequisites

- **Python 3.12** (recommended)
- **Windows 10/11** (tested on Windows)

### Step-by-Step Installation

1. **Install Python 3.12**
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check **"Add python.exe to PATH"** or use the **py launcher** (recommended)

2. **Verify Python Installation**
   ```powershell
   py -3.12 --version
   ```

3. **Create and Activate Virtual Environment**
   ```powershell
   cd "path\to\PROJECT"
   py -3.12 -m venv .venv312
   .\.venv312\Scripts\activate
   ```

4. **Install Dependencies**
   ```powershell
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

### Dependencies

- `opencv-python` - Video processing and computer vision
- `mediapipe` - Face and eye landmark detection
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `scipy` - Scientific computing
- `matplotlib` - Plotting
- `tqdm` - Progress bars
- `pyyaml` - Configuration file parsing

## ⚙️ Configuration

Edit `config.yaml` to adjust parameters:

```yaml
fps_target: 30                    # Target FPS for processing
eye_crop_side: 96                 # Eye crop size
ear_smooth_win: 5                 # EAR smoothing window size
blink_min_frames: 2               # Minimum closed-frames to count as blink
blink_merge_gap: 3                # Merge tiny gaps between closures
baseline:
  short_sec: 30                   # Short baseline duration (seconds)
  extended_sec: 90                # Extended baseline duration (seconds)
paths:
  controlled_dir: "data/controlled"
  noisy_dir: "data/noisy"
  outputs: "outputs"
```

## 📖 Usage

### A) Visualize a Single Video's EAR Curve (Quick Spot Check)

```powershell
python src\visualize_examples.py "data\noisy\HUST-LEBW\some_video.mp4" --cfg config.yaml
```

**Output**: `outputs\some_video_EAR.png`

### B) Extract Features from Videos (NTHU-DDD / HUST-LEBW)

```powershell
python src\run_extract.py --cfg config.yaml
```

**Output**: `outputs\features.csv`

### C) Extract Features from RT-BENE CSVs (Controlled Labels)

RT-BENE uses CSV labels rather than videos:

```powershell
python src\run_extract_rt_bene.py --cfg config.yaml --rt_root "data\controlled" --fps 30 --debug
```

If it prints CSV headers but finds no labels, specify the label column:

```powershell
python src\run_extract_rt_bene.py --cfg config.yaml --rt_root "data\controlled" --fps 30 --label_col label --debug
```

**Output**: `outputs\features_rt_bene.csv`

### D) Merge Feature Files (Recommended)

```powershell
python src\run_merge_features.py --glob "outputs\features*.csv" --out outputs\features_all.csv --drop-duplicates
copy outputs\features_all.csv outputs\features.csv
```

### E) Evaluate Stability/Robustness and AUC (Text Metrics)

```powershell
python src\run_eval.py --cfg config.yaml
```

### F) Produce Plots and Summary Tables

```powershell
python src\make_plots.py --in_csv outputs\features.csv --outdir outputs
```

**Outputs in `outputs\`**:
- `summary_stats.csv` - Summary statistics
- `box_*.png` - Per-feature boxplots (overall + per-environment)
- `stability_bar.png` - Stability comparison
- `robustness_drop.png` - Robustness drop visualization
- `roc_short.png`, `roc_ext.png` - ROC curves for short vs extended baselines

## 🔬 Project Details

### Methodology

1. **Face/Eye Localization**: Uses MediaPipe FaceMesh for robust landmark detection
2. **EAR Signal**: Computes Eye Aspect Ratio (EAR) from 6-point eye landmarks
3. **Blink Segmentation**: Adaptive thresholding with gap merging
4. **Feature Extraction**: 
   - Blink rate (per minute)
   - Mean blink duration (seconds)
   - Inter-blink interval (IBI) and coefficient of variation (IBI-CV)
5. **Baseline Comparison**: Short (30s) vs Extended (90s) rest baselines

### Features Evaluated

- **Stability**: Consistency of features within environments
- **Robustness**: Performance drop when moving from controlled to noisy environments
- **AUC**: Area under ROC curve for cognitive load classification

## 📚 Citation

If you use this repository or parts of it, please cite the following where appropriate:

- **EAR Baseline**: Soukupová & Čech (CVWW 2016) - "Real-Time Eye Blink Detection using Facial Landmarks"
- **RT-BENE Dataset**: RT-BENE (ICCVW 2019) - Controlled eye-patch blink labels
- **HUST-LEBW Dataset**: HUST-LEBW (TIFS 2020) - "Eyeblink in the Wild" in-the-wild blink dataset
- **NTHU Driver Drowsiness**: NTHU Driver Drowsiness Dataset - Driver scenarios with illumination/occlusion

## 🐛 Troubleshooting

- **Split archives**: Always extract from the `.zip` file with all `.z0*` parts present in the same directory
- **Paths with spaces**: Quote them in commands (e.g., `"data\noisy\HUST-LEBW\Video (1).avi"`)
- **No videos found**: Ensure `.mp4` / `.avi` files exist under `data\noisy\...`
- **RT-BENE parsing**: Use `--debug` once to see columns; if needed, pass `--label_col <name>`
- **Install issues**: Upgrade pip/wheels; some environments require recent Microsoft C++ Build Tools for certain packages
- **Large files**: Data files (videos, tar archives) are excluded from git. Download datasets separately.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**NIKHILSOURI**

- GitHub: [@NIKHILSOURI](https://github.com/NIKHILSOURI)
- Repository: [Blink-Detection-in-Noisy-Environments](https://github.com/NIKHILSOURI/Blink-Detection-in-Noisy-Environments)

---

**Note**: This is a 2-week research project evaluating how extended rest baselines improve the stability and robustness of blink features in illumination- and occlusion-rich conditions.

