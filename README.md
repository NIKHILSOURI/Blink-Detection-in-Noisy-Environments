#Blink-Based Cognitive Load Detection in Noisy Environments

This repository contains the implementation, datasets, and analysis scripts for a two-week research project investigating blink-based cognitive load detection under controlled and noisy visual environments. We explore whether using extended rest baselines improves the stability and robustness of blink-derived features such as blink rate, duration, and inter-blink interval (IBI).

#Project Overview

Blink dynamics are a low-cost, non-intrusive indicator of cognitive load, but their reliability decreases under real-world visual noise (e.g., illumination changes, occlusions, pose variation). This project uses existing open datasets and a lightweight landmark-based blink detection pipeline to evaluate:

RQ1: How can blink detection be optimized under visually noisy conditions?

RQ2: Does increasing rest baseline length improve the reliability of blink-based cognitive load detection?

We focus on illumination and occlusion noise as the primary disturbance.

#Datasets Used

RT-BENE – Controlled eye-patch blink dataset for baseline calibration

HUST-LEBW – In-the-wild blink dataset with strong illumination and pose variation

NTHU-DDD – Driver drowsiness dataset with glasses/sunglasses and night scenarios

(Optional: ZJU Eyeblink for additional controlled clips)

#Features and Workflow

Landmark-based blink detection (EAR method)

Blink feature extraction: rate, duration, IBI statistics

Short vs. extended rest baseline comparison

Stability analysis (IBI coefficient of variation)

Robustness analysis (controlled → noisy drop)

Simple classifier for separability evaluation

Visualization scripts for EAR curves, boxplots, robustness bars, and ROC curves

#Repository Structure
blink-load-robustness/
├── data/                # Datasets: controlled (RT-BENE) & noisy (HUST-LEBW/NTHU-DDD)
├── src/                 # Core scripts for detection, feature extraction, evaluation, and visualization
├── outputs/             # Generated feature CSVs, plots, and results
├── config.yaml          # Global settings
├── requirements.txt     # Dependencies
└── README.md            # This file

#How to Run
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract features
python src/run_extract.py --cfg config.yaml
python src/run_extract_rt_bene.py --cfg config.yaml --rt_root data/controlled --fps 30

# 3. Merge features
python src/run_merge_features.py --glob "outputs/features*.csv" --out outputs/features_all.csv

# 4. Evaluate
python src/run_eval.py --cfg config.yaml

# 5. Generate plots
python src/make_plots.py --in_csv outputs/features_all.csv --outdir outputs

#Results

Extended rest baselines reduce feature variability (IBI-CV ↓).

Robustness drop between controlled and noisy conditions decreases when using extended baselines.

A simple classifier achieves higher AUC with extended-baseline-normalized features.
