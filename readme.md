A) Visualize a single video’s EAR curve (quick spot check)

Command:

python src/visualize_examples.py "data/noisy/HUST-LEBW/Video (1).avi" --cfg config.yaml



B) Extract features from videos (HUST-LEBW / NTHU-DDD)

Command:

python src\run_extract.py --cfg config.yaml


C) Extract features from RT-BENE CSVs (controlled labels)

Command (default):

python src\run_extract_rt_bene.py --cfg config.yaml --rt_root "data\controlled" --fps 30 --debug


If it prints your CSV headers but finds no labels, specify the label column explicitly, e.g.:

python src\run_extract_rt_bene.py --cfg config.yaml --rt_root "data\controlled" --fps 30 --label_col label --debug




D) Merge feature files (recommended)

Command:

python src\run_merge_features.py --glob "outputs\features*.csv" --out outputs\features_all.csv --drop-duplicates
copy outputs\features_all.csv outputs\features.csv




E) Evaluate stability, robustness, and AUC (text metrics)

Command:

python src\run_eval.py --cfg config.yaml



F) Produce plots and summary tables

Command:

python src\make_plots.py --in_csv outputs\features.csv --outdir outputs


