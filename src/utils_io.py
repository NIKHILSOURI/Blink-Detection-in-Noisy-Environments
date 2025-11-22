import os, re, json, yaml
import numpy as np

def load_cfg(path="config.yaml"):
    with open(path, "r") as f: return yaml.safe_load(f)

def list_videos(root, exts=(".mp4",".avi",".mov",".mkv")):
    vids=[]
    for d,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(exts):
                vids.append(os.path.join(d,f))
    return sorted(vids)

def choose_baseline_frames(total_frames, fps, short_sec, extended_sec):
    short = min(int(short_sec*fps), total_frames)
    ext   = min(int(extended_sec*fps), total_frames)
    return (0, short-1), (0, ext-1)
