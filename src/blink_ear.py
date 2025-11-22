import cv2, mediapipe as mp
import numpy as np

# MediaPipe FaceMesh eye landmark indices (common subset)
LEFT = [33, 160, 158, 133, 153, 144]    # [outer, up1, up2, inner, low2, low1]
RIGHT = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(pts):
    # pts: 6x2 in order [outer, up1, up2, inner, low2, low1]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

class BlinkEAR:
    def __init__(self, smooth_win=5, blink_min_frames=2, blink_merge_gap=3):
        self.smooth_win = smooth_win
        self.blink_min_frames = blink_min_frames
        self.blink_merge_gap = blink_merge_gap
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_video(self, path, fps_target=30):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        ear_series = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_face.process(rgb)
            if not res.multi_face_landmarks:
                ear_series.append(np.nan)
                continue
            lm = res.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            def gather(idx):
                return np.array([[lm[i].x * w, lm[i].y * h] for i in idx], dtype=np.float32)

            left = eye_aspect_ratio(gather(LEFT))
            right = eye_aspect_ratio(gather(RIGHT))
            ear_series.append((left + right) * 0.5)

        cap.release()
        ear = np.array(ear_series, dtype=np.float32)

        # temporal smoothing (median over window)
        if len(ear) == 0:
            return ear, []
        k = max(1, int(self.smooth_win))
        if k > 1:
            pad = k // 2
            ear_pad = np.pad(ear, (pad, pad), mode="edge")
            ear = np.array([np.median(ear_pad[i:i + k]) for i in range(len(ear))], dtype=np.float32)

        blinks = self._detect_blinks(ear)
        return ear, blinks

    def _detect_blinks(self, ear):
        finite = ear[np.isfinite(ear)]
        if len(finite) < 10:
            return []
        med = np.median(finite)
        thr = max(0.12, med - 0.04)  # adaptive threshold
        closed = (ear < thr).astype(np.int32)

        blinks = []
        start = None
        gap = 0
        for i, c in enumerate(closed):
            if c and start is None:
                start = i
                gap = 0
            elif not c and start is not None:
                gap += 1
                if gap > self.blink_merge_gap:
                    end = i - gap
                    if end - start + 1 >= self.blink_min_frames:
                        blinks.append((start, end))
                    start = None
                    gap = 0

        if start is not None:
            end = len(closed) - 1
            if end - start + 1 >= self.blink_min_frames:
                blinks.append((start, end))
        return blinks

def blink_features(blinks, fps):
    if not blinks:
        return dict(rate_per_min=0.0, mean_dur_s=0.0, ibi_mean_s=np.nan, ibi_cv=np.nan)
    durs = [(e - s + 1) / fps for (s, e) in blinks]
    onsets = [s for (s, _) in blinks]
    if len(onsets) > 1:
        ibis = np.diff(onsets) / fps
        ibi_mean = float(np.mean(ibis))
        ibi_cv = float(np.std(ibis) / ibi_mean) if ibi_mean > 0 else np.nan
    else:
        ibi_mean, ibi_cv = np.nan, np.nan

    # blink rate per minute
    if len(onsets) > 1:
        dur_frames = (onsets[-1] - onsets[0] + 1)
        dur_s = dur_frames / fps
        rate = (len(blinks) / max(dur_s, 1e-6)) * 60.0
    else:
        # fall back if only one blink
        dur_s = (onsets[0] + 1) / fps
        rate = (len(blinks) / max(dur_s, 1e-6)) * 60.0

    return dict(
        rate_per_min=float(rate),
        mean_dur_s=float(np.mean(durs)),
        ibi_mean_s=ibi_mean,
        ibi_cv=ibi_cv
    )
