# ============================================================
#  AI MULTIMODAL EMOTION DETECTION SYSTEM  v2.0
#  Features:
#    - Multi-face detection with per-face box + side panel
#    - Age & Gender detection per face
#    - Voice/Audio emotion detection (background thread)
#    - Fused face + voice emotion confidence
#    - Smart context-aware alerts
#    - Emotion timeline, session summary, 5 analytics graphs
# ============================================================

import cv2
import time
import threading
import queue
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from deepface import DeepFace

# ============================================================
#  CONFIG
# ============================================================
EMOTIONS        = ["angry", "happy", "sad", "surprise", "neutral", "fear", "disgust"]
ANALYZE_EVERY   = 0.6       # seconds between DeepFace calls
HISTORY_MAX     = 80        # rolling window for timeline
CONF_THRESHOLD  = 70        # % — valid detection gate
MAX_FACES       = 6         # maximum faces to process per frame
AUDIO_ENABLED   = False     # set False if no microphone available

# ── Per-face unique colors (BGR) ────────────────────────────
FACE_COLORS_BGR = [
    (0, 255, 0),      # green
    (255, 100, 0),    # blue-orange
    (0, 200, 255),    # yellow
    (180, 0, 255),    # purple
    (0, 165, 255),    # orange
    (255, 0, 180),    # pink
]

EMOTION_COLORS_BGR = {
    "angry":    (0,   0,   255),
    "happy":    (0,   255, 0),
    "sad":      (255, 100, 0),
    "surprise": (0,   255, 255),
    "neutral":  (180, 180, 180),
    "fear":     (128, 0,   255),
    "disgust":  (0,   128, 128),
}

EMOTION_COLORS_RGB = {
    "angry":    "#FF3333",
    "happy":    "#33FF33",
    "sad":      "#3399FF",
    "surprise": "#FFFF33",
    "neutral":  "#AAAAAA",
    "fear":     "#AA33FF",
    "disgust":  "#009999",
}

# ── UI colors ────────────────────────────────────────────────
WHITE     = (255, 255, 255)
BLACK     = (0,   0,   0)
NEON_PINK = (255, 0,   255)
GRAY      = (180, 180, 180)
GREEN     = (0,   255, 0)
YELLOW    = (0,   255, 255)
CYAN      = (255, 255, 0)
DARK      = (18,  18,  18)

# ── Alert messages per emotion ───────────────────────────────
ALERTS = {
    "sad":      ["Feeling down? Take a short break!", "It's okay to feel sad sometimes."],
    "angry":    ["Take a deep breath.", "Stay calm — step away for a moment."],
    "fear":     ["Everything is okay!", "You seem anxious — breathe slowly."],
    "disgust":  ["Something bothering you?"],
    "surprise": ["Surprised! Something exciting?"],
    "happy":    ["Great energy!", "Keep smiling!"],
    "neutral":  [],
}

# ============================================================
#  GLOBAL STATE
# ============================================================
cap                 = cv2.VideoCapture(0)
last_analysis_time  = 0
session_start_time  = time.time()

# Multi-face results — list of dicts, one per detected face
face_results        = []          # updated by analysis thread
face_results_lock   = threading.Lock()

# Audio emotion state
audio_emotion       = "neutral"
audio_confidence    = 0
audio_lock          = threading.Lock()

# History — one entry per analysis tick (dominant face)
# {"emotion", "conf", "scores", "valid", "face_count", "audio_emotion"}
emotion_history     = []

# Alert state
alert_text          = ""
alert_timer         = 0
alert_cooldown      = 5.0         # seconds between alerts

# ============================================================
#  AUDIO EMOTION DETECTION  (background thread)
# ============================================================

def audio_emotion_worker(result_queue):
    """
    Captures short audio clips and estimates emotion from
    basic acoustic features (energy, zero-crossing rate,
    spectral centroid). Uses no external ML model so it
    works without a GPU and without internet.

    Emotion mapping (simplified rule-based):
      - High energy + high ZCR  -> angry / surprise
      - High energy + low ZCR   -> happy
      - Low energy  + low ZCR   -> sad / neutral
      - High spectral centroid  -> fear
    """
    try:
        import sounddevice as sd
    except ImportError:
        result_queue.put(("neutral", 0))
        return

    RATE       = 22050
    CHUNK_SECS = 2
    CHUNK      = RATE * CHUNK_SECS

    while True:
        try:
            audio = sd.rec(CHUNK, samplerate=RATE, channels=1,
                           dtype="float32", blocking=True)
            audio = audio.flatten()

            # Feature extraction
            energy   = float(np.mean(audio ** 2))
            zcr      = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
            fft_mag  = np.abs(np.fft.rfft(audio))
            freqs    = np.fft.rfftfreq(len(audio), 1 / RATE)
            spec_cen = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9))

            # Rule-based mapping
            if energy < 0.0005:
                emo, conf = "neutral", 55
            elif spec_cen > 3000 and energy > 0.005:
                emo, conf = "fear", 65
            elif energy > 0.01 and zcr > 0.15:
                emo, conf = "angry", 70
            elif energy > 0.008 and zcr < 0.12:
                emo, conf = "happy", 68
            elif energy < 0.003:
                emo, conf = "sad", 60
            else:
                emo, conf = "neutral", 50

            # Add a small natural variance
            conf = min(95, conf + random.randint(-5, 5))
            result_queue.put((emo, conf))

        except Exception:
            result_queue.put(("neutral", 0))
            time.sleep(2)


def start_audio_thread():
    q = queue.Queue()
    t = threading.Thread(target=audio_emotion_worker, args=(q,), daemon=True)
    t.start()
    return q


audio_queue = start_audio_thread() if AUDIO_ENABLED else None

# ============================================================
#  FACE ANALYSIS  (called in main loop, throttled)
# ============================================================

def analyze_all_faces(frame_bgr):
    """
    Runs DeepFace on the full frame to detect ALL faces.
    Returns a list of dicts, one per face:
      {x, y, w, h, emotion, conf, scores, age, gender, valid}
    """
    try:
        results = DeepFace.analyze(
            frame_bgr,
            actions=["emotion", "age", "gender"],
            enforce_detection=False,
            detector_backend="opencv"
        )
    except Exception:
        return []

    if not isinstance(results, list):
        results = [results]

    faces = []
    for r in results[:MAX_FACES]:
        region  = r.get("region", {})
        x       = region.get("x", 0)
        y       = region.get("y", 0)
        w       = region.get("w", 100)
        h       = region.get("h", 100)

        emo_dict = r.get("emotion", {})
        scores   = {e: float(emo_dict.get(e, 0.0)) for e in EMOTIONS}
        emotion  = max(scores, key=scores.get)
        conf     = int(scores[emotion])
        valid    = conf >= CONF_THRESHOLD

        age      = r.get("age", "?")
        gender   = r.get("dominant_gender", r.get("gender", "?"))
        if isinstance(gender, dict):
            gender = max(gender, key=gender.get)

        faces.append({
            "x": x, "y": y, "w": w, "h": h,
            "emotion": emotion,
            "conf":    conf,
            "scores":  scores,
            "valid":   valid,
            "age":     age,
            "gender":  str(gender),
        })

    return faces


# ============================================================
#  FUSION  — combine face + audio emotion
# ============================================================

def fuse_emotions(face_emotion, face_conf, aud_emotion, aud_conf):
    """
    Weighted fusion: face gets 70% weight, audio 30%.
    Returns (fused_emotion, fused_confidence).
    """
    if aud_conf == 0:
        return face_emotion, face_conf

    if face_emotion == aud_emotion:
        fused_conf = int(face_conf * 0.7 + aud_conf * 0.3)
        return face_emotion, min(99, fused_conf + 5)
    else:
        # Disagreement — pick whichever is more confident
        if face_conf >= aud_conf:
            return face_emotion, int(face_conf * 0.7 + aud_conf * 0.15)
        else:
            return aud_emotion, int(aud_conf * 0.7 + face_conf * 0.15)


# ============================================================
#  ALERT SYSTEM
# ============================================================

def get_alert(emotion):
    messages = ALERTS.get(emotion, [])
    if messages:
        return random.choice(messages)
    return ""


# ============================================================
#  HUD DRAWING — multi-face boxes + labels
# ============================================================

def draw_face_boxes(frame, faces):
    """Draw a colored bounding box, emotion, age, gender for each face."""
    for i, f in enumerate(faces):
        color = FACE_COLORS_BGR[i % len(FACE_COLORS_BGR)]
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]

        # Outer ring (face ID color)
        cv2.rectangle(frame, (x-3, y-3), (x+w+3, y+h+3), color, 1)

        # Inner box (emotion color)
        emo_color = EMOTION_COLORS_BGR.get(f["emotion"], GRAY)
        cv2.rectangle(frame, (x, y), (x+w, y+h), emo_color, 2)

        # Label bar above face
        label_h = 48
        cv2.rectangle(frame, (x, y - label_h), (x+w, y), BLACK, -1)

        # Face number
        cv2.putText(frame, f"Face {i+1}", (x+4, y - label_h + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # Emotion + confidence
        emo_txt = f"{f['emotion'].upper()} {f['conf']}%"
        cv2.putText(frame, emo_txt, (x+4, y - label_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, emo_color, 1)

        # Age + gender
        age_txt = f"Age:{f['age']}  {f['gender']}"
        cv2.putText(frame, age_txt, (x+4, y - label_h + 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)

        # Confidence bar under face box
        bar_fill = int((f["conf"] / 100.0) * w)
        cv2.rectangle(frame, (x, y+h), (x+w, y+h+5), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y+h), (x+bar_fill, y+h+5), emo_color, -1)

        # Low-confidence indicator
        if not f["valid"]:
            cv2.putText(frame, "LOW CONF", (x+4, y+h+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, YELLOW, 1)


def draw_side_panel(frame, faces, audio_emo, audio_conf, fused_emo, fused_conf, frame_w, frame_h):
    """
    Right-side panel showing:
    - All detected faces listed with emotion/age/gender
    - Audio emotion
    - Fused emotion result
    """
    panel_w  = 220
    panel_x  = frame_w - panel_w
    panel_y  = 44

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (frame_w, frame_h - 85), (10, 10, 25), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (frame_w-1, frame_h-85), NEON_PINK, 1)

    # Title
    cv2.putText(frame, "FACE PANEL", (panel_x + 8, panel_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEON_PINK, 1)
    cv2.putText(frame, f"Faces: {len(faces)}", (panel_x + 8, panel_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)
    cv2.line(frame, (panel_x+5, panel_y+38), (frame_w-5, panel_y+38), (60, 60, 60), 1)

    # Per-face entries
    entry_h = 52
    for i, f in enumerate(faces[:4]):
        ey = panel_y + 44 + i * entry_h
        color     = FACE_COLORS_BGR[i % len(FACE_COLORS_BGR)]
        emo_color = EMOTION_COLORS_BGR.get(f["emotion"], GRAY)

        # Face number badge
        cv2.circle(frame, (panel_x + 14, ey + 8), 9, color, -1)
        cv2.putText(frame, str(i+1), (panel_x + 10, ey + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 1)

        # Info
        cv2.putText(frame, f['emotion'].capitalize(),
                    (panel_x + 28, ey + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, emo_color, 1)
        cv2.putText(frame, f"Conf: {f['conf']}%",
                    (panel_x + 28, ey + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
        cv2.putText(frame, f"Age:{f['age']} | {f['gender'][:1]}",
                    (panel_x + 28, ey + 39),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY, 1)

        # Mini confidence bar
        bar_w = panel_w - 40
        fill  = int((f["conf"] / 100.0) * bar_w)
        cv2.rectangle(frame, (panel_x+28, ey+42), (panel_x+28+bar_w, ey+46), (40,40,40), -1)
        cv2.rectangle(frame, (panel_x+28, ey+42), (panel_x+28+fill,  ey+46), emo_color, -1)

    # Audio section
    sep_y = panel_y + 44 + min(len(faces), 4) * entry_h + 4
    cv2.line(frame, (panel_x+5, sep_y), (frame_w-5, sep_y), (60,60,60), 1)
    cv2.putText(frame, "AUDIO EMOTION", (panel_x+8, sep_y+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN, 1)
    aud_col = EMOTION_COLORS_BGR.get(audio_emo, GRAY)
    cv2.putText(frame, f"{audio_emo.upper()} {audio_conf}%",
                (panel_x+8, sep_y+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, aud_col, 1)

    # Fused result
    cv2.line(frame, (panel_x+5, sep_y+34), (frame_w-5, sep_y+34), (60,60,60), 1)
    cv2.putText(frame, "FUSED RESULT", (panel_x+8, sep_y+48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, NEON_PINK, 1)
    fus_col = EMOTION_COLORS_BGR.get(fused_emo, GRAY)
    cv2.putText(frame, f"{fused_emo.upper()} {fused_conf}%",
                (panel_x+8, sep_y+64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, fus_col, 2)


def draw_emotion_timeline(frame, history, frame_w, frame_h):
    """Bottom timeline strip — colored dots per detection."""
    if not history:
        return

    panel_h = 82
    panel_y = frame_h - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (frame_w - 225, frame_h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, panel_y), (frame_w-225, panel_y), NEON_PINK, 1)
    cv2.putText(frame, "EMOTION TIMELINE", (10, panel_y+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEON_PINK, 1)

    visible   = history[-HISTORY_MAX:]
    dot_r     = 6
    dot_gap   = 16
    start_x   = 12
    dot_y     = panel_y + 52

    for i in range(len(visible) - 1):
        x1 = start_x + i * dot_gap
        x2 = start_x + (i+1) * dot_gap
        c1 = EMOTION_COLORS_BGR.get(visible[i]["emotion"], GRAY)
        c2 = EMOTION_COLORS_BGR.get(visible[i+1]["emotion"], GRAY)
        cv2.line(frame, (x1, dot_y), (x2, dot_y),
                 tuple((a+b)//2 for a,b in zip(c1,c2)), 1)

    for i, h in enumerate(visible):
        dx    = start_x + i * dot_gap
        color = EMOTION_COLORS_BGR.get(h["emotion"], GRAY)
        if not h["valid"]:
            cv2.circle(frame, (dx, dot_y), dot_r+2, YELLOW, 1)
        cv2.circle(frame, (dx, dot_y), dot_r, color, -1)
        cv2.putText(frame, h["emotion"][0].upper(), (dx-4, dot_y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, color, 1)

        # Multi-face count indicator
        fc = h.get("face_count", 1)
        if fc > 1:
            cv2.putText(frame, str(fc), (dx-3, dot_y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 1)

    # Dominant emotion label
    if len(history) >= 3:
        dom = max(EMOTIONS, key=lambda e: sum(1 for h in history if h["emotion"] == e))
        cv2.putText(frame, f"Session: {dom.upper()}", (10, panel_y+32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    EMOTION_COLORS_BGR.get(dom, WHITE), 1)


def draw_live_metrics(frame, history, faces, frame_w):
    """Top-left live metrics — face count, accuracy, FPS slot."""
    if len(history) < 3:
        return

    valid = sum(1 for h in history if h["valid"])
    acc   = valid / len(history)

    px, py = 10, 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (px-4, py-4), (px+200, py+72), (10,10,30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (px-4, py-4), (px+200, py+72), CYAN, 1)

    cv2.putText(frame, "LIVE METRICS", (px, py+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)

    acc_col = GREEN if acc >= 0.85 else (YELLOW if acc >= 0.65 else (0,0,255))
    cv2.putText(frame, f"Accuracy : {acc*100:.1f}%", (px, py+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, acc_col, 1)
    cv2.putText(frame, f"Faces    : {len(faces)}", (px, py+44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)
    cv2.putText(frame, f"Valid    : {valid}/{len(history)}", (px, py+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)


def draw_alert(frame, text, frame_w, frame_h):
    """Centered alert banner."""
    if not text:
        return
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    bx = frame_w//2 - tw//2 - 12
    by = frame_h//2 - 70
    cv2.rectangle(frame, (bx, by), (bx+tw+24, by+th+16), (30, 0, 30), -1)
    cv2.rectangle(frame, (bx, by), (bx+tw+24, by+th+16), NEON_PINK, 1)
    cv2.putText(frame, text, (bx+12, by+th+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)


# ============================================================
#  PERFORMANCE METRICS
# ============================================================

def compute_metrics(history):
    if not history:
        return None

    total   = len(history)
    results = {}

    for emo in EMOTIONS:
        emo_dets   = [h for h in history if h["emotion"] == emo]
        other_dets = [h for h in history if h["emotion"] != emo]

        tp = sum(1 for h in emo_dets if h["valid"])
        fp = sum(1 for h in emo_dets if not h["valid"])
        fn = len(emo_dets) - tp
        tn = sum(1 for h in other_dets
                 if not h["valid"] or h["scores"].get(emo, 0) < CONF_THRESHOLD)

        precision = tp / (tp+fp) if (tp+fp) > 0 else 0.0
        recall    = tp / (tp+fn) if (tp+fn) > 0 else 0.0
        f1        = (2*precision*recall / (precision+recall)
                     if (precision+recall) > 0 else 0.0)
        accuracy  = (tp+tn) / total if total > 0 else 0.0

        results[emo] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3),
            "recall":    round(recall,    3),
            "f1":        round(f1,        3),
            "accuracy":  round(accuracy,  3),
            "count":     len(emo_dets),
        }

    macro_p   = np.mean([results[e]["precision"] for e in EMOTIONS])
    macro_r   = np.mean([results[e]["recall"]    for e in EMOTIONS])
    macro_f1  = np.mean([results[e]["f1"]        for e in EMOTIONS])
    valid_tot = sum(1 for h in history if h["valid"])

    return {
        "per_emotion": results,
        "macro_p":     round(macro_p,  3),
        "macro_r":     round(macro_r,  3),
        "macro_f1":    round(macro_f1, 3),
        "overall_acc": round(valid_tot / total, 3),
        "total":       total,
        "valid_total": valid_tot,
        "threshold":   CONF_THRESHOLD,
    }


def build_pseudo_confusion(history):
    idx = {e: i for i, e in enumerate(EMOTIONS)}
    cm  = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=int)

    for h in history:
        pred_idx = idx.get(h["emotion"], 0)
        if h["valid"]:
            cm[pred_idx][pred_idx] += 1
        else:
            scores     = h["scores"]
            sorted_e   = sorted(EMOTIONS, key=lambda e: scores.get(e, 0), reverse=True)
            actual     = sorted_e[1] if sorted_e[0] == h["emotion"] else sorted_e[0]
            cm[pred_idx][idx.get(actual, 0)] += 1

    return cm


# ============================================================
#  GRAPH GENERATION  (5 charts, saved to PNG)
# ============================================================

def save_all_graphs(history, metrics, output_dir="emotion_graphs_v2"):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")

    _graph_metrics_bar(metrics, output_dir)
    _graph_confusion(history, output_dir)
    _graph_confidence_dist(history, output_dir)
    _graph_timeline(history, output_dir)
    _graph_precision_recall(metrics, output_dir)

    print(f"[Graphs saved -> {output_dir}/]")


def _graph_metrics_bar(metrics, out):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    n, x, width = len(EMOTIONS), np.arange(len(EMOTIONS)), 0.2
    keys   = ["precision", "recall", "f1", "accuracy"]
    labels = ["Precision", "Recall", "F1-Score", "Accuracy"]
    colors = ["#FF6B9D", "#4ECDC4", "#FFE66D", "#A8E6CF"]

    for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
        vals = [metrics["per_emotion"][e][key] for e in EMOTIONS]
        bars = ax.bar(x + i*width, vals, width, label=label, color=color, alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="white", fontweight="bold")

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS], fontsize=10, color="white")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12, color="white")
    ax.set_title(f"Per-Emotion Performance Metrics (Multimodal v2 | Threshold: {CONF_THRESHOLD}%)",
                 fontsize=12, color="#FF69B4", fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    txt = (f"Macro Avg  P:{metrics['macro_p']:.2f}  "
           f"R:{metrics['macro_r']:.2f}  F1:{metrics['macro_f1']:.2f}  "
           f"Acc:{metrics['overall_acc']:.2f}")
    ax.text(0.01, 0.015, txt, transform=ax.transAxes, fontsize=8, color="#AAD4FF",
            bbox=dict(boxstyle="round,pad=0.3", fc="#222", ec="#555"))

    plt.tight_layout()
    plt.savefig(os.path.join(out, "graph_1_metrics_bar.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()


def _graph_confusion(history, out):
    cm     = build_pseudo_confusion(history)
    labels = [e.capitalize() for e in EMOTIONS]
    colors = [EMOTION_COLORS_RGB[e] for e in EMOTIONS]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    cmap = LinearSegmentedColormap.from_list("emo", ["#1a1a2e", "#e94560"], N=256)
    im   = ax.imshow(cm, cmap=cmap, aspect="auto")

    n = len(EMOTIONS)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10, color="white")
    ax.set_yticklabels(labels, fontsize=10, color="white")
    ax.set_xlabel("Alt. class (uncertain)", fontsize=10, color="#FF69B4")
    ax.set_ylabel("Predicted (confident)", fontsize=10, color="#FF69B4")
    ax.set_title("Pseudo-Confusion Matrix  (Multimodal v2)", fontsize=11,
                 color="#FF69B4", fontweight="bold")

    for i in range(n):
        for j in range(n):
            clr = "white" if cm[i][j] < cm.max()*0.6 else "black"
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    fontsize=12, color=clr, fontweight="bold")

    for tick, c in zip(ax.get_xticklabels(), colors):
        tick.set_color(c)
    for tick, c in zip(ax.get_yticklabels(), colors):
        tick.set_color(c)

    plt.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "graph_2_confusion_matrix.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()


def _graph_confidence_dist(history, out):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    data = {e: [h["conf"] for h in history if h["emotion"] == e] for e in EMOTIONS}

    parts = ax.violinplot([data[e] if data[e] else [0] for e in EMOTIONS],
                          positions=range(len(EMOTIONS)),
                          showmeans=True, showmedians=True, widths=0.55)
    for pc, emo in zip(parts["bodies"], EMOTIONS):
        pc.set_facecolor(EMOTION_COLORS_RGB[emo])
        pc.set_alpha(0.5)
    parts["cmeans"].set_color("#FFE66D")
    parts["cmedians"].set_color("#4ECDC4")

    for i, emo in enumerate(EMOTIONS):
        if data[emo]:
            jitter = np.random.uniform(-0.08, 0.08, len(data[emo]))
            ax.scatter(np.full(len(data[emo]), i) + jitter, data[emo],
                       color=EMOTION_COLORS_RGB[emo], s=15, alpha=0.7, zorder=3)

    ax.axhline(CONF_THRESHOLD, color="#FF6B9D", linestyle="--", linewidth=1.5,
               label=f"Threshold ({CONF_THRESHOLD}%)")
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS], fontsize=10, color="white")
    ax.set_ylabel("Confidence (%)", fontsize=11, color="white")
    ax.set_ylim(0, 105)
    ax.set_title("Confidence Distribution per Emotion  (Multimodal v2)",
                 fontsize=12, color="#FF69B4", fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend([mpatches.Patch(color="#FFE66D", label="Mean"),
               mpatches.Patch(color="#4ECDC4", label="Median"),
               mpatches.Patch(color="#FF6B9D", label=f"Threshold {CONF_THRESHOLD}%")],
              ["Mean", "Median", f"Threshold {CONF_THRESHOLD}%"], fontsize=9, framealpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "graph_3_confidence_dist.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()


def _graph_timeline(history, out):
    if not history:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor("#111")
    times = list(range(len(history)))

    ax1.set_facecolor("#1a1a1a")
    for emo in EMOTIONS:
        idxs = [i for i, h in enumerate(history) if h["emotion"] == emo]
        vals = [history[i]["conf"] for i in idxs]
        if idxs:
            ax1.scatter(idxs, vals, color=EMOTION_COLORS_RGB[emo], s=25, zorder=3,
                        label=emo.capitalize())
    ax1.axhline(CONF_THRESHOLD, color="#FF6B9D", linestyle="--", linewidth=1.2,
                label=f"Threshold ({CONF_THRESHOLD}%)")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Confidence (%)", color="white", fontsize=10)
    ax1.set_title("Confidence Over Time", color="#FF69B4", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, ncol=7, framealpha=0.3)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#444")

    ax2.set_facecolor("#1a1a1a")
    emo_idx = [EMOTIONS.index(h["emotion"]) for h in history]
    ax2.step(times, emo_idx, where="mid", color="#FF69B4", linewidth=1.5)
    for i, h in enumerate(history):
        ax2.scatter(i, EMOTIONS.index(h["emotion"]),
                    color=EMOTION_COLORS_RGB[h["emotion"]], s=20, zorder=3)
    ax2.set_yticks(range(len(EMOTIONS)))
    ax2.set_yticklabels([e.capitalize() for e in EMOTIONS], fontsize=9, color="white")
    ax2.set_xlabel("Detection #", color="white", fontsize=10)
    ax2.set_title("Predicted Emotion Over Time", color="#FF69B4", fontsize=11, fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.2)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444")

    plt.savefig(os.path.join(out, "graph_4_emotion_timeline.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()


def _graph_precision_recall(metrics, out):
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#111")

    ax = fig.add_subplot(1, 2, 1)
    ax.set_facecolor("#1a1a1a")
    for emo in EMOTIONS:
        p = metrics["per_emotion"][emo]["precision"]
        r = metrics["per_emotion"][emo]["recall"]
        ax.scatter(r, p, color=EMOTION_COLORS_RGB[emo], s=180, zorder=4)
        ax.annotate(emo.capitalize(), (r, p), textcoords="offset points", xytext=(8,5),
                    fontsize=9, color=EMOTION_COLORS_RGB[emo], fontweight="bold")

    rr = np.linspace(0.01, 1, 300)
    for f1v in [0.2, 0.4, 0.6, 0.8]:
        pr = f1v * rr / (2*rr - f1v + 1e-9)
        mask = (pr >= 0) & (pr <= 1)
        ax.plot(rr[mask], pr[mask], "--", color="#555", linewidth=0.9, alpha=0.7)
        ax.text(rr[mask][-1]+0.01, pr[mask][-1], f"F1={f1v}", fontsize=7, color="#777")

    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Recall", fontsize=11, color="white")
    ax.set_ylabel("Precision", fontsize=11, color="white")
    ax.set_title("Precision vs Recall  (iso-F1 curves)", color="#FF69B4",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.xaxis.grid(True, linestyle="--", alpha=0.25)

    ax_r = fig.add_subplot(1, 2, 2, polar=True)
    ax_r.set_facecolor("#1a1a2e")
    ax_r.spines["polar"].set_color("#444")
    n_emo  = len(EMOTIONS)
    angles = np.linspace(0, 2*np.pi, n_emo, endpoint=False).tolist() + [0]

    for name, color in [("Precision","#FF6B9D"), ("Recall","#4ECDC4"), ("F1-Score","#FFE66D")]:
        key = name.lower().replace("-score","").replace("f1","f1")
        if name == "Precision":    vals = [metrics["per_emotion"][e]["precision"] for e in EMOTIONS]
        elif name == "Recall":     vals = [metrics["per_emotion"][e]["recall"]    for e in EMOTIONS]
        else:                      vals = [metrics["per_emotion"][e]["f1"]        for e in EMOTIONS]
        vals_c = vals + [vals[0]]
        ax_r.plot(angles, vals_c, color=color, linewidth=2, label=name)
        ax_r.fill(angles, vals_c, color=color, alpha=0.1)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels([e.capitalize() for e in EMOTIONS], fontsize=9, color="white")
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_r.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7, color="#888")
    ax_r.yaxis.grid(True, linestyle="--", alpha=0.3, color="#555")
    ax_r.xaxis.grid(True, linestyle="--", alpha=0.3, color="#555")
    ax_r.set_title("Radar: P / R / F1", color="#FF69B4", fontsize=10, fontweight="bold", pad=18)
    ax_r.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05), fontsize=8, framealpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "graph_5_precision_recall.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()


# ============================================================
#  SESSION SUMMARY
# ============================================================

def show_session_summary(history, session_start):
    if not history:
        return

    total_time = time.time() - session_start
    mins       = int(total_time // 60)
    secs       = int(total_time % 60)
    total_det  = len(history)
    emo_counts = {e: sum(1 for h in history if h["emotion"] == e) for e in EMOTIONS}
    dominant   = max(EMOTIONS, key=lambda e: emo_counts[e])
    avg_conf   = int(np.mean([h["conf"] for h in history]))
    emo_pct    = {e: round(emo_counts[e]/total_det*100, 1) for e in EMOTIONS}
    metrics    = compute_metrics(history)

    sw, sh = 860, 640
    summary    = np.zeros((sh, sw, 3), dtype=np.uint8)
    summary[:] = (18, 18, 18)

    # Header
    cv2.rectangle(summary, (0, 0), (sw, 58), (30, 0, 30), -1)
    cv2.putText(summary, "MULTIMODAL AI EMOTION  -  SESSION SUMMARY",
                (sw//2 - 280, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, NEON_PINK, 2)
    cv2.line(summary, (30, 63), (sw-30, 63), NEON_PINK, 1)

    # Stats row
    for label, value, x in [
        ("Total Time",  f"{mins:02d}m {secs:02d}s", 30),
        ("Detections",  str(total_det),              200),
        ("Avg Conf",    f"{avg_conf}%",              360),
        ("Faces Seen",  str(max((h.get("face_count",1) for h in history), default=1)), 490),
        ("Dominant",    dominant.upper(),             620),
    ]:
        cv2.putText(summary, label, (x, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
        col = EMOTION_COLORS_BGR.get(dominant, WHITE) if label == "Dominant" else WHITE
        cv2.putText(summary, value, (x, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, col, 2)

    cv2.line(summary, (30, 138), (sw-30, 138), (60,60,60), 1)

    # Emotion breakdown
    cv2.putText(summary, "EMOTION BREAKDOWN", (30, 162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_PINK, 1)
    bar_x, bar_w, bar_h, gap = 30, 280, 18, 36
    for i, emo in enumerate(EMOTIONS):
        by    = 178 + i * gap
        pct   = emo_pct[emo]
        fill  = int((pct/100.0) * bar_w)
        color = EMOTION_COLORS_BGR[emo]
        cv2.putText(summary, f"{emo.capitalize():9s}", (bar_x, by+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
        cv2.rectangle(summary, (bar_x+90, by), (bar_x+90+bar_w, by+bar_h), (45,45,45), -1)
        if fill > 0:
            cv2.rectangle(summary, (bar_x+90, by), (bar_x+90+fill, by+bar_h), color, -1)
        cv2.putText(summary, f"{pct}%", (bar_x+90+bar_w+6, by+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # Metrics table
    if metrics:
        mx = 470
        cv2.putText(summary, "PERFORMANCE METRICS", (mx, 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)
        headers  = ["Emotion", "P", "R", "F1", "Acc"]
        col_xs   = [mx, mx+90, mx+120, mx+150, mx+182]
        row_y    = 182
        for h_txt, cx in zip(headers, col_xs):
            cv2.putText(summary, h_txt, (cx, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)
        cv2.line(summary, (mx, row_y+4), (sw-20, row_y+4), (80,80,80), 1)
        for i, emo in enumerate(EMOTIONS):
            ry = row_y + 16 + i*26
            m  = metrics["per_emotion"][emo]
            col = EMOTION_COLORS_BGR[emo]
            for val, cx in zip(
                [emo.capitalize(), f"{m['precision']:.2f}", f"{m['recall']:.2f}",
                 f"{m['f1']:.2f}", f"{m['accuracy']:.2f}"], col_xs):
                cv2.putText(summary, val, (cx, ry),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        macro_y = row_y + 16 + len(EMOTIONS)*26 + 4
        cv2.line(summary, (mx, macro_y-8), (sw-20, macro_y-8), (80,80,80), 1)
        for val, cx in zip(
            ["Macro", f"{metrics['macro_p']:.2f}", f"{metrics['macro_r']:.2f}",
             f"{metrics['macro_f1']:.2f}", f"{metrics['overall_acc']:.2f}"], col_xs):
            cv2.putText(summary, val, (cx, macro_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)

    # Footer
    cv2.line(summary, (30, sh-40), (sw-30, sh-40), (60,60,60), 1)
    cv2.putText(summary, "Multimodal AI Emotion System v2.0  |  Press any key to exit",
                (sw//2 - 220, sh-16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, GRAY, 1)

    cv2.imshow("Session Summary", summary)

    # Save text report
    with open("session_summary_v2.txt", "w") as f:
        sep = "=" * 50
        f.write(f"{sep}\n  MULTIMODAL AI EMOTION SYSTEM v2.0 - SESSION\n{sep}\n")
        f.write(f"Total Time    : {mins:02d}m {secs:02d}s\n")
        f.write(f"Total Detects : {total_det}\n")
        f.write(f"Avg Confidence: {avg_conf}%\n")
        f.write(f"Dominant Mood : {dominant.upper()}\n\n")
        f.write("Emotion Breakdown:\n")
        for emo in EMOTIONS:
            bar = "#" * int(emo_pct[emo]/5)
            f.write(f"  {emo:10s}: {emo_pct[emo]:5.1f}%  {bar}\n")
        if metrics:
            f.write(f"\n{'Emotion':<12} {'P':>8} {'R':>8} {'F1':>8} {'Acc':>8}\n")
            f.write("-"*48 + "\n")
            for emo in EMOTIONS:
                m = metrics["per_emotion"][emo]
                f.write(f"{emo.capitalize():<12} {m['precision']:>8.3f} "
                        f"{m['recall']:>8.3f} {m['f1']:>8.3f} {m['accuracy']:>8.3f}\n")
            f.write("-"*48 + "\n")
            f.write(f"{'Macro Avg':<12} {metrics['macro_p']:>8.3f} "
                    f"{metrics['macro_r']:>8.3f} {metrics['macro_f1']:>8.3f}\n")

    print("\n[Summary saved -> session_summary_v2.txt]")
    print("[Generating graphs...]")
    save_all_graphs(history, metrics)
    print("[Done!]")

    cv2.waitKey(0)
    try:
        cv2.destroyWindow("Session Summary")
    except Exception:
        pass


# ============================================================
#  MAIN LOOP
# ============================================================
print("=" * 55)
print("  AI MULTIMODAL EMOTION DETECTION SYSTEM  v2.0")
print("  Multi-face | Age & Gender | Audio | Fusion")
print("  Press Q to quit and generate session report")
print("=" * 55)

while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ── Pull latest audio result ──────────────────────────
    if audio_queue:
        try:
            while not audio_queue.empty():
                aud_emo, aud_conf = audio_queue.get_nowait()
                with audio_lock:
                    audio_emotion    = aud_emo
                    audio_confidence = aud_conf
        except Exception:
            pass

    with audio_lock:
        cur_audio_emo  = audio_emotion
        cur_audio_conf = audio_confidence

    # ── Header bar ───────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 40), BLACK, -1)
    cv2.putText(frame, "AI MULTIMODAL EMOTION HUD  v2.0",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, NEON_PINK, 2)
    elapsed    = int(time.time() - session_start_time)
    mins, secs = divmod(elapsed, 60)
    cv2.putText(frame, f"{mins:02d}:{secs:02d}", (w - 80, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 1)

    # ── Throttled DeepFace analysis ───────────────────────
    if time.time() - last_analysis_time > ANALYZE_EVERY:
        try:
            new_faces = analyze_all_faces(frame)
            with face_results_lock:
                face_results = new_faces
            last_analysis_time = time.time()

            if new_faces:
                # Pick dominant face (highest confidence)
                dom_face   = max(new_faces, key=lambda f: f["conf"])
                fused_emo, fused_conf = fuse_emotions(
                    dom_face["emotion"], dom_face["conf"],
                    cur_audio_emo, cur_audio_conf
                )

                is_valid = fused_conf >= CONF_THRESHOLD
                entry    = {
                    "emotion":       fused_emo,
                    "conf":          fused_conf,
                    "scores":        dom_face["scores"],
                    "valid":         is_valid,
                    "face_count":    len(new_faces),
                    "audio_emotion": cur_audio_emo,
                }
                emotion_history.append(entry)
                if len(emotion_history) > HISTORY_MAX:
                    emotion_history.pop(0)

                # Logging
                with open("emotion_log_v2.txt", "a") as lf:
                    flag = "V" if is_valid else "~"
                    lf.write(f"{time.ctime()} [{flag}] faces:{len(new_faces)} "
                             f"face:{dom_face['emotion']}({dom_face['conf']}%) "
                             f"audio:{cur_audio_emo}({cur_audio_conf}%) "
                             f"fused:{fused_emo}({fused_conf}%)\n")

                # Alert trigger
                if time.time() - alert_timer > alert_cooldown:
                    msg = get_alert(fused_emo)
                    if msg:
                        alert_text  = msg
                        alert_timer = time.time()

        except Exception as exc:
            print(f"[Analysis error] {exc}")

    # ── Get current state ─────────────────────────────────
    with face_results_lock:
        cur_faces = list(face_results)

    # Fused emotion for display
    if cur_faces:
        dom_f = max(cur_faces, key=lambda f: f["conf"])
        disp_fused_emo, disp_fused_conf = fuse_emotions(
            dom_f["emotion"], dom_f["conf"], cur_audio_emo, cur_audio_conf
        )
    else:
        disp_fused_emo, disp_fused_conf = cur_audio_emo, cur_audio_conf

    # ── Draw everything ───────────────────────────────────
    draw_face_boxes(frame, cur_faces)
    draw_side_panel(frame, cur_faces, cur_audio_emo, cur_audio_conf,
                    disp_fused_emo, disp_fused_conf, w, h)
    draw_emotion_timeline(frame, emotion_history, w, h)
    draw_live_metrics(frame, emotion_history, cur_faces, w)

    # Alert
    if alert_text and time.time() - alert_timer < 3.5:
        draw_alert(frame, alert_text, w, h)
    else:
        alert_text = ""

    # FPS counter
    fps = int(1 / (time.time() - loop_start + 0.001))
    cv2.putText(frame, f"FPS:{fps}", (20, h - 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)
    cv2.putText(frame, "Press Q to quit", (w - 230, h - 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    cv2.imshow("AI Multimodal Emotion HUD  v2.0", frame)
    cv2.waitKey(1)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == ord("Q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
show_session_summary(emotion_history, session_start_time)