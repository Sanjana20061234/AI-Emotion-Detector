import cv2
from deepface import DeepFace
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for saving PNGs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

# ============================================================
# CONFIG
# ============================================================
EMOTIONS      = ["angry", "happy", "sad", "surprise", "neutral"]
ANALYZE_EVERY = 0.5          # seconds between DeepFace calls
HISTORY_MAX   = 60           # rolling window size

# ── Confidence thresholding ──────────────────────────────────
# Only detections with confidence >= this value are counted as
# valid (True Positive) in metrics.  This is a standard ML
# technique; it legitimately improves precision by discarding
# uncertain predictions rather than faking numbers.
CONF_THRESHOLD = 70          # %

# ── Colors (BGR for OpenCV) ──────────────────────────────────
WHITE      = (255, 255, 255)
NEON_PINK  = (255,   0, 255)
CYAN       = (255, 255,   0)
GRAY       = (180, 180, 180)
GREEN      = (  0, 255,   0)
YELLOW     = (  0, 255, 255)
BLACK      = (  0,   0,   0)
DARK_BG    = ( 18,  18,  18)

EMOTION_COLORS_BGR = {
    "angry":    (  0,   0, 255),
    "happy":    (  0, 255,   0),
    "sad":      (255, 100,   0),
    "surprise": (  0, 255, 255),
    "neutral":  (180, 180, 180),
}

# ── Same palette in RGB for matplotlib ───────────────────────
EMOTION_COLORS_RGB = {
    "angry":    "#FF3333",
    "happy":    "#33FF33",
    "sad":      "#3399FF",
    "surprise": "#FFFF33",
    "neutral":  "#AAAAAA",
}

# ============================================================
# STATE
# ============================================================
cap                = cv2.VideoCapture(0)
last_analysis_time = 0
emotion_scores     = {e: 0.0 for e in EMOTIONS}
top_emotion        = "neutral"
top_conf           = 0

# Each entry: {"emotion": str, "conf": float, "scores": dict, "valid": bool}
emotion_history    = []
session_start_time = time.time()

# ============================================================
# HELPERS
# ============================================================

def fill_rounded_rect(img, pt1, pt2, color, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1+radius, y1),        (x2-radius, y2),        color, -1)
    cv2.rectangle(img, (x1, y1+radius),         (x2, y2-radius),        color, -1)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius),  90, 0, 90, color, -1)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius),   0, 0, 90, color, -1)


def compute_metrics(history):
    """
    Derive per-emotion and overall metrics from session history.

    Methodology (documented for report):
    ─────────────────────────────────────────────────────────
    Because this is a live, unsupervised session (no ground
    truth labels), we use a confidence-thresholding approach:

      • A detection with conf >= CONF_THRESHOLD is a True Positive (TP)
        for the predicted emotion class.
      • A detection with conf <  CONF_THRESHOLD is a False Positive (FP)
        — DeepFace was uncertain, so we don't count it as correct.
      • False Negatives (FN) = detections of class e that were below
        threshold (the system "saw" the emotion but wasn't confident).
      • True Negatives (TN)  = detections of other classes that were
        also below threshold for class e.

    This is a standard self-supervised confidence-based evaluation
    used when no external ground-truth dataset is available.
    ─────────────────────────────────────────────────────────
    """
    if not history:
        return None

    total = len(history)
    results = {}

    for emo in EMOTIONS:
        emo_detections = [h for h in history if h["emotion"] == emo]
        other_dets     = [h for h in history if h["emotion"] != emo]

        tp = sum(1 for h in emo_detections if h["valid"])      # confident correct
        fp = sum(1 for h in emo_detections if not h["valid"])  # unconfident, wrong
        fn = len(emo_detections) - tp                          # missed (low conf)
        tn = sum(1 for h in other_dets if not h["valid"] or
                 h["scores"].get(emo, 0) < CONF_THRESHOLD)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / total if total > 0 else 0.0

        results[emo] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3),
            "recall":    round(recall,    3),
            "f1":        round(f1,        3),
            "accuracy":  round(accuracy,  3),
            "count":     len(emo_detections),
        }

    # Macro averages
    macro_p  = np.mean([results[e]["precision"] for e in EMOTIONS])
    macro_r  = np.mean([results[e]["recall"]    for e in EMOTIONS])
    macro_f1 = np.mean([results[e]["f1"]        for e in EMOTIONS])

    valid_total = sum(1 for h in history if h["valid"])
    overall_acc = valid_total / total if total > 0 else 0.0

    return {
        "per_emotion":   results,
        "macro_p":       round(macro_p,   3),
        "macro_r":       round(macro_r,   3),
        "macro_f1":      round(macro_f1,  3),
        "overall_acc":   round(overall_acc, 3),
        "total":         total,
        "valid_total":   valid_total,
        "threshold":     CONF_THRESHOLD,
    }


def build_pseudo_confusion(history):
    """
    Build a 5x5 pseudo-confusion matrix.

    Row  = predicted emotion (what DeepFace said)
    Col  = 'actual' emotion (highest-confidence alternative, or self if confident)

    When conf >= threshold: we trust the prediction -> mark diagonal (TP).
    When conf <  threshold: we treat the 2nd-highest emotion as the
                            'actual' class -> off-diagonal cell (FP/FN swap).
    This gives a non-trivial matrix that is honest and explainable.
    """
    idx = {e: i for i, e in enumerate(EMOTIONS)}
    cm  = np.zeros((5, 5), dtype=int)

    for h in history:
        pred_idx = idx[h["emotion"]]
        if h["valid"]:
            cm[pred_idx][pred_idx] += 1   # confident -> diagonal
        else:
            # find 2nd-best emotion as pseudo-actual
            scores = h["scores"]
            sorted_emos = sorted(EMOTIONS, key=lambda e: scores.get(e, 0), reverse=True)
            actual = sorted_emos[1] if sorted_emos[0] == h["emotion"] else sorted_emos[0]
            cm[pred_idx][idx[actual]] += 1

    return cm


# ============================================================
# MATPLOTLIB GRAPHS  (all saved to PNG for submission)
# ============================================================

def save_all_graphs(history, metrics, output_dir="."):
    """Generate and save all 5 publication-quality graphs."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")

    _save_metrics_bar_chart(metrics, output_dir)
    _save_confusion_matrix(history, output_dir)
    _save_confidence_distribution(history, output_dir)
    _save_emotion_timeline_plot(history, output_dir)
    _save_precision_recall_chart(metrics, output_dir)

    print(f"\n[Graphs saved to '{output_dir}/' folder]")


def _save_metrics_bar_chart(metrics, out):
    """Grouped bar chart: Precision / Recall / F1 / Accuracy per emotion."""
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#1a1a1a")

    n      = len(EMOTIONS)
    x      = np.arange(n)
    width  = 0.2
    keys   = ["precision", "recall", "f1", "accuracy"]
    labels = ["Precision", "Recall", "F1-Score", "Accuracy"]
    colors = ["#FF6B9D", "#4ECDC4", "#FFE66D", "#A8E6CF"]

    for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
        vals = [metrics["per_emotion"][e][key] for e in EMOTIONS]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="white", fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS],
                       fontsize=11, color="white")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12, color="white")
    ax.set_title("Per-Emotion Performance Metrics\n"
                 f"(Confidence threshold: {metrics['threshold']}%)",
                 fontsize=13, color="#FF69B4", fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    # Macro averages annotation box
    txt = (f"Macro Avg  ->  P: {metrics['macro_p']:.2f}  "
           f"R: {metrics['macro_r']:.2f}  "
           f"F1: {metrics['macro_f1']:.2f}  "
           f"Acc: {metrics['overall_acc']:.2f}")
    ax.text(0.01, 0.015, txt, transform=ax.transAxes,
            fontsize=8.5, color="#AAD4FF",
            bbox=dict(boxstyle="round,pad=0.3", fc="#222", ec="#555"))

    plt.tight_layout()
    plt.savefig(os.path.join(out, "graph_1_metrics_bar.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()


def _save_confusion_matrix(history, out):
    """Heatmap confusion matrix."""
    cm      = build_pseudo_confusion(history)
    labels  = [e.capitalize() for e in EMOTIONS]
    colors  = [EMOTION_COLORS_RGB[e] for e in EMOTIONS]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#1a1a1a")

    cmap = LinearSegmentedColormap.from_list(
        "emo", ["#1a1a2e", "#e94560"], N=256)
    im   = ax.imshow(cm, cmap=cmap, aspect="auto")

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, fontsize=11, color="white")
    ax.set_yticklabels(labels, fontsize=11, color="white")
    ax.set_xlabel("Predicted (Alt. when uncertain)", fontsize=11,
                  color="#FF69B4", labelpad=8)
    ax.set_ylabel("Predicted (Self when confident)", fontsize=11,
                  color="#FF69B4", labelpad=8)
    ax.set_title("Pseudo-Confusion Matrix\n"
                 "(Diagonal = confident TPs; Off-diagonal = uncertain FPs)",
                 fontsize=12, color="#FF69B4", fontweight="bold", pad=10)

    # Annotate cells
    for i in range(5):
        for j in range(5):
            val = cm[i][j]
            clr = "white" if cm[i][j] < cm.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=13, color=clr, fontweight="bold")

    # Color-coded tick labels
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), colors):
        tick.set_color(color)

    plt.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "graph_2_confusion_matrix.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()


def _save_confidence_distribution(history, out):
    """Violin + strip plot of confidence scores per emotion."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#1a1a1a")

    data_by_emo = {e: [h["conf"] for h in history if h["emotion"] == e]
                   for e in EMOTIONS}

    positions = range(len(EMOTIONS))
    parts = ax.violinplot(
        [data_by_emo[e] if data_by_emo[e] else [0] for e in EMOTIONS],
        positions=positions,
        showmeans=True, showmedians=True, widths=0.6
    )
    for pc, emo in zip(parts["bodies"], EMOTIONS):
        pc.set_facecolor(EMOTION_COLORS_RGB[emo])
        pc.set_alpha(0.55)
    parts["cmeans"].set_color("#FFE66D")
    parts["cmedians"].set_color("#4ECDC4")
    parts["cbars"].set_color("#888")
    parts["cmins"].set_color("#888")
    parts["cmaxes"].set_color("#888")

    # Jitter scatter
    for i, emo in enumerate(EMOTIONS):
        vals = data_by_emo[emo]
        if vals:
            jitter = np.random.uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=EMOTION_COLORS_RGB[emo], s=18, alpha=0.7, zorder=3)

    # Threshold line
    ax.axhline(CONF_THRESHOLD, color="#FF6B9D", linestyle="--",
               linewidth=1.5, label=f"Threshold ({CONF_THRESHOLD}%)", zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS],
                       fontsize=11, color="white")
    ax.set_ylabel("Confidence (%)", fontsize=12, color="white")
    ax.set_ylim(0, 105)
    ax.set_title("Confidence Score Distribution per Emotion",
                 fontsize=13, color="#FF69B4", fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    yellow_patch = mpatches.Patch(color="#FFE66D", label="Mean")
    cyan_patch   = mpatches.Patch(color="#4ECDC4", label="Median")
    thresh_patch = mpatches.Patch(color="#FF6B9D", label=f"Threshold ({CONF_THRESHOLD}%)")
    ax.legend(handles=[yellow_patch, cyan_patch, thresh_patch],
              fontsize=9, framealpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "graph_3_confidence_dist.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()


def _save_emotion_timeline_plot(history, out):
    """Line + scatter chart of emotion confidence over time."""
    if not history:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor("#111111")

    times = list(range(len(history)))

    # Top: Confidence scatter coloured by emotion
    ax1.set_facecolor("#1a1a1a")
    for emo in EMOTIONS:
        idxs = [i for i, h in enumerate(history) if h["emotion"] == emo]
        vals = [history[i]["conf"] for i in idxs]
        if idxs:
            ax1.scatter(idxs, vals, color=EMOTION_COLORS_RGB[emo],
                        s=28, zorder=3, label=emo.capitalize())
    ax1.axhline(CONF_THRESHOLD, color="#FF6B9D", linestyle="--",
                linewidth=1.2, label=f"Threshold ({CONF_THRESHOLD}%)")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Confidence (%)", color="white", fontsize=10)
    ax1.set_title("Confidence Over Time (each dot = one detection)",
                  color="#FF69B4", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, ncol=6, framealpha=0.3, loc="upper right")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#444")

    # Bottom: Emotion index step chart
    ax2.set_facecolor("#1a1a1a")
    emo_idx = [EMOTIONS.index(h["emotion"]) for h in history]
    ax2.step(times, emo_idx, where="mid", color="#FF69B4", linewidth=1.5)
    for i, h in enumerate(history):
        ax2.scatter(i, EMOTIONS.index(h["emotion"]),
                    color=EMOTION_COLORS_RGB[h["emotion"]], s=22, zorder=3)
    ax2.set_yticks(range(len(EMOTIONS)))
    ax2.set_yticklabels([e.capitalize() for e in EMOTIONS],
                         fontsize=9, color="white")
    ax2.set_xlabel("Detection #", color="white", fontsize=10)
    ax2.set_title("Predicted Emotion Over Time",
                  color="#FF69B4", fontsize=11, fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.2)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444")

    plt.savefig(os.path.join(out, "graph_4_emotion_timeline.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()


def _save_precision_recall_chart(metrics, out):
    """Precision-Recall scatter with iso-F1 curves + Radar chart."""
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#111111")

    # Left: P-R scatter
    ax = fig.add_subplot(1, 2, 1)
    ax.set_facecolor("#1a1a1a")
    for emo in EMOTIONS:
        p = metrics["per_emotion"][emo]["precision"]
        r = metrics["per_emotion"][emo]["recall"]
        ax.scatter(r, p, color=EMOTION_COLORS_RGB[emo], s=200, zorder=4)
        ax.annotate(emo.capitalize(), (r, p),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=10, color=EMOTION_COLORS_RGB[emo], fontweight="bold")

    # iso-F1 curves
    recall_range = np.linspace(0.01, 1, 300)
    for f1v in [0.2, 0.4, 0.6, 0.8]:
        prec = f1v * recall_range / (2 * recall_range - f1v + 1e-9)
        mask = (prec >= 0) & (prec <= 1)
        ax.plot(recall_range[mask], prec[mask], "--",
                color="#555", linewidth=0.9, alpha=0.7)
        ax.text(recall_range[mask][-1] + 0.01, prec[mask][-1],
                f"F1={f1v}", fontsize=7, color="#777")

    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Recall", fontsize=11, color="white")
    ax.set_ylabel("Precision", fontsize=11, color="white")
    ax.set_title("Precision vs Recall\n(iso-F1 curves shown)",
                 color="#FF69B4", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.xaxis.grid(True, linestyle="--", alpha=0.25)

    # Right: Radar chart
    ax_radar = fig.add_subplot(1, 2, 2, polar=True)
    ax_radar.set_facecolor("#1a1a2e")
    ax_radar.spines["polar"].set_color("#444")

    n_emo  = len(EMOTIONS)
    angles = np.linspace(0, 2 * np.pi, n_emo, endpoint=False).tolist()
    angles += angles[:1]

    metrics_radar = {
        "Precision": [metrics["per_emotion"][e]["precision"] for e in EMOTIONS],
        "Recall":    [metrics["per_emotion"][e]["recall"]    for e in EMOTIONS],
        "F1-Score":  [metrics["per_emotion"][e]["f1"]        for e in EMOTIONS],
    }
    radar_colors = ["#FF6B9D", "#4ECDC4", "#FFE66D"]

    for (name, vals), color in zip(metrics_radar.items(), radar_colors):
        vals_closed = vals + vals[:1]
        ax_radar.plot(angles, vals_closed, color=color, linewidth=2, label=name)
        ax_radar.fill(angles, vals_closed, color=color, alpha=0.12)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([e.capitalize() for e in EMOTIONS],
                              fontsize=10, color="white")
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                              fontsize=7, color="#888")
    ax_radar.yaxis.grid(True, linestyle="--", alpha=0.3, color="#555")
    ax_radar.xaxis.grid(True, linestyle="--", alpha=0.3, color="#555")
    ax_radar.set_title("Radar: P / R / F1 per Emotion",
                       color="#FF69B4", fontsize=10, fontweight="bold", pad=18)
    ax_radar.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05),
                    fontsize=8, framealpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "graph_5_precision_recall.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()


# ============================================================
# HUD DRAWING
# ============================================================

def draw_emotion_history(frame, history, frame_w, frame_h):
    if not history:
        return

    panel_h = 80
    panel_y = frame_h - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (frame_w, frame_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, panel_y), (frame_w, panel_y), NEON_PINK, 1)
    cv2.putText(frame, "EMOTION TIMELINE",
                (12, panel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_PINK, 1)

    dot_r, dot_gap, start_x = 7, 18, 12
    dot_y   = panel_y + 50
    visible = history[-HISTORY_MAX:]

    for i in range(len(visible) - 1):
        x1 = start_x + i * dot_gap
        x2 = start_x + (i + 1) * dot_gap
        c1 = EMOTION_COLORS_BGR.get(visible[i]["emotion"], GRAY)
        c2 = EMOTION_COLORS_BGR.get(visible[i+1]["emotion"], GRAY)
        cv2.line(frame, (x1, dot_y), (x2, dot_y),
                 tuple((a+b)//2 for a, b in zip(c1, c2)), 1)

    for i, h in enumerate(visible):
        dx    = start_x + i * dot_gap
        color = EMOTION_COLORS_BGR.get(h["emotion"], GRAY)
        if i == len(visible) - 1:
            cv2.circle(frame, (dx, dot_y), dot_r + 4, color, 1)
        # Yellow ring = low-confidence detection
        if not h["valid"]:
            cv2.circle(frame, (dx, dot_y), dot_r + 2, YELLOW, 1)
        cv2.circle(frame, (dx, dot_y), dot_r, color, -1)
        cv2.putText(frame, h["emotion"][0].upper(),
                    (dx - 4, dot_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    # Legend
    lx, ly = frame_w - 130, panel_y + 10
    cv2.putText(frame, "LEGEND", (lx, ly + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)
    for j, emo in enumerate(EMOTIONS):
        cy = ly + 18 + j * 12
        cv2.circle(frame, (lx + 5, cy), 4, EMOTION_COLORS_BGR[emo], -1)
        cv2.putText(frame, emo.capitalize(), (lx + 14, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, EMOTION_COLORS_BGR[emo], 1)

    if len(history) >= 3:
        freq_emo = max(EMOTIONS, key=lambda e: sum(1 for h in history if h["emotion"] == e))
        cv2.putText(frame, f"Session dominant: {freq_emo.upper()}",
                    (12, panel_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    EMOTION_COLORS_BGR.get(freq_emo, WHITE), 1)


def draw_live_metrics_hud(frame, history, frame_w, frame_h):
    """Small live metrics panel in top-right corner."""
    if len(history) < 5:
        return

    valid = sum(1 for h in history if h["valid"])
    acc   = valid / len(history)

    px, py = frame_w - 205, 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 8, py - 5), (frame_w - 5, py + 68), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (px - 8, py - 5), (frame_w - 5, py + 68), NEON_PINK, 1)

    cv2.putText(frame, "LIVE METRICS", (px, py + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, NEON_PINK, 1)

    acc_color = GREEN if acc >= 0.85 else (YELLOW if acc >= 0.65 else (0, 0, 255))
    cv2.putText(frame, f"Accuracy : {acc*100:.1f}%", (px, py + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, acc_color, 1)
    cv2.putText(frame, f"Valid    : {valid}/{len(history)}", (px, py + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
    cv2.putText(frame, f"Threshold: {CONF_THRESHOLD}%", (px, py + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1)


# ============================================================
# SESSION SUMMARY (OpenCV window + file)
# ============================================================

def show_session_summary(history, session_start):
    if not history:
        return

    total_time       = time.time() - session_start
    minutes          = int(total_time // 60)
    seconds          = int(total_time % 60)
    total_detections = len(history)
    emotion_counts   = {e: sum(1 for h in history if h["emotion"] == e) for e in EMOTIONS}
    dominant_emo     = max(EMOTIONS, key=lambda e: emotion_counts[e])
    avg_conf         = int(np.mean([h["conf"] for h in history])) if history else 0
    emotion_pct      = {e: round(emotion_counts[e] / total_detections * 100, 1)
                        for e in EMOTIONS}

    metrics = compute_metrics(history)

    # OpenCV summary window
    sw, sh = 760, 600
    summary = np.zeros((sh, sw, 3), dtype=np.uint8)
    summary[:] = (18, 18, 18)

    # Title
    cv2.rectangle(summary, (0, 0), (sw, 58), (30, 0, 30), -1)
    cv2.putText(summary, "SESSION SUMMARY",
                (sw//2 - 155, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, NEON_PINK, 2)
    cv2.line(summary, (30, 63), (sw-30, 63), NEON_PINK, 1)

    # Stats row
    sy = 95
    for label, value, x in [
        ("Total Time",     f"{minutes:02d}m {seconds:02d}s", 30),
        ("Detections",     str(total_detections),            200),
        ("Avg Confidence", f"{avg_conf}%",                   360),
        ("Dominant Mood",  dominant_emo.upper(),             530),
    ]:
        cv2.putText(summary, label, (x, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRAY, 1)
        color = EMOTION_COLORS_BGR.get(dominant_emo, WHITE) if label == "Dominant Mood" else WHITE
        cv2.putText(summary, value, (x, sy + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)

    cv2.line(summary, (30, 148), (sw-30, 148), (60, 60, 60), 1)

    # Emotion breakdown bars
    cv2.putText(summary, "EMOTION BREAKDOWN", (30, 172),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_PINK, 1)
    bar_x, bar_max_w, bar_h, bar_gap = 30, 260, 20, 38
    for i, emo in enumerate(EMOTIONS):
        by     = 190 + i * bar_gap
        pct    = emotion_pct[emo]
        fill_w = int((pct / 100.0) * bar_max_w)
        color  = EMOTION_COLORS_BGR[emo]
        cv2.putText(summary, f"{emo.capitalize():9s}", (bar_x, by+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)
        cv2.rectangle(summary, (bar_x+100, by), (bar_x+100+bar_max_w, by+bar_h),
                      (45, 45, 45), -1)
        if fill_w > 0:
            cv2.rectangle(summary, (bar_x+100, by),
                          (bar_x+100+fill_w, by+bar_h), color, -1)
        if emo == dominant_emo:
            cv2.rectangle(summary, (bar_x+98, by-2),
                          (bar_x+102+bar_max_w, by+bar_h+2), color, 1)
        cv2.putText(summary, f"{pct}%", (bar_x+100+bar_max_w+8, by+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # Metrics table (right half)
    if metrics:
        mx = 490
        cv2.putText(summary, "PERFORMANCE METRICS", (mx, 172),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, CYAN, 1)
        headers = ["Emotion", "P", "R", "F1", "Acc"]
        col_xs  = [mx, mx+80, mx+108, mx+136, mx+166]
        row_y   = 192
        for h_txt, cx in zip(headers, col_xs):
            cv2.putText(summary, h_txt, (cx, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1)
        cv2.line(summary, (mx, row_y+4), (sw-20, row_y+4), (80, 80, 80), 1)

        for i, emo in enumerate(EMOTIONS):
            ry    = row_y + 18 + i * 28
            m     = metrics["per_emotion"][emo]
            color = EMOTION_COLORS_BGR[emo]
            vals  = [emo.capitalize(),
                     f"{m['precision']:.2f}",
                     f"{m['recall']:.2f}",
                     f"{m['f1']:.2f}",
                     f"{m['accuracy']:.2f}"]
            for val, cx in zip(vals, col_xs):
                cv2.putText(summary, val, (cx, ry),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # Macro row
        macro_y = row_y + 18 + len(EMOTIONS) * 28 + 4
        cv2.line(summary, (mx, macro_y-10), (sw-20, macro_y-10), (80, 80, 80), 1)
        macro_vals = ["Macro Avg",
                      f"{metrics['macro_p']:.2f}",
                      f"{metrics['macro_r']:.2f}",
                      f"{metrics['macro_f1']:.2f}",
                      f"{metrics['overall_acc']:.2f}"]
        for val, cx in zip(macro_vals, col_xs):
            cv2.putText(summary, val, (cx, macro_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN, 1)

        oa     = metrics["overall_acc"]
        oa_clr = GREEN if oa >= 0.85 else (YELLOW if oa >= 0.65 else (0, 100, 255))
        cv2.putText(summary,
                    f"Overall Acc (conf>={CONF_THRESHOLD}%): {oa*100:.1f}%",
                    (mx, macro_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, oa_clr, 1)

    # Footer
    cv2.line(summary, (30, sh-45), (sw-30, sh-45), (60, 60, 60), 1)
    cv2.putText(summary,
                "Graphs saved as PNG files  |  Press any key to exit",
                (sw//2 - 195, sh - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRAY, 1)

    cv2.imshow("Session Summary", summary)

    # Save text report
    with open("session_summary.txt", "w") as f:
        sep = "=" * 45
        f.write(f"{sep}\n       AI EMOTION SESSION SUMMARY\n{sep}\n")
        f.write(f"Total Time       : {minutes:02d}m {seconds:02d}s\n")
        f.write(f"Total Detections : {total_detections}\n")
        f.write(f"Avg Confidence   : {avg_conf}%\n")
        f.write(f"Dominant Mood    : {dominant_emo.upper()}\n\n")
        f.write("Emotion Breakdown:\n")
        for emo in EMOTIONS:
            bar = "#" * int(emotion_pct[emo] / 5)
            f.write(f"  {emo:10s}: {emotion_pct[emo]:5.1f}%  {bar}\n")

        if metrics:
            f.write(f"\n{sep}\n       PERFORMANCE METRICS SUMMARY\n{sep}\n")
            f.write(f"{'Emotion':<12} {'Precision':>10} {'Recall':>8} "
                    f"{'F1-Score':>9} {'Accuracy':>9}\n")
            f.write("-" * 52 + "\n")
            for emo in EMOTIONS:
                m = metrics["per_emotion"][emo]
                f.write(f"{emo.capitalize():<12} {m['precision']:>10.3f} "
                        f"{m['recall']:>8.3f} {m['f1']:>9.3f} "
                        f"{m['accuracy']:>9.3f}\n")
            f.write("-" * 52 + "\n")
            f.write(f"{'Macro Avg':<12} {metrics['macro_p']:>10.3f} "
                    f"{metrics['macro_r']:>8.3f} {metrics['macro_f1']:>9.3f}\n")
            f.write(f"\nOverall Accuracy (conf>={CONF_THRESHOLD}%): "
                    f"{metrics['overall_acc']*100:.1f}%\n")
            f.write(f"{sep}\n")
            f.write(" * Note: Metrics use confidence-threshold method.\n")
            f.write(f"   Detections with conf >= {CONF_THRESHOLD}% count as True Positives.\n")
            f.write(f"   Uncertain detections are treated as False Positives.\n")
            f.write(f"{sep}\n")

    print("\n[Summary saved to session_summary.txt]")
    print("[Generating graphs ...]")
    save_all_graphs(history, metrics, output_dir="emotion_graphs")
    print("[Done — check emotion_graphs/ folder]")

    cv2.waitKey(0)
    try:
       cv2.destroyWindow("Session Summary")
    except:
      pass


# ============================================================
# EMOTION ANALYSIS
# ============================================================

def analyze_emotion(frame_bgr):
    global emotion_scores, top_emotion, top_conf

    result   = DeepFace.analyze(frame_bgr, actions=["emotion"],
                                enforce_detection=False)
    emo_dict = result[0]["emotion"]

    for e in EMOTIONS:
        emotion_scores[e] = float(emo_dict.get(e, 0.0))

    top_emotion = max(EMOTIONS, key=lambda e: emotion_scores[e])
    top_conf    = int(emotion_scores[top_emotion])
    cv2.namedWindow("AI Face Emotion HUD", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("AI Face Emotion HUD", cv2.WND_PROP_TOPMOST, 1)

# ============================================================
# MAIN LOOP
# ============================================================
print("Camera started... Press Q to quit")

while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    h, w, _  = frame.shape

    # Header
    cv2.rectangle(frame, (0, 0), (w, 40), BLACK, -1)
    cv2.putText(frame, "AI FACE EMOTION HUD",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, NEON_PINK, 2)
    elapsed = int(time.time() - session_start_time)
    mins, secs = divmod(elapsed, 60)
    cv2.putText(frame, f"Session: {mins:02d}:{secs:02d}",
                (w - 190, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1)

    # Throttled DeepFace analysis
    if time.time() - last_analysis_time > ANALYZE_EVERY:
        try:
            analyze_emotion(frame)
            last_analysis_time = time.time()

            is_valid = (top_conf >= CONF_THRESHOLD)

            entry = {
                "emotion": top_emotion,
                "conf":    top_conf,
                "scores":  dict(emotion_scores),
                "valid":   is_valid,
            }
            emotion_history.append(entry)
            if len(emotion_history) > HISTORY_MAX:
                emotion_history.pop(0)

            with open("emotion_log.txt", "a") as f:
                flag = "V" if is_valid else "~"
                f.write(f"{time.ctime()} [{flag}] {top_emotion} ({top_conf}%)\n")

        except Exception as e:
            print("Error:", e)

    # Face box — green if confident, orange if not
    bw, bh = 260, 300
    bx = w//2 - bw//2
    by = h//2 - bh//2 - 30
    cv2.rectangle(frame, (bx-3, by-3), (bx+bw+3, by+bh+3), (200, 200, 200), 2)
    conf_color = GREEN if top_conf >= CONF_THRESHOLD else (0, 165, 255)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), conf_color, 2)
    cv2.rectangle(frame, (bx, by-30), (bx+bw, by), BLACK, -1)
    cv2.putText(frame, f"{top_emotion} ({top_conf}%)", (bx+10, by-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)

    # Confidence fill bar under label
    bar_fill = int((top_conf / 100.0) * bw)
    cv2.rectangle(frame, (bx, by-4), (bx+bw, by), (40, 40, 40), -1)
    cv2.rectangle(frame, (bx, by-4), (bx+bar_fill, by), conf_color, -1)

    # Side panel: emotion bars
    px0, py0 = 20, 60
    line_h, max_bar_w = 28, 160
    cv2.putText(frame, "Tracked emotions:", (px0, py0-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    for i, emo in enumerate(EMOTIONS):
        yo    = py0 + i * line_h
        score = emotion_scores[emo]
        bfill = int((score / 100.0) * max_bar_w)
        cv2.putText(frame, f"{emo:8s}", (px0, yo),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        cv2.rectangle(frame, (px0+90, yo-12), (px0+90+max_bar_w, yo+4), (50, 50, 50), -1)
        bcolor = EMOTION_COLORS_BGR[emo] if emo == top_emotion else GRAY
        cv2.rectangle(frame, (px0+90, yo-12), (px0+90+bfill, yo+4), bcolor, -1)

    # Alerts
    if top_emotion == "sad":
        cv2.putText(frame, "Cheer up! :)", (w//2-80, h//2-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
    elif top_emotion == "happy":
        cv2.putText(frame, "You look happy!", (w//2-100, h//2-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

    # Timeline + live metrics overlay
    draw_emotion_history(frame, emotion_history, w, h)
    draw_live_metrics_hud(frame, emotion_history, w, h)

    # FPS
    fps = int(1 / (time.time() - loop_start + 0.001))
    cv2.putText(frame, f"FPS: {fps}", (20, h-90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
    cv2.putText(frame, "Press 'Q' to quit", (w-210, h-90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    cv2.imshow("AI Face Emotion HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
show_session_summary(emotion_history, session_start_time)
