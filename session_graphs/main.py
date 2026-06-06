import cv2
from deepface import DeepFace
import time

# -----------------------------
# CONFIG
# -----------------------------
EMOTIONS = ["angry", "happy", "sad", "surprise", "neutral"]

WHITE      = (255, 255, 255)
NEON_PINK  = (255,   0, 255)
GRAY       = (180, 180, 180)
GREEN      = (  0, 255,   0)
YELLOW     = (  0, 255, 255)
BLACK      = (  0,   0,   0)

EMOTION_COLORS = {
    "angry":    (0,   0,   255),
    "happy":    (0,   255,   0),
    "sad":      (255, 100,   0),
    "surprise": (0,   255, 255),
    "neutral":  (180, 180, 180),
}

EMOTION_EMOJI = {
    "angry":    "ANGRY",
    "happy":    "HAPPY",
    "sad":      "SAD",
    "surprise": "SURPRISE",
    "neutral":  "NEUTRAL",
}

ANALYZE_EVERY = 0.6
HISTORY_MAX   = 40

# -----------------------------
# VARIABLES
# -----------------------------
cap = cv2.VideoCapture(0)

last_analysis_time = 0
emotion_scores     = {e: 0 for e in EMOTIONS}
top_emotion        = "neutral"
top_conf           = 0

emotion_history    = []   # List of (emotion, confidence) tuples
session_start_time = time.time()

# -----------------------------
# HELPER: Fill rounded rect
# -----------------------------
def fill_rounded_rect(img, pt1, pt2, color, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1),          (x2 - radius, y2),          color, -1)
    cv2.rectangle(img, (x1,          y1 + radius),  (x2,          y2 - radius), color, -1)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180,  0, 90, color, -1)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270,  0, 90, color, -1)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),  90,  0, 90, color, -1)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),   0,  0, 90, color, -1)

# -----------------------------
# FUNCTION: Analyze Emotion
# -----------------------------
def analyze_emotion(frame_bgr):
    global emotion_scores, top_emotion, top_conf

    result = DeepFace.analyze(
        frame_bgr,
        actions=["emotion"],
        enforce_detection=False
    )

    emo_dict = result[0]["emotion"]
    for e in EMOTIONS:
        emotion_scores[e] = float(emo_dict.get(e, 0.0))

    top_emotion = max(EMOTIONS, key=lambda e: emotion_scores[e])
    top_conf    = int(emotion_scores[top_emotion])

# -----------------------------
# FUNCTION: Draw Timeline
# -----------------------------
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
                (12, panel_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_PINK, 1)

    dot_r   = 7
    dot_gap = 18
    start_x = 12
    dot_y   = panel_y + 50

    visible = history[-HISTORY_MAX:]

    for i in range(len(visible) - 1):
        x1 = start_x + i * dot_gap
        x2 = start_x + (i + 1) * dot_gap
        c1 = EMOTION_COLORS.get(visible[i][0], GRAY)
        c2 = EMOTION_COLORS.get(visible[i + 1][0], GRAY)
        mid_color = tuple((a + b) // 2 for a, b in zip(c1, c2))
        cv2.line(frame, (x1, dot_y), (x2, dot_y), mid_color, 1)

    for i, (emo, conf) in enumerate(visible):
        dot_x = start_x + i * dot_gap
        color = EMOTION_COLORS.get(emo, GRAY)

        if i == len(visible) - 1:
            cv2.circle(frame, (dot_x, dot_y), dot_r + 4, color, 1)

        cv2.circle(frame, (dot_x, dot_y), dot_r, color, -1)

        initial = emo[0].upper()
        cv2.putText(frame, initial,
                    (dot_x - 4, dot_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    # Legend
    legend_x = frame_w - 130
    legend_y = panel_y + 10
    cv2.putText(frame, "LEGEND",
                (legend_x, legend_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)
    for j, emo in enumerate(EMOTIONS):
        lx = legend_x
        ly = legend_y + 18 + j * 12
        cv2.circle(frame, (lx + 5, ly), 4, EMOTION_COLORS[emo], -1)
        cv2.putText(frame, emo.capitalize(),
                    (lx + 14, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, EMOTION_COLORS[emo], 1)

    # Session dominant
    if len(history) >= 3:
        freq_emo   = max(EMOTIONS, key=lambda e: sum(1 for h in history if h[0] == e))
        freq_color = EMOTION_COLORS.get(freq_emo, WHITE)
        cv2.putText(frame, f"Session dominant: {freq_emo.upper()}",
                    (12, panel_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, freq_color, 1)

# -----------------------------
# FUNCTION: Session Summary Screen
# -----------------------------
def show_session_summary(history, session_start):
    """
    Builds and displays a full summary screen after Q is pressed.
    Shows: total time, detection count, per-emotion % bars,
    dominant emotion, and average confidence.
    Waits for any key press to close.
    """
    if not history:
        return

    # ---- Compute stats ----
    total_time   = time.time() - session_start
    total_detections = len(history)
    minutes      = int(total_time // 60)
    seconds      = int(total_time % 60)

    # Count per emotion
    emotion_counts = {e: sum(1 for h in history if h[0] == e) for e in EMOTIONS}

    # Dominant emotion
    dominant_emo = max(EMOTIONS, key=lambda e: emotion_counts[e])

    # Average confidence across all detections
    avg_conf = int(sum(c for _, c in history) / total_detections) if total_detections else 0

    # Emotion percentages
    emotion_pct = {
        e: round((emotion_counts[e] / total_detections) * 100, 1)
        for e in EMOTIONS
    }

    # ---- Build summary frame ----
    sw, sh = 700, 500
    summary = cv2.imread.__func__ if False else None  # just to suppress warning
    summary = 255 * __import__("numpy").ones((sh, sw, 3), dtype=__import__("numpy").uint8)
    summary[:] = (18, 18, 18)  # Dark background

    # Title bar
    cv2.rectangle(summary, (0, 0), (sw, 60), (30, 0, 30), -1)
    cv2.putText(summary, "SESSION SUMMARY",
                (sw // 2 - 145, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, NEON_PINK, 2)

    # Subtitle line
    cv2.line(summary, (30, 65), (sw - 30, 65), NEON_PINK, 1)

    # ---- Stats row ----
    stats_y = 100
    # Total time
    cv2.putText(summary, "Total Time",
                (50, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1)
    cv2.putText(summary, f"{minutes:02d}m {seconds:02d}s",
                (50, stats_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)

    # Total detections
    cv2.putText(summary, "Detections",
                (230, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1)
    cv2.putText(summary, str(total_detections),
                (230, stats_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)

    # Avg confidence
    cv2.putText(summary, "Avg Confidence",
                (390, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1)
    cv2.putText(summary, f"{avg_conf}%",
                (390, stats_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)

    # Dominant emotion
    dom_color = EMOTION_COLORS.get(dominant_emo, WHITE)
    cv2.putText(summary, "Dominant Mood",
                (530, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1)
    cv2.putText(summary, dominant_emo.upper(),
                (530, stats_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, dom_color, 2)

    # Divider
    cv2.line(summary, (30, 155), (sw - 30, 155), (60, 60, 60), 1)

    # ---- Emotion breakdown bars ----
    cv2.putText(summary, "EMOTION BREAKDOWN",
                (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_PINK, 1)

    bar_x      = 50
    bar_max_w  = 480
    bar_h      = 22
    bar_gap    = 42
    bar_start_y = 210

    for i, emo in enumerate(EMOTIONS):
        by     = bar_start_y + i * bar_gap
        pct    = emotion_pct[emo]
        fill_w = int((pct / 100.0) * bar_max_w)
        color  = EMOTION_COLORS[emo]

        # Label
        cv2.putText(summary, f"{emo.capitalize():10s}",
                    (bar_x, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

        # Bar background
        cv2.rectangle(summary,
                      (bar_x + 110, by),
                      (bar_x + 110 + bar_max_w, by + bar_h),
                      (45, 45, 45), -1)

        # Bar fill
        if fill_w > 0:
            cv2.rectangle(summary,
                          (bar_x + 110, by),
                          (bar_x + 110 + fill_w, by + bar_h),
                          color, -1)

        # Highlight dominant
        if emo == dominant_emo:
            cv2.rectangle(summary,
                          (bar_x + 108, by - 2),
                          (bar_x + 112 + bar_max_w, by + bar_h + 2),
                          color, 1)

        # Percentage text
        cv2.putText(summary, f"{pct}%",
                    (bar_x + 110 + bar_max_w + 10, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # ---- Footer ----
    cv2.line(summary, (30, sh - 45), (sw - 30, sh - 45), (60, 60, 60), 1)
    cv2.putText(summary, "Press any key to exit",
                (sw // 2 - 110, sh - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1)

    # ---- Display ----
    cv2.imshow("Session Summary", summary)

    # ---- Also save summary to file ----
    with open("session_summary.txt", "w") as f:
        f.write("=" * 40 + "\n")
        f.write("       AI EMOTION SESSION SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Time       : {minutes:02d}m {seconds:02d}s\n")
        f.write(f"Total Detections : {total_detections}\n")
        f.write(f"Avg Confidence   : {avg_conf}%\n")
        f.write(f"Dominant Mood    : {dominant_emo.upper()}\n\n")
        f.write("Emotion Breakdown:\n")
        for emo in EMOTIONS:
            bar = "#" * int(emotion_pct[emo] / 5)
            f.write(f"  {emo:10s}: {emotion_pct[emo]:5.1f}%  {bar}\n")
        f.write("=" * 40 + "\n")

    print("\n[Summary saved to session_summary.txt]")
    cv2.waitKey(0)
    cv2.destroyWindow("Session Summary")

# -----------------------------
# MAIN LOOP
# -----------------------------
print("Camera started... Press Q to quit")

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ---- Header ----
    cv2.rectangle(frame, (0, 0), (w, 40), BLACK, -1)
    cv2.putText(frame, "AI FACE EMOTION HUD",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, NEON_PINK, 2)

    # Session timer in header
    elapsed     = int(time.time() - session_start_time)
    mins, secs  = divmod(elapsed, 60)
    cv2.putText(frame, f"Session: {mins:02d}:{secs:02d}",
                (w - 190, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1)

    # ---- AI Analysis (Throttled) ----
    if time.time() - last_analysis_time > ANALYZE_EVERY:
        try:
            analyze_emotion(frame)
            last_analysis_time = time.time()

            with open("emotion_log.txt", "a") as f:
                f.write(f"{time.ctime()} - {top_emotion} ({top_conf}%)\n")

            emotion_history.append((top_emotion, top_conf))
            if len(emotion_history) > HISTORY_MAX:
                emotion_history.pop(0)

        except Exception as e:
            print("Error:", e)

    # ---- Center Face Box ----
    box_w, box_h = 260, 300
    bx = w // 2 - box_w // 2
    by = h // 2 - box_h // 2 - 30

    cv2.rectangle(frame, (bx - 3, by - 3), (bx + box_w + 3, by + box_h + 3),
                  (200, 200, 200), 2)
    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), WHITE, 2)

    label = f"{top_emotion} ({top_conf}%)"
    cv2.rectangle(frame, (bx, by - 30), (bx + box_w, by), BLACK, -1)
    cv2.putText(frame, label, (bx + 10, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # ---- Side Panel: Emotion Bars ----
    panel_x, panel_y = 20, 60
    line_h    = 28
    max_bar_w = 160

    cv2.putText(frame, "Tracked emotions:",
                (panel_x, panel_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    for i, emo in enumerate(EMOTIONS):
        y_off = panel_y + i * line_h
        score = emotion_scores[emo]
        bar_w = int((score / 100.0) * max_bar_w)

        cv2.putText(frame, f"{emo:8s}",
                    (panel_x, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        cv2.rectangle(frame,
                      (panel_x + 90, y_off - 12),
                      (panel_x + 90 + max_bar_w, y_off + 4),
                      (50, 50, 50), -1)

        bar_color = EMOTION_COLORS[emo] if emo == top_emotion else GRAY
        cv2.rectangle(frame,
                      (panel_x + 90, y_off - 12),
                      (panel_x + 90 + bar_w, y_off + 4),
                      bar_color, -1)

    # ---- Alert System ----
    if top_emotion == "sad":
        cv2.putText(frame, "Cheer up! :)",
                    (w // 2 - 80, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
    elif top_emotion == "happy":
        cv2.putText(frame, "You look happy!",
                    (w // 2 - 100, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

    # ---- Emotion History Timeline ----
    draw_emotion_history(frame, emotion_history, w, h)

    # ---- FPS Counter ----
    fps = int(1 / (time.time() - start_time + 0.001))
    cv2.putText(frame, f"FPS: {fps}",
                (20, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

    # ---- Exit hint ----
    cv2.putText(frame, "Press 'Q' to quit",
                (w - 210, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    cv2.imshow("AI Face Emotion HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---- Show summary before closing ----
cap.release()
cv2.destroyAllWindows()
show_session_summary(emotion_history, session_start_time)