import ssl
import urllib.request

# Bypass SSL verification for MediaPipe's initial model downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np

# ============================================================
# RUNTIME SETTINGS
# Keep TensorFlow environment flags before importing TensorFlow.
# This helps avoid unnecessary startup noise and deployment issues.
# ============================================================
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

try:
    cv2.setNumThreads(1)
except Exception:
    pass

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SIGN_ASSETS_DIR = BASE_DIR / "sign_assets"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
MODEL_CANDIDATES = [
    MODEL_DIR / "best_bigru_attention_aug.tflite",
    MODEL_DIR / "model.tflite",
]



# ============================================================
# BRAND AND PRODUCT COPY
# ============================================================
APP_TITLE = "Beyond Words"
APP_SUBTITLE = "Where Silence Meets Understanding"
APP_SUMMARY = (
    "Beyond Words turns Pakistan Sign Language into clear text in real time and "
    "gives people a simple way to prepare sign friendly communication from text."
)
APP_DESCRIPTION = (
    "Built for restaurant ordering and service communication where fast and clear "
    "understanding matters during real customer interactions."
)
PRODUCT_OVERVIEW = (
    "This app helps restaurant staff understand customer signs faster and respond "
    "clearly during ordering, billing, takeaway, and service interactions."
)
PRODUCT_FEATURES = [
    "Real time sign to text translation with one start button",
    "Fast and simple camera flow with live recognition",
    "Text to sign planning area for supported phrases and sign sequences",
    "Clean interface designed for real product use",
    "Consistent visual style based on the uploaded brand theme",
    "Made for respectful and inclusive communication",
]
PRODUCT_USE_CASES = [
    "Restaurant counters and ordering desks",
    "Dining area service support",
    "Takeaway and billing communication",
    "Menu and order change requests",
]
PRODUCT_STEPS = [
    ("1", "Camera", "The camera reads hand and body movement in real time."),
    ("2", "AI Model", "The model detects the sign pattern and scores the best match."),
    ("3", "Text Output", "The recognized sign appears as clear text for staff and users."),
    ("4", "Response", "The team can reply faster and the conversation keeps moving."),
]
FLOW_IMAGE_BASE64 = ""  # Removed huge embedded image for faster deployment startup.


RESTAURANT_CONTEXT = (
    "The current sign library is tailored for restaurant ordering, billing, takeaway, and service communication."
)

# ============================================================
# DEPLOYMENT SETTINGS
# Render/Railway-style hosting requires binding to 0.0.0.0 and the PORT env variable.
# These defaults also still work locally.
# ============================================================
SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
ENABLE_SHARE_LINK = os.environ.get("ENABLE_SHARE_LINK", "false").lower() == "true"
OPEN_IN_BROWSER = os.environ.get("OPEN_IN_BROWSER", "false").lower() == "true"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 225
POSE_DIM = 33 * 3
LEFT_DIM = 21 * 3
RIGHT_DIM = 21 * 3
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
EPS = 1e-6

LIVE_DISPLAY_WIDTH = 640
LIVE_DISPLAY_HEIGHT = 360
LIVE_INPUT_WIDTH = 160
LIVE_INPUT_HEIGHT = 120
LIVE_STREAM_EVERY_SECONDS = float(os.environ.get("LIVE_STREAM_EVERY_SECONDS", "0.15"))
LIVE_PROCESS_MIN_INTERVAL_SECONDS = float(os.environ.get("LIVE_PROCESS_MIN_INTERVAL_SECONDS", "0.10"))
LIVE_ANALYZE_EVERY_N_CALLBACKS = int(os.environ.get("LIVE_ANALYZE_EVERY_N_CALLBACKS", "3"))
LIVE_MIN_SEQUENCE_FOR_PRED = 12
LIVE_PREDICT_EVERY_N_FRAMES = int(os.environ.get("LIVE_PREDICT_EVERY_N_FRAMES", "3"))
LIVE_SMOOTHING_WINDOW = 4
LIVE_MIN_STABLE_COUNT = 2
LIVE_CONFIDENCE_THRESHOLD = 0.78
LIVE_NO_SIGN_THRESHOLD = 0.50
LIVE_IDLE_RESET_FRAMES = 10
LIVE_MAX_HISTORY = 8

VIDEO_SAMPLE_FRAMES = 30
VIDEO_EXPORT_FPS = 8
VIDEO_PREVIEW_WIDTH = 960
VIDEO_PREVIEW_HEIGHT = 540

NO_SIGN_LABEL = "No sign recognized"

FALLBACK_SIGN_CLASSES = [
    "bacon",
    "bill",
    "card",
    "change_order",
    "done",
    "double",
    "green_sauce",
    "grilled",
    "jalapeno",
    "loaded",
    "masala",
    "mayo_garlic",
    "medium",
    "menu",
    "new_york",
    "one",
    "remove",
    "small",
    "soft_drinks",
    "takeaway",
    "three",
    "wait",
    "wings",
    "zinger",
]

# ============================================================
# THEME
# ============================================================
COLOR_BG = "#F8FAFF"
COLOR_SURFACE = "#FFFFFF"
COLOR_TEXT = "#171A24"
COLOR_MUTED = "#5F6476"
COLOR_BORDER = "#E7E8F2"
COLOR_PINK = "#F44DB8"
COLOR_PURPLE = "#7C4DFF"
COLOR_CYAN = "#27C8FF"
COLOR_SOFT_PINK = "#FFF2FB"
COLOR_SOFT_BLUE = "#EFF9FF"
COLOR_SOFT_PURPLE = "#F5F1FF"
COLOR_SUCCESS = "#0C8C5E"
COLOR_WARNING = "#B06C00"
COLOR_DANGER = "#C43D5A"

CUSTOM_CSS = f"""
:root {{
  --bw-bg: {COLOR_BG};
  --bw-surface: {COLOR_SURFACE};
  --bw-text: {COLOR_TEXT};
  --bw-muted: {COLOR_MUTED};
  --bw-border: {COLOR_BORDER};
  --bw-pink: {COLOR_PINK};
  --bw-purple: {COLOR_PURPLE};
  --bw-cyan: {COLOR_CYAN};
  --bw-soft-pink: {COLOR_SOFT_PINK};
  --bw-soft-blue: {COLOR_SOFT_BLUE};
  --bw-soft-purple: {COLOR_SOFT_PURPLE};
  --bw-success: {COLOR_SUCCESS};
  --bw-warning: {COLOR_WARNING};
  --bw-danger: {COLOR_DANGER};
}}

body, .gradio-container {{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
  background:
    radial-gradient(circle at top left, rgba(244,77,184,0.10), transparent 28%),
    radial-gradient(circle at top right, rgba(39,200,255,0.12), transparent 30%),
    linear-gradient(180deg, #F9FBFF 0%, #F6F8FE 100%);
  color: var(--bw-text);
}}

.gradio-container {{
  max-width: 1280px !important;
}}

.bw-hero {{
  background: linear-gradient(135deg, rgba(244,77,184,0.12), rgba(124,77,255,0.12), rgba(39,200,255,0.10));
  border: 1px solid rgba(124,77,255,0.16);
  border-radius: 30px;
  padding: 40px 34px;
  box-shadow: 0 18px 45px rgba(103, 83, 181, 0.08);
}}

.bw-hero-grid {{
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 22px;
  align-items: center;
}}

.bw-brand-kicker {{
  display: inline-block;
  padding: 8px 14px;
  border-radius: 999px;
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(124,77,255,0.14);
  color: var(--bw-purple);
  font-weight: 700;
  font-size: 0.92rem;
  margin-bottom: 14px;
}}

.bw-title {{
  margin: 0;
  font-size: clamp(3.2rem, 6vw, 5.6rem);
  line-height: 0.95;
  font-weight: 900;
  letter-spacing: -0.04em;
  color: #6F3CFF;
  text-shadow: 0 2px 10px rgba(124, 77, 255, 0.10);
}}

.bw-subtitle {{
  margin-top: 14px;
  margin-bottom: 12px;
  font-size: clamp(1.15rem, 2vw, 1.75rem);
  color: var(--bw-text);
  font-weight: 700;
}}

.bw-lead {{
  margin: 0;
  color: var(--bw-muted);
  font-size: 1.02rem;
  line-height: 1.7;
}}

.bw-card {{
  background: rgba(255,255,255,0.92);
  border: 1px solid var(--bw-border);
  border-radius: 22px;
  padding: 24px;
  box-shadow: 0 10px 28px rgba(24, 35, 68, 0.05);
}}

.bw-card h2, .bw-card h3 {{
  margin-top: 0;
  color: var(--bw-text);
}}

.bw-grid-2 {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}}

.bw-grid-3 {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}}

.bw-grid-4 {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}}

.bw-section-title {{
  font-size: 1.25rem;
  font-weight: 800;
  margin: 0 0 10px 0;
}}

.bw-muted {{
  color: var(--bw-muted);
  line-height: 1.7;
}}

.bw-list {{
  margin: 0;
  padding-left: 18px;
  color: var(--bw-muted);
  line-height: 1.8;
}}

.bw-stat {{
  background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(249,250,255,0.92));
  border: 1px solid rgba(124,77,255,0.12);
  border-radius: 20px;
  padding: 18px;
  text-align: left;
}}

.bw-stat-value {{
  font-size: 1.85rem;
  font-weight: 900;
  color: var(--bw-text);
  margin-bottom: 6px;
}}

.bw-stat-label {{
  font-size: 0.96rem;
  color: var(--bw-muted);
}}

.bw-stats-chart {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}}

.bw-chart-card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,249,255,0.95));
  border: 1px solid rgba(124,77,255,0.12);
  border-radius: 22px;
  padding: 20px;
}}

.bw-chart-top {{
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
}}

.bw-chart-value {{
  font-size: 2rem;
  font-weight: 900;
  color: var(--bw-text);
}}

.bw-chart-unit {{
  font-size: 0.92rem;
  color: var(--bw-muted);
  font-weight: 700;
}}

.bw-chart-label {{
  font-size: 0.98rem;
  font-weight: 800;
  color: var(--bw-text);
  margin-bottom: 12px;
}}

.bw-chart-track {{
  width: 100%;
  height: 14px;
  border-radius: 999px;
  background: rgba(124,77,255,0.10);
  overflow: hidden;
  position: relative;
}}

.bw-chart-fill {{
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--bw-pink), var(--bw-purple), var(--bw-cyan));
}}

.bw-chart-note {{
  margin-top: 10px;
  color: var(--bw-muted);
  font-size: 0.90rem;
}}

.bw-chip-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 16px;
}}

.bw-chip {{
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid rgba(124,77,255,0.12);
  background: rgba(255,255,255,0.78);
  color: var(--bw-text);
  font-weight: 600;
  font-size: 0.92rem;
}}

.bw-step {{
  background: rgba(255,255,255,0.92);
  border: 1px solid var(--bw-border);
  border-radius: 22px;
  padding: 20px;
}}

.bw-step-number {{
  width: 42px;
  height: 42px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, rgba(244,77,184,0.16), rgba(39,200,255,0.16));
  color: var(--bw-purple);
  font-weight: 900;
  margin-bottom: 12px;
}}

.bw-action-card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,249,255,0.95));
  border: 1px solid rgba(124,77,255,0.12);
  border-radius: 22px;
  padding: 24px;
}}

.bw-action-title {{
  font-size: 1.18rem;
  font-weight: 800;
  margin-bottom: 8px;
  color: var(--bw-text);
}}

.bw-action-copy {{
  font-size: 0.98rem;
  line-height: 1.7;
  color: var(--bw-muted);
}}

.bw-badge {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 130px;
  padding: 8px 16px;
  border-radius: 999px;
  font-size: 0.90rem;
  font-weight: 800;
}}

.bw-badge.idle {{
  color: #4E5A69;
  background: rgba(109, 121, 138, 0.12);
}}

.bw-badge.success {{
  color: var(--bw-success);
  background: rgba(12, 140, 94, 0.14);
}}

.bw-badge.warning {{
  color: var(--bw-warning);
  background: rgba(176, 108, 0, 0.14);
}}

.bw-badge.danger {{
  color: var(--bw-danger);
  background: rgba(196, 61, 90, 0.14);
}}

.bw-output-box {{
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,247,255,0.96));
  border: 1px solid rgba(124,77,255,0.10);
  border-radius: 22px;
  padding: 22px;
}}

.bw-sequence {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin-top: 10px;
}}

.bw-sequence-card {{
  padding: 18px;
  border-radius: 18px;
  border: 1px solid rgba(124,77,255,0.12);
  background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(245,247,255,0.92));
}}

.bw-sequence-index {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  margin-bottom: 12px;
  background: linear-gradient(135deg, rgba(244,77,184,0.18), rgba(39,200,255,0.18));
  color: var(--bw-purple);
  font-weight: 900;
}}

.bw-sequence-word {{
  font-size: 1.05rem;
  font-weight: 800;
  color: var(--bw-text);
  margin-bottom: 8px;
}}

.bw-sequence-note {{
  color: var(--bw-muted);
  line-height: 1.55;
  font-size: 0.92rem;
}}

.bw-small-note {{
  font-size: 0.92rem;
  color: var(--bw-muted);
  line-height: 1.7;
}}

#home-actions button,
#sign-actions button,
#text-actions button {{
  min-height: 56px !important;
  border-radius: 18px !important;
  font-weight: 800 !important;
}}

.tabs > .tab-nav {{
  background: rgba(255,255,255,0.76);
  border: 1px solid rgba(124,77,255,0.10);
  border-radius: 18px;
  padding: 8px;
  gap: 6px;
}}

.tabs > .tab-nav button {{
  border-radius: 14px !important;
}}

@media (max-width: 900px) {{
  .bw-hero-grid,
  .bw-grid-2,
  .bw-grid-3,
  .bw-grid-4,
  .bw-stats-chart {{
    grid-template-columns: 1fr;
  }}
}}

"""

# ============================================================
# HELPERS
# ============================================================
def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def display_label(value: str) -> str:
    return (value or "").replace("_", " ").title()


def tokenize_text(value: str) -> List[str]:
    cleaned = re.sub(r"[^A-Za-z0-9\s]+", " ", value or "")
    return [token for token in cleaned.split() if token]


def class_phrase_words(class_name: str) -> List[str]:
    return normalize_token(class_name.replace("_", " ")).split()


def load_class_names() -> List[str]:
    canonical = list(FALLBACK_SIGN_CLASSES)
    if CLASS_NAMES_PATH.exists():
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, list) and data:
                loaded = [str(item) for item in data]
                canonical_keys = [normalize_key(item) for item in canonical]
                loaded_keys = [normalize_key(item) for item in loaded]
                if loaded_keys == canonical_keys or set(loaded_keys) == set(canonical_keys):
                    return canonical
                return loaded
        except Exception:
            pass
    return canonical


def locate_model_path() -> Optional[Path]:
    preferred = MODEL_DIR / "best_bigru_attention_aug.tflite"
    if preferred.exists():
        return preferred

    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def badge_html(status: Optional[str]) -> str:
    status_text = (status or "Idle").strip()
    lower = status_text.lower()
    badge_class = "idle"
    if lower in {"recognized", "ready", "running"}:
        badge_class = "success"
    elif lower in {"starting", "warming up", "predicting", "low confidence", "preparing"}:
        badge_class = "warning"
    elif lower in {"error", "camera error", "model missing"}:
        badge_class = "danger"
    return f'<div class="bw-badge {badge_class}">{status_text}</div>'


def history_text(history: List[str]) -> str:
    if not history:
        return "No translations yet."
    return "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(history[-LIVE_MAX_HISTORY:]))


def resize_frame_keep_aspect(frame: np.ndarray, target_w: int, target_h: int, fill_value: int = 255) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return cv2.resize(frame, (target_w, target_h))
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((target_h, target_w, 3), fill_value, dtype=np.uint8)
    x1 = (target_w - new_w) // 2
    y1 = (target_h - new_h) // 2
    canvas[y1:y1 + new_h, x1:x1 + new_w] = resized
    return canvas


def sample_frame_indices(total_frames: int, sequence_length: int = SEQUENCE_LENGTH) -> List[int]:
    if total_frames <= 0:
        return [0] * sequence_length

    if total_frames >= sequence_length:
        indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
    else:
        indices = list(range(total_frames))
        while len(indices) < sequence_length:
            indices.append(total_frames - 1)
        indices = np.array(indices, dtype=int)

    return indices.tolist()


def overlay_prediction_panel(
    frame_rgb: np.ndarray,
    label: str,
    confidence: float,
    status: str,
    helper_text: str = "",
    show_video_background: bool = True,
    show_status_badge: bool = True,
    show_helper_text: bool = True,
) -> np.ndarray:
    if show_video_background:
        display = resize_frame_keep_aspect(frame_rgb, LIVE_DISPLAY_WIDTH, LIVE_DISPLAY_HEIGHT)
    else:
        display = np.full((LIVE_DISPLAY_HEIGHT, LIVE_DISPLAY_WIDTH, 3), 255, dtype=np.uint8)

    cv2.rectangle(display, (18, 18), (LIVE_DISPLAY_WIDTH - 18, 158), (255, 255, 255), -1)
    cv2.rectangle(display, (18, 18), (LIVE_DISPLAY_WIDTH - 18, 158), (234, 235, 244), 1)

    cv2.putText(display, APP_TITLE, (34, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.02, (66, 32, 74), 2)
    cv2.putText(display, APP_SUBTITLE, (34, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (90, 92, 108), 2)
    cv2.putText(display, f"Live Output: {display_label(label)}", (34, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (24, 27, 37), 2)
    cv2.putText(display, f"Confidence: {confidence:.2f}", (34, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (90, 92, 108), 2)

    if show_status_badge:
        status_lower = (status or "Idle").lower()
        status_color = (111, 122, 137)
        if status_lower in {"recognized", "ready", "running"}:
            status_color = (12, 140, 94)
        elif status_lower in {"starting", "warming up", "predicting", "low confidence", "preparing"}:
            status_color = (176, 108, 0)
        elif status_lower in {"error", "model missing"}:
            status_color = (196, 61, 90)

        cv2.rectangle(display, (LIVE_DISPLAY_WIDTH - 236, 28), (LIVE_DISPLAY_WIDTH - 28, 72), status_color, -1)
        cv2.putText(display, status, (LIVE_DISPLAY_WIDTH - 220, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

    if helper_text and show_helper_text:
        cv2.rectangle(display, (18, LIVE_DISPLAY_HEIGHT - 78), (LIVE_DISPLAY_WIDTH - 18, LIVE_DISPLAY_HEIGHT - 24), (255, 255, 255), -1)
        cv2.rectangle(display, (18, LIVE_DISPLAY_HEIGHT - 78), (LIVE_DISPLAY_WIDTH - 18, LIVE_DISPLAY_HEIGHT - 24), (234, 235, 244), 1)
        cv2.putText(display, helper_text[:110], (34, LIVE_DISPLAY_HEIGHT - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (76, 78, 90), 1)

    return display


def blank_preview(message: str) -> np.ndarray:
    return overlay_prediction_panel(
        np.full((LIVE_DISPLAY_HEIGHT, LIVE_DISPLAY_WIDTH, 3), 255, dtype=np.uint8),
        NO_SIGN_LABEL,
        0.0,
        "Idle",
        "",
        show_video_background=False,
        show_status_badge=False,
        show_helper_text=False,
    )


def resolve_uploaded_video_path(video_value: Any) -> str:
    if isinstance(video_value, str) and video_value:
        return video_value

    if isinstance(video_value, dict):
        for key in ("path", "video", "name"):
            value = video_value.get(key)
            if isinstance(value, str) and value:
                return value

    if isinstance(video_value, (list, tuple)) and video_value:
        first = video_value[0]
        if isinstance(first, str) and first:
            return first
        if isinstance(first, dict):
            for key in ("path", "video", "name"):
                value = first.get(key)
                if isinstance(value, str) and value:
                    return value

    raise ValueError("Please upload a prerecorded video.")


def make_browser_playable_video(input_path: str) -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return input_path

    input_file = Path(input_path)
    output_mp4 = str(input_file.with_name(input_file.stem + "_browser.mp4"))

    command = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_mp4,
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0 and Path(output_mp4).exists() and Path(output_mp4).stat().st_size > 0:
            return output_mp4
    except Exception:
        pass

    return input_path

def pad_sequence_to_model_length(sequence: List[np.ndarray]) -> List[np.ndarray]:
    """Used specifically for LIVE camera sliding window (takes LAST 30 frames)"""
    if not sequence:
        return [np.zeros((FEATURE_DIM,), dtype=np.float32) for _ in range(SEQUENCE_LENGTH)]
    seq = list(sequence)
    if len(seq) >= SEQUENCE_LENGTH:
        return seq[-SEQUENCE_LENGTH:]
    last = seq[-1]
    while len(seq) < SEQUENCE_LENGTH:
        seq.append(last.copy())
    return seq


def init_live_state() -> Dict[str, Any]:
    return {
        "running": False,
        "sequence_buffer": [],
        "prediction_queue": [],
        "history": [],
        "frame_count": 0,
        "stream_count": 0,
        "idle_frame_counter": 0,
        "display_label": NO_SIGN_LABEL,
        "display_conf": 0.0,
        "status": "Idle",
        "last_stable_label": None,
        "last_preview_rgb": None,
        "last_process_ts": 0.0,
        "prev_center": None,
        "prev_scale": 1.0,
    }


def start_live_session(_: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, str, str, str, str, Dict[str, Any]]:
    state = init_live_state()
    state["running"] = True
    state["status"] = "Starting"
    preview = blank_preview("Allow camera access, then show a sign and tap Start Live Translation.")
    return preview, NO_SIGN_LABEL, "0.00", badge_html("Starting"), history_text([]), state


def stop_live_session(state: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, str, str, str, str, Dict[str, Any]]:
    next_state = init_live_state()
    if isinstance(state, dict):
        next_state["history"] = list(state.get("history", []))[-LIVE_MAX_HISTORY:]
    next_state["status"] = "Idle"
    preview = blank_preview("Live translation is stopped.")
    return preview, NO_SIGN_LABEL, "0.00", badge_html("Idle"), history_text(next_state["history"]), next_state


def steps_html() -> str:
    cards = []
    for number, title, copy in PRODUCT_STEPS:
        cards.append(
            f"""
            <div class="bw-step">
                <div class="bw-step-number">{number}</div>
                <div class="bw-section-title">{title}</div>
                <div class="bw-muted">{copy}</div>
            </div>
            """
        )
    return f'<div class="bw-grid-4">{"".join(cards)}</div>'


def home_page_html() -> str:
    use_cases = ''.join(f'<span class="bw-chip">{item}</span>' for item in PRODUCT_USE_CASES)
    return f"""
    <div class="bw-hero">
      <div class="bw-hero-grid">
        <div>
          <div class="bw-brand-kicker">Restaurant Sign Communication Assistant</div>
          <h1 class="bw-title">{APP_TITLE}</h1>
          <div class="bw-subtitle">{APP_SUBTITLE}</div>
          <p class="bw-lead">{APP_SUMMARY}</p>
          <p class="bw-lead" style="margin-top: 14px;">{APP_DESCRIPTION}</p>
          <div class="bw-chip-row">{use_cases}</div>
        </div>
        <div class="bw-card">
          <div class="bw-section-title">What this app does</div>
          <div class="bw-muted">{PRODUCT_OVERVIEW}</div>
          <div class="bw-small-note" style="margin-top: 14px;">
            {RESTAURANT_CONTEXT}
          </div>
        </div>
      </div>
    </div>
    """


def app_guide_page_html() -> str:
    feature_items = "".join(f"<li>{item}</li>" for item in PRODUCT_FEATURES)
    restaurant_signs = ", ".join(display_label(item) for item in load_class_names())
    return f"""
    <div class="bw-hero">
      <div class="bw-hero-grid">
        <div>
          <div class="bw-brand-kicker">User Guide</div>
          <h1 class="bw-title">Quick Start</h1>
          <div class="bw-subtitle">Learn the app in three simple workflows</div>
          <p class="bw-lead">
            Use this guide to understand the main app sections before you start live translation, prepare text-to-sign output, or review a recorded video.
          </p>
        </div>
        <div class="bw-card">
          <div class="bw-section-title">How it works</div>
          <div class="bw-muted" style="margin-bottom: 14px;">
            The app captures a sign, reads the movement pattern, predicts the best class, and presents the result in a clean format.
          </div>
          <div class="bw-small-note">
            Current supported restaurant sign classes include {restaurant_signs}.
          </div>
        </div>
      </div>
    </div>
    <div style="height: 18px;"></div>
    <div class="bw-card">
      <div class="bw-section-title">App Flow</div>
      <div class="bw-grid-4" style="margin-top: 14px;">
        <div class="bw-mini-card">
          <div class="bw-step-number">1</div>
          <div class="bw-mini-title">Camera</div>
          <div class="bw-muted">The app captures the user's sign through the camera.</div>
        </div>
        <div class="bw-mini-card">
          <div class="bw-step-number">2</div>
          <div class="bw-mini-title">AI Model</div>
          <div class="bw-muted">MediaPipe landmarks are processed by the TFLite model.</div>
        </div>
        <div class="bw-mini-card">
          <div class="bw-step-number">3</div>
          <div class="bw-mini-title">Prediction</div>
          <div class="bw-muted">The best matching restaurant sign class is selected.</div>
        </div>
        <div class="bw-mini-card">
          <div class="bw-step-number">4</div>
          <div class="bw-mini-title">Text Output</div>
          <div class="bw-muted">The recognized sign appears as clear text for communication.</div>
        </div>
      </div>
    </div>
    <div style="height: 18px;"></div>
    <div class="bw-grid-2">
      <div class="bw-card">
        <div class="bw-section-title">Core features</div>
        <ul class="bw-list">{feature_items}</ul>
      </div>
      <div class="bw-card">
        <div class="bw-section-title">Before you begin</div>
        <ul class="bw-list">
          <li>Use good lighting so the camera can see the hands clearly.</li>
          <li>Keep the full upper body and both hands inside the frame.</li>
          <li>Show one clear sign at a time for better recognition.</li>
          <li>Use short restaurant phrases when working with text-to-sign.</li>
        </ul>
      </div>
    </div>
    <div style="height: 18px;"></div>
    <div class="bw-card">
      <div class="bw-section-title">Sign to Text Tutorial</div>
      <div class="bw-grid-4">
        <div class="bw-step">
          <div class="bw-step-number">1</div>
          <div class="bw-section-title">Open</div>
          <div class="bw-muted">Go to Sign to Text.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">2</div>
          <div class="bw-section-title">Webcam</div>
          <div class="bw-muted">Click to Access Webcam and allow camera permission when the browser asks.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">3</div>
          <div class="bw-section-title">Start</div>
          <div class="bw-muted">Click Start Live Translation so the app begins reading camera input.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">4</div>
          <div class="bw-section-title">Record</div>
          <div class="bw-muted">Click Record, keep your hands visible, and perform one supported restaurant sign clearly.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">5</div>
          <div class="bw-section-title">Prediction</div>
          <div class="bw-muted">The model predicts the sign and shows the recognized text, confidence score, status badge, and recent translations.</div>
        </div>
      </div>
    </div>
    <div style="height: 18px;"></div>
    <div class="bw-card">
      <div class="bw-section-title">Text to Sign Tutorial</div>
      <div class="bw-grid-4">
        <div class="bw-step">
          <div class="bw-step-number">1</div>
          <div class="bw-section-title">Open</div>
          <div class="bw-muted">Go to Text to Sign.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">2</div>
          <div class="bw-section-title">Type</div>
          <div class="bw-muted">Enter a short restaurant phrase, such as menu zinger takeaway.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">3</div>
          <div class="bw-section-title">Start</div>
          <div class="bw-muted">Click Start Text to Sign to prepare the sign sequence.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">4</div>
          <div class="bw-section-title">Review</div>
          <div class="bw-muted">Check the generated sign sequence and read the summary.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">5</div>
          <div class="bw-section-title">Play</div>
          <div class="bw-muted">Select a recorded sign from the dropdown to play its video.</div>
        </div>
      </div>
    </div>
    <div style="height: 18px;"></div>
    <div class="bw-card">
      <div class="bw-section-title">Recorded Videos Prediction Tutorial</div>
      <div class="bw-grid-4">
        <div class="bw-step">
          <div class="bw-step-number">1</div>
          <div class="bw-section-title">Open</div>
          <div class="bw-muted">Go to Recorded Videos Prediction.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">2</div>
          <div class="bw-section-title">Upload</div>
          <div class="bw-muted">Upload a clear sign video from your computer.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">3</div>
          <div class="bw-section-title">Run</div>
          <div class="bw-muted">Click Run Recorded Videos Prediction to start analysis.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">4</div>
          <div class="bw-section-title">Preview</div>
          <div class="bw-muted">Wait for the processed preview video and prediction result.</div>
        </div>
        <div class="bw-step">
          <div class="bw-step-number">5</div>
          <div class="bw-section-title">Result</div>
          <div class="bw-muted">Check the predicted text, confidence value, status, and summary.</div>
        </div>
      </div>
    </div>
    """


# ============================================================
# TEXT TO SIGN HELPERS
# ============================================================
def build_sign_asset_index(class_names: List[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not SIGN_ASSETS_DIR.exists():
        return index

    allowed_extensions = {".mp4", ".webm"}
    for file in SIGN_ASSETS_DIR.rglob("*"):
        if file.is_file() and file.suffix.lower() in allowed_extensions:
            index[normalize_key(file.stem)] = file

    for class_name in class_names:
        key = normalize_key(class_name)
        if key in index:
            continue
        class_folder = SIGN_ASSETS_DIR / key
        if class_folder.exists() and class_folder.is_dir():
            for file in class_folder.iterdir():
                if file.is_file() and file.suffix.lower() in allowed_extensions:
                    index[key] = file
                    break
    return index


def find_sign_matches(text: str, class_names: List[str]) -> List[Dict[str, Any]]:
    tokens = tokenize_text(text)
    if not tokens:
        return []

    patterns: List[Tuple[List[str], str]] = []
    for class_name in class_names:
        patterns.append((class_phrase_words(class_name), class_name))
    patterns.sort(key=lambda item: len(item[0]), reverse=True)

    items: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        matched = False
        for words, class_name in patterns:
            width = len(words)
            if width == 0 or i + width > len(tokens):
                continue
            if [token.lower() for token in tokens[i:i + width]] == words:
                items.append({
                    "type": "class",
                    "class_name": class_name,
                    "display": display_label(class_name),
                    "source_text": " ".join(tokens[i:i + width]),
                })
                i += width
                matched = True
                break
        if not matched:
            items.append({
                "type": "text",
                "class_name": None,
                "display": tokens[i].title(),
                "source_text": tokens[i],
            })
            i += 1
    return items


def resolve_sign_media(selection: Optional[str], asset_index: Dict[str, Path]) -> Tuple[Optional[str], str]:
    if not selection:
        return None, "No sign asset selected."
    key = normalize_key(selection)
    file_path = asset_index.get(key)
    if file_path is None:
        return None, f"No asset file found for {display_label(selection)}."
    return str(file_path), f"Playing recorded sign for {display_label(selection)}."


def default_text_to_sign_state() -> Tuple[str, str, str, Any, Optional[str], str]:
    class_names = load_class_names()
    supported_signs_text = ", ".join(display_label(item) for item in class_names)
    asset_index = build_sign_asset_index(class_names)
    asset_choices = [display_label(item) for item in class_names if normalize_key(item) in asset_index]
    empty_html = (
        '<div class="bw-output-box"><div class="bw-section-title">Text to Sign</div>'
        '<div class="bw-muted">Enter a phrase, then tap Start Text to Sign.</div></div>'
    )
    if asset_choices:
        first_selection = asset_choices[0]
        video_value, preview_note = resolve_sign_media(first_selection, asset_index)
        dropdown_update = gr.update(choices=asset_choices, value=first_selection)
    else:
        video_value, preview_note = None, "Add recorded sign files inside the sign_assets folder to enable playback."
        dropdown_update = gr.update(choices=[], value=None)
    return empty_html, "Ready.", supported_signs_text, dropdown_update, video_value, preview_note


def prepare_text_to_sign(text: str):
    raw = (text or "").strip()
    class_names = load_class_names()
    asset_index = build_sign_asset_index(class_names)

    if not raw:
        return default_text_to_sign_state()

    items = find_sign_matches(raw, class_names)
    if not items:
        return default_text_to_sign_state()

    cards = []
    matched_classes: List[str] = []
    asset_ready_classes: List[str] = []

    for idx, item in enumerate(items, start=1):
        if item["type"] == "class" and item["class_name"]:
            class_name = str(item["class_name"])
            matched_classes.append(class_name)
            if normalize_key(class_name) in asset_index:
                asset_ready_classes.append(class_name)
                note = f"Matched to supported class {display_label(class_name)}. Recorded sign asset is ready."
            else:
                note = f"Matched to supported class {display_label(class_name)}. Add a sign asset file to enable playback."
        else:
            note = "No direct sign class match was found for this part of the phrase."

        cards.append(
            f"""
            <div class="bw-sequence-card">
              <div class="bw-sequence-index">{idx}</div>
              <div class="bw-sequence-word">{item["display"]}</div>
              <div class="bw-sequence-note">{note}</div>
            </div>
            """
        )

    html = f"""
    <div class="bw-output-box">
      <div class="bw-section-title">Text to Sign Output</div>
      <div class="bw-muted">
        The phrase is split into a simple sign sequence. Use the dropdown below to play the recorded sign videos for supported classes.
      </div>
      <div class="bw-sequence">{"".join(cards)}</div>
    </div>
    """

    summary = (
        f"Prepared {len(items)} sequence items. "
        f"{len(matched_classes)} matched supported sign classes. "
        f"{len(asset_ready_classes)} have recorded sign assets ready."
    )
    supported_signs_text = ", ".join(display_label(item) for item in class_names)

    unique_asset_classes = []
    seen = set()
    for class_name in asset_ready_classes:
        key = normalize_key(class_name)
        if key not in seen:
            seen.add(key)
            unique_asset_classes.append(class_name)

    if unique_asset_classes:
        asset_choices = [display_label(item) for item in unique_asset_classes]
        default_selection = asset_choices[0]
        dropdown_update = gr.update(choices=asset_choices, value=default_selection)
        video_value, preview_note = resolve_sign_media(default_selection, asset_index)
    else:
        all_asset_classes = [item for item in class_names if normalize_key(item) in asset_index]
        asset_choices = [display_label(item) for item in all_asset_classes]
        if asset_choices:
            default_selection = asset_choices[0]
            dropdown_update = gr.update(choices=asset_choices, value=default_selection)
            video_value, preview_note = resolve_sign_media(default_selection, asset_index)
            preview_note = (
                "No recorded sign asset matched the full phrase yet. "
                + preview_note
            )
        else:
            dropdown_update = gr.update(choices=[], value=None)
            video_value, preview_note = None, "Add recorded sign files inside the sign_assets folder to enable playback."

    return html, summary, supported_signs_text, dropdown_update, video_value, preview_note


def update_sign_asset_preview(selection: Optional[str]):
    class_names = load_class_names()
    asset_index = build_sign_asset_index(class_names)
    return resolve_sign_media(selection, asset_index)

# ============================================================
# INFERENCE ENGINE
# ============================================================
@dataclass
class PredictionResult:
    label: str
    confidence: float
    status: str


class PSLInferenceEngine:
    def __init__(self) -> None:
        self.class_names = load_class_names()
        model_path = locate_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"No TFLite model was found in {MODEL_DIR}. Add model.tflite or best_bigru_attention_aug.tflite."
            )

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]["shape"]
        dummy = np.zeros(input_shape, dtype=np.float32)
        self._set_input_tensor(dummy)
        self.interpreter.invoke()
        _ = self._get_output_tensor()

        self.mp_holistic = mp.solutions.holistic
        self.live_holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.video_holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )

    def _set_input_tensor(self, arr: np.ndarray) -> None:
        input_info = self.input_details[0]
        input_dtype = input_info["dtype"]
        tensor = np.asarray(arr, dtype=np.float32)

        if np.issubdtype(input_dtype, np.floating):
            tensor = tensor.astype(input_dtype)
        elif np.issubdtype(input_dtype, np.integer):
            scale, zero_point = input_info.get("quantization", (0.0, 0))
            if scale and scale > 0:
                tensor = np.round(tensor / scale + zero_point).astype(input_dtype)
            else:
                tensor = tensor.astype(input_dtype)
        else:
            tensor = tensor.astype(input_dtype)

        self.interpreter.set_tensor(input_info["index"], tensor)

    def _get_output_tensor(self) -> np.ndarray:
        output_info = self.output_details[0]
        output = self.interpreter.get_tensor(output_info["index"])
        output_dtype = output_info["dtype"]

        if np.issubdtype(output_dtype, np.integer):
            scale, zero_point = output_info.get("quantization", (0.0, 0))
            if scale and scale > 0:
                output = (output.astype(np.float32) - zero_point) * scale
            else:
                output = output.astype(np.float32)
        else:
            output = output.astype(np.float32)

        return output

    def close(self) -> None:
        for item in (getattr(self, "live_holistic", None), getattr(self, "video_holistic", None)):
            try:
                if item is not None:
                    item.close()
            except Exception:
                pass

    def extract_landmarks(self, results: Any) -> np.ndarray:
        pose = np.zeros((33, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark[:33]):
                pose[idx] = [landmark.x, landmark.y, landmark.z]

        if results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark[:21]):
                left_hand[idx] = [landmark.x, landmark.y, landmark.z]

        if results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark[:21]):
                right_hand[idx] = [landmark.x, landmark.y, landmark.z]

        return np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()]).astype(np.float32)

    def is_missing_landmark(self, point: np.ndarray, eps: float = 1e-6) -> bool:
        return bool(np.all(np.abs(point) < eps))

    def normalize_frame_with_state(
        self,
        frame_225: np.ndarray,
        prev_center: Optional[np.ndarray] = None,
        prev_scale: float = 1.0,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        pose = frame_225[:POSE_DIM].reshape(33, 3)
        left_hand = frame_225[POSE_DIM:POSE_DIM + LEFT_DIM].reshape(21, 3)
        right_hand = frame_225[POSE_DIM + LEFT_DIM:].reshape(21, 3)

        left_shoulder = pose[LEFT_SHOULDER_IDX]
        right_shoulder = pose[RIGHT_SHOULDER_IDX]

        shoulders_valid = (
            not self.is_missing_landmark(left_shoulder, eps=eps)
            and not self.is_missing_landmark(right_shoulder, eps=eps)
        )

        if shoulders_valid:
            center = (left_shoulder + right_shoulder) / 2.0
            scale = float(np.linalg.norm(left_shoulder - right_shoulder))
            if scale < eps:
                scale = float(prev_scale) if prev_scale and prev_scale > eps else 1.0
        else:
            center = prev_center.copy() if prev_center is not None else np.zeros(3, dtype=np.float32)
            scale = float(prev_scale) if prev_scale and prev_scale > eps else 1.0

        def normalize_block(block: np.ndarray) -> np.ndarray:
            output = np.zeros_like(block, dtype=np.float32)
            mask = np.any(np.abs(block) > eps, axis=1)
            if np.any(mask):
                output[mask] = (block[mask] - center) / scale
            return output

        normalized = np.concatenate(
            [normalize_block(pose), normalize_block(left_hand), normalize_block(right_hand)],
            axis=0,
        )

        return normalized.flatten().astype(np.float32), center.astype(np.float32), float(scale)

    def normalize_frame(self, frame_225: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        normalized, _, _ = self.normalize_frame_with_state(
            np.asarray(frame_225, dtype=np.float32),
            prev_center=None,
            prev_scale=1.0,
            eps=eps,
        )
        return normalized

    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            fixed = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
            rows = min(sequence.shape[0], SEQUENCE_LENGTH)
            cols = min(sequence.shape[1], FEATURE_DIM) if sequence.ndim > 1 else 0
            if cols > 0:
                fixed[:rows, :cols] = sequence[:rows, :cols]
            sequence = fixed

        normalized_sequence = np.zeros_like(sequence, dtype=np.float32)
        prev_center = None
        prev_scale = 1.0

        for i in range(SEQUENCE_LENGTH):
            normalized_frame, prev_center, prev_scale = self.normalize_frame_with_state(
                sequence[i],
                prev_center=prev_center,
                prev_scale=prev_scale,
                eps=EPS,
            )
            normalized_sequence[i] = normalized_frame

        return normalized_sequence

    def predict_sequence(self, sequence_buffer: List[np.ndarray]) -> PredictionResult:
        seq = np.asarray(sequence_buffer, dtype=np.float32)
        if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            raise ValueError(f"Expected sequence shape ({SEQUENCE_LENGTH}, {FEATURE_DIM}), got {seq.shape}")

        seq = np.expand_dims(seq, axis=0)
        self._set_input_tensor(seq)
        self.interpreter.invoke()
        probs = self._get_output_tensor()[0]

        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])
        label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"Class {pred_idx}"
        return PredictionResult(label=label, confidence=pred_conf, status="Predicting")

    def process_frame_for_live(self, frame_rgb: Optional[np.ndarray], state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if frame_rgb is None:
            return blank_preview("Waiting for camera input."), state

        frame_rgb = np.asarray(frame_rgb).astype(np.uint8)
        display_rgb = resize_frame_keep_aspect(frame_rgb, LIVE_DISPLAY_WIDTH, LIVE_DISPLAY_HEIGHT)

        if not state.get("running", False):
            preview = overlay_prediction_panel(
                display_rgb.copy(),
                NO_SIGN_LABEL,
                0.0,
                "Idle",
                "Tap Start Live Translation and allow camera access.",
                show_video_background=False,
                show_status_badge=False,
                show_helper_text=False,
            )
            state["last_preview_rgb"] = preview
            return preview, state

        state["stream_count"] = int(state.get("stream_count", 0)) + 1
        should_analyze = (state["stream_count"] % LIVE_ANALYZE_EVERY_N_CALLBACKS) == 0

        if should_analyze:
            proc_rgb = cv2.resize(frame_rgb, (LIVE_INPUT_WIDTH, LIVE_INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
            results = self.live_holistic.process(proc_rgb)

            norm_feat, next_center, next_scale = self.normalize_frame_with_state(
                self.extract_landmarks(results),
                prev_center=state.get("prev_center"),
                prev_scale=float(state.get("prev_scale", 1.0)),
                eps=EPS,
            )
            state["prev_center"] = next_center
            state["prev_scale"] = next_scale

            seq_buffer = list(state.get("sequence_buffer", []))
            seq_buffer.append(norm_feat)
            state["sequence_buffer"] = seq_buffer[-SEQUENCE_LENGTH:]
            state["frame_count"] = int(state.get("frame_count", 0)) + 1

            hands_visible = bool(results.left_hand_landmarks or results.right_hand_landmarks)
            state["idle_frame_counter"] = 0 if hands_visible else int(state.get("idle_frame_counter", 0)) + 1

            if state["idle_frame_counter"] >= LIVE_IDLE_RESET_FRAMES:
                state.update(
                    {
                        "sequence_buffer": [],
                        "prediction_queue": [],
                        "display_label": NO_SIGN_LABEL,
                        "display_conf": 0.0,
                        "status": "Ready",
                        "prev_center": None,
                        "prev_scale": 1.0,
                    }
                )
            elif (
                state["frame_count"] % LIVE_PREDICT_EVERY_N_FRAMES == 0
                and len(state["sequence_buffer"]) >= LIVE_MIN_SEQUENCE_FOR_PRED
                and hands_visible
            ):
                pred = self.predict_sequence(pad_sequence_to_model_length(state["sequence_buffer"]))
                if pred.confidence >= LIVE_CONFIDENCE_THRESHOLD:
                    queue = list(state.get("prediction_queue", [])) + [(pred.label, pred.confidence)]
                    state["prediction_queue"] = queue[-LIVE_SMOOTHING_WINDOW:]
                    counts = Counter(label for label, _ in state["prediction_queue"])
                    best_label, best_count = counts.most_common(1)[0]
                    avg_conf = float(np.mean([conf for label, conf in state["prediction_queue"] if label == best_label]))
                    if best_count >= LIVE_MIN_STABLE_COUNT:
                        state.update(
                            {
                                "display_label": best_label,
                                "display_conf": avg_conf,
                                "status": "Recognized",
                            }
                        )
                    else:
                        state.update(
                            {
                                "display_label": NO_SIGN_LABEL,
                                "display_conf": avg_conf,
                                "status": "Predicting",
                            }
                        )
                else:
                    state.update(
                        {
                            "prediction_queue": [],
                            "display_label": NO_SIGN_LABEL,
                            "display_conf": pred.confidence,
                            "status": "Low confidence",
                        }
                    )
            elif len(state["sequence_buffer"]) < LIVE_MIN_SEQUENCE_FOR_PRED:
                state["status"] = "Warming up"
            elif not hands_visible:
                state["status"] = "Ready"

            if (
                state.get("display_label", NO_SIGN_LABEL) != NO_SIGN_LABEL
                and state["display_label"] != state.get("last_stable_label")
            ):
                state["history"] = (state.get("history", []) + [display_label(state["display_label"])])[-LIVE_MAX_HISTORY:]
                state["last_stable_label"] = state["display_label"]
            elif state.get("display_label", NO_SIGN_LABEL) == NO_SIGN_LABEL:
                state["last_stable_label"] = None

        preview = overlay_prediction_panel(
            display_rgb.copy(),
            state.get("display_label", NO_SIGN_LABEL),
            float(state.get("display_conf", 0.0)),
            str(state.get("status", "Running")),
            "Keep your hands visible and show one clear sign.",
            show_video_background=False,
            show_status_badge=False,
            show_helper_text=False,
        )
        state["last_preview_rgb"] = preview
        return preview, state

    def extract_video_sequence(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = sample_frame_indices(total_frames, SEQUENCE_LENGTH)

        current_frame_idx = 0
        target_set = set(frame_indices)
        frame_dict: Dict[int, np.ndarray] = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx in target_set:
                frame_dict[current_frame_idx] = frame.copy()

            current_frame_idx += 1

        cap.release()

        if total_frames == 0 or len(frame_dict) == 0:
            raise RuntimeError(f"No frames could be read from video: {video_path}")

        sequence: List[np.ndarray] = []

        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            available_keys = sorted(frame_dict.keys())

            for idx in frame_indices:
                if idx in frame_dict:
                    frame = frame_dict[idx]
                else:
                    nearest = available_keys[0]
                    for key in available_keys:
                        if key <= idx:
                            nearest = key
                        else:
                            break
                    frame = frame_dict[nearest]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                features = self.extract_landmarks(results)
                sequence.append(features)

        sequence_array = np.asarray(sequence, dtype=np.float32)

        if sequence_array.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            fixed = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
            rows = min(sequence_array.shape[0], SEQUENCE_LENGTH)
            cols = min(sequence_array.shape[1], FEATURE_DIM) if sequence_array.ndim > 1 else 0
            if cols > 0:
                fixed[:rows, :cols] = sequence_array[:rows, :cols]
            sequence_array = fixed

        sequence_array = self.normalize_sequence(sequence_array)
        return sequence_array

    def analyze_video(self, video_path: str) -> Tuple[str, str, float, str, List[str]]:
        resolved_video_path = resolve_uploaded_video_path(video_path)

        final_sequence = self.extract_video_sequence(resolved_video_path)
        pred = self.predict_sequence(final_sequence)
        final_label = pred.label

        cap = cv2.VideoCapture(resolved_video_path)
        if not cap.isOpened():
            raise RuntimeError("The uploaded video could not be read.")

        preview_source_frames: List[np.ndarray] = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            preview_source_frames.append(frame)
        cap.release()

        if not preview_source_frames:
            raise RuntimeError("The uploaded video did not contain readable frames.")

        if len(preview_source_frames) > VIDEO_SAMPLE_FRAMES:
            preview_indices = sample_frame_indices(len(preview_source_frames), VIDEO_SAMPLE_FRAMES)
            preview_frames = [preview_source_frames[i] for i in preview_indices]
        else:
            preview_frames = preview_source_frames

        temp_dir = Path(tempfile.mkdtemp())
        raw_temp_path = str(temp_dir / "beyond_words_preview_raw.mp4")

        preview_overlay_frames: List[np.ndarray] = []
        for frame in preview_frames:
            rgb_frame = cv2.cvtColor(
                resize_frame_keep_aspect(frame, VIDEO_PREVIEW_WIDTH, VIDEO_PREVIEW_HEIGHT),
                cv2.COLOR_BGR2RGB,
            )
            overlay = overlay_prediction_panel(
                rgb_frame,
                final_label,
                pred.confidence,
                "Recognized" if final_label != NO_SIGN_LABEL else "Low confidence",
                "Video review completed.",
            )
            preview_overlay_frames.append(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if not preview_overlay_frames:
            raise RuntimeError("Could not build the processed preview video frames.")

        out_h, out_w = preview_overlay_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            raw_temp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_EXPORT_FPS,
            (out_w, out_h),
        )

        if not writer.isOpened():
            raise RuntimeError("Could not create the processed preview video.")

        for overlay_bgr in preview_overlay_frames:
            writer.write(overlay_bgr)
        writer.release()

        temp_path = make_browser_playable_video(raw_temp_path)

        summary = (
            f"Video processed successfully. The strongest match was "
            f"{display_label(final_label)} with {pred.confidence:.4f} confidence."
        )
        return temp_path, display_label(final_label), pred.confidence, summary, [display_label(final_label)]

class EngineManager:
    def __init__(self) -> None:
        self._engine: Optional[PSLInferenceEngine] = None
        self._error: Optional[str] = None
        self._lock = Lock()

    def ensure_loaded(self) -> PSLInferenceEngine:
        with self._lock:
            if self._engine is None:
                try:
                    self._engine = PSLInferenceEngine()
                    self._error = None
                except Exception as exc:
                    self._error = str(exc)
                    raise
            return self._engine

    @property
    def last_error(self) -> Optional[str]:
        return self._error


ENGINE_MANAGER = EngineManager()


# ============================================================
# UI CALLBACKS
# ============================================================
def live_stream_handler(frame: Optional[np.ndarray], state: Dict[str, Any]):
    if state is None:
        state = init_live_state()

    if not state.get("running", False):
        preview = blank_preview("Tap Start Live Translation.")
        return preview, NO_SIGN_LABEL, "0.00", badge_html("Idle"), history_text(state.get("history", [])), state

    try:
        engine = ENGINE_MANAGER.ensure_loaded()
    except Exception as exc:
        preview = blank_preview("Model files are missing.")
        return preview, NO_SIGN_LABEL, "0.00", badge_html("Model missing"), str(exc), state

    preview, next_state = engine.process_frame_for_live(frame, state)
    recognized = display_label(next_state.get("display_label", NO_SIGN_LABEL))
    return (
        preview,
        recognized,
        f"{next_state.get('display_conf', 0.0):.2f}",
        badge_html(next_state.get("status", "Idle")),
        history_text(next_state.get("history", [])),
        next_state,
    )


def run_video_prediction(video_path: str):
    try:
        engine = ENGINE_MANAGER.ensure_loaded()
        path, label, conf, summary, _ = engine.analyze_video(video_path)
        return path, label, f"{conf:.4f}", badge_html("Recognized"), summary
    except Exception as exc:
        return None, NO_SIGN_LABEL, "0.00", badge_html("Error"), str(exc)

# ============================================================
# APP LAYOUT
# ============================================================
def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="fuchsia", secondary_hue="violet", neutral_hue="slate")

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title=APP_TITLE) as demo:
        with gr.Tabs(selected="home") as tabs:
            with gr.Tab("🏠 Home", id="home"):
                gr.HTML(home_page_html())

                with gr.Row(elem_id="home-actions"):
                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="bw-action-card">
                              <div class="bw-action-title">Sign to Text</div>
                              <div class="bw-action-copy">
                                Open the live camera experience and start translation with one button.
                              </div>
                            </div>
                            """
                        )
                        btn_to_sign = gr.Button("Open Sign to Text", variant="primary")

                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="bw-action-card">
                              <div class="bw-action-title">Text to Sign</div>
                              <div class="bw-action-copy">
                                Enter a phrase and play the recorded sign assets from a dropdown.
                              </div>
                            </div>
                            """
                        )
                        btn_to_text = gr.Button("Open Text to Sign", variant="primary")

                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="bw-action-card">
                              <div class="bw-action-title">Recorded Videos Prediction</div>
                              <div class="bw-action-copy">
                                Upload a saved sign video and review the prediction with a processed preview.
                              </div>
                            </div>
                            """
                        )
                        btn_to_video = gr.Button("Open Recorded Videos Prediction", variant="primary")

            with gr.Tab("✨ Quick Start Guide", id="quick_start_guide"):
                gr.HTML(app_guide_page_html())

            with gr.Tab("🤟 Sign to Text", id="sign_to_text"):
                live_state = gr.State(value=init_live_state())

                gr.HTML(
                    f"""
                    <div class="bw-card">
                      <div class="bw-section-title">Live Sign to Text</div>
                      <div class="bw-muted">
                        Click Start Live Translation. Your browser will ask for camera access. After that the app starts reading signs in real time.
                      </div>
                      <div class="bw-small-note" style="margin-top: 10px;">
                        {RESTAURANT_CONTEXT}
                      </div>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        live_camera_input = gr.Image(
                            sources=["webcam"],
                            type="numpy",
                            label="Camera",
                            height=340,
                        )
                        with gr.Row(elem_id="sign-actions"):
                            start_btn = gr.Button("Start Live Translation", variant="primary")
                            stop_btn = gr.Button("Stop", variant="secondary")

                    with gr.Column(scale=7):
                        live_preview = gr.Image(
                            label="Live Preview",
                            type="numpy",
                            value=blank_preview("Tap Start Live Translation."),
                            height=540,
                        )

                with gr.Row():
                    live_prediction = gr.Textbox(label="Recognized Text", value=NO_SIGN_LABEL)
                    live_confidence = gr.Textbox(label="Confidence", value="0.00")
                    live_status = gr.HTML(value=badge_html("Idle"))

                live_history = gr.Textbox(label="Recent Translations", lines=5, value="No translations yet.")

                start_btn.click(
                    start_live_session,
                    inputs=live_state,
                    outputs=[live_preview, live_prediction, live_confidence, live_status, live_history, live_state],
                    queue=False,
                    show_progress="hidden",
                )
                stop_btn.click(
                    stop_live_session,
                    inputs=live_state,
                    outputs=[live_preview, live_prediction, live_confidence, live_status, live_history, live_state],
                    queue=False,
                    show_progress="hidden",
                )
                live_camera_input.stream(
                    live_stream_handler,
                    inputs=[live_camera_input, live_state],
                    outputs=[live_preview, live_prediction, live_confidence, live_status, live_history, live_state],
                    stream_every=LIVE_STREAM_EVERY_SECONDS,
                    queue=False,
                    show_progress="hidden",
                )
                gr.HTML(
                    """
                    <div class="bw-card" style="margin-top: 10px;">
                      <div class="bw-section-title">Live mode notes</div>
                      <div class="bw-small-note">
                        Keep your hands visible. Show one sign clearly. Good front lighting helps the model stay fast and stable.
                      </div>
                    </div>
                    """
                )

            with gr.Tab("✍️ Text to Sign", id="text_to_sign"):
                gr.HTML(
                    f"""
                    <div class="bw-card">
                      <div class="bw-section-title">Text to Sign</div>
                      <div class="bw-muted">
                        Type a phrase, then click Start Text to Sign. The app prepares a simple sign sequence view and lets you play matching recorded sign assets.
                      </div>
                      <div class="bw-small-note" style="margin-top: 10px;">
                        Place your recorded sign videos inside a folder named sign_assets next to this script. Name each file like bacon.mp4, bill.mp4, green_sauce.mp4, or new_york.mp4.
                      </div>
                    </div>
                    """
                )

                text_input = gr.Textbox(
                    label="Enter text",
                    placeholder="Example: menu zinger wings takeaway",
                    lines=3,
                )

                with gr.Row(elem_id="text-actions"):
                    run_text_to_sign_btn = gr.Button("Start Text to Sign", variant="primary")
                    clear_text_btn = gr.Button("Clear", variant="secondary")

                text_sign_output = gr.HTML(
                    '<div class="bw-output-box"><div class="bw-section-title">Text to Sign</div><div class="bw-muted">Enter a phrase, then tap Start Text to Sign.</div></div>'
                )
                text_sign_summary = gr.Textbox(label="Summary", value="Ready.")
                supported_signs = gr.Textbox(label="Available Sign Classes", lines=4, value=", ".join(display_label(item) for item in load_class_names()))

                sign_asset_dropdown = gr.Dropdown(
                    label="Recorded Sign Assets",
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                )
                sign_asset_video = gr.Video(label="Sign Asset Video", height=360)
                sign_asset_note = gr.Textbox(label="Asset Preview Note", value="Add sign assets to preview them.")

                run_text_to_sign_btn.click(
                    prepare_text_to_sign,
                    inputs=text_input,
                    outputs=[
                        text_sign_output,
                        text_sign_summary,
                        supported_signs,
                        sign_asset_dropdown,
                        sign_asset_video,
                        sign_asset_note,
                    ],
                )
                clear_text_btn.click(
                    default_text_to_sign_state,
                    outputs=[
                        text_sign_output,
                        text_sign_summary,
                        supported_signs,
                        sign_asset_dropdown,
                        sign_asset_video,
                        sign_asset_note,
                    ],
                )
                sign_asset_dropdown.change(
                    update_sign_asset_preview,
                    inputs=sign_asset_dropdown,
                    outputs=[sign_asset_video, sign_asset_note],
                )

                gr.HTML(
                    """
                    <div class="bw-card" style="margin-top: 10px;">
                      <div class="bw-section-title">Text mode notes</div>
                      <div class="bw-small-note">
                        This dropdown is for your recorded sign videos. When a phrase matches supported classes, you can pick a sign from the list and play its video directly inside the app.
                      </div>
                    </div>
                    """
                )

            with gr.Tab("🎬 Recorded Videos Prediction", id="video_review"):
                gr.HTML(
                    """
                    <div class="bw-card">
                      <div class="bw-section-title">Recorded Videos Prediction</div>
                      <div class="bw-muted">
                        Upload a recorded sign clip to predict the class from saved videos in a simple and clean format.
                      </div>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=7):
                        video_input = gr.Video(label="Upload recorded video", height=420)
                        predict_video_btn = gr.Button("Run Recorded Videos Prediction", variant="primary")
                        video_output = gr.Video(label="Processed Preview", height=420, format="mp4")

                    with gr.Column(scale=5):
                        video_prediction = gr.Textbox(label="Predicted Text", value=NO_SIGN_LABEL)
                        video_confidence = gr.Textbox(label="Confidence", value="0.00")
                        video_status = gr.HTML(value=badge_html("Idle"))
                        video_summary = gr.Textbox(label="Summary", lines=8, value="Ready.")

                predict_video_btn.click(
                    run_video_prediction,
                    inputs=video_input,
                    outputs=[video_output, video_prediction, video_confidence, video_status, video_summary],
                    queue=False,
                    show_progress="hidden",
                )


        btn_to_sign.click(fn=lambda: gr.Tabs(selected="sign_to_text"), outputs=tabs)
        btn_to_text.click(fn=lambda: gr.Tabs(selected="text_to_sign"), outputs=tabs)
        btn_to_video.click(fn=lambda: gr.Tabs(selected="video_review"), outputs=tabs)

        demo.load(
            default_text_to_sign_state,
            outputs=[
                text_sign_output,
                text_sign_summary,
                supported_signs,
                sign_asset_dropdown,
                sign_asset_video,
                sign_asset_note,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=ENABLE_SHARE_LINK,
        inbrowser=OPEN_IN_BROWSER,
        show_error=True,
    )
