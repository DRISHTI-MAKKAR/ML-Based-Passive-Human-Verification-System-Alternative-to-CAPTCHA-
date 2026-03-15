"""
BotGuard ML Backend
Passive human verification using behavioral biometrics
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import math
import time
import json
import hashlib
import uvicorn

app = FastAPI(title="BotGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

class MouseEvent(BaseModel):
    t: int        # timestamp ms
    x: float
    y: float
    type: str     # move | click | enter | leave

class KeyEvent(BaseModel):
    t: int
    type: str     # down | up
    code: str

class ScrollEvent(BaseModel):
    t: int
    dy: float
    total: float

class SignalPayload(BaseModel):
    session_id: str
    mouse: List[MouseEvent] = []
    keys: List[KeyEvent] = []
    scrolls: List[ScrollEvent] = []
    device: dict = {}
    timing: dict = {}

class VerifyResponse(BaseModel):
    score: float
    decision: str   # pass | challenge | block
    flags: List[str]
    signals: dict
    ms: float

# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────

def extract_mouse_features(events: List[MouseEvent]) -> dict:
    if len(events) < 3:
        return {"mouse_entropy": 0, "avg_velocity": 0, "velocity_variance": 0,
                "straightness": 0, "direction_changes": 0, "micro_pauses": 0,
                "bot_like_linearity": 1.0}

    moves = [e for e in events if e.type == "move"]
    if len(moves) < 3:
        return {"mouse_entropy": 0, "avg_velocity": 0, "velocity_variance": 0,
                "straightness": 0, "direction_changes": 0, "micro_pauses": 0,
                "bot_like_linearity": 1.0}

    # Velocities
    velocities = []
    for i in range(1, len(moves)):
        dt = max(moves[i].t - moves[i-1].t, 1)
        dx = moves[i].x - moves[i-1].x
        dy = moves[i].y - moves[i-1].y
        v = math.sqrt(dx*dx + dy*dy) / dt
        velocities.append(v)

    avg_v = np.mean(velocities) if velocities else 0
    var_v = np.var(velocities) if velocities else 0

    # Direction changes (human = many small changes)
    angles = []
    for i in range(1, len(moves) - 1):
        dx1 = moves[i].x - moves[i-1].x
        dy1 = moves[i].y - moves[i-1].y
        dx2 = moves[i+1].x - moves[i].x
        dy2 = moves[i+1].y - moves[i].y
        if dx1 == 0 and dy1 == 0: continue
        if dx2 == 0 and dy2 == 0: continue
        dot = dx1*dx2 + dy1*dy2
        mag1 = math.sqrt(dx1**2 + dy1**2)
        mag2 = math.sqrt(dx2**2 + dy2**2)
        if mag1 * mag2 == 0: continue
        cos_a = max(-1, min(1, dot / (mag1 * mag2)))
        angles.append(math.acos(cos_a))

    direction_changes = len([a for a in angles if a > 0.1]) / max(len(angles), 1)

    # Straightness: ratio of total path length to Euclidean start→end
    total_dist = sum(
        math.sqrt((moves[i].x-moves[i-1].x)**2 + (moves[i].y-moves[i-1].y)**2)
        for i in range(1, len(moves))
    )
    euclidean = math.sqrt(
        (moves[-1].x - moves[0].x)**2 + (moves[-1].y - moves[0].y)**2
    )
    straightness = euclidean / max(total_dist, 1)  # near 1 = bot-like

    # Micro-pauses (human naturally pauses briefly)
    micro_pauses = len([
        moves[i].t - moves[i-1].t
        for i in range(1, len(moves))
        if 80 < (moves[i].t - moves[i-1].t) < 400
    ])

    # Shannon entropy of velocity distribution
    if velocities:
        hist, _ = np.histogram(velocities, bins=10, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist + 1e-9)) / 10
    else:
        entropy = 0

    # Bot linearity score: consistent velocity + straight path = suspicious
    cv = (math.sqrt(var_v) / avg_v) if avg_v > 0 else 0  # coefficient of variation
    bot_like_linearity = straightness * (1 - min(cv, 1))

    return {
        "mouse_entropy": float(entropy),
        "avg_velocity": float(avg_v),
        "velocity_variance": float(var_v),
        "straightness": float(straightness),
        "direction_changes": float(direction_changes),
        "micro_pauses": int(micro_pauses),
        "bot_like_linearity": float(bot_like_linearity),
    }


def extract_keyboard_features(events: List[KeyEvent]) -> dict:
    if len(events) < 4:
        return {"avg_dwell": 0, "dwell_variance": 0, "avg_flight": 0,
                "flight_variance": 0, "typing_rhythm_score": 0, "superhuman_speed": 0}

    downs = {e.code: e.t for e in events if e.type == "down"}
    ups = {e.code: e.t for e in events if e.type == "up"}

    # Dwell times (key held duration)
    dwells = []
    for code, down_t in downs.items():
        if code in ups:
            dwell = ups[code] - down_t
            if 0 < dwell < 500:
                dwells.append(dwell)

    # Flight times (time between key releases and next press)
    down_times = sorted([e.t for e in events if e.type == "down"])
    up_times = sorted([e.t for e in events if e.type == "up"])
    flights = []
    for i in range(len(up_times) - 1):
        if i < len(down_times) - 1:
            flight = down_times[i+1] - up_times[i]
            if -50 < flight < 600:
                flights.append(flight)

    avg_dwell = float(np.mean(dwells)) if dwells else 0
    dwell_var = float(np.var(dwells)) if dwells else 0
    avg_flight = float(np.mean(flights)) if flights else 0
    flight_var = float(np.var(flights)) if flights else 0

    # Superhuman detection: >12 keypresses/second sustained
    if len(down_times) > 5:
        span = max(down_times[-1] - down_times[0], 1)
        kps = len(down_times) / (span / 1000)
        superhuman_speed = 1.0 if kps > 12 else 0.0
    else:
        superhuman_speed = 0.0

    # Rhythm score: humans have natural variance in timing
    rhythm_score = min(1.0, (dwell_var ** 0.5) / 30) if dwell_var > 0 else 0

    return {
        "avg_dwell": avg_dwell,
        "dwell_variance": dwell_var,
        "avg_flight": avg_flight,
        "flight_variance": flight_var,
        "typing_rhythm_score": float(rhythm_score),
        "superhuman_speed": superhuman_speed,
    }


def extract_device_features(device: dict) -> dict:
    score = 0
    flags = []

    # Canvas fingerprint present
    if device.get("canvas_hash"): score += 15
    else: flags.append("no_canvas")

    # WebGL support
    if device.get("webgl"): score += 10
    else: flags.append("no_webgl")

    # Realistic screen dimensions
    w = device.get("screen_w", 0)
    h = device.get("screen_h", 0)
    if w > 800 and h > 600 and w < 8000: score += 10
    else: flags.append("suspicious_screen")

    # Plugin/font count (headless browsers have 0)
    fonts = device.get("font_count", 0)
    if fonts > 5: score += 15
    else: flags.append("no_fonts")

    # Platform present
    if device.get("platform"): score += 10

    # Timezone coherent
    if device.get("timezone"): score += 5

    # Touch vs mouse mismatch
    if device.get("touch_capable") and device.get("mouse_events"):
        score += 5  # Consistent

    # Audio context
    if device.get("audio_hash"): score += 10

    # Normalized 0-1
    return {
        "device_entropy_score": min(1.0, score / 80),
        "device_flags": flags,
    }


def extract_timing_features(timing: dict, mouse: List[MouseEvent]) -> dict:
    # Time from page load to first interaction
    tti = timing.get("time_to_interact", 9999)

    # Bots often interact immediately (< 100ms) or after exact delays
    immediate_action = 1.0 if tti < 100 else 0.0
    suspiciously_exact = 1.0 if tti > 0 and tti % 1000 < 10 else 0.0

    # Request rate
    req_count = timing.get("request_count", 0)
    session_ms = max(timing.get("session_duration", 1), 1)
    req_rate = req_count / (session_ms / 1000)
    high_req_rate = 1.0 if req_rate > 50 else 0.0

    # Human natural dwell (> 2 seconds before submitting)
    natural_dwell = 1.0 if session_ms > 2000 else 0.0

    return {
        "time_to_interact_ms": tti,
        "immediate_action": immediate_action,
        "suspiciously_exact_timing": suspiciously_exact,
        "high_request_rate": high_req_rate,
        "natural_dwell": natural_dwell,
        "session_duration_ms": session_ms,
    }

# ─────────────────────────────────────────────
# ML Classifier (Rule-based ensemble simulating
# the XGBoost + LSTM + IsoForest pipeline)
# ─────────────────────────────────────────────

def classify(features: dict) -> tuple[float, list]:
    """
    Weighted ensemble scoring.
    Returns (confidence_human_score 0-1, flags)
    """
    flags = []
    score = 0.5  # neutral prior

    mf = features.get("mouse", {})
    kf = features.get("keyboard", {})
    df = features.get("device", {})
    tf = features.get("timing", {})

    # ── MOUSE SIGNALS (weight: 0.30) ──────────────
    mouse_score = 0.5

    linearity = mf.get("bot_like_linearity", 0.5)
    if linearity > 0.85:
        mouse_score -= 0.3
        flags.append("linear_mouse_path")
    elif linearity < 0.3:
        mouse_score += 0.2

    entropy = mf.get("mouse_entropy", 0)
    if entropy > 0.3:
        mouse_score += 0.15
    elif entropy == 0 and mf.get("avg_velocity", -1) >= 0:
        mouse_score -= 0.2
        flags.append("zero_mouse_entropy")

    direction_changes = mf.get("direction_changes", 0)
    if direction_changes > 0.4:
        mouse_score += 0.1
    elif direction_changes < 0.05 and mf.get("avg_velocity", 0) > 0:
        mouse_score -= 0.15
        flags.append("no_direction_variance")

    micro_pauses = mf.get("micro_pauses", 0)
    if micro_pauses >= 2:
        mouse_score += 0.1

    mouse_score = max(0, min(1, mouse_score))

    # ── KEYBOARD SIGNALS (weight: 0.25) ───────────
    key_score = 0.5

    if kf.get("superhuman_speed", 0) > 0:
        key_score -= 0.4
        flags.append("superhuman_typing_speed")

    dwell_var = kf.get("dwell_variance", 0)
    if dwell_var > 100:
        key_score += 0.2
    elif dwell_var == 0 and kf.get("avg_dwell", 0) > 0:
        key_score -= 0.25
        flags.append("robotic_dwell_uniformity")

    rhythm = kf.get("typing_rhythm_score", 0)
    key_score += rhythm * 0.2

    key_score = max(0, min(1, key_score))

    # ── DEVICE ENTROPY (weight: 0.20) ─────────────
    dev_score = df.get("device_entropy_score", 0.5)
    for f in df.get("device_flags", []):
        flags.append(f"device:{f}")
    if len(df.get("device_flags", [])) >= 3:
        dev_score = max(0, dev_score - 0.2)
        flags.append("headless_browser_suspected")

    # ── TIMING SIGNALS (weight: 0.25) ─────────────
    timing_score = 0.5

    if tf.get("immediate_action", 0):
        timing_score -= 0.3
        flags.append("immediate_form_action")

    if tf.get("suspiciously_exact_timing", 0):
        timing_score -= 0.15
        flags.append("exact_timing_interval")

    if tf.get("high_request_rate", 0):
        timing_score -= 0.3
        flags.append("high_request_rate")

    if tf.get("natural_dwell", 0):
        timing_score += 0.2

    timing_score = max(0, min(1, timing_score))

    # ── DATA AVAILABILITY ADJUSTMENT ─────────────
    # If very few signals, be conservative (avoid false positives)
    has_mouse = mf.get("avg_velocity", -1) >= 0 and len(features.get("_raw_mouse", [])) > 5
    has_keys = kf.get("avg_dwell", 0) > 0
    has_device = df.get("device_entropy_score", 0) > 0

    if not has_mouse and not has_keys:
        # Very limited signals — rely on device + timing only, with penalty
        final = 0.55 * timing_score + 0.35 * dev_score + 0.1 * 0.5
        flags.append("limited_behavioral_signals")
    else:
        # Full weighted ensemble
        w_mouse = 0.30 if has_mouse else 0.05
        w_key   = 0.25 if has_keys else 0.05
        w_dev   = 0.20
        w_time  = 1.0 - w_mouse - w_key - w_dev

        final = (
            w_mouse * mouse_score +
            w_key   * key_score +
            w_dev   * dev_score +
            w_time  * timing_score
        )

    final = max(0.0, min(1.0, float(final)))
    return final, flags


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/v1/verify", response_model=VerifyResponse)
async def verify(payload: SignalPayload):
    t0 = time.time()

    mouse_features = extract_mouse_features(payload.mouse)
    key_features   = extract_keyboard_features(payload.keys)
    device_features = extract_device_features(payload.device)
    timing_features = extract_timing_features(payload.timing, payload.mouse)

    all_features = {
        "mouse": mouse_features,
        "keyboard": key_features,
        "device": device_features,
        "timing": timing_features,
        "_raw_mouse": payload.mouse,
    }

    score, flags = classify(all_features)

    if score >= 0.85:
        decision = "pass"
    elif score >= 0.55:
        decision = "challenge"
    else:
        decision = "block"

    elapsed = (time.time() - t0) * 1000

    return VerifyResponse(
        score=round(score, 4),
        decision=decision,
        flags=flags,
        signals={
            "mouse": mouse_features,
            "keyboard": key_features,
            "device": device_features,
            "timing": timing_features,
        },
        ms=round(elapsed, 2),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "BotGuard ML API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
