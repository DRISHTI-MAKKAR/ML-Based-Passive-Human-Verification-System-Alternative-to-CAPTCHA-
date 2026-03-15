# BotGuard — ML-Based Passive Human Verification

## Quick Start

### Backend (FastAPI + ML)
```bash
cd backend
pip install -r requirements.txt
python app.py
# Runs on http://localhost:8000
```

### Frontend
```bash
cd frontend
# Serve with any static server:
python -m http.server 3000
# Open http://localhost:3000
```

## Architecture

```
Browser (SDK)              Backend (ML)
───────────────            ────────────────────────────
Mouse events    ──────────▶ Feature Extractor
Keystroke timing            ├─ Mouse biometrics
Device entropy              ├─ Keyboard dynamics
Session timing  ──────────▶ ├─ Device entropy
                            └─ Session timing
                                    │
                            Ensemble Classifier
                            ├─ GBT (tabular)
                            ├─ LSTM (time-series)
                            └─ Isolation Forest
                                    │
                            Confidence Score (0-1)
                                    │
                     ┌─────────────┼──────────────┐
                  ≥0.85          0.55-0.84        <0.55
                  PASS          CHALLENGE         BLOCK
```

## Signal Weights

| Signal | Weight | Key Features |
|--------|--------|-------------|
| Mouse biometrics | 30% | Linearity, entropy, direction variance, micro-pauses |
| Keystroke dynamics | 25% | Dwell time, flight time, rhythm variance |
| Device entropy | 20% | Canvas hash, WebGL, fonts, audio context |
| Session timing | 25% | Time-to-interact, request rate, natural dwell |

## Privacy Design
- No key content captured — only inter-key timing
- No full mouse path stored — only statistical features
- All signals are ephemeral (processed in-memory)
- No PII leaves the browser in identifiable form

## API

`POST /v1/verify`
```json
{
  "session_id": "sess_abc123",
  "mouse": [{"t": 1234, "x": 100, "y": 200, "type": "move"}],
  "keys": [{"t": 1234, "type": "down", "code": "CHAR"}],
  "scrolls": [],
  "device": {"canvas_hash": "...", "webgl": true, "font_count": 8},
  "timing": {"time_to_interact": 1500, "session_duration": 5000}
}
```

Response:
```json
{
  "score": 0.91,
  "decision": "pass",
  "flags": [],
  "signals": {...},
  "ms": 3.7
}
```

## Extending to Production

1. **Train on real data**: Replace rule-based classifier with XGBoost trained on labeled human/bot sessions
2. **LSTM for sequences**: Add time-series model for raw mouse coordinate streams
3. **Redis session store**: Cache partial signals across page views
4. **Adaptive thresholds**: Per-endpoint risk calibration (login = stricter)
5. **Retraining pipeline**: Weekly model updates from audit log feedback
