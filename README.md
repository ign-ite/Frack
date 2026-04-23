# Frack: Real-Time Body Framing Guidance

A webcam-based Python application that guides a user into a camera-ready full-body position for assessment workflows.

This version prioritizes high-value product features over cosmetic add-ons:
- Robust framing analysis for both front-facing and side-facing body orientations
- Hands-free controls with gestures
- Remote control panel from a phone browser
- Runtime calibration and session logging
- Cleaner on-screen guidance with directional arrows and confidence telemetry

## High-Value Features Implemented

### 1) Orientation-Robust Framing (Front + Side)
The classifier now estimates orientation (`FRONT` vs `SIDE`) using torso width/height geometry and keeps framing guidance stable for either pose orientation.

### 2) Gesture Controls
No extra libraries required; uses existing pose keypoints:
- Left-hand wave (while raised): mute/unmute
- Both hands above head (hold): save screenshot
- T-pose (hold ~2s): run calibration

### 3) Phone Web Control Panel
Optional local Flask server provides controls at `http://<host>:5000`:
- Mute/unmute
- Screenshot
- Calibrate baseline

### 4) Timed Launch
A startup countdown is rendered for 10 seconds. Pose analysis begins after countdown completes.

### 5) Calibration Mode
Calibration captures the user’s current “good” pose baseline and stores dynamic thresholds in:
- `body_framing_guidance/calibration_profile.json`

### 6) Product Telemetry / Observability
Session events are written to CSV:
- Folder: `body_framing_guidance/session_logs/`
- Includes state changes, orientation, key metrics, and action sources (keyboard, gesture, remote, auto)

### 7) Cleaner UI Overlay
Overlay now includes:
- State and orientation
- Directional arrows for left/right correction
- FPS, audio status, confidence metrics
- Lean-angle warning for posture stability
- Countdown timer

## Keyboard Controls
- `q` quit
- `s` screenshot
- `m` mute/unmute
- `d` toggle debug telemetry overlay
- `c` calibration

## Requirements
- Python 3.10-3.12 (recommended: 3.12)
- Webcam

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Windows (automated)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

### macOS (automated)

```bash
chmod +x ./setup_macos.sh
./setup_macos.sh
```

Setup-only mode:

```bash
./setup_macos.sh --setup-only
```

## Run

```bash
python body_framing_guidance/main.py
```

Optional camera and panel arguments:

```bash
python body_framing_guidance/main.py --camera-index 0 --frame-width 1280 --frame-height 720
python body_framing_guidance/main.py --no-web-panel
python body_framing_guidance/main.py --web-panel-host 0.0.0.0 --web-panel-port 5000
python body_framing_guidance/main.py --list-cameras
```

`--list-cameras` now prints index plus friendly device name when available.

## Project Structure

```text
README.md
requirements.txt
setup.ps1
setup_macos.sh
TRADEOFFS.md
body_framing_guidance/
  main.py
  config.py
  pose_detector.py
  framing_logic.py
  debounce_controller.py
  gesture_controller.py
  remote_control.py
  session_logger.py
  audio_engine.py
  utils.py
  audio_clips/
```

## Notes
- Voice commands were intentionally not included in this pass. In noisy real-world rooms they can be less reliable than gesture + phone control for this use case.
- The app remains single-person by design.
