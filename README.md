# Eidovis Body Framing Guidance Prototype

## Project summary
This repository contains a real-time body framing guidance prototype built in Python. The codebase uses a live webcam feed, MediaPipe Pose for full-body keypoint estimation, deterministic framing heuristics, and audio prompts to help a user position their body correctly in frame.

All source code is contained in the `body_framing_guidance/` folder. This README is the entry point for GitHub and is intentionally placed at the repository root so the project can be uploaded and shared easily.

## What this app does
- Captures a live webcam feed
- Detects a person using MediaPipe Pose with 33 landmarks
- Computes framing quality based on body span and hip centering
- Provides real-time spoken guidance such as:
  - "Please step into frame"
  - "Please move back"
  - "Please come closer"
  - "Move to your left"
  - "Move to your right"
  - "You're in frame. Hold this position."
- Uses a layered audio fallback chain for reliability
- Implements stability buffering and cooldowns to avoid noisy repeated prompts
- Recovers gracefully when the camera is unavailable

## Repository structure
```
/eidovis
  |-- .gitignore
  |-- README.md
  |-- TRADEOFFS.md
  |-- requirements.txt
  |-- setup.ps1
  |-- body_framing_guidance/
       |-- main.py
       |-- config.py
       |-- pose_detector.py
       |-- framing_logic.py
       |-- debounce_controller.py
       |-- audio_engine.py
       |-- utils.py
       |-- audio_clips/
            |-- README.txt
```

## Getting started
### Recommended installation
From the repository root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This script will:
- detect or install Python 3.12 when needed
- create a local `.venv`
- install runtime dependencies
- launch the app by default

### If you only want to install dependencies

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -SetupOnly
```

### Run the app directly
After setup, activate the virtual environment and run the main program:

```powershell
.venv\Scripts\activate
python body_framing_guidance/main.py
```

### Optional arguments

```powershell
python body_framing_guidance/main.py --camera-index 1 --frame-width 1280 --frame-height 720
python body_framing_guidance/main.py --list-cameras
```

## Camera and missing-camera behavior
- The app can detect multiple connected cameras.
- Use `--list-cameras` to enumerate devices.
- If no camera is available, the app displays a black recovery screen and provides instructions.
- Recovery mode supports:
  - `r` to rescan and reconnect
  - `c` to choose a camera from detected devices
  - `l` to list cameras again
  - `q` to quit

## How it works
### MediaPipe Pose for person detection
The app uses MediaPipe Pose because it provides 33 landmarks, including ankles and feet, which are essential for full-body framing checks. This is a stronger fit than 17-keypoint models when determining whether a user is too close, too far, or partially cropped.

### Framing heuristics
The classifier follows a strict priority order:
1. NO_PERSON
2. TOO_CLOSE
3. TOO_FAR
4. SHIFTED_LEFT / SHIFTED_RIGHT
5. GOOD

The key metric is `body_span_ratio`, computed from the nose to ankle average in pixel coordinates. Hip midpoint is used for centering because it is more stable under head and shoulder motion.

### Audio feedback design
The app uses a layered audio engine:
- Layer 1: gTTS + pygame (preferred when internet is available)
- Layer 2: local pre-recorded MP3 clips in `body_framing_guidance/audio_clips/`
- Layer 3: pyttsx3 offline TTS

This chain ensures the system remains usable even if internet or one audio backend is unavailable.

### Debouncing
Audio is only spoken when a state is stable for several frames and when the state changes. A cooldown prevents rapid re-triggering. This makes the guidance feel much more polished than speaking every frame.

## Design decisions and why they matter
### Why MediaPipe Pose
- CPU-friendly
- 33 landmarks including ankles/feet
- Easy Python API
- Strong fit for fitness-style applications

### Why keypoint heuristics instead of bounding boxes
- Bounding boxes alone cannot reliably detect foot crop or user centering.
- Keypoints allow reasoning about ankle visibility, head position, and hip alignment.

### Why layered audio fallbacks
- Real-time audio should not depend on a single network or playback path.
- The app prefers higher-quality speech but can still function offline.

### Why a black recovery screen instead of corrupted camera output
- If the camera fails, showing a static informative screen is much clearer than an unreliable image.
- This improves usability and prevents the user from assuming the app is simply broken.

## Known limitations
- Front-view only; side/rear poses are unsupported.
- Single-person assumption only.
- Requires reasonably good lighting.
- Baggy clothing can reduce pose landmark visibility.
- The app is not a production-grade posture classifier; it is a framing guidance prototype.

## Future directions
### Immediate improvements
- Add a calibration mode for different room/camera setups.
- Use MediaPipe world landmarks or depth estimation for actual distance measurements.
- Add a small GUI overlay menu for camera selection and setup.
- Add a more natural offline TTS model or local voice pack.

### Longer-term enhancements
- Add a learned framing quality model on top of pose heuristics.
- Add gesture-based controls for silent setup.
- Add a multi-person detection mode with explicit single-user tracking.
- Add a visual demo mode that records a short guidance session.

## Uploading to GitHub
This repository is already structured for GitHub. Upload the entire `eidovis` folder as a single repo. The code lives under `body_framing_guidance/`, and the root `README.md` and `TRADEOFFS.md` provide the main documentation.

## Useful files
- `README.md` — this document
- `TRADEOFFS.md` — decision tables and tradeoff analysis
- `requirements.txt` — pinned runtime dependencies
- `setup.ps1` — Windows-first setup and launch script
- `body_framing_guidance/` — contains all source code and utilities
- `body_framing_guidance/audio_clips/README.txt` — instructions for local fallback clips
