# Real-Time Body Framing Guidance (Python + MediaPipe)

## 1. Overview
This prototype captures a live webcam feed, runs MediaPipe Pose (BlazePose) to estimate 33 body landmarks, evaluates framing quality with deterministic keypoint heuristics, and provides real-time spoken guidance so a user can position themselves correctly for fitness-oriented observation. The system is designed for CPU-only execution, robust behavior under noisy frame-to-frame pose jitter, and practical audio UX via layered fallback speech output.

## 2. Assumptions
- Front-view only: user is facing the camera (not side-view or back-view).
- Single person in frame.
- Standard laptop webcam (RGB/BGR camera stream via OpenCV).
- Reasonably lit indoor environment.
- CPU-only execution target (no GPU required).

## 3. Setup Instructions
### Python version
- Required: Python 3.10-3.12.
- Recommended on Windows: Python 3.12.
- Note: Python 3.13 currently installs MediaPipe builds that do not expose
  `mediapipe.solutions.pose`, which this assignment requires.

### Single setup file (recommended)
From the project root:

```bash
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

What this script now does automatically:
- Detects Python 3.12.
- If missing, installs Python 3.12 (tries `winget`, then python.org installer fallback).
- Creates/repairs `.venv` with Python 3.12.
- Installs dependencies from `requirements.txt`.
- Launches the app.

Setup only (do not launch app):

```bash
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -SetupOnly
```

Run app with a specific camera index:

```bash
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -RunApp -CameraIndex 1 -FrameWidth 1280 -FrameHeight 720
```

### Manual setup with requirements.txt (alternative)
From the project root:

```bash
py -3.12 -m venv .venv
```

Activate environment:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the app
```bash
python body_framing_guidance/main.py
```

Optional webcam arguments:

```bash
python body_framing_guidance/main.py --camera-index 0 --frame-width 640 --frame-height 480
```

List available cameras and exit:

```bash
python body_framing_guidance/main.py --list-cameras
```

Controls:
- Press `q` in the video window to exit.
- `Ctrl+C` in terminal also shuts down gracefully.
- If no camera is available, a static black status screen is shown.
- In no-camera mode: press `r` to rescan, `c` to choose camera, `l` to print camera list, `q` to quit.

### Live webcam runtime behavior
- Camera warm-up: startup reads and discards a small number of frames so auto-exposure and focus stabilize.
- Read-failure tolerance: occasional dropped reads are ignored without crashing the loop.
- Auto-reconnect: after repeated consecutive read failures, the app releases and reopens the webcam automatically.
- Reconnect fallback: if reconnect attempts fail, the app enters a black-screen recovery mode instead of showing corrupted visuals.
- Missing camera voice prompt: a periodic audio line announces that no camera was detected.

### Audio fallback behavior and optional clips
The app automatically generates and caches voice prompts when internet is available. This cache is stored in `body_framing_guidance/audio_cache/`, so you do not need to provide your own MP3 files manually.

If you want to override or provide alternative voice prompts, you may still place MP3 files in `body_framing_guidance/audio_clips/` with these names:
- `no_person.mp3`
- `too_close.mp3`
- `too_far.mp3`
- `move_left.mp3`
- `move_right.mp3`
- `good.mp3`

Optional:
- `hold.mp3`
- `missing_camera.mp3`

The app will prefer cached generated audio first, then manual clips if present.

## 4. Tech Stack & Why
### Vision stack: MediaPipe Pose (BlazePose)
Chosen:
- `mediapipe.solutions.pose.Pose` with:
  - `min_detection_confidence = 0.5`
  - `min_tracking_confidence = 0.5`

Alternatives considered and why not used:
- MoveNet:
  - Efficient and fast, but typically exposed with 17 keypoints.
  - Less detailed lower-body/foot coverage than BlazePose 33-landmark schema for framing checks.
- YOLOv8-Pose:
  - Strong model family, but heavier install/runtime and often GPU-oriented for smooth realtime throughput.
  - Overkill for a single-person framing prototype in a 4-6h assignment scope.
- OpenPose:
  - Historically strong but setup/runtime complexity is much higher for a quick CPU-first prototype.

Why MediaPipe Pose specifically:
- CPU-friendly real-time performance.
- 33 landmarks include ankles, heels, foot index points useful for full-body coverage heuristics.
- Mature Python API and low integration overhead.
- Commonly used in fitness-context demos and products.

### Framing logic: keypoint-based heuristics (not box-only)
Chosen:
- Rule-based checks using specific pose landmarks (nose, shoulders, hips, knees, ankles, heels, foot tips).

Alternative:
- Bounding-box-only framing logic.

Why keypoint logic:
- Distinguishes meaningful body structure from coarse silhouette extent.
- Supports explicit checks like “ankles missing while shoulders clear” for cropped-feet detection.
- Better aligned with downstream fitness pose assessment needs.

### Audio stack: layered fallback chain
Chosen order:
1. Layer 1: `gTTS + pygame` (best natural quality, internet required).
2. Layer 2: pre-recorded MP3 clips + `pygame`.
3. Layer 3: `pyttsx3` offline TTS (fully local fallback).

Why this design:
- Maximizes quality when internet is available.
- Preserves functionality offline.
- Handles environment variability (missing internet, missing clips, or missing playback backend).

### Debounce strategy: combined, not single-mechanism
Chosen:
- Stability buffer (8 frames) + state-change gate + 2.5s cooldown.

Alternative single approaches:
- Cooldown-only: still allows repeated prompts when state alternates after cooldown.
- State-change-only: can still chatter at boundary oscillations.

Why combined:
- Stability suppresses transient threshold crossings.
- State gate avoids repeating unchanged instructions.
- Cooldown limits bursty alternation prompts.

## 5. Heuristic Design & Threshold Rationale
### Landmark set used in logic
- `NOSE (0)`
- `LEFT_SHOULDER (11), RIGHT_SHOULDER (12)`
- `LEFT_HIP (23), RIGHT_HIP (24)`
- `LEFT_KNEE (25), RIGHT_KNEE (26)`
- `LEFT_ANKLE (27), RIGHT_ANKLE (28)`
- `LEFT_HEEL (29), RIGHT_HEEL (30)`
- `LEFT_FOOT_INDEX (31), RIGHT_FOOT_INDEX (32)`

### State priority order (exact)
1. `NO_PERSON`
2. `TOO_CLOSE`
3. `TOO_FAR`
4. `SHIFTED_LEFT` / `SHIFTED_RIGHT`
5. `GOOD`

### Core metric definitions
- `body_span_ratio = (ankle_y_avg - nose_y) / frame_height` (pixel coords)
- `mid_hip_x = (left_hip_x + right_hip_x) / 2` (normalized)

### Threshold table and rationale
| Constant | Value | Why this value |
|---|---:|---|
| `MIN_DETECTION_CONFIDENCE` | 0.5 | Balanced detection sensitivity vs false positives in unconstrained webcam input. |
| `MIN_TRACKING_CONFIDENCE` | 0.5 | Stable enough temporal tracking without over-filtering legitimate motion. |
| `MIN_VISIBILITY_THRESHOLD` | 0.3 | Conservative cutoff for “landmark is present enough” in noisy detections. |
| `ANKLE_VISIBILITY_THRESHOLD` | 0.5 | Stricter confidence needed before using ankles to claim full-body visibility. |
| `TOO_CLOSE_SPAN_RATIO` | 0.88 | If body occupies >88% of frame height, practical head/feet crop risk is high. |
| `TOO_FAR_SPAN_RATIO` | 0.40 | Below 40% body span, landmarks are often too small/noisy for reliable assessment even if fully visible. |
| `HEAD_TOP_MARGIN` | 0.05 | Nose in top 5% strongly indicates head near/crossing frame edge. |
| `HIP_LEFT_THRESHOLD` | 0.35 | Mid-hip left of 35% means user is substantially off-center. |
| `HIP_RIGHT_THRESHOLD` | 0.65 | Mid-hip right of 65% means user is substantially off-center. |
| `STABILITY_FRAMES` | 8 | ~0.25s at ~30 FPS; enough to suppress flicker while retaining responsiveness. |
| `AUDIO_COOLDOWN_SECONDS` | 2.5 | Prevents rapid-fire prompts in oscillating boundary conditions. |
| `TOO_CLOSE_ANKLE_VISIBILITY_THRESHOLD` | 0.4 | Low ankle visibility with visible shoulders suggests feet crop due to closeness. |
| `TOO_CLOSE_SHOULDER_VISIBILITY_THRESHOLD` | 0.7 | Requires shoulders to be confidently visible before interpreting weak ankles as crop. |

### Why `TOO_FAR_SPAN_RATIO = 0.40`
At spans below 40% of frame height, joints become too small for robust keypoint confidence in common laptop webcams, especially ankles/feet. This threshold intentionally prioritizes “assessment quality” over “barely visible full body”, matching the assignment requirement.

### Why `TOO_CLOSE_SPAN_RATIO = 0.88`
Once the body fills roughly 88%+ of frame height, small posture changes often push head/feet beyond frame boundaries. The threshold gives a small safety margin before actual clipping dominates.

### Why mid-hip for centering (not nose or shoulders)
- Hip midpoint is more stable than nose under head movement (looking down/up).
- Shoulders can skew with arm motion or torso rotation.
- Hips better represent body mass center for framing quality.

### Stability buffer effect
Without buffering, single-frame jitter around thresholds creates frequent state flips. Requiring 8 consecutive matching frames makes transitions deliberate and reduces audio flicker while keeping feedback near real-time.

## 6. System Architecture
```text
+--------------------------+
| Webcam (OpenCV capture)  |
+------------+-------------+
             |
             v
+--------------------------+
| MediaPipe Pose Detector  |
| - 33 landmarks           |
| - visibility scores      |
+------------+-------------+
             |
             v
+-------------------------------+
| Framing Heuristics            |
| - NO_PERSON / TOO_CLOSE       |
| - TOO_FAR / SHIFTED / GOOD    |
| - body_span_ratio, mid_hip_x  |
+---------------+---------------+
                |
                v
+-----------------------------------------------+
| DebounceController                             |
| - Stability buffer (8 frames)                  |
| - State-change gate                            |
| - 2.5s cooldown                                |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| AudioEngine                                    |
| Layer 1: gTTS + pygame                         |
| Layer 2: pre-recorded clips + pygame           |
| Layer 3: pyttsx3 offline                       |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| Overlay + Display                              |
| - State label, metrics, audio layer            |
| - skeleton, center line, hip-zone guide        |
+-----------------------------------------------+
```

## 7. Known Limitations
- Side-view and back-view poses are outside scope; logic is front-view tuned.
- Poor lighting can reduce landmark visibility and trigger noisy classifications.
- Layer 1 (gTTS) needs internet to generate startup clips; fallback chain handles outages.
- Baggy/occluding clothing can reduce keypoint confidence.
- No multi-person support (deliberate scope choice).
- Extreme distances weaken ankle visibility; ankle checks are the weakest link in far-range framing reliability.

## 8. What I Would Improve With More Time
- Use MediaPipe world landmarks (3D) for more direct distance estimation.
- Add startup calibration to adapt thresholds to camera FOV and room geometry.
- Add person/body segmentation for precise body coverage beyond sparse keypoints.
- Replace basic offline TTS with higher-quality local neural TTS (e.g., Coqui TTS).
- Add explicit calibration mode where user stands in ideal position to set personalized baseline.

## 9. Tradeoffs Made
- Chose speed/integration simplicity over maximal model accuracy (MediaPipe vs heavier pose stacks).
- Chose resilient audio fallback behavior over always-best voice quality.
- Chose fixed thresholds over adaptive calibration to stay within assignment scope.
- Chose interpretable rule-based heuristics over an ML framing-quality classifier for transparency and quick iteration.

## Implementation Notes
- Main loop logs FPS every 30 frames.
- Audio calls are dispatched via daemon threads to avoid blocking camera capture.
- Webcam runtime includes warm-up, consecutive read-failure tracking, and auto-reconnect attempts.
- Every module is separated by responsibility:
  - `config.py`: constants only
  - `pose_detector.py`: MediaPipe wrapper and landmark extraction
  - `framing_logic.py`: deterministic classification
  - `debounce_controller.py`: state stabilization + cooldown
  - `audio_engine.py`: layered speech backends
  - `utils.py`: overlays and coordinate helpers
  - `main.py`: orchestration loop
