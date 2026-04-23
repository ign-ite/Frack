# Runtime Tradeoffs (Current)

## Product Direction
This pass intentionally prioritized features that reduce setup friction and improve autonomous operation:
- hands-free control
- orientation-robust guidance
- automation and logging


## Vision model comparison
| Option | Speed on CPU | Keypoints | Ease of Setup | Accuracy | Fitness App Suitability | Our Choice |
|---|---|---|---|---|---|---|
| MediaPipe Pose | High | 33 | Easy | Good for real-time pose and body coverage | High | Yes |
| MoveNet | High | 17 | Easy to moderate | Good for coarse pose | Medium | No |
| YOLOv8-Pose | Medium on CPU | 17+ | Moderate to heavy | High with GPU | Medium | No |
| OpenPose | Low on CPU | 25+ | Hard | Historically strong | Medium | No |

## Audio option comparison
| Option | Quality | Offline? | Latency | Setup Complexity | Our Choice |
|---|---|---|---|---|---|
| gTTS + pygame | High | No | Low once loaded | Moderate | Primary |
| Pre-recorded clips | Variable | Yes | Very low | Moderate | Fallback |
| pyttsx3 | Medium | Yes | Moderate | Easy | Final fallback |

## Debounce design decision
A single debounce strategy is not sufficient for a live webcam guidance loop. Cooldown-only still allows repeated prompts when the state alternates after the timer expires. State-change-only still triggers repeated audio if the state jitters at a boundary. The chosen design combines:
- Stability buffer (8 frames)
- State-change gating
- Minimum cooldown (2.5 seconds)

This combination provides both responsiveness and audible stability: the system ignores transient noise, only speaks when the state truly changes, and avoids rapid back-and-forth prompts.

## Tradeoff Summary

## 1) Gesture + Web Controls vs Voice Commands
Decision:
- Implemented gesture and remote web controls first.

Why:
- They are more deterministic in noisy home/gym environments than speech recognition.
- They align better with demo reliability and practical control from a phone.

Cost:
- Voice command convenience is deferred to a later iteration.

## 2) Orientation Heuristics vs Full 3D Orientation Estimation
Decision:
- Implemented torso geometry heuristics to classify `FRONT` vs `SIDE`.

Why:
- Keeps runtime fast and interpretable on CPU.
- Improves resilience for side-facing users without model complexity.

Cost:
- Extreme oblique angles can still be ambiguous.

## 3) Deterministic Framing Rules vs Learned Quality Model
Decision:
- Retained deterministic keypoint rules with calibration.

Why:
- Fast, debuggable behavior and low engineering overhead.
- Calibration personalizes thresholds without requiring training data.

Cost:
- Rules are less adaptive than a trained model for unusual body/camera setups.

## 4) Local Flask Control Panel vs Native UI App
Decision:
- Added lightweight Flask panel.

Why:
- Works instantly from a phone browser on local network.
- Minimal implementation complexity and no packaging burden.

Cost:
- No native mobile app affordances.

## 5) CSV Logging vs Full Telemetry Pipeline
Decision:
- Added local CSV session logs.

Why:
- Immediate observability and reproducibility for QA/interviews.
- Zero infrastructure requirements.

Cost:
- No centralized analytics or dashboards.

## 6) Startup Countdown vs Immediate Analysis
Decision:
- Added strict startup countdown before analysis begins.

Why:
- Better user experience for one-person setup.
- Reduces unstable early-frame noise while user is moving into position.

Cost:
- Adds fixed startup latency.


