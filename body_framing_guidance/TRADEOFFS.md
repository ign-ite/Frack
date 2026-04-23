# Runtime Tradeoffs (Current)

## Product Direction
This pass intentionally prioritized features that reduce setup friction and improve autonomous operation:
- hands-free control
- orientation-robust guidance
- automation and logging

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
