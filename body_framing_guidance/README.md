# Body Framing Guidance Runtime

This folder contains the full runtime for real-time webcam framing guidance.

## What Changed in This Iteration

The runtime now focuses on high-impact product behavior:
- Orientation-aware guidance (`FRONT` and `SIDE` body poses)
- Gesture control actions (mute, screenshot, calibration)
- Remote control API + phone web panel
- Startup countdown before analysis
- User-specific calibration with persisted thresholds
- Session CSV logging for observability
- Cleaner visual overlay with directional arrows and live metrics

## Runtime Modules

- `main.py`: main loop orchestration, automation, controls, camera recovery
- `pose_detector.py`: MediaPipe pose inference wrapper
- `framing_logic.py`: orientation-aware deterministic framing classifier + calibration persistence
- `debounce_controller.py`: state stabilization and audio cooldown logic
- `gesture_controller.py`: gesture-to-command mapping
- `remote_control.py`: Flask panel and action endpoints
- `session_logger.py`: CSV session event logging
- `audio_engine.py`: layered speech fallback engine
- `utils.py`: drawing helpers and UI overlay rendering
- `config.py`: all tunable constants

## Runtime Outputs

Generated during use:
- `calibration_profile.json` (saved thresholds)
- `captures/` (screenshots)
- `session_logs/` (CSV logs)
- `audio_cache/` (generated voice clips)

## Controls Summary

Keyboard:
- `q`, `s`, `m`, `d`, `c`, `a`, `x`

Gestures:
- left wave: mute/unmute
- both hands up: screenshot
- T-pose hold: calibration

Remote web panel:
- `http://<host>:5000`
- start / stop / mute / screenshot / calibrate
