Body Framing Guidance - Layer 2 Audio Clips
===========================================

This folder is used by Audio Layer 2 (pre-recorded fallback).

Required files (exact names):
- no_person.mp3
- too_close.mp3
- too_far.mp3
- move_left.mp3
- move_right.mp3
- good.mp3

Optional file:
- hold.mp3
- missing_camera.mp3

Suggested phrase content:
- no_person.mp3  -> "Please step into frame"
- too_close.mp3  -> "Please move back"
- too_far.mp3    -> "Please come closer"
- move_left.mp3  -> "Move to your left"
- move_right.mp3 -> "Move to your right"
- good.mp3       -> "You're in frame. Hold this position."
- hold.mp3       -> "Hold this position."
- missing_camera.mp3 -> "No camera detected. Please connect a camera or choose another input."

Recording guidance:
1) Use clear speech in a quiet room.
2) Keep each clip under ~1.5 seconds for responsive feedback.
3) Normalize loudness so all prompts have similar volume.
4) Export as MP3 and keep a sample rate of 44.1 kHz or 48 kHz.

When Layer 1 (gTTS+pygame) fails, the app automatically checks this folder.
