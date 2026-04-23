# Tradeoff Analysis

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
