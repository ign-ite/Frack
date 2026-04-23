# Tradeoff Analysis

## Vision Model Comparison
| Option | Speed on CPU | Keypoints | Ease of Setup | Accuracy | Fitness App Suitability | Our Choice |
|---|---|---:|---|---|---|---|
| MediaPipe Pose (BlazePose) | High (real-time on typical laptop CPU) | 33 | Very easy (`pip install mediapipe`) | Good for real-time pose and body coverage heuristics | High (includes ankles/feet and fitness-oriented examples) | Yes |
| MoveNet | High | 17 | Easy to moderate (depends on runtime wrapper) | Good for coarse pose | Medium (fewer landmarks, less lower-body detail) | No |
| YOLOv8-Pose | Medium to low on CPU for real-time target | Typically 17 (variant-dependent) | Moderate to heavy | High potential, especially with GPU | Medium to high, but heavy for assignment scope | No |
| OpenPose | Low on CPU (better with GPU) | Rich skeleton outputs | Hard (complex dependencies/build) | Historically strong | Medium (engineering overhead too high for quick prototype) | No |

## Audio Option Comparison
| Option | Quality | Offline? | Latency | Setup Complexity | Our Choice |
|---|---|---|---|---|---|
| gTTS + pygame | High (natural voice) | No (internet required for synthesis) | Low at runtime after startup generation | Moderate | Primary (Layer 1) |
| Pre-recorded clips + pygame | Variable (depends on recordings) | Yes | Very low | Moderate (manual recording + file management) | Fallback (Layer 2) |
| pyttsx3 | Medium to low (robotic but understandable) | Yes | Low to moderate | Easy | Final fallback (Layer 3) |

## Debounce Design Decision
A single debounce mechanism is not enough for noisy real-time pose streams. If only cooldown is used, the system can still chatter whenever state alternation aligns with timer expiry. If only state-change gating is used, rapid threshold oscillation still produces repeated alternations (`TOO_FAR -> GOOD -> TOO_FAR`) and frequent prompts. The chosen hybrid design applies three layers: (1) an 8-frame stability buffer to suppress transient jitter, (2) a state-change gate so unchanged states are not re-spoken, and (3) a 2.5-second minimum cooldown to cap prompt frequency even under sustained boundary oscillation. This combination gave the best balance between responsiveness and user-comfortable audio behavior.
