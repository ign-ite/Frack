"""Central configuration for the body framing guidance prototype.

All numeric thresholds and tunable constants live here so the logic modules avoid
magic numbers and can be tuned quickly during testing.
"""

# Pose detection thresholds
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MIN_VISIBILITY_THRESHOLD = 0.3
ANKLE_VISIBILITY_THRESHOLD = 0.5

# Framing heuristics
TOO_CLOSE_SPAN_RATIO = 0.88  # Body fills >88% of frame height
TOO_FAR_SPAN_RATIO = 0.40  # Body fills <40% of frame height
HEAD_TOP_MARGIN = 0.05  # Nose within 5% of top edge means potential head crop
HIP_LEFT_THRESHOLD = 0.35  # mid_hip_x < this -> shifted left (tell user to move right)
HIP_RIGHT_THRESHOLD = 0.65  # mid_hip_x > this -> shifted right (tell user to move left)
TOO_CLOSE_ANKLE_VISIBILITY_THRESHOLD = 0.4
TOO_CLOSE_SHOULDER_VISIBILITY_THRESHOLD = 0.7

# Stability buffer
STABILITY_FRAMES = 8  # Frames required before a state change is accepted

# Audio debounce
AUDIO_COOLDOWN_SECONDS = 2.5

# Display
FONT_SCALE = 0.8
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SMALL_FONT_SCALE = 0.55
TEXT_MARGIN_X = 12
TEXT_MARGIN_Y = 28
TEXT_LINE_SPACING = 30
FONT_THICKNESS = 2
SMALL_FONT_THICKNESS = 1
CENTER_LINE_THICKNESS = 2
ZONE_BORDER_THICKNESS = 1
GOOD_ZONE_BORDER_THICKNESS = ZONE_BORDER_THICKNESS
ZONE_ALPHA = 0.18

# Colors in BGR (OpenCV format)
STATE_GOOD_COLOR = (0, 200, 0)
STATE_BAD_COLOR = (0, 0, 255)
DEBUG_TEXT_COLOR = (255, 255, 255)
AUDIO_LAYER_TEXT_COLOR = (225, 225, 225)
CENTER_LINE_COLOR = (255, 255, 0)
GOOD_ZONE_COLOR = (120, 120, 120)
GOOD_ZONE_BORDER_COLOR = (180, 180, 180)

# Landmark indices from MediaPipe Pose
LANDMARK_NOSE = 0
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24
LANDMARK_LEFT_KNEE = 25
LANDMARK_RIGHT_KNEE = 26
LANDMARK_LEFT_ANKLE = 27
LANDMARK_RIGHT_ANKLE = 28
LANDMARK_LEFT_HEEL = 29
LANDMARK_RIGHT_HEEL = 30
LANDMARK_LEFT_FOOT_INDEX = 31
LANDMARK_RIGHT_FOOT_INDEX = 32

CRITICAL_JOINT_INDICES = (
    LANDMARK_NOSE,
    LANDMARK_LEFT_SHOULDER,
    LANDMARK_RIGHT_SHOULDER,
    LANDMARK_LEFT_HIP,
    LANDMARK_RIGHT_HIP,
    LANDMARK_LEFT_KNEE,
    LANDMARK_RIGHT_KNEE,
    LANDMARK_LEFT_ANKLE,
    LANDMARK_RIGHT_ANKLE,
)

# Runtime and window behavior
WINDOW_TITLE = "Body Framing Guidance"
WEBCAM_INDEX = 0
FPS_LOG_INTERVAL = 30
EXIT_KEY = "q"
WAIT_KEY_DELAY_MS = 1
CAMERA_SCAN_MAX_INDEX = 8

# Webcam runtime resilience
CAMERA_WARMUP_FRAMES = 12
MAX_CONSECUTIVE_READ_FAILURES = 12
CAMERA_RECONNECT_ATTEMPTS = 5
CAMERA_RECONNECT_WAIT_SECONDS = 0.5
NO_CAMERA_AUDIO_INTERVAL_SECONDS = 10.0
NO_CAMERA_RETRY_KEY = "r"
NO_CAMERA_CHOOSE_KEY = "c"
NO_CAMERA_LIST_KEY = "l"
NO_CAMERA_QUIT_KEY = "q"
NO_CAMERA_TITLE_TEXT = "No Camera Feed"
NO_CAMERA_TEXT_COLOR = (255, 255, 255)
NO_CAMERA_HINT_COLOR = (180, 180, 180)
NO_CAMERA_ERROR_COLOR = (0, 0, 255)
NO_CAMERA_TEXT_START_Y = 120
NO_CAMERA_TEXT_LINE_SPACING = 34

# Audio phrases and files
AUDIO_CACHE_FOLDER = "audio_cache"
AUDIO_CLIPS_FOLDER = "audio_clips"
TEMP_AUDIO_PREFIX = "body_framing_guidance_audio_"
AUDIO_TTS_RATE = 180

INSTRUCTION_PHRASES = {
    "no_person": "Please step into frame",
    "too_close": "Please move back",
    "too_far": "Please come closer",
    "move_left": "Move to your left",
    "move_right": "Move to your right",
    "good": "You're in frame. Hold this position.",
    "hold": "Hold this position.",
    "missing_camera": "No camera detected. Please connect a camera or choose another input.",
}

# Layer 2 requires these files; hold.mp3 is optional.
LAYER2_REQUIRED_FILES = (
    "no_person.mp3",
    "too_close.mp3",
    "too_far.mp3",
    "move_left.mp3",
    "move_right.mp3",
    "good.mp3",
)
