"""Gesture-based control actions derived from MediaPipe pose landmarks."""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from config import (
    GESTURE_HANDS_UP_COOLDOWN_SECONDS,
    GESTURE_HANDS_UP_HOLD_SECONDS,
    GESTURE_T_POSE_COOLDOWN_SECONDS,
    GESTURE_T_POSE_HOLD_SECONDS,
    GESTURE_WAVE_COOLDOWN_SECONDS,
    GESTURE_WAVE_MIN_AMPLITUDE,
    GESTURE_WAVE_MIN_DIRECTION_CHANGES,
    GESTURE_WAVE_WINDOW_SECONDS,
    LANDMARK_LEFT_ELBOW,
    LANDMARK_LEFT_SHOULDER,
    LANDMARK_LEFT_WRIST,
    LANDMARK_NOSE,
    LANDMARK_RIGHT_ELBOW,
    LANDMARK_RIGHT_SHOULDER,
    LANDMARK_RIGHT_WRIST,
    T_POSE_MIN_SPAN,
    T_POSE_VERTICAL_TOLERANCE,
    WRIST_VISIBILITY_THRESHOLD,
)
from pose_detector import LandmarkPoint


class GestureController:
    """Map body gestures to high-level in-app control commands.

    Emitted command strings:
    - ``toggle_mute``: left-hand wave
    - ``screenshot``: both hands above head
    - ``calibrate``: T-pose hold
    """

    def __init__(self) -> None:
        self._left_wrist_history: Deque[Tuple[float, float]] = deque(maxlen=40)
        self._last_wave_trigger = 0.0

        self._hands_up_started_at: Optional[float] = None
        self._last_hands_up_trigger = 0.0

        self._t_pose_started_at: Optional[float] = None
        self._last_t_pose_trigger = 0.0

    def update(
        self,
        landmarks: Optional[Dict[int, LandmarkPoint]],
        now: float,
    ) -> List[str]:
        """Consume current-frame landmarks and return zero or more command strings."""
        if landmarks is None:
            self._hands_up_started_at = None
            self._t_pose_started_at = None
            self._prune_wrist_history(now)
            return []

        commands: List[str] = []
        self._prune_wrist_history(now)

        if self._detect_left_wave(landmarks, now):
            commands.append("toggle_mute")

        if self._detect_hands_up(landmarks, now):
            commands.append("screenshot")

        if self._detect_t_pose(landmarks, now):
            commands.append("calibrate")

        return commands

    def _prune_wrist_history(self, now: float) -> None:
        while self._left_wrist_history:
            timestamp, _ = self._left_wrist_history[0]
            if now - timestamp <= GESTURE_WAVE_WINDOW_SECONDS:
                break
            self._left_wrist_history.popleft()

    @staticmethod
    def _is_visible(point: Optional[LandmarkPoint], threshold: float) -> bool:
        return point is not None and point.visibility >= threshold

    def _detect_left_wave(
        self,
        landmarks: Dict[int, LandmarkPoint],
        now: float,
    ) -> bool:
        left_wrist = landmarks.get(LANDMARK_LEFT_WRIST)
        left_elbow = landmarks.get(LANDMARK_LEFT_ELBOW)
        left_shoulder = landmarks.get(LANDMARK_LEFT_SHOULDER)

        if (
            not self._is_visible(left_wrist, WRIST_VISIBILITY_THRESHOLD)
            or left_elbow is None
            or left_shoulder is None
        ):
            return False

        self._left_wrist_history.append((now, left_wrist.x_norm))
        if len(self._left_wrist_history) < 6:
            return False

        hand_raised = left_wrist.y_norm < min(left_elbow.y_norm, left_shoulder.y_norm)
        if not hand_raised:
            return False

        x_values = [value for _, value in self._left_wrist_history]
        amplitude = max(x_values) - min(x_values)
        direction_changes = 0
        previous_direction = 0
        for index in range(1, len(x_values)):
            delta = x_values[index] - x_values[index - 1]
            if abs(delta) < 0.01:
                continue
            direction = 1 if delta > 0 else -1
            if previous_direction != 0 and direction != previous_direction:
                direction_changes += 1
            previous_direction = direction

        if (
            amplitude >= GESTURE_WAVE_MIN_AMPLITUDE
            and direction_changes >= GESTURE_WAVE_MIN_DIRECTION_CHANGES
            and now - self._last_wave_trigger >= GESTURE_WAVE_COOLDOWN_SECONDS
        ):
            self._last_wave_trigger = now
            self._left_wrist_history.clear()
            return True

        return False

    def _detect_hands_up(
        self,
        landmarks: Dict[int, LandmarkPoint],
        now: float,
    ) -> bool:
        nose = landmarks.get(LANDMARK_NOSE)
        left_wrist = landmarks.get(LANDMARK_LEFT_WRIST)
        right_wrist = landmarks.get(LANDMARK_RIGHT_WRIST)

        hands_up = (
            nose is not None
            and self._is_visible(left_wrist, WRIST_VISIBILITY_THRESHOLD)
            and self._is_visible(right_wrist, WRIST_VISIBILITY_THRESHOLD)
            and left_wrist is not None
            and right_wrist is not None
            and left_wrist.y_norm < (nose.y_norm - 0.02)
            and right_wrist.y_norm < (nose.y_norm - 0.02)
        )

        if hands_up:
            if self._hands_up_started_at is None:
                self._hands_up_started_at = now
            held_seconds = now - self._hands_up_started_at
            if (
                held_seconds >= GESTURE_HANDS_UP_HOLD_SECONDS
                and now - self._last_hands_up_trigger >= GESTURE_HANDS_UP_COOLDOWN_SECONDS
            ):
                self._hands_up_started_at = None
                self._last_hands_up_trigger = now
                return True
        else:
            self._hands_up_started_at = None

        return False

    def _detect_t_pose(
        self,
        landmarks: Dict[int, LandmarkPoint],
        now: float,
    ) -> bool:
        left_shoulder = landmarks.get(LANDMARK_LEFT_SHOULDER)
        right_shoulder = landmarks.get(LANDMARK_RIGHT_SHOULDER)
        left_elbow = landmarks.get(LANDMARK_LEFT_ELBOW)
        right_elbow = landmarks.get(LANDMARK_RIGHT_ELBOW)
        left_wrist = landmarks.get(LANDMARK_LEFT_WRIST)
        right_wrist = landmarks.get(LANDMARK_RIGHT_WRIST)

        required_points = [
            left_shoulder,
            right_shoulder,
            left_elbow,
            right_elbow,
            left_wrist,
            right_wrist,
        ]
        if any(point is None for point in required_points):
            self._t_pose_started_at = None
            return False

        left_horizontal = (
            abs(left_elbow.y_norm - left_shoulder.y_norm) <= T_POSE_VERTICAL_TOLERANCE
            and abs(left_wrist.y_norm - left_shoulder.y_norm) <= T_POSE_VERTICAL_TOLERANCE
        )
        right_horizontal = (
            abs(right_elbow.y_norm - right_shoulder.y_norm) <= T_POSE_VERTICAL_TOLERANCE
            and abs(right_wrist.y_norm - right_shoulder.y_norm) <= T_POSE_VERTICAL_TOLERANCE
        )
        arm_span = abs(right_wrist.x_norm - left_wrist.x_norm)
        t_pose = left_horizontal and right_horizontal and arm_span >= T_POSE_MIN_SPAN

        if t_pose:
            if self._t_pose_started_at is None:
                self._t_pose_started_at = now
            held_seconds = now - self._t_pose_started_at
            if (
                held_seconds >= GESTURE_T_POSE_HOLD_SECONDS
                and now - self._last_t_pose_trigger >= GESTURE_T_POSE_COOLDOWN_SECONDS
            ):
                self._t_pose_started_at = None
                self._last_t_pose_trigger = now
                return True
        else:
            self._t_pose_started_at = None

        return False
