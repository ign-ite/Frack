"""Rule-based framing heuristics built on MediaPipe Pose landmarks."""

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from config import (
    ANKLE_VISIBILITY_THRESHOLD,
    CALIBRATION_CENTER_MARGIN,
    CALIBRATION_CLOSE_SCALE,
    CALIBRATION_FAR_SCALE,
    CALIBRATION_PROFILE_FILE,
    HEAD_TOP_MARGIN,
    HIP_LEFT_THRESHOLD,
    HIP_RIGHT_THRESHOLD,
    LANDMARK_LEFT_ANKLE,
    LANDMARK_LEFT_HIP,
    LANDMARK_LEFT_SHOULDER,
    LANDMARK_NOSE,
    LANDMARK_RIGHT_ANKLE,
    LANDMARK_RIGHT_HIP,
    LANDMARK_RIGHT_SHOULDER,
    MIN_VISIBILITY_THRESHOLD,
    SIDE_ORIENTATION_RATIO_THRESHOLD,
    TOO_CLOSE_ANKLE_VISIBILITY_THRESHOLD,
    TOO_CLOSE_SHOULDER_VISIBILITY_THRESHOLD,
    TOO_CLOSE_SPAN_RATIO,
    TOO_FAR_SPAN_RATIO,
)
from pose_detector import LandmarkPoint


class FramingState(str, Enum):
    """Finite set of framing states produced by heuristic logic."""

    NO_PERSON = "NO_PERSON"
    TOO_CLOSE = "TOO_CLOSE"
    TOO_FAR = "TOO_FAR"
    SHIFTED_LEFT = "SHIFTED_LEFT"
    SHIFTED_RIGHT = "SHIFTED_RIGHT"
    GOOD = "GOOD"


@dataclass
class FramingAnalysis:
    """Classification output and debug metrics for the current frame.

    Attributes:
        state: Classified framing state for the current frame.
        body_span_ratio: Fraction of frame height covered from nose to ankle average.
        mid_hip_x: Normalized horizontal midpoint of hips in [0, 1].
        critical_joints_confident: Visibility-based confidence summary from detector.
    """

    state: FramingState
    body_span_ratio: Optional[float]
    mid_hip_x: Optional[float]
    critical_joints_confident: bool
    orientation_label: str
    shoulder_confidence: Optional[float]
    ankle_confidence: Optional[float]
    lean_angle_deg: Optional[float]


STATE_TO_INSTRUCTION_KEY = {
    FramingState.NO_PERSON: "no_person",
    FramingState.TOO_CLOSE: "too_close",
    FramingState.TOO_FAR: "too_far",
    FramingState.SHIFTED_LEFT: "move_right",
    FramingState.SHIFTED_RIGHT: "move_left",
    FramingState.GOOD: "good",
}


class FramingLogic:
    """Assignment-specified pose-keypoint framing classifier.

    The classifier follows the exact priority order requested in the assignment:
    1) NO_PERSON, 2) TOO_CLOSE, 3) TOO_FAR, 4) SHIFTED_LEFT/SHIFTED_RIGHT, 5) GOOD.
    """

    def __init__(self) -> None:
        self._too_close_span_ratio = TOO_CLOSE_SPAN_RATIO
        self._too_far_span_ratio = TOO_FAR_SPAN_RATIO
        self._hip_left_threshold = HIP_LEFT_THRESHOLD
        self._hip_right_threshold = HIP_RIGHT_THRESHOLD
        self._calibration_file = Path(__file__).resolve().parent / CALIBRATION_PROFILE_FILE
        self._is_calibrated = False
        self._load_calibration_profile()

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def hip_thresholds(self) -> Tuple[float, float]:
        return self._hip_left_threshold, self._hip_right_threshold

    def classify(
        self,
        landmarks: Optional[Dict[int, LandmarkPoint]],
        frame_height: int,
        critical_joints_confident: bool,
    ) -> FramingAnalysis:
        """Classify the current frame into one framing state.

        Args:
            landmarks: Landmark dictionary keyed by MediaPipe index. None when no
                person was detected.
            frame_height: Current frame height in pixels.
            critical_joints_confident: Visibility confidence summary from detector.

        Returns:
            FramingAnalysis containing the state and debug metrics.

        Notes:
            body_span_ratio uses pixel coordinates, as required:
            (ankle_y_avg - nose_y) / frame_height
        """
        if landmarks is None:
            return self._analysis(
                state=FramingState.NO_PERSON,
                body_span_ratio=None,
                mid_hip_x=None,
                critical_joints_confident=False,
                orientation_label="UNKNOWN",
                shoulder_confidence=None,
                ankle_confidence=None,
                lean_angle_deg=None,
            )

        visibilities = [point.visibility for point in landmarks.values()]
        if visibilities and all(value < MIN_VISIBILITY_THRESHOLD for value in visibilities):
            return self._analysis(
                state=FramingState.NO_PERSON,
                body_span_ratio=None,
                mid_hip_x=None,
                critical_joints_confident=False,
                orientation_label="UNKNOWN",
                shoulder_confidence=None,
                ankle_confidence=None,
                lean_angle_deg=None,
            )

        nose = landmarks.get(LANDMARK_NOSE)
        left_shoulder = landmarks.get(LANDMARK_LEFT_SHOULDER)
        right_shoulder = landmarks.get(LANDMARK_RIGHT_SHOULDER)
        left_hip = landmarks.get(LANDMARK_LEFT_HIP)
        right_hip = landmarks.get(LANDMARK_RIGHT_HIP)
        left_ankle = landmarks.get(LANDMARK_LEFT_ANKLE)
        right_ankle = landmarks.get(LANDMARK_RIGHT_ANKLE)

        if nose is None:
            return self._analysis(
                state=FramingState.NO_PERSON,
                body_span_ratio=None,
                mid_hip_x=None,
                critical_joints_confident=False,
                orientation_label="UNKNOWN",
                shoulder_confidence=None,
                ankle_confidence=None,
                lean_angle_deg=None,
            )

        ankle_points = self._visible_points((left_ankle, right_ankle), MIN_VISIBILITY_THRESHOLD)
        if not ankle_points:
            ankle_points = self._existing_points((left_ankle, right_ankle))
        if not ankle_points:
            return self._analysis(
                state=FramingState.NO_PERSON,
                body_span_ratio=None,
                mid_hip_x=None,
                critical_joints_confident=False,
                orientation_label="UNKNOWN",
                shoulder_confidence=self._mean_visibility((left_shoulder, right_shoulder)),
                ankle_confidence=None,
                lean_angle_deg=None,
            )

        ankle_y_avg = sum(point.y_px for point in ankle_points) / float(len(ankle_points))
        body_span_ratio = (ankle_y_avg - float(nose.y_px)) / float(max(1, frame_height))

        shoulder_confidence = self._mean_visibility((left_shoulder, right_shoulder))
        ankle_confidence = self._mean_visibility((left_ankle, right_ankle))
        mid_hip_x, mid_hip_y = self._midpoint_norm((left_hip, right_hip), fallback=(nose.x_norm, nose.y_norm))
        mid_shoulder_x, mid_shoulder_y = self._midpoint_norm((left_shoulder, right_shoulder), fallback=None)

        orientation_label = self._estimate_orientation(
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            left_hip=left_hip,
            right_hip=right_hip,
        )

        lean_angle_deg = self._estimate_lean_angle(
            mid_shoulder_x=mid_shoulder_x,
            mid_shoulder_y=mid_shoulder_y,
            mid_hip_x=mid_hip_x,
            mid_hip_y=mid_hip_y,
        )

        visible_ankles_for_far = any(
            point.visibility >= ANKLE_VISIBILITY_THRESHOLD
            for point in self._existing_points((left_ankle, right_ankle))
        )

        # Priority 2: too close if body almost fills frame, head touches top margin,
        # or shoulders are clear while ankles are weak (feet likely cropped).
        if (
            body_span_ratio > self._too_close_span_ratio
            or nose.y_norm < HEAD_TOP_MARGIN
            or (
                (ankle_confidence or 0.0) < TOO_CLOSE_ANKLE_VISIBILITY_THRESHOLD
                and (shoulder_confidence or 0.0) > TOO_CLOSE_SHOULDER_VISIBILITY_THRESHOLD
            )
        ):
            return self._analysis(
                state=FramingState.TOO_CLOSE,
                body_span_ratio=body_span_ratio,
                mid_hip_x=mid_hip_x,
                critical_joints_confident=critical_joints_confident,
                orientation_label=orientation_label,
                shoulder_confidence=shoulder_confidence,
                ankle_confidence=ankle_confidence,
                lean_angle_deg=lean_angle_deg,
            )

        # Priority 3: too far even when full body is visible, if person appears small.
        if body_span_ratio < self._too_far_span_ratio and visible_ankles_for_far:
            return self._analysis(
                state=FramingState.TOO_FAR,
                body_span_ratio=body_span_ratio,
                mid_hip_x=mid_hip_x,
                critical_joints_confident=critical_joints_confident,
                orientation_label=orientation_label,
                shoulder_confidence=shoulder_confidence,
                ankle_confidence=ankle_confidence,
                lean_angle_deg=lean_angle_deg,
            )

        # Priority 4: lateral centering by hip midpoint, which is torso-stable.
        if mid_hip_x < self._hip_left_threshold:
            return self._analysis(
                state=FramingState.SHIFTED_LEFT,
                body_span_ratio=body_span_ratio,
                mid_hip_x=mid_hip_x,
                critical_joints_confident=critical_joints_confident,
                orientation_label=orientation_label,
                shoulder_confidence=shoulder_confidence,
                ankle_confidence=ankle_confidence,
                lean_angle_deg=lean_angle_deg,
            )

        if mid_hip_x > self._hip_right_threshold:
            return self._analysis(
                state=FramingState.SHIFTED_RIGHT,
                body_span_ratio=body_span_ratio,
                mid_hip_x=mid_hip_x,
                critical_joints_confident=critical_joints_confident,
                orientation_label=orientation_label,
                shoulder_confidence=shoulder_confidence,
                ankle_confidence=ankle_confidence,
                lean_angle_deg=lean_angle_deg,
            )

        return self._analysis(
            state=FramingState.GOOD,
            body_span_ratio=body_span_ratio,
            mid_hip_x=mid_hip_x,
            critical_joints_confident=critical_joints_confident,
            orientation_label=orientation_label,
            shoulder_confidence=shoulder_confidence,
            ankle_confidence=ankle_confidence,
            lean_angle_deg=lean_angle_deg,
        )

    def calibrate(
        self,
        landmarks: Optional[Dict[int, LandmarkPoint]],
        frame_height: int,
    ) -> bool:
        """Calibrate dynamic framing thresholds from the current user pose."""
        analysis = self.classify(
            landmarks=landmarks,
            frame_height=frame_height,
            critical_joints_confident=True,
        )
        if analysis.body_span_ratio is None or analysis.mid_hip_x is None:
            return False

        span = analysis.body_span_ratio
        center = analysis.mid_hip_x

        self._too_close_span_ratio = min(0.98, max(0.68, span * CALIBRATION_CLOSE_SCALE))
        self._too_far_span_ratio = min(0.78, max(0.20, span * CALIBRATION_FAR_SCALE))
        self._hip_left_threshold = max(0.10, min(0.49, center - CALIBRATION_CENTER_MARGIN))
        self._hip_right_threshold = min(0.90, max(0.51, center + CALIBRATION_CENTER_MARGIN))
        if self._too_far_span_ratio >= self._too_close_span_ratio - 0.05:
            self._too_far_span_ratio = max(0.20, self._too_close_span_ratio - 0.08)

        self._is_calibrated = True
        self._save_calibration_profile()
        return True

    def _load_calibration_profile(self) -> None:
        if not self._calibration_file.exists():
            return

        try:
            payload = json.loads(self._calibration_file.read_text(encoding="utf-8"))
            self._too_close_span_ratio = float(payload.get("too_close_span_ratio", TOO_CLOSE_SPAN_RATIO))
            self._too_far_span_ratio = float(payload.get("too_far_span_ratio", TOO_FAR_SPAN_RATIO))
            self._hip_left_threshold = float(payload.get("hip_left_threshold", HIP_LEFT_THRESHOLD))
            self._hip_right_threshold = float(payload.get("hip_right_threshold", HIP_RIGHT_THRESHOLD))
            self._is_calibrated = True
        except Exception:
            self._is_calibrated = False

    def _save_calibration_profile(self) -> None:
        payload = {
            "too_close_span_ratio": self._too_close_span_ratio,
            "too_far_span_ratio": self._too_far_span_ratio,
            "hip_left_threshold": self._hip_left_threshold,
            "hip_right_threshold": self._hip_right_threshold,
        }
        try:
            self._calibration_file.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    @staticmethod
    def _analysis(
        state: FramingState,
        body_span_ratio: Optional[float],
        mid_hip_x: Optional[float],
        critical_joints_confident: bool,
        orientation_label: str,
        shoulder_confidence: Optional[float],
        ankle_confidence: Optional[float],
        lean_angle_deg: Optional[float],
    ) -> FramingAnalysis:
        return FramingAnalysis(
            state=state,
            body_span_ratio=body_span_ratio,
            mid_hip_x=mid_hip_x,
            critical_joints_confident=critical_joints_confident,
            orientation_label=orientation_label,
            shoulder_confidence=shoulder_confidence,
            ankle_confidence=ankle_confidence,
            lean_angle_deg=lean_angle_deg,
        )

    @staticmethod
    def _existing_points(points: Iterable[Optional[LandmarkPoint]]) -> Tuple[LandmarkPoint, ...]:
        return tuple(point for point in points if point is not None)

    @staticmethod
    def _visible_points(
        points: Iterable[Optional[LandmarkPoint]],
        threshold: float,
    ) -> Tuple[LandmarkPoint, ...]:
        return tuple(
            point for point in points if point is not None and point.visibility >= threshold
        )

    @staticmethod
    def _mean_visibility(points: Iterable[Optional[LandmarkPoint]]) -> Optional[float]:
        existing = [point.visibility for point in points if point is not None]
        if not existing:
            return None
        return sum(existing) / float(len(existing))

    @staticmethod
    def _midpoint_norm(
        points: Iterable[Optional[LandmarkPoint]],
        fallback: Optional[Tuple[float, float]],
    ) -> Tuple[Optional[float], Optional[float]]:
        visible = [point for point in points if point is not None and point.visibility >= MIN_VISIBILITY_THRESHOLD]
        if not visible:
            visible = [point for point in points if point is not None]

        if visible:
            x_value = sum(point.x_norm for point in visible) / float(len(visible))
            y_value = sum(point.y_norm for point in visible) / float(len(visible))
            return x_value, y_value

        if fallback is not None:
            return fallback

        return None, None

    @staticmethod
    def _estimate_orientation(
        left_shoulder: Optional[LandmarkPoint],
        right_shoulder: Optional[LandmarkPoint],
        left_hip: Optional[LandmarkPoint],
        right_hip: Optional[LandmarkPoint],
    ) -> str:
        if any(point is None for point in (left_shoulder, right_shoulder, left_hip, right_hip)):
            return "UNKNOWN"

        shoulder_width = abs(left_shoulder.x_norm - right_shoulder.x_norm)
        hip_width = abs(left_hip.x_norm - right_hip.x_norm)
        torso_width = max(shoulder_width, hip_width)

        shoulder_y = (left_shoulder.y_norm + right_shoulder.y_norm) / 2.0
        hip_y = (left_hip.y_norm + right_hip.y_norm) / 2.0
        torso_height = max(0.05, abs(hip_y - shoulder_y))
        width_height_ratio = torso_width / torso_height

        return "SIDE" if width_height_ratio < SIDE_ORIENTATION_RATIO_THRESHOLD else "FRONT"

    @staticmethod
    def _estimate_lean_angle(
        mid_shoulder_x: Optional[float],
        mid_shoulder_y: Optional[float],
        mid_hip_x: Optional[float],
        mid_hip_y: Optional[float],
    ) -> Optional[float]:
        if (
            mid_shoulder_x is None
            or mid_shoulder_y is None
            or mid_hip_x is None
            or mid_hip_y is None
        ):
            return None

        vertical_component = max(1e-6, mid_hip_y - mid_shoulder_y)
        horizontal_component = mid_shoulder_x - mid_hip_x
        return abs(math.degrees(math.atan2(horizontal_component, vertical_component)))


def state_to_instruction_key(state: FramingState) -> str:
    """Map framing state to speech instruction key.

    Args:
        state: Framing state produced by FramingLogic.

    Returns:
        Instruction key used by AudioEngine.

    Notes:
        SHIFTED_LEFT -> "move_right" and SHIFTED_RIGHT -> "move_left" intentionally.
        The assignment defines movement from the user's perspective while facing
        the camera; comments are explicit to avoid camera/user perspective confusion.
    """
    return STATE_TO_INSTRUCTION_KEY[state]
