"""Rule-based framing heuristics built on MediaPipe Pose landmarks."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from config import (
    ANKLE_VISIBILITY_THRESHOLD,
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
            return FramingAnalysis(
                state=FramingState.NO_PERSON,
                body_span_ratio=None,
                mid_hip_x=None,
                critical_joints_confident=False,
            )

        visibilities = [point.visibility for point in landmarks.values()]
        if visibilities and all(value < MIN_VISIBILITY_THRESHOLD for value in visibilities):
            return FramingAnalysis(
                state=FramingState.NO_PERSON,
                body_span_ratio=None,
                mid_hip_x=None,
                critical_joints_confident=False,
            )

        nose = landmarks[LANDMARK_NOSE]
        left_shoulder = landmarks[LANDMARK_LEFT_SHOULDER]
        right_shoulder = landmarks[LANDMARK_RIGHT_SHOULDER]
        left_hip = landmarks[LANDMARK_LEFT_HIP]
        right_hip = landmarks[LANDMARK_RIGHT_HIP]
        left_ankle = landmarks[LANDMARK_LEFT_ANKLE]
        right_ankle = landmarks[LANDMARK_RIGHT_ANKLE]

        ankle_y_avg = (left_ankle.y_px + right_ankle.y_px) / 2.0
        # Required metric: body coverage from nose to ankle relative to frame height.
        body_span_ratio = (ankle_y_avg - float(nose.y_px)) / float(frame_height)
        mid_hip_x = (left_hip.x_norm + right_hip.x_norm) / 2.0

        ankle_visibility_avg = (left_ankle.visibility + right_ankle.visibility) / 2.0
        shoulder_visibility_avg = (
            left_shoulder.visibility + right_shoulder.visibility
        ) / 2.0
        both_ankles_visible = (
            left_ankle.visibility > ANKLE_VISIBILITY_THRESHOLD
            and right_ankle.visibility > ANKLE_VISIBILITY_THRESHOLD
        )

        # Priority 2: too close if body almost fills frame, head touches top margin,
        # or shoulders are clear while ankles are weak (feet likely cropped).
        if (
            body_span_ratio > TOO_CLOSE_SPAN_RATIO
            or nose.y_norm < HEAD_TOP_MARGIN
            or (
                ankle_visibility_avg < TOO_CLOSE_ANKLE_VISIBILITY_THRESHOLD
                and shoulder_visibility_avg > TOO_CLOSE_SHOULDER_VISIBILITY_THRESHOLD
            )
        ):
            return FramingAnalysis(
                state=FramingState.TOO_CLOSE,
                body_span_ratio=body_span_ratio,
                mid_hip_x=mid_hip_x,
                critical_joints_confident=critical_joints_confident,
            )

        # Priority 3: too far even when full body is visible, if person appears small.
        if body_span_ratio < TOO_FAR_SPAN_RATIO and both_ankles_visible:
            return FramingAnalysis(
                state=FramingState.TOO_FAR,
                body_span_ratio=body_span_ratio,
                mid_hip_x=mid_hip_x,
                critical_joints_confident=critical_joints_confident,
            )

        # Priority 4: lateral centering by hip midpoint, which is torso-stable.
        if mid_hip_x < HIP_LEFT_THRESHOLD:
            return FramingAnalysis(
                state=FramingState.SHIFTED_LEFT,
                body_span_ratio=body_span_ratio,
                mid_hip_x=mid_hip_x,
                critical_joints_confident=critical_joints_confident,
            )

        if mid_hip_x > HIP_RIGHT_THRESHOLD:
            return FramingAnalysis(
                state=FramingState.SHIFTED_RIGHT,
                body_span_ratio=body_span_ratio,
                mid_hip_x=mid_hip_x,
                critical_joints_confident=critical_joints_confident,
            )

        return FramingAnalysis(
            state=FramingState.GOOD,
            body_span_ratio=body_span_ratio,
            mid_hip_x=mid_hip_x,
            critical_joints_confident=critical_joints_confident,
        )


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
