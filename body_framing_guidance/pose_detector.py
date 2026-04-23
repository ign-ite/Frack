"""MediaPipe Pose wrapper for real-time landmark extraction.

This module assumes a single person in frame and supports both front-facing
and side-facing orientations for downstream framing logic.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import mediapipe as mp

from config import (
    CRITICAL_JOINT_INDICES,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MIN_VISIBILITY_THRESHOLD,
)
from utils import normalized_to_pixel_coordinates


@dataclass
class LandmarkPoint:
    """A single pose landmark with normalized and pixel coordinates.

    Attributes:
        x_norm: Normalized x coordinate in [0, 1] relative to image width.
        y_norm: Normalized y coordinate in [0, 1] relative to image height.
        z_norm: Relative depth from MediaPipe Pose (not metric distance).
        visibility: MediaPipe confidence score for this landmark.
        x_px: Pixel x coordinate clamped to image bounds.
        y_px: Pixel y coordinate clamped to image bounds.
    """

    x_norm: float
    y_norm: float
    z_norm: float
    visibility: float
    x_px: int
    y_px: int


@dataclass
class PoseDetectionResult:
    """Structured output from a single pose inference pass.

    Attributes:
        landmarks: Mapping from landmark index to LandmarkPoint. None when no
            person is detected.
        pose_landmarks_proto: Raw MediaPipe landmark proto for drawing.
        critical_joints_confident: True when all assignment-critical joints meet
            the minimum visibility threshold.
    """

    landmarks: Optional[Dict[int, LandmarkPoint]]
    pose_landmarks_proto: Optional[object]
    critical_joints_confident: bool


class PoseDetector:
    """Thin object-oriented wrapper around MediaPipe Pose.

    The detector is configured for CPU-friendly real-time inference and returns
    all 33 pose landmarks on each frame when available.
    """

    def __init__(self) -> None:
        """Initialize MediaPipe Pose with assignment-required thresholds.

        Raises:
            RuntimeError: If installed MediaPipe build does not expose the legacy
                `solutions.pose` API required by this prototype.
        """
        self._mp_pose = self._resolve_pose_api()
        self._pose = self._mp_pose.Pose(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    @staticmethod
    def _resolve_pose_api():
        """Resolve MediaPipe Pose API namespace with explicit compatibility checks.

        Returns:
            MediaPipe pose module namespace exposing the `Pose` class.

        Raises:
            RuntimeError: When MediaPipe package does not provide the
                `mediapipe.solutions.pose` API.
        """
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return mp.solutions.pose

        mp_version = getattr(mp, "__version__", "unknown")
        raise RuntimeError(
            "Installed mediapipe package does not expose 'mediapipe.solutions.pose' "
            f"(detected version: {mp_version}). Use Python 3.12 with "
            "mediapipe==0.10.14 and run setup.ps1 from the project root."
        )

    @property
    def mp_pose(self):
        """Expose MediaPipe pose namespace for drawing connections.

        Returns:
            The `mediapipe.solutions.pose` namespace.
        """
        return self._mp_pose

    def detect(self, frame_bgr) -> PoseDetectionResult:
        """Run pose estimation on a BGR frame.

        Args:
            frame_bgr: OpenCV BGR image for inference.

        Returns:
            PoseDetectionResult containing landmark dictionaries in normalized and
            pixel coordinate space, plus a confidence summary flag.

        Notes:
            This method converts BGR to RGB before inference because MediaPipe
            Pose expects RGB input.
        """
        if frame_bgr is None:
            return PoseDetectionResult(
                landmarks=None,
                pose_landmarks_proto=None,
                critical_joints_confident=False,
            )

        frame_height, frame_width = frame_bgr.shape[:2]

        # MediaPipe Pose expects RGB images.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(frame_rgb)

        if result.pose_landmarks is None:
            return PoseDetectionResult(
                landmarks=None,
                pose_landmarks_proto=None,
                critical_joints_confident=False,
            )

        landmarks: Dict[int, LandmarkPoint] = {}
        for index, landmark in enumerate(result.pose_landmarks.landmark):
            x_px, y_px = normalized_to_pixel_coordinates(
                landmark.x,
                landmark.y,
                frame_width,
                frame_height,
            )
            landmarks[index] = LandmarkPoint(
                x_norm=landmark.x,
                y_norm=landmark.y,
                z_norm=landmark.z,
                visibility=landmark.visibility,
                x_px=x_px,
                y_px=y_px,
            )

        critical_joints_confident = self._are_critical_joints_confident(landmarks)

        return PoseDetectionResult(
            landmarks=landmarks,
            pose_landmarks_proto=result.pose_landmarks,
            critical_joints_confident=critical_joints_confident,
        )

    def close(self) -> None:
        """Release MediaPipe resources associated with the pose graph."""
        self._pose.close()

    @staticmethod
    def _are_critical_joints_confident(landmarks: Dict[int, LandmarkPoint]) -> bool:
        """Check whether assignment-critical joints are detected reliably.

        Args:
            landmarks: Landmark map keyed by MediaPipe landmark index.

        Returns:
            True if all critical joints have visibility >= MIN_VISIBILITY_THRESHOLD.
        """
        return all(
            landmarks[index].visibility >= MIN_VISIBILITY_THRESHOLD
            for index in CRITICAL_JOINT_INDICES
            if index in landmarks
        )
