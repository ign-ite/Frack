"""Utility helpers for coordinate conversion and OpenCV overlays."""

from typing import Optional, Tuple

import cv2
import mediapipe as mp

from config import (
    AUDIO_LAYER_TEXT_COLOR,
    CENTER_LINE_COLOR,
    CENTER_LINE_THICKNESS,
    DEBUG_TEXT_COLOR,
    FONT_SCALE,
    FONT_THICKNESS,
    GOOD_ZONE_BORDER_COLOR,
    GOOD_ZONE_BORDER_THICKNESS,
    GOOD_ZONE_COLOR,
    HIP_LEFT_THRESHOLD,
    HIP_RIGHT_THRESHOLD,
    SMALL_FONT_SCALE,
    SMALL_FONT_THICKNESS,
    STATE_BAD_COLOR,
    STATE_GOOD_COLOR,
    TEXT_LINE_SPACING,
    TEXT_MARGIN_X,
    TEXT_MARGIN_Y,
    ZONE_ALPHA,
)


def normalized_to_pixel_coordinates(
    x_norm: float,
    y_norm: float,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int]:
    """Convert normalized MediaPipe coordinates to clamped pixel coordinates.

    Args:
        x_norm: Normalized x value in approximately [0, 1].
        y_norm: Normalized y value in approximately [0, 1].
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Tuple of integer pixel coordinates (x_px, y_px) clamped to image bounds.
    """
    x_px = int(x_norm * frame_width)
    y_px = int(y_norm * frame_height)

    # Clamp to avoid out-of-range values when landmarks jitter near borders.
    x_px = max(0, min(frame_width - 1, x_px))
    y_px = max(0, min(frame_height - 1, y_px))
    return x_px, y_px


def draw_pose_skeleton(frame_bgr, pose_landmarks_proto) -> None:
    """Draw MediaPipe pose landmarks and edges on a frame.

    Args:
        frame_bgr: OpenCV BGR frame to mutate in place.
        pose_landmarks_proto: Raw MediaPipe landmark proto from pose inference.

    Returns:
        None. The input frame is modified directly.
    """
    if pose_landmarks_proto is None:
        return

    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose_module = mp.solutions.pose

    drawing_utils.draw_landmarks(
        frame_bgr,
        pose_landmarks_proto,
        pose_module.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
    )


def draw_guidance_overlay(
    frame_bgr,
    state_label: str,
    body_span_ratio: Optional[float],
    mid_hip_x: Optional[float],
    audio_layer_label: str,
) -> None:
    """Render assignment-required debug and guidance overlays.

    Args:
        frame_bgr: Frame to draw on (mutated in place).
        state_label: Current state label to display.
        body_span_ratio: Debug metric for body scale in frame.
        mid_hip_x: Debug metric for lateral centering.
        audio_layer_label: Active audio layer descriptor.

    Returns:
        None. The input frame is modified directly.
    """
    frame_height, frame_width = frame_bgr.shape[:2]

    zone_left = int(HIP_LEFT_THRESHOLD * frame_width)
    zone_right = int(HIP_RIGHT_THRESHOLD * frame_width)

    overlay = frame_bgr.copy()
    cv2.rectangle(
        overlay,
        (zone_left, 0),
        (zone_right, frame_height - 1),
        GOOD_ZONE_COLOR,
        -1,
    )
    # Alpha blending creates a visible target area without hiding landmarks.
    cv2.addWeighted(overlay, ZONE_ALPHA, frame_bgr, 1.0 - ZONE_ALPHA, 0.0, frame_bgr)

    cv2.rectangle(
        frame_bgr,
        (zone_left, 0),
        (zone_right, frame_height - 1),
        GOOD_ZONE_BORDER_COLOR,
        GOOD_ZONE_BORDER_THICKNESS,
    )

    center_x = frame_width // 2
    cv2.line(
        frame_bgr,
        (center_x, 0),
        (center_x, frame_height - 1),
        CENTER_LINE_COLOR,
        CENTER_LINE_THICKNESS,
    )

    is_good_state = state_label.upper() == "GOOD"
    state_color = STATE_GOOD_COLOR if is_good_state else STATE_BAD_COLOR
    state_text = f"STATE: {state_label}"
    cv2.putText(
        frame_bgr,
        state_text,
        (TEXT_MARGIN_X, TEXT_MARGIN_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        state_color,
        FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    span_text = (
        f"body_span_ratio: {body_span_ratio:.3f}"
        if body_span_ratio is not None
        else "body_span_ratio: N/A"
    )
    cv2.putText(
        frame_bgr,
        span_text,
        (TEXT_MARGIN_X, TEXT_MARGIN_Y + TEXT_LINE_SPACING),
        cv2.FONT_HERSHEY_SIMPLEX,
        SMALL_FONT_SCALE,
        DEBUG_TEXT_COLOR,
        SMALL_FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    hip_text = f"mid_hip_x: {mid_hip_x:.3f}" if mid_hip_x is not None else "mid_hip_x: N/A"
    cv2.putText(
        frame_bgr,
        hip_text,
        (TEXT_MARGIN_X, TEXT_MARGIN_Y + (2 * TEXT_LINE_SPACING)),
        cv2.FONT_HERSHEY_SIMPLEX,
        SMALL_FONT_SCALE,
        DEBUG_TEXT_COLOR,
        SMALL_FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    audio_text = f"Audio: {audio_layer_label}"
    (audio_text_width, _), _ = cv2.getTextSize(
        audio_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        SMALL_FONT_SCALE,
        SMALL_FONT_THICKNESS,
    )
    audio_x = max(TEXT_MARGIN_X, frame_width - audio_text_width - TEXT_MARGIN_X)
    cv2.putText(
        frame_bgr,
        audio_text,
        (audio_x, TEXT_MARGIN_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        SMALL_FONT_SCALE,
        AUDIO_LAYER_TEXT_COLOR,
        SMALL_FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )
