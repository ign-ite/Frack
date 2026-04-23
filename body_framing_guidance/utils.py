"""Utility helpers for coordinate conversion and OpenCV overlays."""

from typing import Optional, Tuple

import cv2
import mediapipe as mp

from config import (
    ARROW_COLOR,
    AUDIO_LAYER_TEXT_COLOR,
    CENTER_LINE_COLOR,
    CENTER_LINE_THICKNESS,
    COUNTDOWN_COLOR,
    DEBUG_TEXT_COLOR,
    FONT_SCALE,
    FONT_THICKNESS,
    GOOD_ZONE_BORDER_COLOR,
    GOOD_ZONE_BORDER_THICKNESS,
    GOOD_ZONE_COLOR,
    HEADER_ALPHA,
    HIP_LEFT_THRESHOLD,
    HIP_RIGHT_THRESHOLD,
    SMALL_FONT_SCALE,
    SMALL_FONT_THICKNESS,
    STATE_BAD_COLOR,
    STATE_GOOD_COLOR,
    TEXT_LINE_SPACING,
    TEXT_MARGIN_X,
    TEXT_MARGIN_Y,
    WARNING_COLOR,
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
    orientation_label: str,
    shoulder_confidence: Optional[float],
    ankle_confidence: Optional[float],
    lean_angle_deg: Optional[float],
    fps_value: Optional[float],
    is_muted: bool,
    debug_enabled: bool,
    hip_left_threshold: float = HIP_LEFT_THRESHOLD,
    hip_right_threshold: float = HIP_RIGHT_THRESHOLD,
    countdown_remaining: Optional[float] = None,
    posture_warning: bool = False,
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

    zone_left = int(hip_left_threshold * frame_width)
    zone_right = int(hip_right_threshold * frame_width)

    header_height = 96 if debug_enabled else 70
    header = frame_bgr.copy()
    cv2.rectangle(header, (0, 0), (frame_width - 1, header_height), (20, 20, 20), -1)
    cv2.addWeighted(header, HEADER_ALPHA, frame_bgr, 1.0 - HEADER_ALPHA, 0.0, frame_bgr)

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
    state_text = f"STATE: {state_label}  |  ORIENTATION: {orientation_label} |"
    (state_text_width, _), _ = cv2.getTextSize(
        state_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_THICKNESS,
    )
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

    mute_text = "MUTED" if is_muted else "AUDIO LIVE"
    fps_text = f"FPS: {fps_value:.1f}" if fps_value is not None else "FPS: N/A"
    status_text = f"{mute_text}  |  {fps_text}"
    (status_text_width, _), _ = cv2.getTextSize(
        status_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        SMALL_FONT_SCALE,
        SMALL_FONT_THICKNESS,
    )
    cv2.putText(
        frame_bgr,
        status_text,
        (TEXT_MARGIN_X, TEXT_MARGIN_Y + TEXT_LINE_SPACING),
        cv2.FONT_HERSHEY_SIMPLEX,
        SMALL_FONT_SCALE,
        AUDIO_LAYER_TEXT_COLOR,
        SMALL_FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    if debug_enabled:
        span_text = (
            f"body_span_ratio: {body_span_ratio:.3f}"
            if body_span_ratio is not None
            else "body_span_ratio: N/A"
        )
        hip_text = (
            f"mid_hip_x: {mid_hip_x:.3f}"
            if mid_hip_x is not None
            else "mid_hip_x: N/A"
        )
        confidence_text = (
            "conf(shoulder/ankle): "
            f"{(shoulder_confidence or 0.0):.2f}/{(ankle_confidence or 0.0):.2f}"
        )
        lean_text = (
            f"lean_angle: {lean_angle_deg:.1f} deg"
            if lean_angle_deg is not None
            else "lean_angle: N/A"
        )

        cv2.putText(
            frame_bgr,
            span_text,
            (TEXT_MARGIN_X, TEXT_MARGIN_Y + (2 * TEXT_LINE_SPACING)),
            cv2.FONT_HERSHEY_SIMPLEX,
            SMALL_FONT_SCALE,
            DEBUG_TEXT_COLOR,
            SMALL_FONT_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            hip_text,
            (TEXT_MARGIN_X, TEXT_MARGIN_Y + (3 * TEXT_LINE_SPACING)),
            cv2.FONT_HERSHEY_SIMPLEX,
            SMALL_FONT_SCALE,
            DEBUG_TEXT_COLOR,
            SMALL_FONT_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            confidence_text,
            (TEXT_MARGIN_X, TEXT_MARGIN_Y + (4 * TEXT_LINE_SPACING)),
            cv2.FONT_HERSHEY_SIMPLEX,
            SMALL_FONT_SCALE,
            DEBUG_TEXT_COLOR,
            SMALL_FONT_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            lean_text,
            (TEXT_MARGIN_X, TEXT_MARGIN_Y + (5 * TEXT_LINE_SPACING)),
            cv2.FONT_HERSHEY_SIMPLEX,
            SMALL_FONT_SCALE,
            DEBUG_TEXT_COLOR,
            SMALL_FONT_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    audio_text = f"Audio: {audio_layer_label}"

    candidate_rows = [
        (TEXT_MARGIN_Y, state_text_width),
        (TEXT_MARGIN_Y + TEXT_LINE_SPACING, status_text_width),
        (header_height - 10, 0),
    ]

    placed_audio_text = audio_text
    audio_x = TEXT_MARGIN_X
    audio_y = candidate_rows[-1][0]
    for row_y, left_text_width in candidate_rows:
        available_width = max(40, frame_width - (left_text_width + (3 * TEXT_MARGIN_X)))
        candidate_text = _truncate_text_to_width(
            text=audio_text,
            max_width=available_width,
            scale=SMALL_FONT_SCALE,
            thickness=SMALL_FONT_THICKNESS,
        )
        (candidate_width, _), _ = cv2.getTextSize(
            candidate_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            SMALL_FONT_SCALE,
            SMALL_FONT_THICKNESS,
        )
        candidate_x = max(TEXT_MARGIN_X, frame_width - candidate_width - TEXT_MARGIN_X)
        left_edge = left_text_width + (2 * TEXT_MARGIN_X)
        if candidate_x >= left_edge:
            placed_audio_text = candidate_text
            audio_x = candidate_x
            audio_y = row_y
            break

    cv2.putText(
        frame_bgr,
        placed_audio_text,
        (audio_x, audio_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        SMALL_FONT_SCALE,
        AUDIO_LAYER_TEXT_COLOR,
        SMALL_FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    if posture_warning:
        cv2.putText(
            frame_bgr,
            "Posture warning: reduce forward/backward lean",
            (TEXT_MARGIN_X, frame_height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            SMALL_FONT_SCALE,
            WARNING_COLOR,
            FONT_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    if state_label.upper() == "SHIFTED_LEFT":
        _draw_horizontal_arrow(
            frame_bgr=frame_bgr,
            start=(int(frame_width * 0.28), frame_height // 2),
            end=(int(frame_width * 0.72), frame_height // 2),
            label="Move right",
        )
    elif state_label.upper() == "SHIFTED_RIGHT":
        _draw_horizontal_arrow(
            frame_bgr=frame_bgr,
            start=(int(frame_width * 0.72), frame_height // 2),
            end=(int(frame_width * 0.28), frame_height // 2),
            label="Move left",
        )

    if countdown_remaining is not None and countdown_remaining > 0:
        countdown_text = f"Starting in {int(countdown_remaining)}"
        _draw_center_text(
            frame_bgr=frame_bgr,
            text=countdown_text,
            y=int(frame_height * 0.45),
            color=COUNTDOWN_COLOR,
            scale=1.1,
            thickness=2,
        )

def _draw_horizontal_arrow(frame_bgr, start: Tuple[int, int], end: Tuple[int, int], label: str) -> None:
    cv2.arrowedLine(frame_bgr, start, end, ARROW_COLOR, 4, tipLength=0.08)
    label_x = min(start[0], end[0]) + 12
    label_y = start[1] - 12
    cv2.putText(
        frame_bgr,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        ARROW_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )


def _draw_center_text(
    frame_bgr,
    text: str,
    y: int,
    color: Tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    frame_width = frame_bgr.shape[1]
    (text_width, _), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    x = max(12, (frame_width - text_width) // 2)
    cv2.putText(
        frame_bgr,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _truncate_text_to_width(text: str, max_width: int, scale: float, thickness: int) -> str:
    """Trim text with ellipsis so it fits within max_width in pixels."""
    if max_width <= 0:
        return "..."

    (full_width, _), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    if full_width <= max_width:
        return text

    ellipsis = "..."
    (ellipsis_width, _), _ = cv2.getTextSize(
        ellipsis,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    if ellipsis_width >= max_width:
        return ellipsis

    trimmed = text
    while trimmed:
        trimmed = trimmed[:-1]
        candidate = f"{trimmed}{ellipsis}"
        (candidate_width, _), _ = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )
        if candidate_width <= max_width:
            return candidate

    return ellipsis
