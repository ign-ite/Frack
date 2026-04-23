"""Entry point for real-time body framing guidance."""

import argparse
import platform
import queue
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

from audio_engine import AudioEngine
from config import (
    AUTO_ASSESS_BAD_HOLD_SECONDS,
    AUTO_ASSESS_GOOD_HOLD_SECONDS,
    CALIBRATE_KEY,
    CAMERA_RECONNECT_ATTEMPTS,
    CAMERA_RECONNECT_WAIT_SECONDS,
    CAMERA_SCAN_MAX_INDEX,
    CAMERA_WARMUP_FRAMES,
    DEBUG_TOGGLE_KEY,
    EXIT_KEY,
    FPS_LOG_INTERVAL,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LEAN_ANGLE_WARNING_DEGREES,
    MAX_CONSECUTIVE_READ_FAILURES,
    MUTE_KEY,
    NO_CAMERA_AUDIO_INTERVAL_SECONDS,
    NO_CAMERA_CHOOSE_KEY,
    NO_CAMERA_ERROR_COLOR,
    NO_CAMERA_HINT_COLOR,
    NO_CAMERA_LIST_KEY,
    NO_CAMERA_QUIT_KEY,
    NO_CAMERA_RETRY_KEY,
    NO_CAMERA_TEXT_COLOR,
    NO_CAMERA_TEXT_LINE_SPACING,
    NO_CAMERA_TEXT_START_Y,
    NO_CAMERA_TITLE_TEXT,
    POSTURE_WARNING_COOLDOWN_SECONDS,
    SCREENSHOT_FOLDER,
    SCREENSHOT_KEY,
    SMALL_FONT_SCALE,
    START_ASSESS_KEY,
    STARTUP_COUNTDOWN_SECONDS,
    STOP_ASSESS_KEY,
    WAIT_KEY_DELAY_MS,
    WEB_PANEL_ENABLED_DEFAULT,
    WEB_PANEL_HOST,
    WEB_PANEL_PORT,
    WEBCAM_INDEX,
    WINDOW_TITLE,
)
from debounce_controller import DebounceController
from framing_logic import FramingLogic, FramingState, state_to_instruction_key
from gesture_controller import GestureController
from pose_detector import PoseDetector
from remote_control import RemoteControlServer
from session_logger import SessionLogger
from utils import draw_guidance_overlay, draw_pose_skeleton


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for live webcam runtime.

    Returns:
        Parsed argument namespace containing camera index and frame size options.
    """
    parser = argparse.ArgumentParser(
        description="Real-time body framing guidance with webcam pose estimation."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help=(
            "OpenCV camera index. If omitted, available cameras are auto-detected "
            "and you can choose when multiple are found."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List detected camera indices and exit.",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=CAMERA_SCAN_MAX_INDEX,
        help="Highest camera index to scan when listing/detecting devices.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=FRAME_WIDTH,
        help="Requested capture width in pixels",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=FRAME_HEIGHT,
        help="Requested capture height in pixels",
    )
    parser.add_argument(
        "--no-web-panel",
        action="store_true",
        help="Disable the local web control panel.",
    )
    parser.add_argument(
        "--web-panel-host",
        type=str,
        default=WEB_PANEL_HOST,
        help="Host for web control panel.",
    )
    parser.add_argument(
        "--web-panel-port",
        type=int,
        default=WEB_PANEL_PORT,
        help="Port for web control panel.",
    )
    return parser.parse_args()


def initialize_camera(
    index: int,
    frame_width: int,
    frame_height: int,
) -> Optional[cv2.VideoCapture]:
    """Create and configure webcam capture.

    Args:
        index: Webcam index passed to OpenCV VideoCapture.
        frame_width: Requested frame width for capture.
        frame_height: Requested frame height for capture.

    Returns:
        Initialized cv2.VideoCapture object or None if the camera fails to open.
    """
    camera = _create_video_capture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not camera.isOpened():
        print(f"[Main] Unable to open webcam at index {index}")
        camera.release()
        return None

    warmup_camera(camera, CAMERA_WARMUP_FRAMES)
    return camera


def discover_available_cameras(
    max_camera_index: int,
    frame_width: int,
    frame_height: int,
) -> List[int]:
    """Probe camera indices and return those that provide readable frames.

    Args:
        max_camera_index: Highest index to probe, inclusive.
        frame_width: Requested frame width for probe reads.
        frame_height: Requested frame height for probe reads.

    Returns:
        List of camera indices that opened and produced at least one frame.
    """
    available_indices: List[int] = []
    for index in range(0, max_camera_index + 1):
        probe = _create_video_capture(index)
        if not probe.isOpened():
            probe.release()
            continue

        for _ in range(3):
            success, frame = probe.read()
            if success and frame is not None:
                available_indices.append(index)
                break

        probe.release()

    return available_indices


def _create_video_capture(index: int) -> cv2.VideoCapture:
    """Create cross-platform camera capture with backend preferences."""
    system_name = platform.system().lower()
    if "windows" in system_name:
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if "darwin" in system_name:
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def print_available_cameras(available_indices: List[int]) -> None:
    """Print camera discovery results to the console.

    Args:
        available_indices: List of detected camera indices.

    Returns:
        None.
    """
    if not available_indices:
        print("[Main] No cameras detected.")
        return

    camera_names = discover_camera_names(max(available_indices))
    print("[Main] Detected camera devices:")
    for index in available_indices:
        device_name = camera_names.get(index, "Unknown device")
        print(f"  - {index}: {device_name}")


def discover_camera_names(max_camera_index: int) -> Dict[int, str]:
    """Best-effort camera name discovery keyed by OpenCV index.

    This is intentionally fail-soft: if backend APIs are unavailable, callers
    still receive stable index-based behavior.
    """
    names: Dict[int, str] = {}

    try:
        from cv2_enumerate_cameras import enumerate_cameras  # type: ignore

        for camera_info in enumerate_cameras():
            raw_index = getattr(camera_info, "index", None)
            if not isinstance(raw_index, int) or raw_index < 0:
                continue

            device_name = str(getattr(camera_info, "name", "")).strip()
            if not device_name:
                device_name = str(getattr(camera_info, "path", "")).strip()

            if not device_name:
                continue

            for candidate_index in _candidate_base_indices(raw_index):
                if 0 <= candidate_index <= max_camera_index and candidate_index not in names:
                    names[candidate_index] = device_name
    except Exception:
        # Optional dependency: ignore and try platform-specific fallback below.
        pass

    if names:
        return names

    if platform.system().lower().startswith("windows"):
        try:
            from pygrabber.dshow_graph import FilterGraph  # type: ignore

            graph = FilterGraph()
            for index, device_name in enumerate(graph.get_input_devices()):
                if index > max_camera_index:
                    break
                cleaned_name = str(device_name).strip()
                if cleaned_name:
                    names[index] = cleaned_name
        except Exception:
            pass

    return names


def _candidate_base_indices(raw_index: int) -> List[int]:
    """Return likely OpenCV base indices for backend-offset camera indices."""
    candidates = [raw_index]

    # Common backend offsets for OpenCV camera APIs.
    for offset in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_AVFOUNDATION):
        if raw_index >= int(offset):
            candidates.append(raw_index - int(offset))

    # Legacy compact normalization fallback.
    candidates.append(raw_index % 100)

    unique_candidates: List[int] = []
    for index in candidates:
        if index not in unique_candidates:
            unique_candidates.append(index)
    return unique_candidates


def prompt_for_camera_choice(available_indices: List[int]) -> Optional[int]:
    """Prompt user to choose one camera index from detected devices.

    Args:
        available_indices: Camera indices that are currently available.

    Returns:
        Chosen camera index, or None if the user cancels selection.
    """
    if not available_indices:
        print("[Main] Camera choice requested, but no cameras are currently detected.")
        return None

    print_available_cameras(available_indices)
    # No default; require user to type a numeric index or quit.
    prompt = "Select camera index (enter number, or 'q' to cancel): "


    while True:
        try:
            raw_value = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("[Main] Camera selection cancelled by user input stream.")
            return None

        if raw_value == "":
            print("[Main] You must enter a valid camera index, or 'q' to quit.")
            continue

        lowered = raw_value.lower()
        if lowered in {"q", "quit", "exit"}:
            return None

        try:
            selected_index = int(raw_value)
        except ValueError:
            print("[Main] Invalid input. Enter a numeric camera index or 'q'.")
            continue

        if selected_index in available_indices:
            return selected_index

        print("[Main] Selected index is not currently available. Try again.")


def resolve_camera_index(
    requested_index: Optional[int],
    available_indices: List[int],
    prompt_on_multiple: bool,
) -> Optional[int]:
    """Resolve the camera index to open based on request and availability.

    Args:
        requested_index: Explicitly requested camera index from CLI, if any.
        available_indices: Camera indices currently detected.
        prompt_on_multiple: Whether to prompt user when multiple cameras exist.

    Returns:
        Resolved camera index, or None if no suitable selection exists.
    """
    if requested_index is not None:
        if requested_index in available_indices:
            return requested_index

        print(
            "[Main] "
            f"Requested camera index {requested_index} is not currently available."
        )
        return None

    if not available_indices:
        return None

    if len(available_indices) == 1:
        return available_indices[0]

    if prompt_on_multiple:
        selected_index = prompt_for_camera_choice(available_indices)
        if selected_index is not None:
            return selected_index

        fallback_index = available_indices[0]
        print(
            "[Main] "
            "Camera selection cancelled. "
            f"Falling back to first detected camera index {fallback_index}."
        )
        return fallback_index

    # Non-interactive fallback: choose the first stable index.
    return available_indices[0]


def warmup_camera(camera: cv2.VideoCapture, warmup_frames: int) -> None:
    """Read a few startup frames to stabilize auto-exposure and focus.

    Args:
        camera: Open OpenCV capture handle.
        warmup_frames: Number of frames to read and discard.

    Returns:
        None.
    """
    for _ in range(warmup_frames):
        camera.read()


def build_no_camera_frame(
    frame_width: int,
    frame_height: int,
    preferred_camera_index: Optional[int],
    available_indices: List[int],
    camera_names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Create a static black frame with missing-camera status instructions.

    Args:
        frame_width: Output frame width in pixels.
        frame_height: Output frame height in pixels.
        preferred_camera_index: Preferred camera index to reconnect, if any.
        available_indices: Currently detected camera indices.

    Returns:
        OpenCV BGR image containing camera-missing status text.
    """
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    preferred_text = (
        str(preferred_camera_index)
        if preferred_camera_index is not None
        else f"auto ({WEBCAM_INDEX} fallback)"
    )
    if camera_names is None:
        camera_names = discover_camera_names(max(available_indices) if available_indices else 0)

    detected_parts: List[str] = []
    for index in available_indices:
        device_name = camera_names.get(index, "Unknown")
        detected_parts.append(f"{index}:{device_name}")
    detected_text = ", ".join(detected_parts) if detected_parts else "none"

    lines = [
        (NO_CAMERA_TITLE_TEXT, NO_CAMERA_ERROR_COLOR),
        (f"Preferred camera: {preferred_text}", NO_CAMERA_TEXT_COLOR),
        (f"Detected cameras: {detected_text}", NO_CAMERA_TEXT_COLOR),
        (
            f"Press '{NO_CAMERA_RETRY_KEY.upper()}' to rescan and reconnect",
            NO_CAMERA_HINT_COLOR,
        ),
        (
            f"Press '{NO_CAMERA_CHOOSE_KEY.upper()}' to choose camera",
            NO_CAMERA_HINT_COLOR,
        ),
        (
            f"Press '{NO_CAMERA_LIST_KEY.upper()}' to print camera list",
            NO_CAMERA_HINT_COLOR,
        ),
        (
            f"Press '{NO_CAMERA_QUIT_KEY.upper()}' to quit",
            NO_CAMERA_HINT_COLOR,
        ),
    ]

    y_pos = NO_CAMERA_TEXT_START_Y
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (30, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            SMALL_FONT_SCALE,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
        y_pos += NO_CAMERA_TEXT_LINE_SPACING

    return frame


def run_no_camera_mode(
    audio_engine: AudioEngine,
    frame_width: int,
    frame_height: int,
    max_camera_index: int,
    preferred_camera_index: Optional[int],
) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
    """Run a black-screen recovery mode until a camera becomes available.

    Args:
        audio_engine: Audio engine used for missing-camera voice prompt.
        frame_width: Capture width for camera probing and reconnect.
        frame_height: Capture height for camera probing and reconnect.
        max_camera_index: Highest camera index to probe.
        preferred_camera_index: Camera index to prioritize for reconnection.

    Returns:
        Tuple of (camera_handle, selected_index). Both None when user exits.
    """
    available_indices = discover_available_cameras(
        max_camera_index=max_camera_index,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    camera_names = discover_camera_names(max_camera_index)
    last_scan_time = 0.0
    last_audio_time = 0.0

    print("[Main] Entering no-camera recovery mode.")
    while True:
        now = time.time()

        if now - last_audio_time >= NO_CAMERA_AUDIO_INTERVAL_SECONDS:
            audio_engine.speak("missing_camera")
            last_audio_time = now

        # Periodic scan allows hot-plug camera recovery without restarting the app.
        if now - last_scan_time >= CAMERA_RECONNECT_WAIT_SECONDS:
            available_indices = discover_available_cameras(
                max_camera_index=max_camera_index,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            camera_names = discover_camera_names(max_camera_index)
            last_scan_time = now

            if (
                preferred_camera_index is not None
                and preferred_camera_index in available_indices
            ):
                camera = initialize_camera(
                    index=preferred_camera_index,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                if camera is not None:
                    print(
                        "[Main] "
                        f"Recovered preferred camera at index {preferred_camera_index}."
                    )
                    return camera, preferred_camera_index

            if preferred_camera_index is None and len(available_indices) == 1:
                auto_index = available_indices[0]
                camera = initialize_camera(
                    index=auto_index,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                if camera is not None:
                    print(f"[Main] Auto-connected camera index {auto_index}.")
                    return camera, auto_index

        frame = build_no_camera_frame(
            frame_width=frame_width,
            frame_height=frame_height,
            preferred_camera_index=preferred_camera_index,
            available_indices=available_indices,
            camera_names=camera_names,
        )
        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
        if key in {ord(NO_CAMERA_QUIT_KEY), ord(EXIT_KEY)}:
            print("[Main] User exited from no-camera recovery mode.")
            return None, None

        if key == ord(NO_CAMERA_LIST_KEY):
            print_available_cameras(available_indices)
            continue

        if key == ord(NO_CAMERA_RETRY_KEY):
            print("[Main] Manual camera rescan requested.")
            available_indices = discover_available_cameras(
                max_camera_index=max_camera_index,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            camera_names = discover_camera_names(max_camera_index)
            target_index = resolve_camera_index(
                requested_index=preferred_camera_index,
                available_indices=available_indices,
                prompt_on_multiple=False,
            )
            if target_index is None:
                continue

            camera = initialize_camera(
                index=target_index,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if camera is not None:
                return camera, target_index
            continue

        if key == ord(NO_CAMERA_CHOOSE_KEY):
            available_indices = discover_available_cameras(
                max_camera_index=max_camera_index,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            camera_names = discover_camera_names(max_camera_index)
            selected_index = prompt_for_camera_choice(available_indices)
            if selected_index is None:
                continue

            camera = initialize_camera(
                index=selected_index,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if camera is not None:
                print(f"[Main] Connected selected camera index {selected_index}.")
                return camera, selected_index


def attempt_camera_reconnect(
    index: int,
    frame_width: int,
    frame_height: int,
) -> Optional[cv2.VideoCapture]:
    """Attempt to reconnect the camera after repeated frame read failures.

    Args:
        index: Webcam index passed to OpenCV VideoCapture.
        frame_width: Requested frame width for capture.
        frame_height: Requested frame height for capture.

    Returns:
        Reconnected cv2.VideoCapture object, or None if all attempts fail.
    """
    for attempt in range(1, CAMERA_RECONNECT_ATTEMPTS + 1):
        print(
            "[Main] "
            f"Attempting camera reconnect ({attempt}/{CAMERA_RECONNECT_ATTEMPTS})..."
        )
        camera = initialize_camera(
            index=index,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if camera is not None:
            print("[Main] Camera reconnect successful")
            return camera

        # Small backoff avoids tight reconnect loops if camera is unavailable.
        time.sleep(CAMERA_RECONNECT_WAIT_SECONDS)

    return None


def log_fps(frame_count: int, last_tick: float) -> float:
    """Log FPS every configured frame interval.

    Args:
        frame_count: Number of frames processed so far.
        last_tick: Timestamp from the previous FPS log checkpoint.

    Returns:
        Updated timestamp to use for the next FPS interval.
    """
    if frame_count % FPS_LOG_INTERVAL != 0:
        return last_tick

    now = time.time()
    elapsed = max(1e-6, now - last_tick)
    fps = FPS_LOG_INTERVAL / elapsed
    print(f"[Main] FPS: {fps:.2f}")
    return now


def main() -> None:
    """Run live webcam framing guidance until user exits.

    Returns:
        None. The function owns the realtime loop and cleanup sequence.
    """
    args = parse_arguments()

    available_cameras = discover_available_cameras(
        max_camera_index=args.max_camera_index,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )

    if args.list_cameras:
        print_available_cameras(available_cameras)
        return

    audio_engine = AudioEngine()

    selected_camera_index = resolve_camera_index(
        requested_index=args.camera_index,
        available_indices=available_cameras,
        prompt_on_multiple=True,
    )

    camera = None
    if selected_camera_index is not None:
        camera = initialize_camera(
            index=selected_camera_index,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
        )

    if camera is None:
        camera, selected_camera_index = run_no_camera_mode(
            audio_engine=audio_engine,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            max_camera_index=args.max_camera_index,
            preferred_camera_index=args.camera_index,
        )
        if camera is None:
            audio_engine.shutdown()
            cv2.destroyAllWindows()
            return

    pose_detector = PoseDetector()
    framing_logic = FramingLogic()
    debounce_controller = DebounceController()
    gesture_controller = GestureController()
    session_logger = SessionLogger()

    command_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    remote_server = None
    if WEB_PANEL_ENABLED_DEFAULT and not args.no_web_panel:
        remote_server = RemoteControlServer(
            command_queue=command_queue,
            host=args.web_panel_host,
            port=args.web_panel_port,
        )
        if remote_server.start():
            print(f"[Main] Web control panel active at {remote_server.url}")
        else:
            print(f"[Main] Web panel disabled: {remote_server.startup_error}")
            remote_server = None

    frame_count = 0
    fps_tick = time.time()
    fps_value: Optional[float] = None
    consecutive_read_failures = 0
    is_muted = False
    debug_enabled = True
    assessment_mode = False
    startup_time = time.time()
    screenshot_dir = Path(__file__).resolve().parent / SCREENSHOT_FOLDER
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    last_stable_state = None
    good_hold_started_at: Optional[float] = None
    bad_hold_started_at: Optional[float] = None
    last_posture_warning_at = 0.0

    print("[Main] Starting real-time framing guidance. Press 'q' to exit.")

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures == 1:
                    print("[Main] Frame capture failed; monitoring for recovery")

                if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    print(
                        "[Main] "
                        f"Frame capture failed for {consecutive_read_failures} consecutive "
                        "reads. Reconnecting camera..."
                    )
                    camera.release()

                    reconnect_index = (
                        selected_camera_index
                        if selected_camera_index is not None
                        else WEBCAM_INDEX
                    )
                    reconnected_camera = attempt_camera_reconnect(
                        index=reconnect_index,
                        frame_width=args.frame_width,
                        frame_height=args.frame_height,
                    )
                    if reconnected_camera is None:
                        print("[Main] Camera reconnect failed. Switching to recovery mode.")
                        camera, selected_camera_index = run_no_camera_mode(
                            audio_engine=audio_engine,
                            frame_width=args.frame_width,
                            frame_height=args.frame_height,
                            max_camera_index=args.max_camera_index,
                            preferred_camera_index=selected_camera_index,
                        )
                        if camera is None:
                            print("[Main] No camera recovered. Exiting application.")
                            break

                        debounce_controller.reset()
                        consecutive_read_failures = 0
                        continue

                    camera = reconnected_camera
                    consecutive_read_failures = 0

                key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
                if key == ord(EXIT_KEY):
                    print("[Main] Exit key detected while camera was unavailable.")
                    break
                continue

            consecutive_read_failures = 0

            now = time.time()
            countdown_remaining = max(0.0, STARTUP_COUNTDOWN_SECONDS - (now - startup_time))
            if countdown_remaining > 0:
                draw_guidance_overlay(
                    frame_bgr=frame,
                    state_label="WARMUP",
                    body_span_ratio=None,
                    mid_hip_x=None,
                    audio_layer_label=audio_engine.active_layer_label,
                    orientation_label="UNKNOWN",
                    shoulder_confidence=None,
                    ankle_confidence=None,
                    lean_angle_deg=None,
                    fps_value=fps_value,
                    is_muted=is_muted,
                    assessment_mode=False,
                    debug_enabled=debug_enabled,
                    hip_left_threshold=framing_logic.hip_thresholds[0],
                    hip_right_threshold=framing_logic.hip_thresholds[1],
                    countdown_remaining=countdown_remaining,
                    good_hold_remaining=None,
                    posture_warning=False,
                )
                cv2.imshow(WINDOW_TITLE, frame)

                frame_count += 1
                if frame_count % FPS_LOG_INTERVAL == 0:
                    now_fps = time.time()
                    elapsed = max(1e-6, now_fps - fps_tick)
                    fps_value = FPS_LOG_INTERVAL / elapsed
                    print(f"[Main] FPS: {fps_value:.2f}")
                    fps_tick = now_fps

                key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
                if key == ord(EXIT_KEY):
                    print("[Main] Exit key detected. Shutting down.")
                    break
                if key == ord(SCREENSHOT_KEY):
                    screenshot_path = _save_screenshot(frame, screenshot_dir, source="keyboard")
                    session_logger.log_action("screenshot", source="keyboard", details=screenshot_path)
                if key == ord(MUTE_KEY):
                    is_muted = not is_muted
                    session_logger.log_action("toggle_mute", source="keyboard", details=str(is_muted))
                if key == ord(DEBUG_TOGGLE_KEY):
                    debug_enabled = not debug_enabled
                    session_logger.log_action("toggle_debug", source="keyboard", details=str(debug_enabled))
                continue

            detection = pose_detector.detect(frame)
            analysis = framing_logic.classify(
                landmarks=detection.landmarks,
                frame_height=frame.shape[0],
                critical_joints_confident=detection.critical_joints_confident,
            )

            if last_stable_state != analysis.state:
                session_logger.log_state_change(
                    state=analysis.state.value,
                    orientation=analysis.orientation_label,
                    body_span_ratio=analysis.body_span_ratio,
                    mid_hip_x=analysis.mid_hip_x,
                    lean_angle_deg=analysis.lean_angle_deg,
                    assessment_mode=assessment_mode,
                )
                last_stable_state = analysis.state

            if detection.landmarks is not None:
                gesture_actions = gesture_controller.update(detection.landmarks, now=now)
                for action in gesture_actions:
                    command_queue.put(("gesture", action))
            else:
                gesture_controller.update(None, now=now)

            decision = debounce_controller.update(analysis.state)
            stable_state = (
                decision.stable_state if decision.stable_state is not None else analysis.state
            )

            if stable_state == analysis.state == FramingState.GOOD:
                if good_hold_started_at is None:
                    good_hold_started_at = now
                if not assessment_mode and (now - good_hold_started_at) >= AUTO_ASSESS_GOOD_HOLD_SECONDS:
                    assessment_mode = True
                    session_logger.log_action("assessment_start", source="auto")
                    _save_screenshot(frame, screenshot_dir, source="auto_start")
                    if not is_muted:
                        audio_engine.speak("good")
            else:
                good_hold_started_at = None

            if assessment_mode and stable_state != FramingState.GOOD:
                if bad_hold_started_at is None:
                    bad_hold_started_at = now
                if (now - bad_hold_started_at) >= AUTO_ASSESS_BAD_HOLD_SECONDS:
                    assessment_mode = False
                    session_logger.log_action("assessment_stop", source="auto")
            else:
                bad_hold_started_at = None

            posture_warning = (
                analysis.lean_angle_deg is not None
                and analysis.lean_angle_deg >= LEAN_ANGLE_WARNING_DEGREES
            )
            if (
                posture_warning
                and not is_muted
                and now - last_posture_warning_at >= POSTURE_WARNING_COOLDOWN_SECONDS
            ):
                audio_engine.speak("hold")
                last_posture_warning_at = now

            if decision.should_speak and decision.state_to_speak is not None and not is_muted:
                instruction_key = state_to_instruction_key(decision.state_to_speak)
                audio_engine.speak(instruction_key)

            while True:
                try:
                    source, action = command_queue.get_nowait()
                except queue.Empty:
                    break

                if action in {"toggle_mute", "mute"}:
                    is_muted = not is_muted
                    session_logger.log_action("toggle_mute", source=source, details=str(is_muted))
                elif action == "screenshot":
                    screenshot_path = _save_screenshot(frame, screenshot_dir, source=source)
                    session_logger.log_action("screenshot", source=source, details=screenshot_path)
                elif action == "calibrate":
                    if framing_logic.calibrate(detection.landmarks, frame.shape[0]):
                        session_logger.log_action("calibrate", source=source, details="ok")
                    else:
                        session_logger.log_action("calibrate", source=source, details="failed")
                elif action == "start":
                    assessment_mode = True
                    session_logger.log_action("assessment_start", source=source)
                elif action == "stop":
                    assessment_mode = False
                    session_logger.log_action("assessment_stop", source=source)

            draw_pose_skeleton(frame, detection.pose_landmarks_proto)

            good_hold_remaining = None
            if not assessment_mode:
                if good_hold_started_at is None:
                    good_hold_remaining = AUTO_ASSESS_GOOD_HOLD_SECONDS
                else:
                    elapsed_good_hold = now - good_hold_started_at
                    good_hold_remaining = max(0.0, AUTO_ASSESS_GOOD_HOLD_SECONDS - elapsed_good_hold)

            draw_guidance_overlay(
                frame_bgr=frame,
                state_label=stable_state.value,
                body_span_ratio=analysis.body_span_ratio,
                mid_hip_x=analysis.mid_hip_x,
                audio_layer_label=audio_engine.active_layer_label,
                orientation_label=analysis.orientation_label,
                shoulder_confidence=analysis.shoulder_confidence,
                ankle_confidence=analysis.ankle_confidence,
                lean_angle_deg=analysis.lean_angle_deg,
                fps_value=fps_value,
                is_muted=is_muted,
                assessment_mode=assessment_mode,
                debug_enabled=debug_enabled,
                hip_left_threshold=framing_logic.hip_thresholds[0],
                hip_right_threshold=framing_logic.hip_thresholds[1],
                countdown_remaining=countdown_remaining if countdown_remaining > 0 else None,
                good_hold_remaining=good_hold_remaining,
                posture_warning=posture_warning,
            )

            cv2.imshow(WINDOW_TITLE, frame)

            frame_count += 1
            if frame_count % FPS_LOG_INTERVAL == 0:
                now_fps = time.time()
                elapsed = max(1e-6, now_fps - fps_tick)
                fps_value = FPS_LOG_INTERVAL / elapsed
                print(f"[Main] FPS: {fps_value:.2f}")
                fps_tick = now_fps

            key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
            if key == ord(EXIT_KEY):
                print("[Main] Exit key detected. Shutting down.")
                break
            if key == ord(SCREENSHOT_KEY):
                screenshot_path = _save_screenshot(frame, screenshot_dir, source="keyboard")
                session_logger.log_action("screenshot", source="keyboard", details=screenshot_path)
            if key == ord(MUTE_KEY):
                is_muted = not is_muted
                session_logger.log_action("toggle_mute", source="keyboard", details=str(is_muted))
            if key == ord(DEBUG_TOGGLE_KEY):
                debug_enabled = not debug_enabled
                session_logger.log_action("toggle_debug", source="keyboard", details=str(debug_enabled))
            if key == ord(CALIBRATE_KEY):
                if framing_logic.calibrate(detection.landmarks, frame.shape[0]):
                    session_logger.log_action("calibrate", source="keyboard", details="ok")
                else:
                    session_logger.log_action("calibrate", source="keyboard", details="failed")
            if key == ord(START_ASSESS_KEY):
                assessment_mode = True
                session_logger.log_action("assessment_start", source="keyboard")
            if key == ord(STOP_ASSESS_KEY):
                assessment_mode = False
                session_logger.log_action("assessment_stop", source="keyboard")

    except KeyboardInterrupt:
        # Ctrl+C should always trigger a clean shutdown path.
        print("\n[Main] Keyboard interrupt received. Shutting down.")
    finally:
        if remote_server is not None:
            remote_server.stop()
        audio_engine.shutdown()
        pose_detector.close()
        session_logger.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("[Main] Cleanup complete.")


def _save_screenshot(frame, screenshot_dir: Path, source: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = screenshot_dir / f"capture_{source}_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


if __name__ == "__main__":
    main()
