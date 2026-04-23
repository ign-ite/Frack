"""Entry point for real-time body framing guidance."""

import argparse
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from audio_engine import AudioEngine
from config import (
    CAMERA_RECONNECT_ATTEMPTS,
    CAMERA_RECONNECT_WAIT_SECONDS,
    CAMERA_SCAN_MAX_INDEX,
    CAMERA_WARMUP_FRAMES,
    EXIT_KEY,
    FPS_LOG_INTERVAL,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_CONSECUTIVE_READ_FAILURES,
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
    SMALL_FONT_SCALE,
    WAIT_KEY_DELAY_MS,
    WEBCAM_INDEX,
    WINDOW_TITLE,
)
from debounce_controller import DebounceController
from framing_logic import FramingLogic, state_to_instruction_key
from pose_detector import PoseDetector
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
    camera = cv2.VideoCapture(index)
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
        probe = cv2.VideoCapture(index)
        if not probe.isOpened():
            probe.release()
            continue

        probe.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        probe.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        success, frame = probe.read()
        if success and frame is not None:
            available_indices.append(index)

        probe.release()

    return available_indices


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

    print("[Main] Detected camera indices:")
    for index in available_indices:
        print(f"  - {index}")


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
    default_index = available_indices[0]
    prompt = (
        f"Select camera index (Enter for {default_index}, 'q' to cancel): "
    )

    while True:
        try:
            raw_value = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("[Main] Camera selection cancelled by user input stream.")
            return None

        if raw_value == "":
            return default_index

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
        success, _ = camera.read()
        if not success:
            break


def build_no_camera_frame(
    frame_width: int,
    frame_height: int,
    preferred_camera_index: Optional[int],
    available_indices: List[int],
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
    detected_text = ", ".join(str(index) for index in available_indices) or "none"

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

    frame_count = 0
    fps_tick = time.time()
    consecutive_read_failures = 0

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

            detection = pose_detector.detect(frame)
            analysis = framing_logic.classify(
                landmarks=detection.landmarks,
                frame_height=frame.shape[0],
                critical_joints_confident=detection.critical_joints_confident,
            )

            decision = debounce_controller.update(analysis.state)
            stable_state = (
                decision.stable_state if decision.stable_state is not None else analysis.state
            )

            if decision.should_speak and decision.state_to_speak is not None:
                instruction_key = state_to_instruction_key(decision.state_to_speak)
                audio_engine.speak(instruction_key)

            draw_pose_skeleton(frame, detection.pose_landmarks_proto)
            draw_guidance_overlay(
                frame_bgr=frame,
                state_label=stable_state.value,
                body_span_ratio=analysis.body_span_ratio,
                mid_hip_x=analysis.mid_hip_x,
                audio_layer_label=audio_engine.active_layer_label,
            )

            cv2.imshow(WINDOW_TITLE, frame)

            frame_count += 1
            fps_tick = log_fps(frame_count, fps_tick)

            key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
            if key == ord(EXIT_KEY):
                print("[Main] Exit key detected. Shutting down.")
                break

    except KeyboardInterrupt:
        # Ctrl+C should always trigger a clean shutdown path.
        print("\n[Main] Keyboard interrupt received. Shutting down.")
    finally:
        audio_engine.shutdown()
        pose_detector.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("[Main] Cleanup complete.")


if __name__ == "__main__":
    main()
