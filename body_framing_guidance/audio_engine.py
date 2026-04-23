"""Layered audio output engine with internet and offline fallbacks."""

import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional

from config import (
    AUDIO_CACHE_FOLDER,
    AUDIO_CLIPS_FOLDER,
    AUDIO_TTS_RATE,
    INSTRUCTION_PHRASES,
    LAYER2_REQUIRED_FILES,
    TEMP_AUDIO_PREFIX,
)


class AudioEngine:
    """Non-blocking speech engine with three prioritized layers.

    Layer 1 (primary): gTTS + pygame using startup-generated MP3 files.
    Layer 2 (fallback): pre-recorded MP3 files in ./audio_clips played by pygame.
    Layer 3 (final): pyttsx3 offline TTS in a daemon thread.
    """

    def __init__(self) -> None:
        """Initialize audio layer stack and select the best available option."""
        self._project_root = Path(__file__).resolve().parent
        self._playback_lock = threading.Lock()

        self._active_layer = "none"
        self._active_layer_label = "None"

        self._pygame_module = None
        self._pygame_ready = False
        self._tts_engine = None
        self._audio_files: Dict[str, Path] = {}
        self._temp_audio_dir: Optional[Path] = None
        self._audio_cache_dir = self._project_root / AUDIO_CACHE_FOLDER

        self._ensure_audio_cache_dir()
        self._initialize_layers()

    @property
    def active_layer_label(self) -> str:
        """Get a human-readable label of the currently active audio layer.

        Returns:
            Current audio layer label for overlay/debug display.
        """
        return self._active_layer_label

    def speak(self, instruction_key: str) -> None:
        """Queue a non-blocking speech/playback request.

        Args:
            instruction_key: Instruction key, e.g. "too_close" or "move_left".

        Notes:
            The work is dispatched to a daemon thread so camera capture never blocks.
        """
        if instruction_key not in INSTRUCTION_PHRASES:
            print(f"[AudioEngine] Unknown instruction key: {instruction_key}")
            return

        worker = threading.Thread(
            target=self._speak_worker,
            args=(instruction_key,),
            daemon=True,
        )
        worker.start()

    def shutdown(self) -> None:
        """Release audio resources used by pygame and pyttsx3, if initialized."""
        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
            except Exception:
                pass

        if self._pygame_ready and self._pygame_module is not None:
            try:
                self._pygame_module.mixer.quit()
            except Exception:
                pass

    def _initialize_layers(self) -> None:
        """Try each audio layer in required priority order."""
        self._initialize_pygame_if_available()

        layer1_error = self._try_initialize_layer1()
        if layer1_error is None:
            return
        print(
            "[AudioEngine] "
            f"Audio Layer 1 failed: {layer1_error}. "
            "Falling back to Layer 2 (pre-recorded clips)."
        )

        layer2_error = self._try_initialize_layer2()
        if layer2_error is None:
            return
        print(
            "[AudioEngine] "
            f"Audio Layer 2 failed: {layer2_error}. "
            "Falling back to Layer 3 (pyttsx3)."
        )

        layer3_error = self._try_initialize_layer3()
        if layer3_error is None:
            return

        self._active_layer = "none"
        self._active_layer_label = "None (audio disabled)"
        print(
            "[AudioEngine] "
            f"Audio Layer 3 failed: {layer3_error}. Audio output disabled."
        )

    def _initialize_pygame_if_available(self) -> None:
        """Initialize pygame mixer when installed and usable.

        Notes:
            Layer 1 and Layer 2 depend on pygame for MP3 playback.
        """
        try:
            import pygame  # type: ignore
        except Exception as exc:
            print(
                "[AudioEngine] "
                f"pygame import failed ({exc}). Layer 1 and Layer 2 may be unavailable."
            )
            return

        self._pygame_module = pygame
        try:
            pygame.mixer.init()
            self._pygame_ready = True
        except Exception as exc:
            print(
                "[AudioEngine] "
                f"pygame.mixer initialization failed ({exc}). "
                "Layer 1 and Layer 2 may be unavailable."
            )
            self._pygame_ready = False

    def _ensure_audio_cache_dir(self) -> None:
        """Create a persistent audio cache directory for generated voice clips."""
        try:
            self._audio_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[AudioEngine] Could not create audio cache directory: {exc}")

    def _try_initialize_layer1(self) -> Optional[str]:
        """Initialize Layer 1 by generating gTTS MP3 files and binding pygame.

        Returns:
            None on success, otherwise an error string describing the failure.
        """
        if not self._pygame_ready:
            return "pygame is not ready"

        try:
            from gtts import gTTS  # type: ignore
        except Exception as exc:
            return f"gTTS import failed ({exc})"

        try:
            self._temp_audio_dir = Path(tempfile.mkdtemp(prefix=TEMP_AUDIO_PREFIX))
            self._audio_files.clear()

            for instruction_key, text in INSTRUCTION_PHRASES.items():
                output_path = self._temp_audio_dir / f"{instruction_key}.mp3"
                cache_path = self._audio_cache_dir / f"{instruction_key}.mp3"

                # Generate the runtime playback file for immediate use.
                gTTS(text=text, lang="en").save(str(output_path))
                self._audio_files[instruction_key] = output_path

                try:
                    if not cache_path.exists():
                        gTTS(text=text, lang="en").save(str(cache_path))
                except Exception:
                    # Cache generation is best-effort; playback still works from temp files.
                    pass

            self._active_layer = "layer1"
            self._active_layer_label = "Layer 1 (gTTS+pygame)"
            print("[AudioEngine] Audio Layer 1 (gTTS+pygame) initialized successfully")
            return None
        except Exception as exc:
            self._audio_files.clear()
            return str(exc)

    def _try_initialize_layer2(self) -> Optional[str]:
        """Initialize Layer 2 from local cached or pre-recorded clips.
 
        Returns:
            None on success, otherwise an error string describing the failure.
        """
        if not self._pygame_ready:
            return "pygame is not ready"

        cache_dir = self._audio_cache_dir
        clip_dir = self._project_root / AUDIO_CLIPS_FOLDER

        source_dir = None
        for candidate_dir in (cache_dir, clip_dir):
            missing_files = [
                file_name
                for file_name in LAYER2_REQUIRED_FILES
                if not (candidate_dir / file_name).exists()
            ]
            if not missing_files:
                source_dir = candidate_dir
                break

        if source_dir is None:
            return (
                f"missing required clip files in either {cache_dir} or {clip_dir}. "
                "Add manual clips or run with internet to generate cached audio."
            )

        self._audio_files.clear()
        for instruction_key in INSTRUCTION_PHRASES:
            clip_path = source_dir / f"{instruction_key}.mp3"
            self._audio_files[instruction_key] = clip_path

        label = "audio cache" if source_dir == cache_dir else "pre-recorded clips"
        self._active_layer = "layer2"
        self._active_layer_label = f"Layer 2 ({label}+pygame)"
        print(
            "[AudioEngine] "
            f"Audio Layer 2 ({label} + pygame) initialized successfully"
        )
        return None

    def _try_initialize_layer3(self) -> Optional[str]:
        """Initialize Layer 3 with offline pyttsx3 speech synthesis.

        Returns:
            None on success, otherwise an error string describing the failure.
        """
        try:
            import pyttsx3  # type: ignore
        except Exception as exc:
            return f"pyttsx3 import failed ({exc})"

        try:
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", AUDIO_TTS_RATE)
            self._active_layer = "layer3"
            self._active_layer_label = "Layer 3 (pyttsx3)"
            print("[AudioEngine] Audio Layer 3 (pyttsx3) initialized successfully")
            return None
        except Exception as exc:
            self._tts_engine = None
            return str(exc)

    def _speak_worker(self, instruction_key: str) -> None:
        """Execute one speech request in a thread-safe worker.

        Args:
            instruction_key: Requested instruction key.
        """
        with self._playback_lock:
            if self._active_layer in {"layer1", "layer2"}:
                self._play_with_pygame(instruction_key)
                return

            if self._active_layer == "layer3":
                self._speak_with_pyttsx3(instruction_key)
                return

            # Fail-soft behavior if every layer failed.
            print(f"[AudioEngine] Audio disabled. Instruction: {INSTRUCTION_PHRASES[instruction_key]}")

    def _play_with_pygame(self, instruction_key: str) -> None:
        """Play an instruction clip through pygame without blocking the caller.

        Args:
            instruction_key: Instruction key to play.
        """
        if self._pygame_module is None:
            print("[AudioEngine] pygame module unavailable during playback")
            return

        clip_path = self._audio_files.get(instruction_key)
        if clip_path is None:
            print(
                "[AudioEngine] "
                f"No clip available for key '{instruction_key}' in {self._active_layer_label}"
            )
            return

        try:
            # Stop current playback to avoid layered overlapping prompts.
            self._pygame_module.mixer.music.stop()
            self._pygame_module.mixer.music.load(str(clip_path))
            self._pygame_module.mixer.music.play()
        except Exception as exc:
            print(f"[AudioEngine] pygame playback failed: {exc}")

    def _speak_with_pyttsx3(self, instruction_key: str) -> None:
        """Speak instruction text with pyttsx3.

        Args:
            instruction_key: Instruction key mapped to phrase text.

        Notes:
            pyttsx3 `runAndWait` is blocking, but this method is always called from
            a daemon worker thread so the camera loop stays responsive.
        """
        if self._tts_engine is None:
            print("[AudioEngine] pyttsx3 engine unavailable during speech")
            return

        phrase = INSTRUCTION_PHRASES[instruction_key]
        try:
            self._tts_engine.say(phrase)
            self._tts_engine.runAndWait()
        except Exception as exc:
            print(f"[AudioEngine] pyttsx3 playback failed: {exc}")
