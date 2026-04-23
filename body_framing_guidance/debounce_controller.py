"""Debouncing and throttling controller for real-time audio feedback."""

import time
from dataclasses import dataclass
from typing import Optional

from config import AUDIO_COOLDOWN_SECONDS, STABILITY_FRAMES
from framing_logic import FramingState


@dataclass
class DebounceDecision:
    """Output of one debounce update step.

    Attributes:
        raw_state: Immediate state from current-frame heuristics.
        stable_state: Accepted state after stability filtering.
        should_speak: Whether audio should be triggered now.
        state_to_speak: State to vocalize when `should_speak` is True.
        consecutive_count: Number of consecutive frames observed for raw_state.
        cooldown_remaining: Remaining cooldown seconds before next allowed speak.
    """

    raw_state: FramingState
    stable_state: Optional[FramingState]
    should_speak: bool
    state_to_speak: Optional[FramingState]
    consecutive_count: int
    cooldown_remaining: float


class DebounceController:
    """Combine stability, state-change, and cooldown audio gates.

    Why both state-change gate + cooldown:
    - State-change-only can still spam when state oscillates at boundaries.
    - Cooldown-only can still repeatedly re-speak the same state after timer expiry.
    - Combined gating removes both failure modes.
    """

    def __init__(
        self,
        stability_frames: int = STABILITY_FRAMES,
        cooldown_seconds: float = AUDIO_COOLDOWN_SECONDS,
    ) -> None:
        """Initialize the debounce controller.

        Args:
            stability_frames: Frames required before accepting a new stable state.
            cooldown_seconds: Minimum seconds between spoken instructions.
        """
        self._stability_frames = stability_frames
        self._cooldown_seconds = cooldown_seconds

        self._candidate_state: Optional[FramingState] = None
        self._candidate_count = 0

        self._last_stable_state: Optional[FramingState] = None
        self._last_spoken_state: Optional[FramingState] = None
        self._last_spoken_time = 0.0

    def update(self, raw_state: FramingState, now: Optional[float] = None) -> DebounceDecision:
        """Consume one raw state and decide if audio should trigger.

        Args:
            raw_state: Immediate frame-level state from FramingLogic.
            now: Optional timestamp override in seconds.

        Returns:
            DebounceDecision containing stable-state tracking and speak decision.

        Notes:
            A stable state must persist for `stability_frames` before it is accepted.
            Audio is emitted only when a newly accepted stable state differs from the
            last spoken state and cooldown has elapsed.
        """
        timestamp = now if now is not None else time.time()

        if raw_state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = raw_state
            self._candidate_count = 1

        state_became_stable = False
        if (
            self._candidate_count >= self._stability_frames
            and self._candidate_state != self._last_stable_state
        ):
            # Strategy C: accept only after N consecutive frames.
            self._last_stable_state = self._candidate_state
            state_became_stable = True

        elapsed = timestamp - self._last_spoken_time
        cooldown_remaining = max(0.0, self._cooldown_seconds - elapsed)

        should_speak = False
        state_to_speak: Optional[FramingState] = None
        if state_became_stable and self._last_stable_state != self._last_spoken_state:
            if cooldown_remaining <= 0.0:
                # Strategy A + B: state transition plus minimum spacing in time.
                should_speak = True
                state_to_speak = self._last_stable_state
                self._last_spoken_state = self._last_stable_state
                self._last_spoken_time = timestamp
                cooldown_remaining = self._cooldown_seconds

        return DebounceDecision(
            raw_state=raw_state,
            stable_state=self._last_stable_state,
            should_speak=should_speak,
            state_to_speak=state_to_speak,
            consecutive_count=self._candidate_count,
            cooldown_remaining=cooldown_remaining,
        )

    def reset(self) -> None:
        """Reset debounce state, useful when restarting a capture session."""
        self._candidate_state = None
        self._candidate_count = 0
        self._last_stable_state = None
        self._last_spoken_state = None
        self._last_spoken_time = 0.0
