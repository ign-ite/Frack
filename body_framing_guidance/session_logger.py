"""CSV session logging for state changes and control actions."""

import csv
import time
from pathlib import Path
from typing import Optional

from config import SESSION_LOG_FOLDER


class SessionLogger:
    """Persist runtime events to CSV for debugging and QA evidence."""

    def __init__(self) -> None:
        root = Path(__file__).resolve().parent
        self._log_dir = root / SESSION_LOG_FOLDER
        self._log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = self._log_dir / f"session_{timestamp}.csv"
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "timestamp",
                "event_type",
                "state",
                "orientation",
                "body_span_ratio",
                "mid_hip_x",
                "lean_angle_deg",
                "source",
                "details",
            ]
        )
        self._file.flush()

    def log_state_change(
        self,
        state: str,
        orientation: str,
        body_span_ratio: Optional[float],
        mid_hip_x: Optional[float],
        lean_angle_deg: Optional[float],
    ) -> None:
        self._write_row(
            event_type="state_change",
            state=state,
            orientation=orientation,
            body_span_ratio=body_span_ratio,
            mid_hip_x=mid_hip_x,
            lean_angle_deg=lean_angle_deg,
            source="system",
            details="",
        )

    def log_action(self, action: str, source: str, details: str = "") -> None:
        self._write_row(
            event_type="action",
            state="",
            orientation="",
            body_span_ratio=None,
            mid_hip_x=None,
            lean_angle_deg=None,
            source=source,
            details=f"{action}:{details}" if details else action,
        )

    def _write_row(
        self,
        event_type: str,
        state: str,
        orientation: str,
        body_span_ratio: Optional[float],
        mid_hip_x: Optional[float],
        lean_angle_deg: Optional[float],
        source: str,
        details: str,
    ) -> None:
        self._writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                event_type,
                state,
                orientation,
                "" if body_span_ratio is None else f"{body_span_ratio:.4f}",
                "" if mid_hip_x is None else f"{mid_hip_x:.4f}",
                "" if lean_angle_deg is None else f"{lean_angle_deg:.2f}",
                source,
                details,
            ]
        )
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass
