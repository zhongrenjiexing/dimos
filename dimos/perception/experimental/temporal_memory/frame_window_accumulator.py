# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frame buffering and windowing for temporal memory.

Pure logic component — no VLM, no I/O.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dimos.msgs.sensor_msgs import Image


def _debug(msg: str, *args: object) -> None:
    """DEBUG: 输出到 stdout，便于在终端中查找。"""
    print(f"[frame-acc DEBUG] {msg % args if args else msg}", flush=True)


@dataclass
class Frame:
    """A single buffered video frame."""

    frame_index: int
    timestamp_s: float
    image: Image


class FrameWindowAccumulator:
    """Bounded frame buffer with windowing extraction.

    Thread-safe: a single lock protects all mutable state.

    Parameters
    ----------
    max_buffer_frames:
        Maximum frames kept in memory.  Oldest frames are evicted when full.
    window_s:
        Minimum elapsed time (in seconds) a window must span.
    stride_s:
        Minimum interval between successive window extractions.
    fps:
        Expected ingest rate — used to compute ``frames_needed``.
    """

    def __init__(
        self,
        *,
        max_buffer_frames: int = 100,
        window_s: float = 5.0,
        stride_s: float = 5.0,
        fps: float = 1.0,
    ) -> None:
        self._lock = threading.Lock()
        self._buffer: deque[Frame] = deque(maxlen=max_buffer_frames)
        self._frame_count = 0
        self._last_analysis_time = -float("inf")
        self._video_start_wall_time: float | None = None

        self.window_s = window_s
        self.stride_s = stride_s
        self.fps = fps
        self._stride_skip_count = 0  # DEBUG: throttle stride-skip logs

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def set_start_time(self, wall_time: float) -> None:
        with self._lock:
            if self._video_start_wall_time is None:
                self._video_start_wall_time = wall_time
                _debug(
                    "set_start_time: _video_start_wall_time=%.3f (wall_time=%.3f)",
                    self._video_start_wall_time,
                    wall_time,
                )

    def add_frame(self, image: Image, wall_time: float) -> None:
        """Add a frame to the buffer.

        ``wall_time`` is the monotonic wall-clock time at ingest.
        The frame's ``timestamp_s`` is computed relative to the
        video start time.
        """
        with self._lock:
            if self._video_start_wall_time is None:
                _debug("add_frame: dropped (video_start not set yet)")
                return
            # 修复: 当 image.ts (机器人时钟) 与 video_start (PC 时钟) 不同步时，
            # timestamp_s 会异常，导致 stride 检查失败。改用 wall_time 保证 stride 按真实时间推进。
            # 原逻辑 (保留): if image.ts: timestamp_s = image.ts - video_start; else: wall_time - video_start
            timestamp_s = wall_time - self._video_start_wall_time
            frame = Frame(
                frame_index=self._frame_count,
                timestamp_s=timestamp_s,
                image=image,
            )
            self._buffer.append(frame)
            self._frame_count += 1
            # DEBUG: every 20th frame or first frame, log timestamp sources
            if self._frame_count == 1 or self._frame_count % 20 == 0:
                _debug(
                    "add_frame #%d: wall_time=%.3f video_start=%.3f timestamp_s=%.3f (fixed: wall_time)",
                    self._frame_count,
                    wall_time,
                    self._video_start_wall_time,
                    timestamp_s,
                )

    # ------------------------------------------------------------------
    # Window extraction
    # ------------------------------------------------------------------

    def try_extract_window(self) -> list[Frame] | None:
        """Try to extract a window of frames.

        Returns ``None`` if insufficient data or stride hasn't elapsed.
        On success, updates ``_last_analysis_time`` and returns frames.
        """
        with self._lock:
            if not self._buffer:
                _debug("try_extract_window: SKIP (buffer empty)")
                return None
            current_time = self._buffer[-1].timestamp_s
            delta = abs(current_time - self._last_analysis_time)
            frames_needed = max(1, int(self.fps * self.window_s))

            if delta < self.stride_s:
                self._stride_skip_count += 1
                if self._stride_skip_count <= 3 or self._stride_skip_count % 10 == 0:
                    _debug(
                        "try_extract_window: SKIP stride #%d "
                        "(delta=%.3f < stride_s=%.1f) current_time=%.3f "
                        "_last_analysis_time=%.3f buffered=%d",
                        self._stride_skip_count,
                        delta,
                        self.stride_s,
                        current_time,
                        self._last_analysis_time,
                        len(self._buffer),
                    )
                return None
            if len(self._buffer) < frames_needed:
                _debug(
                    "try_extract_window: SKIP frames_needed (buffered=%d < %d)",
                    len(self._buffer),
                    frames_needed,
                )
                return None

            frames = list(self._buffer)[-frames_needed:]
            self._last_analysis_time = frames[-1].timestamp_s
            self._stride_skip_count = 0  # reset on successful extract
            _debug(
                "try_extract_window: OK extracted %d frames [%.3f-%.3f]s _last_analysis_time->%.3f",
                len(frames),
                frames[0].timestamp_s,
                frames[-1].timestamp_s,
                self._last_analysis_time,
            )
            return frames

    def mark_analysis_time(self, t: float) -> None:
        """Manually advance the last-analysis timestamp (e.g. after a skip)."""
        with self._lock:
            self._last_analysis_time = t

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def latest_frame(self) -> Frame | None:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def frames_list(self) -> list[Frame]:
        """Return a snapshot of all buffered frames."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
