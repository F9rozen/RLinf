#!/usr/bin/env python3
"""Test Intel RealSense color stream: optional PNG snapshot and/or MP4 recording.

Recording defaults to piping raw BGR frames into **ffmpeg** (libx264 + yuv420p), which
produces playable MP4 on all typical players. OpenCV-only muxing often writes broken
files when OpenCV is built without ffmpeg codecs — use ``--record-backend opencv``
only if you must (see script output).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

try:
    import cv2
except ImportError as e:
    raise SystemExit("Install opencv-python: pip install opencv-python") from e

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
WARMUP_FRAMES = 15


def record_video_ffmpeg(
    pipeline,
    first_bgr: np.ndarray,
    duration_sec: float,
    fps: float,
    out_path: Path,
) -> tuple[int, str]:
    """Encode BGR frames to H.264 MP4 via stdin rawvideo — playable on all common players."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found in PATH")

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() not in (".mp4", ".mkv", ".mov"):
        out_path = out_path.with_suffix(".mp4")

    h, w = first_bgr.shape[:2]
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    def write_frame(arr: np.ndarray) -> None:
        b = np.ascontiguousarray(arr, dtype=np.uint8)
        if b.shape[0] != h or b.shape[1] != w or b.shape[2] != 3:
            raise ValueError(f"unexpected frame shape {b.shape}, expected ({h},{w},3)")
        proc.stdin.write(b.tobytes())

    n_frames = 0
    err: bytes | None = None
    try:
        write_frame(first_bgr)
        n_frames = 1
        end_t = time.monotonic() + max(0.0, duration_sec)
        while time.monotonic() < end_t:
            frames = pipeline.wait_for_frames()
            c = frames.get_color_frame()
            if not c:
                continue
            write_frame(np.asanyarray(c.get_data()))
            n_frames += 1
    finally:
        proc.stdin.close()
        err = proc.stderr.read() if proc.stderr else b""
        rc = proc.wait()
        if rc != 0:
            msg = err.decode(errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg exited with {rc}. Install ffmpeg with libx264. Last stderr:\n{msg}"
            )

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty or missing file: {out_path}")

    return n_frames, str(out_path)


def open_video_writer(
    preferred_path: str, fps: float, frame_size: tuple[int, int]
) -> tuple[cv2.VideoWriter, str]:
    """Open cv2.VideoWriter with codecs that work across builds (ARM often lacks MP4)."""
    path = Path(preferred_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = frame_size

    # (fourcc as 4 chars, file suffix) — try MJPG/AVI early: works without FFmpeg MP4 muxer
    attempts: list[tuple[str, str]] = [
        ("mp4v", ".mp4"),
        ("MJPG", ".avi"),
        ("XVID", ".avi"),
    ]

    stem = path.stem
    parent = path.parent
    last_err: str | None = None
    for fourcc_str, suffix in attempts:
        out_path = parent / f"{stem}{suffix}"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
        if writer.isOpened():
            return writer, str(out_path)
        last_err = f"{fourcc_str}/{suffix}"
        writer.release()

    raise RuntimeError(
        "Could not open cv2.VideoWriter with any codec "
        f"(tried mp4v, MJPG, XVID). Last: {last_err}. "
        "Install opencv with ffmpeg support or use a machine with libavcodec."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RealSense color camera test / record")
    p.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording length in seconds (default: 5)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output video path (.mp4). Default: realsense_clip_<timestamp>.mp4",
    )
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Stream / encode FPS")
    p.add_argument(
        "--png",
        type=str,
        nargs="?",
        const="realsense_frame.png",
        default=None,
        metavar="PATH",
        help="Also save a PNG snapshot (default path: realsense_frame.png)",
    )
    p.add_argument(
        "--png-only",
        action="store_true",
        help="Only grab one PNG, do not record video",
    )
    p.add_argument(
        "--record-backend",
        choices=("auto", "ffmpeg", "opencv"),
        default="auto",
        help="auto: prefer ffmpeg (reliable MP4); fallback opencv if ffmpeg missing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, DEFAULT_WIDTH, DEFAULT_HEIGHT, rs.format.bgr8, args.fps
    )

    pipeline.start(config)
    try:
        # Warm-up: auto-exposure often needs a few frames
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame (try another USB port / cable).")

        image = np.asanyarray(color.get_data())
        h, w = image.shape[:2]

        if args.png is not None:
            cv2.imwrite(args.png, image)
            print(f"Saved PNG {args.png} ({w}x{h})")

        if args.png_only:
            return

        out_path = args.output
        if out_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"realsense_clip_{stamp}.mp4"
        out_p = Path(out_path).expanduser()

        use_ffmpeg = args.record_backend == "ffmpeg" or (
            args.record_backend == "auto" and shutil.which("ffmpeg")
        )
        if args.record_backend == "opencv":
            use_ffmpeg = False

        if use_ffmpeg:
            n_frames, used_path = record_video_ffmpeg(
                pipeline, image, args.duration, float(args.fps), out_p
            )
            print(
                f"Saved video {used_path} ({w}x{h} @ {args.fps} Hz, ffmpeg libx264, "
                f"{n_frames} frames, ~{args.duration}s target)"
            )
        else:
            if args.record_backend == "ffmpeg":
                raise SystemExit(
                    "ffmpeg not found in PATH. Install e.g. `sudo apt install ffmpeg` "
                    "or use --record-backend opencv (file may not play in some players)."
                )
            writer, used_path = open_video_writer(
                str(out_p), float(args.fps), (w, h)
            )

            def bgr_frame(arr: np.ndarray) -> np.ndarray:
                return np.ascontiguousarray(arr, dtype=np.uint8)

            writer.write(bgr_frame(image))
            end_t = time.monotonic() + max(0.0, args.duration)
            n_extra = 1
            while time.monotonic() < end_t:
                frames = pipeline.wait_for_frames()
                c = frames.get_color_frame()
                if not c:
                    continue
                writer.write(bgr_frame(np.asanyarray(c.get_data())))
                n_extra += 1

            writer.release()
            sz = Path(used_path).stat().st_size
            print(
                f"Saved video {used_path} ({w}x{h} @ {args.fps} Hz, opencv, "
                f"~{n_extra} frames, {sz} bytes — if unreadable, install ffmpeg and rerun."
            )
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
