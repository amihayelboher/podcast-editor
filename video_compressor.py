"""Compress video with FFmpeg (re-encode video, copy audio)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def compressed_path_for(video_path: str) -> str:
    """Derive ``<stem>_compressed.mp4`` next to the input video."""
    stem, _ = os.path.splitext(video_path)
    return f"{stem}_compressed.mp4"


def compress_video(
    input_path: str,
    output_path: str | None = None,
    codec: str = "libx264",
    crf: int = 23,
    preset: str = "slow",
) -> str:
    """
    Compress a video with FFmpeg.

    Re-encodes video with ``codec`` / ``crf`` / ``preset`` and stream-copies audio.
    If ``output_path`` is None, writes ``<stem>_compressed.mp4`` next to the input.

    Returns the output file path.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output_path = compressed_path_for(input_path)

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-c:a",
        "copy",
        output_path,
    ]

    print(
        f"[video_compressor] Compressing: {input_path}\n"
        f"[video_compressor] Settings: codec={codec}, crf={crf}, preset={preset}, "
        f"audio=copy"
    )
    print(f"[video_compressor] Output: {output_path}")
    print("[video_compressor] Running FFmpeg...")
    subprocess.run(cmd, check=True)
    print(f"[video_compressor] Done! Saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress video with FFmpeg (re-encode video, copy audio)."
    )
    parser.add_argument(
        "input",
        help="Input video path",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output video path (default: <stem>_compressed.mp4)",
    )
    parser.add_argument(
        "--codec",
        default="libx264",
        help="Video codec (default: libx264)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Constant Rate Factor (default: 23; lower = higher quality)",
    )
    parser.add_argument(
        "--preset",
        default="slow",
        help="x264/x265 preset (default: slow)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    compress_video(
        args.input,
        output_path=args.output,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
    )
