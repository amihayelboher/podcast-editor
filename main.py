"""End-to-end podcast editor pipeline orchestrator."""

from __future__ import annotations

import os
import tempfile

from audio_denoise import denoise_video
from transcribe import transcribe_audio
from video_compressor import compress_video
from video_cutter import cut_video_from_transcript

VIDEO_FOLDER = "test_videos"
INPUT_VIDEO = "static_noise.mp4"


def edited_path_for(input_path: str) -> str:
    """Derive ``<stem>_edited<ext>`` next to the input video."""
    stem, ext = os.path.splitext(input_path)
    return f"{stem}_edited{ext}"


def main() -> None:
    input_path = os.path.join(VIDEO_FOLDER, INPUT_VIDEO)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    final_path = edited_path_for(input_path)

    print("=" * 60)
    print("[main] Podcast editor pipeline")
    print(f"[main] Input: {input_path}")
    print(f"[main] Final output: {final_path}")
    print("=" * 60)

    # Intermediates live in a temp dir and are deleted when the pipeline finishes.
    with tempfile.TemporaryDirectory() as tmpdir:
        denoised_path = os.path.join(tmpdir, "audio_denoised.mp4")
        trimmed_path = os.path.join(tmpdir, "silence_trimmed.mp4")

        # Step 1 — Denoise
        print("\n[main] Step 1/4: Denoising audio...")
        denoise_video(input_path, denoised_path, methods=["static"])
        print("[main] Step 1/4 done")

        # Step 2 — Transcribe + align (keep result in memory; no JSON on disk)
        print("\n[main] Step 2/4: Transcribing + aligning (WhisperX)...")
        result = transcribe_audio(denoised_path, model_size="medium", language=None)
        print("[main] Step 2/4 done")

        # Step 3 — Silence trim
        print("\n[main] Step 3/4: Cutting silence...")
        cut_video_from_transcript(denoised_path, result, trimmed_path)
        print("[main] Step 3/4 done")

        # Step 4 — Compress trimmed video → final edited output
        print("\n[main] Step 4/4: Compressing trimmed video...")
        compress_video(
            trimmed_path,
            output_path=final_path,
            codec="libx264",
            crf=23,
            preset="slow",
        )
        print(f"[main] Step 4/4 done: {final_path}")

    print("\n" + "=" * 60)
    print(f"[main] Pipeline complete. Edited video: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
