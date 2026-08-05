import argparse
import os

from audio_denoise import denoise_video
from transcribe import save_transcription_json, transcribe_audio
from video_cutter import cut_video_from_transcript, trimmed_path_for

VIDEO_FOLDER = "test_videos"
INPUT_VIDEO = "static_noise.mp4"


def output_path_for(input_path: str) -> str:
    stem, ext = os.path.splitext(input_path)
    return f"{stem}_audio_denoised{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Podcast editor: denoise + WhisperX align + silence trim"
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip denoising; use existing denoised video if present",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip WhisperX transcription/alignment (and silence trim)",
    )
    parser.add_argument(
        "--no-cut",
        action="store_true",
        help="Skip silence-gap trimming after alignment",
    )
    parser.add_argument(
        "--model-size",
        default="medium",
        help="WhisperX model size (default: medium)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language for ASR + alignment (e.g. en, he). Default: auto-detect",
    )
    args = parser.parse_args()

    input_path = os.path.join(VIDEO_FOLDER, INPUT_VIDEO)
    output_video = output_path_for(input_path)

    if args.no_denoise:
        if not os.path.exists(output_video):
            raise FileNotFoundError(
                f"--no-denoise requires existing file: {output_video}"
            )
        print(f"Skipping denoise; using {output_video}")
    else:
        denoise_video(input_path, output_video, methods=["static"])

    if args.no_transcribe:
        print("Skipping transcription (--no-transcribe).")
        return

    # WhisperX: ASR + load_align_model + forced alignment → precise word times.
    result = transcribe_audio(
        output_video,
        model_size=args.model_size,
        language=args.language,
    )
    stem, _ = os.path.splitext(input_path)
    out_json = f"{stem}_transcript.json"
    save_transcription_json(result, out_json)

    if args.no_cut:
        print("Skipping silence trim (--no-cut).")
        return

    # Pass WhisperX-aligned result dict directly (same data written to JSON).
    trimmed_video = trimmed_path_for(output_video)
    cut_video_from_transcript(output_video, result, trimmed_video)


if __name__ == "__main__":
    main()
