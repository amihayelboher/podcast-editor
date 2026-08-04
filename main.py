import argparse
import os

from audio_denoise import denoise_video
from transcribe import save_transcription_json, transcribe_audio

VIDEO_FOLDER = "test_videos"
INPUT_VIDEO = "static_noise.mp4"


def output_path_for(input_path: str) -> str:
    stem, ext = os.path.splitext(input_path)
    return f"{stem}_audio_denoised{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Podcast editor Stage 1: denoise + transcribe")
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip transcription after audio denoising",
    )
    parser.add_argument(
        "--model-size",
        default="small",
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force transcription language (e.g. en, he). Default: auto-detect",
    )
    args = parser.parse_args()

    input_path = os.path.join(VIDEO_FOLDER, INPUT_VIDEO)
    output_video = output_path_for(input_path)
    denoise_video(input_path, output_video, methods=["static"])

    if args.no_transcribe:
        print("Skipping transcription (--no-transcribe).")
        return

    result = transcribe_audio(
        output_video,
        model_size=args.model_size,
        language=args.language,
    )
    stem, _ = os.path.splitext(input_path)
    out_json = f"{stem}_transcript.json"
    save_transcription_json(result, out_json)


if __name__ == "__main__":
    main()
