import os

from audio_denoise import remove_static_noise_from_video

INPUT_VIDEO = "test_non_static_noise.mp4"


def output_path_for(input_path: str) -> str:
    stem, ext = os.path.splitext(input_path)
    return f"{stem}_audio_denoised{ext}"


def main() -> None:
    output_video = output_path_for(INPUT_VIDEO)
    remove_static_noise_from_video(INPUT_VIDEO, output_video, device="cuda")


if __name__ == "__main__":
    main()
