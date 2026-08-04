import os

from audio_denoise import denoise_video

VIDEO_FOLDER = "test_videos"
INPUT_VIDEO = "non_static_noise.mp4"


def output_path_for(input_path: str) -> str:
    stem, ext = os.path.splitext(input_path)
    return f"{stem}_audio_denoised{ext}"


def main() -> None:
    input_path = os.path.join(VIDEO_FOLDER, INPUT_VIDEO)
    output_video = output_path_for(input_path)
    denoise_video(input_path, output_video, methods=["static"])


if __name__ == "__main__":
    main()
