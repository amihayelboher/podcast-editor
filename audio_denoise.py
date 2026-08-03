import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
from df.enhance import enhance, init_df
from torchaudio.functional import resample


def _resolve_device(device: str) -> str:
    """Return a usable torch device string, falling back to CPU if CUDA is unavailable."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return device


def _extract_audio(video_path: str, audio_path: str) -> None:
    """Extract mono WAV audio from a video file at 48 kHz."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-c:a",
        "pcm_s16le",
        audio_path,
    ]
    subprocess.run(cmd, check=True)


def _mux_audio_into_video(video_path: str, audio_path: str, output_path: str) -> None:
    """Replace the video's audio track with denoised audio; copy video stream."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def _load_audio_tensor(path: str, target_sr: int) -> torch.Tensor:
    """Load audio with soundfile and return shape [C, T] float32 tensor at target_sr."""
    data, sr = sf.read(path, always_2d=True, dtype="float32")  # [T, C]
    audio = torch.from_numpy(data.T)  # [C, T]
    if sr != target_sr:
        print(f"Resampling audio from {sr} Hz to {target_sr} Hz...")
        audio = resample(audio, sr, target_sr)
    return audio.contiguous()


def _save_audio_tensor(path: str, audio: torch.Tensor, sr: int) -> None:
    """Save [C, T] or [T] float tensor as WAV via soundfile."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        data = audio.numpy().T  # [T, C]
    else:
        data = np.asarray(audio, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
    sf.write(path, data, sr)


def remove_static_noise_from_video(input_path: str, output_path: str, device: str = "cuda") -> None:
    """
    Denoise a video's audio with DeepFilterNet and write a new video file.

    Extracts audio, runs DeepFilterNet noise reduction, then muxes the cleaned
    audio back with the original video stream.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    device = _resolve_device(device)
    print(f"Denoising audio with DeepFilterNet on device: {device}")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_audio = os.path.join(tmpdir, "audio_raw.wav")
        clean_audio = os.path.join(tmpdir, "audio_clean.wav")

        print("Extracting audio...")
        _extract_audio(input_path, raw_audio)

        print("Loading DeepFilterNet model...")
        model, df_state, _ = init_df()
        model = model.to(device)
        model.eval()

        print("Running noise reduction...")
        audio = _load_audio_tensor(raw_audio, target_sr=df_state.sr())
        with torch.no_grad():
            enhanced = enhance(model, df_state, audio)

        _save_audio_tensor(clean_audio, enhanced, df_state.sr())
        print("Muxing denoised audio back into video...")
        _mux_audio_into_video(input_path, clean_audio, output_path)

    print(f"Done! Denoised video saved to: {output_path}")
