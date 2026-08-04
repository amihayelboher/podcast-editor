import os
import subprocess
import tempfile

import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
from df.enhance import enhance, init_df
from torchaudio.functional import resample

# Silero Speech Enhancer expects 24 kHz mono input; output is typically 48 kHz.
_SILERO_INPUT_SR = 24000
_SILERO_OUTPUT_SR = 48000


def _resolve_device(device: str) -> str:
    """Return a usable torch device string, falling back to CPU if CUDA is unavailable."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return device


def _extract_audio(
    video_path: str,
    audio_path: str,
    sample_rate: int = 48000,
    channels: int = 1,
) -> None:
    """Extract PCM WAV audio from a video file."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
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
        # Silero may return [1, 1, T]
        while audio.ndim > 2:
            audio = audio.squeeze(0)
        data = audio.numpy().T  # [T, C]
    else:
        data = np.asarray(audio, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
    sf.write(path, data, sr)


def _mute_non_speech_intervals(
    audio: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.65,
    min_silence_duration_ms: int = 50,
    speech_pad_ms: int = 30,
) -> torch.Tensor:
    """
    Mute non-speech gaps with Silero VAD; keep the exact sample count.

    Applies a 10 ms linear fade-in/out at each speech boundary to avoid pops.
    Expects audio as [C, T] (or [T]); returns the same layout and length.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float()
    else:
        audio = torch.as_tensor(audio, dtype=torch.float32)

    original_1d = audio.ndim == 1
    if original_1d:
        audio = audio.unsqueeze(0)
    while audio.ndim > 2:
        audio = audio.squeeze(0)

    n_samples = audio.shape[-1]
    mono = audio.mean(dim=0) if audio.shape[0] > 1 else audio[0]

    print("[vad] Loading Silero VAD model...")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]

    print(
        f"[vad] Detecting speech timestamps "
        f"(threshold={threshold}, min_silence_duration_ms={min_silence_duration_ms}, "
        f"speech_pad_ms={speech_pad_ms})..."
    )
    speech_timestamps = get_speech_timestamps(
        mono,
        vad_model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    print(f"[vad] Found {len(speech_timestamps)} speech segment(s)")

    # Default silent; speech segments get gain of 1 with short edge fades.
    gain = torch.zeros(n_samples, dtype=torch.float32)
    fade_len = max(1, int(round(0.010 * sample_rate)))  # 10 ms

    for seg in speech_timestamps:
        start = int(max(0, min(seg["start"], n_samples)))
        end = int(max(0, min(seg["end"], n_samples)))
        if end <= start:
            continue

        gain[start:end] = 1.0
        # Keep fades within the segment so they cannot re-open muted gaps oddly.
        n_fade = min(fade_len, max(1, (end - start) // 2))
        if n_fade > 0:
            fade_in = torch.linspace(0.0, 1.0, n_fade)
            fade_out = torch.linspace(1.0, 0.0, n_fade)
            gain[start : start + n_fade] = fade_in
            gain[end - n_fade : end] = fade_out

    gated = audio * gain.unsqueeze(0)
    if gated.shape[-1] != n_samples:
        raise RuntimeError(
            f"VAD gating changed length ({n_samples} -> {gated.shape[-1]}); sync would break"
        )

    if original_1d:
        return gated.squeeze(0)
    return gated.contiguous()


def denoise_deepfilter(
    input_path: str,
    output_path: str,
    device: str = "cuda",
    mute_non_speech_intervals: bool = True,
    threshold: float = 0.65,
    min_silence_duration_ms: int = 50,
) -> None:
    """
    Denoise a video's audio with DeepFilterNet (stationary noise).

    Extracts audio, runs DeepFilterNet, optionally mutes non-speech gaps via
    Silero VAD (length-preserving), then muxes without re-encoding the video stream.

    VAD tuning (when mute_non_speech_intervals is True):
      - threshold: speech probability threshold (higher = strict, ignores clicks)
      - min_silence_duration_ms: minimum gap to split speech (lower = more word gaps)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    device = _resolve_device(device)
    print(f"[static] Denoising with DeepFilterNet on device: {device}")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_audio = os.path.join(tmpdir, "audio_raw.wav")
        clean_audio = os.path.join(tmpdir, "audio_clean.wav")

        print("[static] Extracting mono audio at 48 kHz...")
        _extract_audio(input_path, raw_audio, sample_rate=48000, channels=1)

        print("[static] Loading DeepFilterNet model...")
        model, df_state, _ = init_df()
        model = model.to(device)
        model.eval()

        print("[static] Running stationary noise reduction...")
        audio = _load_audio_tensor(raw_audio, target_sr=df_state.sr())
        with torch.no_grad():
            enhanced = enhance(model, df_state, audio)

        if mute_non_speech_intervals:
            print("[static] Muting non-speech intervals (Silero VAD, length-preserving)...")
            enhanced = _mute_non_speech_intervals(
                enhanced,
                sample_rate=df_state.sr(),
                threshold=threshold,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=30,
            )
        else:
            print("[static] Skipping non-speech mute (mute_non_speech_intervals=False)")

        _save_audio_tensor(clean_audio, enhanced, df_state.sr())
        print("[static] Muxing denoised audio back into video...")
        _mux_audio_into_video(input_path, clean_audio, output_path)

    print(f"[static] Done! Saved to: {output_path}")


def denoise_transients(input_path: str, output_path: str) -> None:
    """
    Reduce non-stationary noise (keyboard/mouse clicks) with noisereduce.

    Extracts mono audio, applies noisereduce with stationary=False, then muxes
    the cleaned audio back into the video without re-encoding the video stream.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    print("[transients] Denoising with noisereduce (non-stationary)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_audio = os.path.join(tmpdir, "audio_raw.wav")
        clean_audio = os.path.join(tmpdir, "audio_clean.wav")

        print("[transients] Extracting mono audio at 16 kHz...")
        _extract_audio(input_path, raw_audio, sample_rate=16000, channels=1)

        print("[transients] Running reduce_noise(stationary=False)...")
        data, sr = sf.read(raw_audio, dtype="float32")
        reduced = nr.reduce_noise(y=data, sr=sr, stationary=False)
        sf.write(clean_audio, reduced, sr)

        print("[transients] Muxing cleaned audio back into video...")
        _mux_audio_into_video(input_path, clean_audio, output_path)

    print(f"[transients] Done! Saved to: {output_path}")


def denoise_silero(
    input_path: str,
    output_path: str,
    device: str = "cuda",
) -> None:
    """
    Enhance speech / suppress non-human voice with Silero Speech Enhancer.

    Extracts mono audio at 16 kHz, resamples to the model rate (24 kHz), runs
    silero_denoise, then muxes the enhanced audio (48 kHz) without re-encoding video.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    device = _resolve_device(device)
    print(f"[silero] Denoising with Silero Speech Enhancer on device: {device}")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_audio = os.path.join(tmpdir, "audio_raw.wav")
        clean_audio = os.path.join(tmpdir, "audio_clean.wav")

        print("[silero] Extracting mono audio at 16 kHz...")
        _extract_audio(input_path, raw_audio, sample_rate=16000, channels=1)

        print("[silero] Loading Silero silero_denoise model...")
        # Official hub entry returns (model, samples, utils).
        loaded = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_denoise",
            force_reload=False,
            trust_repo=True,
        )
        if len(loaded) == 3:
            model, _samples, _utils = loaded
        else:
            model, _utils = loaded

        model = model.to(device)
        model.eval()

        print("[silero] Running speech enhancement...")
        # 16 kHz extract -> upsample to Silero's 24 kHz input.
        audio = _load_audio_tensor(raw_audio, target_sr=_SILERO_INPUT_SR)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = (audio * 0.95).to(device)

        with torch.no_grad():
            enhanced = model(audio)

        # Model output is typically [1, 1, T] at 48 kHz.
        _save_audio_tensor(clean_audio, enhanced, _SILERO_OUTPUT_SR)
        print("[silero] Muxing enhanced audio back into video...")
        _mux_audio_into_video(input_path, clean_audio, output_path)

    print(f"[silero] Done! Saved to: {output_path}")


_METHOD_ALIASES = {
    "static": "deepfilter",
    "deepfilter": "deepfilter",
    "transients": "transients",
    "non-static": "transients",
    "non-human voice": "silero",
    "silero": "silero",
}

_METHOD_HANDLERS = {
    "deepfilter": denoise_deepfilter,
    "transients": denoise_transients,
    "silero": denoise_silero,
}

def _normalize_methods(methods: str | list[str]) -> list[str]:
    """
    Accept a single method name or an ordered list; return canonical method keys.

    If methods is or contains "non-human voice" (or silero aliases), return only
    the Silero stage — no chaining with other denoisers.
    """
    if isinstance(methods, str):
        methods = [methods]
    if not methods:
        raise ValueError("methods must contain at least one denoise step")

    raw = [str(name).strip() for name in methods]
    raw_lower = [name.lower() for name in raw]

    if "silero" in raw_lower or "non-human voice" in raw_lower:
        return ["silero"]

    normalized: list[str] = []
    for name in raw_lower:
        key = _METHOD_ALIASES.get(name)
        if key is None:
            allowed = ", ".join(sorted(_METHOD_ALIASES))
            raise ValueError(f"Unknown denoise method {name!r}. Expected one of: {allowed}")
        normalized.append(key)
    return normalized


def denoise_video(
    input_path: str,
    output_path: str,
    methods: str | list[str] = "static",
    mute_non_speech_intervals: bool = True,
    threshold: float = 0.65,
    min_silence_duration_ms: int = 50,
) -> None:
    """
    Denoise video audio with one or more stages, applied in order.

    Parameters
    ----------
    methods:
        A single method name (str) or an ordered list of method names.
        Accepted names (aliases in parentheses):
          - "static" / "deepfilter" -> DeepFilterNet
          - "transients" / "non-static" -> noisereduce clicks
          - "non-human voice" / "silero" -> Silero Speech Enhancer only
            (if present, only Silero runs; other methods in the list are ignored)
    mute_non_speech_intervals:
        When True (default), DeepFilterNet stages mute non-speech gaps via Silero VAD
        without changing the audio length (video sync preserved).
    threshold:
        Silero VAD speech probability threshold (default 0.65; higher ignores clicks).
    min_silence_duration_ms:
        Minimum silence gap to split speech segments (default 50 ms between words).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    pipeline = _normalize_methods(methods)
    print(f"denoise_video: pipeline {pipeline}")

    # Chain stages through intermediate video files when more than one method is used.
    with tempfile.TemporaryDirectory() as tmpdir:
        current = input_path
        for i, method in enumerate(pipeline):
            is_last = i == len(pipeline) - 1
            stage_out = output_path if is_last else os.path.join(tmpdir, f"stage_{i}_{method}.mp4")
            print(f"denoise_video: stage {i + 1}/{len(pipeline)} -> {method}")
            if method == "deepfilter":
                denoise_deepfilter(
                    current,
                    stage_out,
                    mute_non_speech_intervals=mute_non_speech_intervals,
                    threshold=threshold,
                    min_silence_duration_ms=min_silence_duration_ms,
                )
            else:
                _METHOD_HANDLERS[method](current, stage_out)
            current = stage_out
