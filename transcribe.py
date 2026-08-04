"""Stage 1 transcription: word-level timestamps via faster-whisper (or openai-whisper)."""

from __future__ import annotations

import json
import os
import time
from typing import Any

# Prefer faster-whisper; fall back to openai-whisper.
_BACKEND: str | None = None
try:
    from faster_whisper import WhisperModel  # type: ignore

    _BACKEND = "faster-whisper"
except ImportError:
    try:
        import whisper as openai_whisper  # type: ignore

        _BACKEND = "openai-whisper"
    except ImportError:
        openai_whisper = None  # type: ignore


def _require_backend() -> str:
    if _BACKEND is None:
        raise ImportError(
            "No Whisper backend found. Install one of:\n"
            "  pip install faster-whisper\n"
            "  pip install openai-whisper"
        )
    return _BACKEND


def _normalize_word(word: str, start: float, end: float, probability: float) -> dict[str, Any]:
    return {
        "word": word,
        "start": float(start),
        "end": float(end),
        "probability": float(probability),
    }


def _transcribe_faster_whisper(
    path: str,
    model_size: str,
    language: str | None,
) -> dict[str, Any]:
    device = "cuda"
    compute_type = "float16"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
            compute_type = "int8"
    except ImportError:
        device = "cpu"
        compute_type = "int8"

    print(f"[transcribe] Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("[transcribe] Transcription started (word_timestamps=True)...")
    t0 = time.perf_counter()
    segments_iter, info = model.transcribe(
        path,
        language=language,
        word_timestamps=True,
    )

    segments: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(
                    _normalize_word(
                        w.word,
                        w.start,
                        w.end,
                        getattr(w, "probability", 0.0),
                    )
                )
        segment_text = (seg.text or "").strip()
        text_parts.append(segment_text)
        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": segment_text,
                "words": words,
            }
        )

    elapsed = time.perf_counter() - t0
    detected = getattr(info, "language", language)
    print(
        f"[transcribe] Done in {elapsed:.1f}s"
        + (f" (detected language: {detected})" if detected else "")
    )
    return {
        "text": " ".join(p for p in text_parts if p).strip(),
        "segments": segments,
        "language": detected,
        "backend": "faster-whisper",
        "model_size": model_size,
    }


def _transcribe_openai_whisper(
    path: str,
    model_size: str,
    language: str | None,
) -> dict[str, Any]:
    print(f"[transcribe] Loading openai-whisper model '{model_size}'...")
    model = openai_whisper.load_model(model_size)

    print("[transcribe] Transcription started (word_timestamps=True)...")
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {"word_timestamps": True, "verbose": False}
    if language:
        kwargs["language"] = language
    result = model.transcribe(path, **kwargs)
    elapsed = time.perf_counter() - t0

    segments: list[dict[str, Any]] = []
    for seg in result.get("segments") or []:
        words = []
        for w in seg.get("words") or []:
            words.append(
                _normalize_word(
                    w.get("word", ""),
                    w.get("start", 0.0),
                    w.get("end", 0.0),
                    w.get("probability", 0.0),
                )
            )
        segments.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or "").strip(),
                "words": words,
            }
        )

    detected = result.get("language") or language
    print(
        f"[transcribe] Done in {elapsed:.1f}s"
        + (f" (detected language: {detected})" if detected else "")
    )
    return {
        "text": (result.get("text") or "").strip(),
        "segments": segments,
        "language": detected,
        "backend": "openai-whisper",
        "model_size": model_size,
    }


def transcribe_audio(
    video_or_audio_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
) -> dict[str, Any]:
    """
    Transcribe video/audio with word-level timestamps.

    Uses faster-whisper when available, otherwise openai-whisper.
    Whisper backends accept video paths directly (ffmpeg extracts audio).

    Returns
    -------
    dict with keys:
      - text: full transcript
      - segments: list of {start, end, text, words: [{word, start, end, probability}]}
      - language, backend, model_size (metadata)
    """
    if not os.path.exists(video_or_audio_path):
        raise FileNotFoundError(f"Input not found: {video_or_audio_path}")

    backend = _require_backend()
    print(f"[transcribe] Input: {video_or_audio_path}")
    print(f"[transcribe] Backend: {backend}")

    if backend == "faster-whisper":
        return _transcribe_faster_whisper(video_or_audio_path, model_size, language)
    return _transcribe_openai_whisper(video_or_audio_path, model_size, language)


def save_transcription_json(result: dict, output_json_path: str) -> None:
    """Write transcription result as pretty-printed JSON."""
    parent = os.path.dirname(os.path.abspath(output_json_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[transcribe] Saved transcript JSON to: {output_json_path}")


def transcript_path_for(video_or_audio_path: str) -> str:
    """Derive ``<stem>_transcript.json`` next to the input file."""
    stem, _ = os.path.splitext(video_or_audio_path)
    return f"{stem}_transcript.json"
