"""Stage 1 transcription: WhisperX ASR + forced word-level alignment (CUDA)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any


def _require_whisperx():
    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "whisperx is required for word-level forced alignment.\n"
            "  pip install whisperx\n"
            "Requires PyTorch with CUDA for GPU acceleration."
        ) from exc
    return whisperx


def _resolve_device(prefer_cuda: bool = True) -> tuple[str, str]:
    """
    Return (device, preferred_compute_type) for WhisperX.

    GPU uses float16 (with int8 OOM fallback at load time); CPU uses int8.
    """
    try:
        import torch
    except ImportError:
        return "cpu", "int8"

    if prefer_cuda and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[transcribe] Using CUDA: {name}")
        return "cuda", "float16"

    print("[transcribe] CUDA not available — using CPU (int8)")
    return "cpu", "int8"


def _is_oom_error(exc: BaseException) -> bool:
    """True if exception looks like CUDA / allocator out-of-memory."""
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "out of memory",
            "cuda out of memory",
            "cudnn_status_alloc_failed",
            "failed to allocate",
        )
    )


def _clear_cuda() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_whisperx_model(
    whisperx: Any,
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None,
    asr_options: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """
    Load WhisperX ASR model.

    On CUDA OOM with float16, retries once with compute_type=\"int8\".
    Returns (model, compute_type_used).
    """
    kwargs: dict[str, Any] = {
        "compute_type": compute_type,
        "language": language,
    }
    if asr_options is not None:
        kwargs["asr_options"] = asr_options

    try:
        model = whisperx.load_model(model_size, device, **kwargs)
        return model, compute_type
    except Exception as exc:
        if (
            device == "cuda"
            and compute_type == "float16"
            and _is_oom_error(exc)
        ):
            print(
                f"[transcribe] OOM loading model with float16 "
                f"({exc}); falling back to int8..."
            )
            _clear_cuda()
            kwargs["compute_type"] = "int8"
            model = whisperx.load_model(model_size, device, **kwargs)
            return model, "int8"
        raise


def _normalize_word(
    word: str,
    start: float,
    end: float,
    score: float | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "word": word,
        "start": float(start),
        "end": float(end),
    }
    if score is not None:
        item["score"] = float(score)
        item["probability"] = float(score)
    return item


def _segment_words(seg: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize WhisperX word entries from an aligned segment."""
    words: list[dict[str, Any]] = []
    for w in seg.get("words") or []:
        text = w.get("word")
        if text is None:
            text = w.get("text", "")
        # Aligned words may omit start/end when alignment failed for that token.
        if w.get("start") is None or w.get("end") is None:
            continue
        words.append(
            _normalize_word(
                str(text),
                float(w["start"]),
                float(w["end"]),
                w.get("score"),
            )
        )
    return words


def _normalize_aligned_result(
    aligned: dict[str, Any],
    *,
    language: str | None,
    model_size: str,
    device: str,
    compute_type: str,
) -> dict[str, Any]:
    """Convert WhisperX align() output into our stable transcript schema."""
    segments_out: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for seg in aligned.get("segments") or []:
        words = _segment_words(seg)
        segment_text = (seg.get("text") or "").strip()
        if not segment_text and words:
            segment_text = " ".join(w["word"].strip() for w in words).strip()
        text_parts.append(segment_text)

        start = seg.get("start")
        end = seg.get("end")
        if start is None and words:
            start = words[0]["start"]
        if end is None and words:
            end = words[-1]["end"]

        segments_out.append(
            {
                "start": float(start or 0.0),
                "end": float(end or 0.0),
                "text": segment_text,
                "words": words,
            }
        )

    # Prefer word-level full text when available (aligned word list).
    word_level = aligned.get("word_segments")
    if word_level:
        full_text = " ".join(
            str(w.get("word") or w.get("text") or "").strip()
            for w in word_level
            if (w.get("word") or w.get("text"))
        ).strip()
    else:
        full_text = " ".join(p for p in text_parts if p).strip()

    return {
        "text": full_text,
        "segments": segments_out,
        "language": language,
        "backend": "whisperx",
        "model_size": model_size,
        "device": device,
        "compute_type": compute_type,
        "aligned": True,
    }


def transcribe_audio(
    video_or_audio_path: str,
    model_size: str = "medium",
    language: str | None = None,
    *,
    batch_size: int = 8,
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    """
    Transcribe with WhisperX, then forced-align for precise word timestamps.

    Pipeline:
      1. load_model (float16 on GPU; int8 OOM fallback) + load_audio + transcribe
      2. load_align_model for the target/detected language
      3. whisperx.align → hyper-precise word start/end times

    Returns
    -------
    dict with keys:
      - text: full transcript
      - segments: list of {start, end, text, words: [{word, start, end, score?}]}
      - language, backend=whisperx, model_size, device, compute_type, aligned=True
    """
    if not os.path.exists(video_or_audio_path):
        raise FileNotFoundError(f"Input not found: {video_or_audio_path}")

    whisperx = _require_whisperx()
    device, compute_type = _resolve_device(prefer_cuda=prefer_cuda)

    # Raw decoding: keep fillers/hesitations; avoid prior-window suppression of repeats.
    asr_options = {
        "suppress_tokens": [],
        "condition_on_previous_text": False,
    }

    print(f"[transcribe] Input: {video_or_audio_path}")
    print(
        f"[transcribe] Backend: whisperx | model={model_size} "
        f"| device={device} | compute_type={compute_type} (preferred)"
    )

    t0 = time.perf_counter()

    print(f"[transcribe] Loading WhisperX model '{model_size}' ({compute_type})...")
    print(
        "[transcribe] ASR options: suppress_tokens=[] "
        "condition_on_previous_text=False"
    )
    model, compute_type = _load_whisperx_model(
        whisperx,
        model_size,
        device,
        compute_type,
        language,
        asr_options=asr_options,
    )
    print(f"[transcribe] Model ready (compute_type={compute_type})")

    print("[transcribe] Loading audio...")
    audio = whisperx.load_audio(video_or_audio_path)

    print("[transcribe] Running ASR (WhisperX)...")
    try:
        asr_result = model.transcribe(
            audio, batch_size=batch_size, language=language
        )
    except Exception as exc:
        # Batch inference can OOM even if load succeeded; retry int8 + smaller batch.
        if device == "cuda" and compute_type == "float16" and _is_oom_error(exc):
            print(
                f"[transcribe] OOM during ASR (batch_size={batch_size}); "
                "reloading model as int8 with smaller batch..."
            )
            del model
            _clear_cuda()
            model, compute_type = _load_whisperx_model(
                whisperx,
                model_size,
                device,
                "int8",
                language,
                asr_options=asr_options,
            )
            reduced_batch = max(1, batch_size // 2)
            asr_result = model.transcribe(
                audio, batch_size=reduced_batch, language=language
            )
        else:
            raise

    detected = asr_result.get("language") or language
    if not detected:
        raise RuntimeError(
            "WhisperX could not detect language; pass language= explicitly "
            "(e.g. language='en')."
        )
    print(f"[transcribe] Detected language: {detected}")

    # Free ASR model VRAM before loading the alignment model (important on 4GB GPUs).
    del model
    _clear_cuda()

    print(f"[transcribe] Loading alignment model for '{detected}'...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected,
        device=device,
    )

    print("[transcribe] Forced word-level alignment...")
    aligned = whisperx.align(
        asr_result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    del align_model
    _clear_cuda()

    elapsed = time.perf_counter() - t0
    result = _normalize_aligned_result(
        aligned,
        language=detected,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
    n_words = sum(len(s.get("words") or []) for s in result["segments"])
    print(
        f"[transcribe] Done in {elapsed:.1f}s - "
        f"{len(result['segments'])} segment(s), {n_words} aligned word(s)"
    )
    return result


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video with WhisperX ASR + forced word alignment."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=os.path.join("test_videos", "static_noise_audio_denoised.mp4"),
        help=(
            "Input audio or video path "
            "(default: test_videos/static_noise_audio_denoised.mp4)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default: <stem>_transcript.json)",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="WhisperX ASR batch size (default: 8)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU instead of CUDA",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or transcript_path_for(input_path)

    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = transcribe_audio(
        input_path,
        model_size=args.model_size,
        language=args.language,
        batch_size=args.batch_size,
        prefer_cuda=not args.cpu,
    )
    save_transcription_json(result, output_path)
