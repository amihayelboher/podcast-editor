"""Trim excess silence between words using WhisperX-aligned word timestamps."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

# Safety padding kept around each kept speech region (seconds).
PAD_BEFORE_AFTER = 0.12
# Max silent pause between words inside a sentence (seconds).
MAX_MID_SENTENCE_PAUSE = 0.25
# Max silent pause after punctuation marks (seconds).
MAX_PUNCTUATION_PAUSE = 0.50

# Drop words whose duration is outside these bounds (seconds).
# Stretched "ghost" words from mis-alignment often exceed MAX_WORD_DURATION.
MIN_WORD_DURATION = 0.02
MAX_WORD_DURATION = 2.5
# Drop very low-confidence alignments when score is present (WhisperX score ~0–1).
MIN_WORD_SCORE = 0.0

_PUNCTUATION_CHARS = ".!?:;,"


def load_transcript(transcript: str | dict[str, Any]) -> dict[str, Any]:
    """Load Whisper / WhisperX JSON from a path or return a dict as-is."""
    if isinstance(transcript, dict):
        return transcript
    with open(transcript, encoding="utf-8") as f:
        return json.load(f)


def extract_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten aligned segment word arrays into chronological word dicts.

    Handles WhisperX output (``word`` + optional ``score``) and legacy schemas.
    Each word is normalized to: {text, start, end, score?}.
    """
    words: list[dict[str, Any]] = []

    # Prefer segment-level words (primary WhisperX align output).
    for segment in transcript.get("segments") or []:
        for w in segment.get("words") or []:
            parsed = _parse_word_entry(w)
            if parsed is not None:
                words.append(parsed)

    # Some WhisperX dumps also expose a flat word_segments list.
    if not words:
        for w in transcript.get("word_segments") or []:
            parsed = _parse_word_entry(w)
            if parsed is not None:
                words.append(parsed)

    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def _parse_word_entry(w: dict[str, Any]) -> dict[str, Any] | None:
    """Parse one word dict; return None if timestamps are missing or invalid."""
    text = w.get("word") if w.get("word") is not None else w.get("text")
    if text is None:
        return None
    text = str(text)
    if w.get("start") is None or w.get("end") is None:
        return None
    start = float(w["start"])
    end = float(w["end"])
    if end < start:
        return None
    out: dict[str, Any] = {"text": text, "start": start, "end": end}
    score = w.get("score", w.get("probability"))
    if score is not None:
        out["score"] = float(score)
    return out


def filter_aligned_words(
    words: list[dict[str, Any]],
    *,
    min_duration: float = MIN_WORD_DURATION,
    max_duration: float = MAX_WORD_DURATION,
    min_score: float = MIN_WORD_SCORE,
) -> list[dict[str, Any]]:
    """
    Drop abnormal word timings that forced alignment can still produce.

    Filters:
      - zero / negative / tiny durations
      - stretched durations (silence hidden inside a long "word")
      - low alignment scores when present
      - non-positive timeline (after filter, keep chronological order)
    """
    kept: list[dict[str, Any]] = []
    dropped = 0

    for w in words:
        duration = float(w["end"]) - float(w["start"])
        if duration < min_duration or duration > max_duration:
            dropped += 1
            continue
        score = w.get("score")
        if score is not None and float(score) < min_score:
            dropped += 1
            continue
        # Reject empty / punctuation-only tokens? Keep punctuation-bearing words
        # (e.g. "Hello.") so pause rules still apply.
        if not str(w.get("text", "")).strip():
            dropped += 1
            continue
        kept.append(w)

    # Collapse duplicate / nested overlaps that can appear after filtering.
    cleaned: list[dict[str, Any]] = []
    for w in kept:
        if cleaned and w["start"] < cleaned[-1]["end"]:
            # If this word is almost entirely inside the previous, drop the longer stretched one.
            prev = cleaned[-1]
            if w["end"] <= prev["end"]:
                # Nested inside previous — keep the tighter of the two.
                prev_dur = prev["end"] - prev["start"]
                cur_dur = w["end"] - w["start"]
                if cur_dur < prev_dur:
                    cleaned[-1] = w
                dropped += 1
                continue
            # Partial overlap: snap current start to previous end (tiny clip, no silence invent).
            if w["end"] - prev["end"] >= min_duration:
                w = {**w, "start": prev["end"]}
            else:
                dropped += 1
                continue
        cleaned.append(w)

    if dropped:
        print(
            f"[video_cutter] Filtered {dropped} abnormal word(s) "
            f"({len(cleaned)} kept)"
        )
    return cleaned


def ends_with_punctuation(text: str) -> bool:
    """True if the word ends with a sentence/clause punctuation mark."""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _PUNCTUATION_CHARS


def allowed_pause_after(word_text: str) -> float:
    """Return the max allowed silent gap after this word."""
    if ends_with_punctuation(word_text):
        return MAX_PUNCTUATION_PAUSE
    return MAX_MID_SENTENCE_PAUSE


def compute_keep_intervals(
    words: list[dict[str, Any]],
    *,
    pad: float = PAD_BEFORE_AFTER,
    media_duration: float | None = None,
) -> list[tuple[float, float]]:
    """
    Build keep intervals by splitting wherever inter-word silence exceeds limits.

    Contiguous words (gap <= allowed pause) stay in one interval. Excess gaps
    start a new interval: previous ends at current.end + pad, next starts at
    next.start - pad.
    """
    if not words:
        return []

    pad = max(0.0, float(pad))
    intervals: list[tuple[float, float]] = []

    seg_start = max(0.0, words[0]["start"] - pad)
    seg_end = words[0]["end"] + pad

    for i in range(len(words) - 1):
        current = words[i]
        nxt = words[i + 1]
        gap = float(nxt["start"]) - float(current["end"])
        limit = allowed_pause_after(current["text"])

        if gap > limit:
            # Close current region after this word (+ pad).
            seg_end = float(current["end"]) + pad
            if media_duration is not None:
                seg_end = min(seg_end, media_duration)
            if seg_end > seg_start:
                intervals.append((seg_start, seg_end))

            # Open next region before the following word (- pad).
            seg_start = max(0.0, float(nxt["start"]) - pad)
            seg_end = float(nxt["end"]) + pad
        else:
            # Keep contiguous; extend through the next word.
            seg_end = float(nxt["end"]) + pad

    if media_duration is not None:
        seg_end = min(seg_end, media_duration)
    if seg_end > seg_start:
        intervals.append((seg_start, seg_end))

    return _merge_intervals(intervals)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent intervals (can happen when pad > half the cut gap)."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item[0])
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def build_concat_filter(intervals: list[tuple[float, float]]) -> str:
    """
    Build filter_complex: per-interval trim/atrim + setpts, then concat.

    Using PTS-STARTPTS on both streams keeps A/V in sync per clip and after join.
    """
    if not intervals:
        raise ValueError("No intervals to cut")

    parts: list[str] = []
    for i, (start, end) in enumerate(intervals):
        # Explicit start/end keeps filtergraph readable and avoids off-by-one issues.
        parts.append(
            f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS[v{i}]"
        )
        parts.append(
            f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{i}]"
        )

    # Interleave video/audio labels as required by the concat filter.
    labels = "".join(f"[v{i}][a{i}]" for i in range(len(intervals)))
    parts.append(f"{labels}concat=n={len(intervals)}:v=1:a=1[outv][outa]")
    return ";".join(parts)


def cut_video_by_intervals(
    input_path: str,
    output_path: str,
    intervals: list[tuple[float, float]],
) -> None:
    """Cut and join keep-intervals with FFmpeg concat filter (A/V locked)."""
    if not intervals:
        raise ValueError("No keep intervals — nothing to export")

    filter_complex = build_concat_filter(intervals)
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
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-map",
        "[outa]",
        # Stream-copy is not compatible with filter_complex; re-encode for sync.
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        output_path,
    ]
    print(f"[video_cutter] Cutting {len(intervals)} segment(s)...")
    subprocess.run(cmd, check=True)


def cut_video_from_transcript(
    input_path: str,
    transcript: str | dict[str, Any],
    output_path: str,
    *,
    pad: float = PAD_BEFORE_AFTER,
) -> list[tuple[float, float]]:
    """
    Parse WhisperX-aligned JSON (or path), compute silence-trimmed intervals, export.

    Returns the keep intervals that were written.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    data = load_transcript(transcript)
    words = extract_words(data)
    if not words:
        print("[video_cutter] No words in transcript — skipping cut.")
        return []

    words = filter_aligned_words(words)
    if not words:
        print("[video_cutter] All words filtered as abnormal — skipping cut.")
        return []

    backend = data.get("backend", "unknown")
    aligned = data.get("aligned", False)
    print(
        f"[video_cutter] Using {len(words)} word timestamp(s) "
        f"(backend={backend}, aligned={aligned})"
    )

    intervals = compute_keep_intervals(words, pad=pad)
    if not intervals:
        print("[video_cutter] No keep intervals produced — skipping cut.")
        return []

    kept = sum(end - start for start, end in intervals)
    print(
        f"[video_cutter] {len(words)} words -> {len(intervals)} keep interval(s) "
        f"({kept:.2f}s kept)"
    )
    for i, (start, end) in enumerate(intervals, 1):
        print(f"  [{i}] {start:.3f}s - {end:.3f}s ({end - start:.3f}s)")

    cut_video_by_intervals(input_path, output_path, intervals)
    print(f"[video_cutter] Saved trimmed video to: {output_path}")
    return intervals


def trimmed_path_for(video_path: str) -> str:
    """Derive ``<stem>_silence_trimmed<ext>`` next to the input video."""
    stem, ext = os.path.splitext(video_path)
    return f"{stem}_silence_trimmed{ext}"


def _default_transcript_path(video_path: str) -> str:
    """Prefer ``<stem>_transcript.json``; also try dropping ``_audio_denoised``."""
    stem, _ = os.path.splitext(video_path)
    candidates = [f"{stem}_transcript.json"]
    if stem.endswith("_audio_denoised"):
        base = stem[: -len("_audio_denoised")]
        candidates.append(f"{base}_transcript.json")
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Trim excess silence using WhisperX-aligned transcript JSON."
    )
    parser.add_argument(
        "video",
        help="Input video path (timestamps should match this media)",
    )
    parser.add_argument(
        "-t",
        "--transcript",
        default=None,
        help="WhisperX JSON path (default: <video_stem>_transcript.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output video path (default: <video_stem>_silence_trimmed<ext>)",
    )
    args = parser.parse_args()

    video_path = args.video
    transcript_path = args.transcript or _default_transcript_path(video_path)
    output_path = args.output or trimmed_path_for(video_path)

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(transcript_path):
        print(f"Transcript not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    cut_video_from_transcript(video_path, transcript_path, output_path)
