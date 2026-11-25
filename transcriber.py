"""Audio extraction, transcription (Gemini Audio / faster-whisper), and script forced alignment."""

from __future__ import annotations

import difflib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import load_api_keys


def get_audio_duration(media_path: Path) -> float:
    """Return media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return max(0.1, float(res.stdout.strip()))


def extract_wav_16k(input_path: Path, output_wav: Path) -> Path:
    """Convert any uploaded audio/video into 16kHz mono WAV for reliable transcription."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return output_wav


def _interpolate_words_from_text(text: str, start: float, end: float) -> List[Dict[str, Any]]:
    """Split segment text into words with character-proportional timestamps."""
    tokens = [t for t in text.strip().split() if t]
    if not tokens:
        return []
    total_chars = sum(max(1, len(re.sub(r"\W+", "", t))) for t in tokens)
    duration = max(0.05, end - start)
    cursor = start
    words: List[Dict[str, Any]] = []
    for tok in tokens:
        weight = max(1, len(re.sub(r"\W+", "", tok))) / total_chars
        w_dur = duration * weight
        words.append(
            {
                "word": tok,
                "start": round(cursor, 3),
                "end": round(cursor + w_dur, 3),
            }
        )
        cursor += w_dur
    return words


def transcribe_local_whisper(wav_path: Path, model_size: str = "base") -> List[Dict[str, Any]]:
    """Transcribe audio locally with faster-whisper for exact word-level timestamps."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav_path), word_timestamps=True)
    words: List[Dict[str, Any]] = []
    for seg in segments:
        if getattr(seg, "words", None):
            for w in seg.words:
                tok = w.word.strip()
                if tok:
                    words.append(
                        {
                            "word": tok,
                            "start": round(float(w.start), 3),
                            "end": round(float(w.end), 3),
                        }
                    )
        else:
            words.extend(_interpolate_words_from_text(seg.text, float(seg.start), float(seg.end)))

    if not words:
        raise RuntimeError("Local faster-whisper returned no transcribed words.")
    return words


def transcribe_gemini(wav_path: Path) -> List[Dict[str, Any]]:
    """Transcribe audio using Gemini 2.5 Flash with word-level timestamps."""
    from google import genai
    from google.genai import types

    keys = load_api_keys()
    api_key = keys.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    duration = get_audio_duration(wav_path)
    client = genai.Client(api_key=api_key)
    prompt = (
        f"Transcribe this audio (total duration {duration:.2f} seconds) word-by-word. "
        "Return ONLY a JSON array of objects with exact timestamps in seconds: "
        '[{"word": "Hello", "start": 0.10, "end": 0.42}, ...]. '
        "No markdown, no explanation."
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=wav_path.read_bytes(), mime_type="audio/wav"),
            prompt,
        ],
    )
    raw = (resp.text or "").strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise RuntimeError("Gemini did not return valid word-timestamp JSON.")

    items = json.loads(match.group(0))
    words: List[Dict[str, Any]] = []
    for it in items:
        tok = str(it.get("word", "")).strip()
        if not tok:
            continue
        s = max(0.0, min(duration, float(it.get("start", 0.0))))
        e = max(s + 0.05, min(duration, float(it.get("end", s + 0.25))))
        words.append({"word": tok, "start": round(s, 3), "end": round(e, 3)})

    if not words:
        raise RuntimeError("Gemini returned no transcribed words.")
    return words


def _normalize_token(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def align_script_to_words(script_text: str, asr_words: List[Dict[str, Any]], total_duration: float) -> List[Dict[str, Any]]:
    """
    Force-align a user-pasted script onto ASR word timestamps using SequenceMatcher.
    Preserves 100% of the user's exact spelling, capitalization, and punctuation
    while anchoring timings to the actual spoken audio.
    """
    script_tokens = [t for t in script_text.strip().split() if t]
    if not script_tokens:
        return asr_words
    if not asr_words:
        return _interpolate_words_from_text(script_text, 0.0, total_duration)

    norm_script = [_normalize_token(t) for t in script_tokens]
    norm_asr = [_normalize_token(w["word"]) for w in asr_words]

    matcher = difflib.SequenceMatcher(None, norm_asr, norm_script, autojunk=False)
    aligned: List[Dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1):
                asr_w = asr_words[i1 + offset]
                aligned.append(
                    {
                        "word": script_tokens[j1 + offset],
                        "start": asr_w["start"],
                        "end": asr_w["end"],
                    }
                )
        elif tag == "replace":
            span_start = asr_words[i1]["start"]
            span_end = asr_words[i2 - 1]["end"]
            sub_text = " ".join(script_tokens[j1:j2])
            aligned.extend(_interpolate_words_from_text(sub_text, span_start, span_end))
        elif tag == "insert":
            prev_end = asr_words[i1 - 1]["end"] if i1 > 0 else 0.0
            next_start = asr_words[i1]["start"] if i1 < len(asr_words) else total_duration
            if next_start <= prev_end:
                next_start = min(total_duration, prev_end + 0.25 * (j2 - j1))
            sub_text = " ".join(script_tokens[j1:j2])
            aligned.extend(_interpolate_words_from_text(sub_text, prev_end, next_start))
        # 'delete' (extra ASR hallucination/filler not in script) is omitted

    # Ensure monotonic non-overlapping timestamps
    for idx in range(len(aligned)):
        if aligned[idx]["end"] <= aligned[idx]["start"]:
            aligned[idx]["end"] = round(aligned[idx]["start"] + 0.08, 3)
        if idx > 0 and aligned[idx]["start"] < aligned[idx - 1]["end"]:
            aligned[idx]["start"] = aligned[idx - 1]["end"]
            if aligned[idx]["end"] <= aligned[idx]["start"]:
                aligned[idx]["end"] = round(aligned[idx]["start"] + 0.08, 3)

    return aligned


def transcribe_and_align(
    wav_path: Path,
    engine: str = "auto",
    script_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe voiceover using Gemini Audio or local faster-whisper,
    and optionally force-align a pasted script to the audio timestamps.
    """
    duration = get_audio_duration(wav_path)
    keys = load_api_keys()
    has_gemini = bool(keys.get("GEMINI_API_KEY"))
    has_local_whisper = importlib.util.find_spec("faster_whisper") is not None

    used_engine = engine
    if engine == "gemini":
        raw_words = transcribe_gemini(wav_path)
        used_engine = "gemini"
    elif engine == "local" or (engine == "auto" and has_local_whisper):
        if not has_local_whisper:
            raise ValueError(
                "Local Whisper is not installed on cloud hosting. Use Auto or Gemini 2.5 Audio."
            )
        raw_words = transcribe_local_whisper(wav_path)
        used_engine = "local_whisper"
    elif has_gemini:
        raw_words = transcribe_gemini(wav_path)
        used_engine = "gemini"
    else:
        raise ValueError(
            "No transcription engine available. Please add your Gemini API Key in API Keys."
        )

    if script_text and script_text.strip():
        final_words = align_script_to_words(script_text, raw_words, duration)
        mode = "script_synced"
    else:
        final_words = raw_words
        mode = "voiceover_only"

    return {
        "duration": round(duration, 3),
        "engine": used_engine,
        "mode": mode,
        "words": final_words,
    }
