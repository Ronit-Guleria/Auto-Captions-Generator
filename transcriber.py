"""Audio extraction, transcription (Cloudflare Workers AI / faster-whisper), and script forced alignment."""

from __future__ import annotations

import base64
import difflib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

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


def transcribe_cloudflare(wav_path: Path) -> List[Dict[str, Any]]:
    """Transcribe audio using Cloudflare Workers AI Whisper."""
    keys = load_api_keys()
    account_id = keys.get("CF_ACCOUNT_ID", "").strip()
    api_token = keys.get("CF_API_TOKEN", "").strip()
    if not account_id or not api_token:
        raise ValueError("Cloudflare Workers AI credentials (CF_ACCOUNT_ID and CF_API_TOKEN) are not set.")

    audio_bytes = wav_path.read_bytes()
    duration = get_audio_duration(wav_path)

    # Try @cf/openai/whisper-large-v3-turbo first (supports base64 audio + segments)
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/openai/whisper-large-v3-turbo"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {"audio": base64.b64encode(audio_bytes).decode("ascii")}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        # Fallback to @cf/openai/whisper binary upload
        url_fallback = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/openai/whisper"
        resp = requests.post(
            url_fallback,
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/octet-stream"},
            data=audio_bytes,
            timeout=120,
        )

    resp.raise_for_status()
    data = resp.json()
    if not data.get("success", True) and "result" not in data:
        raise RuntimeError(f"Cloudflare AI error: {json.dumps(data)}")

    result = data.get("result", data)
    words: List[Dict[str, Any]] = []

    if isinstance(result.get("words"), list) and result["words"]:
        for w in result["words"]:
            tok = str(w.get("word", "")).strip()
            if tok:
                words.append(
                    {
                        "word": tok,
                        "start": round(float(w.get("start", 0.0)), 3),
                        "end": round(float(w.get("end", 0.0)), 3),
                    }
                )
    elif isinstance(result.get("segments"), list) and result["segments"]:
        for seg in result["segments"]:
            seg_words = seg.get("words")
            if isinstance(seg_words, list) and seg_words:
                for w in seg_words:
                    tok = str(w.get("word", "")).strip()
                    if tok:
                        words.append(
                            {
                                "word": tok,
                                "start": round(float(w.get("start", 0.0)), 3),
                                "end": round(float(w.get("end", 0.0)), 3),
                            }
                        )
            else:
                seg_text = str(seg.get("text", "")).strip()
                s_start = float(seg.get("start", 0.0))
                s_end = float(seg.get("end", s_start + 0.5))
                words.extend(_interpolate_words_from_text(seg_text, s_start, s_end))
    elif result.get("text"):
        words = _interpolate_words_from_text(str(result["text"]), 0.0, duration)

    if not words:
        raise RuntimeError("Cloudflare Whisper returned no transcribed words.")
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
    Transcribe voiceover using Cloudflare Workers AI or local faster-whisper,
    and optionally force-align a pasted script to the audio timestamps.
    """
    duration = get_audio_duration(wav_path)
    keys = load_api_keys()
    has_cf = bool(keys.get("CF_ACCOUNT_ID") and keys.get("CF_API_TOKEN"))

    used_engine = engine
    if engine == "cloudflare" or (engine == "auto" and has_cf):
        try:
            raw_words = transcribe_cloudflare(wav_path)
            used_engine = "cloudflare"
        except Exception as exc:
            if engine == "cloudflare":
                raise
            # Fallback to local faster-whisper if Cloudflare fails in auto mode
            raw_words = transcribe_local_whisper(wav_path)
            used_engine = f"local_whisper (CF fallback: {exc})"
    else:
        raw_words = transcribe_local_whisper(wav_path)
        used_engine = "local_whisper"

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
