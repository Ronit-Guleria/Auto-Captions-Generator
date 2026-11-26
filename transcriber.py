"""Audio extraction, multi-provider transcription (ElevenLabs -> Universal -> Gemini -> Grok -> Faster-Whisper), and script forced alignment."""

from __future__ import annotations

import difflib
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


HINGLISH_PROMPT = (
    "Transcribe Hindi and English spoken words in Romanized Hinglish using ONLY English letters (a-z), "
    "for example: Five Star ne Squid Game ke sath mil kar ek campaign launch kiya."
)


def _run_cf_whisper(wav_path: Path, token: str, language_mode: str = "hinglish", script_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run Cloudflare Workers AI Whisper using a specific Cloudflare token."""
    import base64

    keys = load_api_keys()
    account_id = keys.get("CF_ACCOUNT_ID", "").strip()
    if not account_id or not token:
        raise ValueError("CF_ACCOUNT_ID and Cloudflare token are required.")

    audio_bytes = wav_path.read_bytes()
    duration = get_audio_duration(wav_path)
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/openai/whisper-large-v3-turbo"
    payload: Dict[str, Any] = {"audio": base64.b64encode(audio_bytes).decode("ascii")}
    if script_hint and script_hint.strip():
        payload["initial_prompt"] = script_hint.strip()[:600]
        if re.search(r"[a-zA-Z]", script_hint):
            payload["language"] = "en"
    elif language_mode == "hinglish":
        payload["language"] = "en"
        payload["initial_prompt"] = HINGLISH_PROMPT
    elif language_mode in ("en", "hi"):
        payload["language"] = language_mode

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    result = data.get("result", data)
    words: List[Dict[str, Any]] = []
    if isinstance(result.get("words"), list) and result["words"]:
        for w in result["words"]:
            tok = str(w.get("word", "")).strip()
            if tok:
                words.append({"word": tok, "start": round(float(w.get("start", 0)), 3), "end": round(float(w.get("end", 0)), 3)})
    elif isinstance(result.get("segments"), list) and result["segments"]:
        for seg in result["segments"]:
            words.extend(_interpolate_words_from_text(str(seg.get("text", "")), float(seg.get("start", 0)), float(seg.get("end", 0.5))))
    elif result.get("text"):
        words = _interpolate_words_from_text(str(result["text"]), 0.0, duration)
    if not words:
        raise RuntimeError("Cloudflare Workers AI returned no transcribed words.")
    return words


def transcribe_cf_elevenlabs(wav_path: Path, language_mode: str = "hinglish", script_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """Transcribe via Cloudflare Workers AI using CF_ELEVENLABS_TOKEN."""
    keys = load_api_keys()
    token = keys.get("CF_ELEVENLABS_TOKEN", "").strip() or keys.get("CF_API_TOKEN", "").strip()
    if not token:
        raise ValueError("CF_ELEVENLABS_TOKEN is not set.")
    return _run_cf_whisper(wav_path, token, language_mode=language_mode, script_hint=script_hint)


def transcribe_cf_universal(wav_path: Path, language_mode: str = "hinglish", script_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """Transcribe via Cloudflare Workers AI using CF_UNIVERSAL_TOKEN."""
    keys = load_api_keys()
    token = keys.get("CF_UNIVERSAL_TOKEN", "").strip() or keys.get("CF_API_TOKEN", "").strip()
    if not token:
        raise ValueError("CF_UNIVERSAL_TOKEN is not set.")
    return _run_cf_whisper(wav_path, token, language_mode=language_mode, script_hint=script_hint)


def transcribe_elevenlabs(wav_path: Path) -> List[Dict[str, Any]]:
    """Transcribe audio using ElevenLabs Scribe Speech-to-Text API."""
    keys = load_api_keys()
    api_key = keys.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is not set.")

    duration = get_audio_duration(wav_path)
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": api_key}

    with wav_path.open("rb") as f:
        resp = requests.post(
            url,
            headers=headers,
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"model_id": "scribe_v1", "timestamps_granularity": "word"},
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()

    words: List[Dict[str, Any]] = []
    for w in data.get("words", []):
        if w.get("type", "word") != "word":
            continue
        tok = str(w.get("text", w.get("word", ""))).strip()
        if tok:
            s = max(0.0, float(w.get("start", 0.0)))
            e = max(s + 0.05, float(w.get("end", s + 0.2)))
            words.append({"word": tok, "start": round(s, 3), "end": round(e, 3)})

    if not words and data.get("text"):
        words = _interpolate_words_from_text(str(data["text"]), 0.0, duration)
    if not words:
        raise RuntimeError("ElevenLabs Scribe returned no transcribed words.")
    return words


def transcribe_universal(wav_path: Path) -> List[Dict[str, Any]]:
    """Transcribe audio using AssemblyAI Universal STT API."""
    keys = load_api_keys()
    api_key = keys.get("UNIVERSAL_3.5_PRO", "").strip()
    if not api_key:
        raise ValueError("UNIVERSAL_3.5_PRO is not set.")

    duration = get_audio_duration(wav_path)
    headers = {"authorization": api_key}

    # 1. Upload audio
    with wav_path.open("rb") as f:
        up_resp = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            data=f,
            timeout=120,
        )
    up_resp.raise_for_status()
    upload_url = up_resp.json()["upload_url"]

    # 2. Request transcription
    tr_resp = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={**headers, "content-type": "application/json"},
        json={"audio_url": upload_url, "speech_models": ["universal-2"]},
        timeout=30,
    )
    tr_resp.raise_for_status()
    transcript_id = tr_resp.json()["id"]

    # 3. Poll until completed
    poll_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    deadline = time.time() + 180
    result: Dict[str, Any] = {}
    while time.time() < deadline:
        p_resp = requests.get(poll_url, headers=headers, timeout=30)
        p_resp.raise_for_status()
        result = p_resp.json()
        status = result.get("status")
        if status == "completed":
            break
        if status == "error":
            raise RuntimeError(f"Universal STT error: {result.get('error', 'unknown')}")
        time.sleep(1.5)
    else:
        raise TimeoutError("Universal STT timed out waiting for completion.")

    words: List[Dict[str, Any]] = []
    for w in result.get("words") or []:
        tok = str(w.get("text", "")).strip()
        if tok:
            s = max(0.0, float(w.get("start", 0)) / 1000.0)
            e = max(s + 0.05, float(w.get("end", 0)) / 1000.0)
            words.append({"word": tok, "start": round(s, 3), "end": round(e, 3)})

    if not words and result.get("text"):
        words = _interpolate_words_from_text(str(result["text"]), 0.0, duration)
    if not words:
        raise RuntimeError("Universal STT returned no transcribed words.")
    return words


def transcribe_local_whisper(
    wav_path: Path,
    model_size: str = "base",
    language_mode: str = "hinglish",
    script_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Transcribe audio locally with faster-whisper for exact word-level timestamps."""
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    kwargs: Dict[str, Any] = {"word_timestamps": True}
    if script_hint and script_hint.strip():
        kwargs["initial_prompt"] = script_hint.strip()[:600]
        if re.search(r"[a-zA-Z]", script_hint):
            kwargs["language"] = "en"
    elif language_mode == "hinglish":
        kwargs["language"] = "en"
        kwargs["initial_prompt"] = HINGLISH_PROMPT
    elif language_mode in ("en", "hi"):
        kwargs["language"] = language_mode

    segments, _ = model.transcribe(str(wav_path), **kwargs)
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


class GeminiRetryableError(RuntimeError):
    """Raised when Gemini fails (e.g. high demand/rate limit) and can be retried."""


def transcribe_gemini(wav_path: Path, language_mode: str = "hinglish") -> List[Dict[str, Any]]:
    """Transcribe audio using Gemini 3.8 Flash with word-level timestamps."""
    from google import genai
    from google.genai import types

    keys = load_api_keys()
    api_key = keys.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    duration = get_audio_duration(wav_path)
    client = genai.Client(api_key=api_key)
    lang_instruction = (
        "Write all Hindi/Hinglish and English words in Romanized Hinglish using ONLY English letters (A-Z), NEVER Urdu/Arabic or Devanagari script. "
        if language_mode == "hinglish"
        else "Write Hindi words in Devanagari script. "
        if language_mode == "hi"
        else "Transcribe in English. "
        if language_mode == "en"
        else ""
    )
    prompt = (
        f"Transcribe this audio (total duration {duration:.2f} seconds) word-by-word. "
        f"{lang_instruction}"
        "Return ONLY a JSON array of objects with exact timestamps in seconds: "
        '[{"word": "Hello", "start": 0.10, "end": 0.42}, ...]. '
        "No markdown, no explanation."
    )
    try:
        resp = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                types.Part.from_bytes(data=wav_path.read_bytes(), mime_type="audio/wav"),
                prompt,
            ],
        )
    except Exception as exc:
        msg = str(exc)
        low = msg.lower()
        if any(k in low for k in ("503", "429", "unavailable", "resource_exhausted", "overloaded", "high demand", "quota")):
            raise GeminiRetryableError(
                "Gemini is currently facing high demand or rate limits."
            ) from exc
        raise GeminiRetryableError(f"Gemini transcription failed: {msg}") from exc

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
    if matcher.ratio() < 0.25 and len(asr_words) > 0:
        # Fallback when ASR returned a different script (e.g. Devanagari/Urdu) than the Romanized pasted script:
        # map each script word proportionally onto the real ASR word timestamps so pauses stay synced.
        n_asr = len(asr_words)
        n_scr = len(script_tokens)
        aligned_slots: List[Dict[str, Any]] = []
        for j, tok in enumerate(script_tokens):
            idx_a = min(n_asr - 1, int(j * n_asr / n_scr))
            aligned_slots.append(
                {
                    "word": tok,
                    "start": asr_words[idx_a]["start"],
                    "end": asr_words[idx_a]["end"],
                }
            )
        return aligned_slots

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

    for idx in range(len(aligned)):
        if aligned[idx]["end"] <= aligned[idx]["start"]:
            aligned[idx]["end"] = round(aligned[idx]["start"] + 0.08, 3)
        if idx > 0 and aligned[idx]["start"] < aligned[idx - 1]["end"]:
            aligned[idx]["start"] = aligned[idx - 1]["end"]
            if aligned[idx]["end"] <= aligned[idx]["start"]:
                aligned[idx]["end"] = round(aligned[idx]["start"] + 0.08, 3)

    return aligned


_URDU_DEVANAGARI_MAP = {
    # Urdu / Arabic script -> Romanized phonetic letters
    "ا": "a", "آ": "aa", "أ": "a", "إ": "i", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t",
    "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z",
    "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z",
    "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "گ": "g",
    "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "o", "ہ": "h", "ھ": "h", "ء": "",
    "ی": "i", "ے": "e", "ئ": "i", "ؤ": "o", "ة": "h", "ى": "a",
    "َ": "a", "ِ": "i", "ُ": "u", "ً": "an", "ٍ": "in", "ٌ": "un", "ْ": "", "ّ": "", "ٰ": "a",
    # Devanagari -> Romanized phonetic letters
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ए": "e", "ऐ": "ai",
    "ओ": "o", "औ": "au", "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t",
    "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
    "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s",
    "ह": "h", "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ं": "n", "ँ": "n", "्": "", "ः": "h",
}


def _fallback_romanize(text: str) -> str:
    out = []
    for ch in text:
        if ch in _URDU_DEVANAGARI_MAP:
            out.append(_URDU_DEVANAGARI_MAP[ch])
        elif ord(ch) < 128:
            out.append(ch)
    res = "".join(out).strip()
    return res if res else text


def _transliterate_words_to_hinglish(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If ASR returned Urdu/Arabic or Devanagari words while language_mode='hinglish',
    transliterate them 1-to-1 into Romanized Hinglish (English letters) via Gemini
    (with an instant offline phonetic fallback if Gemini is busy) while preserving
    100% of the original word timestamps.
    """
    if not words:
        return words
    has_non_ascii = any(re.search(r"[^\x00-\x7F]", w["word"]) for w in words)
    if not has_non_ascii:
        return words

    keys = load_api_keys()
    gem_key = keys.get("GEMINI_API_KEY", "").strip()
    if gem_key:
        try:
            from google import genai

            client = genai.Client(api_key=gem_key)
            raw_list = [w["word"] for w in words]
            prompt = (
                f"Transliterate each of these {len(raw_list)} Urdu/Hindi words 1-to-1 into natural Romanized Hinglish "
                "(using ONLY English letters A-Z and punctuation, e.g. 'ایک' -> 'Ek', 'آدمی' -> 'aadmi', 'نے' -> 'ne'). "
                f"Return ONLY a JSON array of strings of exact length {len(raw_list)}.\n\n"
                + json.dumps(raw_list, ensure_ascii=False)
            )
            resp = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
            raw = (resp.text or "").strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                converted = json.loads(match.group(0))
                if isinstance(converted, list) and len(converted) == len(words):
                    for w, new_tok in zip(words, converted):
                        cleaned = str(new_tok).strip()
                        if cleaned:
                            w["word"] = cleaned
                    return words
        except Exception:
            pass

    # Deterministic offline fallback so Urdu/Devanagari never leaks into Hinglish mode
    for w in words:
        if re.search(r"[^\x00-\x7F]", w["word"]):
            w["word"] = _fallback_romanize(w["word"])
    return words


def _run_with_one_retry(fn: Callable[[Path], List[Dict[str, Any]]], wav_path: Path) -> List[Dict[str, Any]]:
    """Try a transcription provider once, and on failure retry once before raising."""
    try:
        return fn(wav_path)
    except Exception:
        time.sleep(0.8)
        return fn(wav_path)


def transcribe_and_align(
    wav_path: Path,
    engine: str = "auto",
    script_text: Optional[str] = None,
    language_mode: str = "hinglish",
) -> Dict[str, Any]:
    """
    Transcribe voiceover in ordered fallback sequence:
    1. Cloudflare AI (ElevenLabs Token)
    2. Cloudflare AI (Universal 3.5 Pro Token)
    3. Gemini (gemini-3.8-flash)
    4. Faster-Whisper (offline mode only, when installed)

    Each cloud model retries once before shifting to the next model.
    """
    duration = get_audio_duration(wav_path)
    keys = load_api_keys()
    has_local_whisper = importlib.util.find_spec("faster_whisper") is not None

    cf_el_fn = lambda p: transcribe_cf_elevenlabs(p, language_mode=language_mode, script_hint=script_text)
    cf_uni_fn = lambda p: transcribe_cf_universal(p, language_mode=language_mode, script_hint=script_text)
    gem_fn = lambda p: transcribe_gemini(p, language_mode=language_mode)

    single_map: Dict[str, Tuple[str, Callable[[Path], List[Dict[str, Any]]]]] = {
        "cf_elevenlabs": ("Cloudflare AI (ElevenLabs Token)", cf_el_fn),
        "cf_universal": ("Cloudflare AI (Universal 3.5 Pro Token)", cf_uni_fn),
        "gemini": ("Gemini (gemini-3.8-flash)", gem_fn),
        "elevenlabs": ("ElevenLabs Scribe", transcribe_elevenlabs),
        "universal": ("Universal 3.5 Pro", transcribe_universal),
    }

    used_engine = ""
    raw_words: List[Dict[str, Any]] = []

    if engine == "local":
        if not has_local_whisper:
            raise ValueError("Faster-Whisper is not installed on cloud hosting.")
        raw_words = transcribe_local_whisper(wav_path, language_mode=language_mode, script_hint=script_text)
        used_engine = "Faster-Whisper (Offline)"
    elif engine in single_map:
        label, fn = single_map[engine]
        raw_words = _run_with_one_retry(fn, wav_path)
        used_engine = label
    else:
        # Auto sequence: CF ElevenLabs -> CF Universal -> Gemini -> Faster-Whisper (offline)
        cf_acc = keys.get("CF_ACCOUNT_ID", "").strip()
        has_cf_el = bool(cf_acc and (keys.get("CF_ELEVENLABS_TOKEN") or keys.get("CF_API_TOKEN")))
        has_cf_uni = bool(cf_acc and (keys.get("CF_UNIVERSAL_TOKEN") or keys.get("CF_API_TOKEN")))
        pipeline: List[Tuple[str, bool, Callable[[Path], List[Dict[str, Any]]]]] = [
            ("Cloudflare AI (ElevenLabs Token)", has_cf_el, cf_el_fn),
            ("Cloudflare AI (Universal 3.5 Pro Token)", has_cf_uni, cf_uni_fn),
            ("Gemini (gemini-3.8-flash)", bool(keys.get("GEMINI_API_KEY")), gem_fn),
            ("ElevenLabs Scribe", bool(keys.get("ELEVENLABS_API_KEY")), transcribe_elevenlabs),
            ("Universal 3.5 Pro", bool(keys.get("UNIVERSAL_3.5_PRO")), transcribe_universal),
        ]

        errors: List[str] = []
        for label, is_configured, fn in pipeline:
            if not is_configured:
                continue
            try:
                raw_words = _run_with_one_retry(fn, wav_path)
                if errors:
                    used_engine = f"{label} (fallback after {', '.join(e.split(':')[0] for e in errors)})"
                else:
                    used_engine = label
                break
            except Exception as exc:
                errors.append(f"{label}: {exc}")

        if not raw_words:
            if has_local_whisper:
                raw_words = transcribe_local_whisper(wav_path)
                used_engine = (
                    f"Faster-Whisper (offline fallback after {', '.join(e.split(':')[0] for e in errors)})"
                    if errors
                    else "Faster-Whisper (Offline)"
                )
            else:
                detail = " | ".join(errors) if errors else "No API keys configured."
                raise RuntimeError(f"All cloud transcription models failed: {detail}")

    if script_text and script_text.strip():
        final_words = align_script_to_words(script_text, raw_words, duration)
        mode = "script_synced"
    else:
        if language_mode == "hinglish":
            raw_words = _transliterate_words_to_hinglish(raw_words)
        final_words = raw_words
        mode = "voiceover_only"

    return {
        "duration": round(duration, 3),
        "engine": used_engine,
        "mode": mode,
        "words": final_words,
    }


def check_all_api_keys() -> List[Dict[str, Any]]:
    """Test each configured API key with a lightweight read-only request."""
    keys = load_api_keys()
    results: List[Dict[str, Any]] = []

    # 1. Cloudflare Workers AI tokens
    cf_acc = keys.get("CF_ACCOUNT_ID", "").strip()
    cf_token_names = [k for k in ("CF_ELEVENLABS_TOKEN", "CF_UNIVERSAL_TOKEN", "CF_API_TOKEN") if keys.get(k, "").strip()]
    if not cf_acc or not cf_token_names:
        results.append({"name": "Cloudflare Workers AI (CF_ACCOUNT_ID)", "ok": False, "status": "Not configured"})
    else:
        for k in cf_token_names:
            tok = keys[k].strip()
            try:
                r = requests.get(
                    f"https://api.cloudflare.com/client/v4/accounts/{cf_acc}/ai/models/search?per_page=1",
                    headers={"Authorization": f"Bearer {tok}"},
                    timeout=12,
                )
                if r.status_code == 200:
                    results.append({"name": f"Cloudflare AI ({k})", "ok": True, "status": "Working"})
                else:
                    results.append({"name": f"Cloudflare AI ({k})", "ok": False, "status": f"Failed (HTTP {r.status_code})"})
            except Exception as exc:
                results.append({"name": f"Cloudflare AI ({k})", "ok": False, "status": f"Error: {exc}"})

    # 3. Gemini 3.8 Flash
    gem_key = keys.get("GEMINI_API_KEY", "").strip()
    if not gem_key:
        results.append({"name": "Gemini 3.8 Flash (GEMINI_API_KEY)", "ok": False, "status": "Not configured"})
    else:
        try:
            from google import genai

            client = genai.Client(api_key=gem_key)
            client.models.get(model="gemini-3.8-flash")
            results.append({"name": "Gemini 3.8 Flash (GEMINI_API_KEY)", "ok": True, "status": "Working"})
        except Exception as exc:
            results.append({"name": "Gemini 3.8 Flash (GEMINI_API_KEY)", "ok": False, "status": f"Failed ({exc})"})

    # 4. Optional direct ElevenLabs / Universal keys (shown only if set)
    el_key = keys.get("ELEVENLABS_API_KEY", "").strip()
    if el_key:
        try:
            r = requests.get("https://api.elevenlabs.io/v1/user", headers={"xi-api-key": el_key}, timeout=12)
            results.append({"name": "ElevenLabs Scribe (ELEVENLABS_API_KEY)", "ok": r.status_code == 200, "status": "Working" if r.status_code == 200 else f"Failed (HTTP {r.status_code})"})
        except Exception as exc:
            results.append({"name": "ElevenLabs Scribe (ELEVENLABS_API_KEY)", "ok": False, "status": f"Error: {exc}"})

    uni_key = keys.get("UNIVERSAL_3.5_PRO", "").strip()
    if uni_key:
        try:
            r = requests.get("https://api.assemblyai.com/v2/transcript?limit=1", headers={"authorization": uni_key}, timeout=12)
            results.append({"name": "Universal 3.5 Pro (UNIVERSAL_3.5_PRO)", "ok": r.status_code == 200, "status": "Working" if r.status_code == 200 else f"Failed (HTTP {r.status_code})"})
        except Exception as exc:
            results.append({"name": "Universal 3.5 Pro (UNIVERSAL_3.5_PRO)", "ok": False, "status": f"Error: {exc}"})

    # 5. Faster-Whisper (Offline Local)
    has_local = importlib.util.find_spec("faster_whisper") is not None
    results.append({
        "name": "Faster-Whisper (Offline Local)",
        "ok": has_local,
        "status": "Available (Local PC)" if has_local else "Not installed (Cloud Mode)",
    })

    return results
