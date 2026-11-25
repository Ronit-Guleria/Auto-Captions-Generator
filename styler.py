"""Caption grouping (1-word, 3-word, line), Gemini emphasis tagging, and style presets."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from config import load_api_keys

CAPTION_STYLES: Dict[str, Dict[str, Any]] = {
    "purple_pill": {
        "id": "purple_pill",
        "name": "Purple Pill",
        "description": "Bold white text with a rounded purple pill behind the active word (Screenshot 1)",
        "font_role": "sans_bold",
        "accent_font_role": "sans_bold",
        "uppercase": False,
        "text_color": "#FFFFFF",
        "accent_color": "#FFFFFF",
        "pill_color": "#5B42F3",
        "outline_color": "#000000",
        "outline_width": 0,
        "shadow": True,
        "mode_type": "word_pill",
    },
    "bold_impact": {
        "id": "bold_impact",
        "name": "Bold Impact",
        "description": "Giant condensed uppercase typography with punchy pop-in (READY / MONTH)",
        "font_role": "condensed",
        "accent_font_role": "condensed",
        "uppercase": True,
        "text_color": "#FFFFFF",
        "accent_color": "#FFE600",
        "pill_color": None,
        "outline_color": "#000000",
        "outline_width": 5,
        "shadow": True,
        "mode_type": "highlight_word",
    },
    "editorial_serif": {
        "id": "editorial_serif",
        "name": "Editorial Serif",
        "description": "Clean sans-serif paired with italic serif accent words (bought / usual)",
        "font_role": "sans_bold",
        "accent_font_role": "serif_italic",
        "uppercase": False,
        "text_color": "#FFFFFF",
        "accent_color": "#FFF066",
        "pill_color": None,
        "outline_color": "#000000",
        "outline_width": 2,
        "shadow": True,
        "mode_type": "highlight_word",
    },
    "box_highlight": {
        "id": "box_highlight",
        "name": "Box Badge",
        "description": "High-contrast bold text inside a solid orange/red badge (TRADE. / FIRED)",
        "font_role": "condensed",
        "accent_font_role": "condensed",
        "uppercase": True,
        "text_color": "#111111",
        "accent_color": "#FFFFFF",
        "pill_color": "#FF5A1F",
        "outline_color": "#000000",
        "outline_width": 0,
        "shadow": True,
        "mode_type": "full_box",
    },
    "dual_color": {
        "id": "dual_color",
        "name": "Dual-Color Hook",
        "description": "Crisp white text with neon yellow emphasis words (TEAMS LOSE / TWICE)",
        "font_role": "sans_bold",
        "accent_font_role": "sans_bold",
        "uppercase": True,
        "text_color": "#FFFFFF",
        "accent_color": "#D4FF00",
        "pill_color": None,
        "outline_color": "#000000",
        "outline_width": 4,
        "shadow": True,
        "mode_type": "highlight_word",
    },
    "stacked_magazine": {
        "id": "stacked_magazine",
        "name": "Stacked Magazine",
        "description": "Oversized red background word behind crisp foreground text (this / super simple)",
        "font_role": "sans_bold",
        "accent_font_role": "serif_bold",
        "uppercase": False,
        "text_color": "#FFFFFF",
        "accent_color": "#FF1E2D",
        "pill_color": None,
        "outline_color": "#000000",
        "outline_width": 3,
        "shadow": True,
        "mode_type": "stacked_bg",
    },
    "glass_pill": {
        "id": "glass_pill",
        "name": "Dark Pill",
        "description": "Clean subtitle inside a sleek dark rounded container",
        "font_role": "sans_bold",
        "accent_font_role": "sans_bold",
        "uppercase": False,
        "text_color": "#FFFFFF",
        "accent_color": "#7DD3FC",
        "pill_color": "#141418CC",
        "outline_color": "#000000",
        "outline_width": 0,
        "shadow": False,
        "mode_type": "full_box",
    },
    "minimal_clean": {
        "id": "minimal_clean",
        "name": "Minimal Clean",
        "description": "Modern talking-head subtitle with soft shadow and active-word glow",
        "font_role": "sans_regular",
        "accent_font_role": "sans_bold",
        "uppercase": False,
        "text_color": "#E5E7EB",
        "accent_color": "#FFFFFF",
        "pill_color": None,
        "outline_color": "#000000",
        "outline_width": 2,
        "shadow": True,
        "mode_type": "highlight_word",
    },
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "at", "for", "and", "or", "but", "it", "its", "this", "that",
    "with", "as", "by", "from", "just", "so",
}


def group_words_into_segments(
    words: List[Dict[str, Any]],
    chunk_mode: str = "three_word",
) -> List[Dict[str, Any]]:
    """
    Group word-level timestamps into caption segments based on chunk_mode:
    - 'single': 1 word per segment
    - 'three_word': up to 3 words per segment (breaks on punctuation or pause > 0.45s)
    - 'line': up to 7 words or ~36 chars per segment (breaks on punctuation or pause > 0.55s)
    """
    if not words:
        return []

    if chunk_mode == "single":
        segments = []
        for idx, w in enumerate(words):
            segments.append(
                {
                    "id": idx + 1,
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["word"],
                    "words": [w],
                    "emphasis_index": 0,
                }
            )
        return segments

    max_words = 3 if chunk_mode == "three_word" else 7
    max_chars = 24 if chunk_mode == "three_word" else 38
    pause_threshold = 0.45 if chunk_mode == "three_word" else 0.55

    segments: List[Dict[str, Any]] = []
    bucket: List[Dict[str, Any]] = []

    for w in words:
        if bucket:
            prev = bucket[-1]
            gap = w["start"] - prev["end"]
            current_chars = sum(len(x["word"]) + 1 for x in bucket) + len(w["word"])
            ended_sentence = bool(re.search(r"[.!?,;:]$", prev["word"]))
            if (
                len(bucket) >= max_words
                or current_chars > max_chars
                or gap > pause_threshold
                or ended_sentence
            ):
                segments.append(_finalize_segment(len(segments) + 1, bucket))
                bucket = []
        bucket.append(w)

    if bucket:
        segments.append(_finalize_segment(len(segments) + 1, bucket))

    return segments


def _pick_heuristic_emphasis(bucket: List[Dict[str, Any]]) -> int:
    """Pick the most impactful word index in a phrase without calling an LLM."""
    best_idx = len(bucket) - 1
    best_score = -1
    for i, w in enumerate(bucket):
        clean = re.sub(r"[^a-zA-Z0-9]", "", w["word"]).lower()
        score = len(clean)
        if clean not in STOPWORDS:
            score += 6
        if w["word"].isupper() and len(clean) > 1:
            score += 4
        if score >= best_score:
            best_score = score
            best_idx = i
    return best_idx


def _finalize_segment(seg_id: int, bucket: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": seg_id,
        "start": bucket[0]["start"],
        "end": bucket[-1]["end"],
        "text": " ".join(w["word"] for w in bucket),
        "words": bucket,
        "emphasis_index": _pick_heuristic_emphasis(bucket),
    }


def enhance_emphasis_with_gemini(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use Gemini API (if configured) to pick the single most impactful hook/emphasis
    word index in each multi-word segment. Falls back silently to heuristic if unavailable.
    """
    keys = load_api_keys()
    api_key = keys.get("GEMINI_API_KEY", "").strip()
    if not api_key or not segments:
        return segments

    multi = [
        {"id": s["id"], "words": [w["word"] for w in s["words"]]}
        for s in segments
        if len(s["words"]) > 1
    ]
    if not multi:
        return segments

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = (
            "You are a viral short-form video caption editor. "
            "For each caption segment below, pick the 0-based index of the ONE best word to highlight "
            "(prefer strong verbs, nouns, numbers, or emotional hook words). "
            "Return ONLY a JSON object mapping segment id (as string) to integer index, e.g. {\"1\": 2}.\n\n"
            + json.dumps(multi)
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = (resp.text or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            mapping = json.loads(match.group(0))
            for seg in segments:
                val = mapping.get(str(seg["id"]))
                if isinstance(val, int) and 0 <= val < len(seg["words"]):
                    seg["emphasis_index"] = val
    except Exception:
        # Keep heuristic emphasis if Gemini quota/network fails
        pass

    return segments
