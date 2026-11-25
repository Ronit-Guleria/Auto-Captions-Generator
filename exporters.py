"""Subtitle exporters (.srt, .vtt, .ass, .json) synced to voiceover timestamps."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from config import ASPECT_RATIOS
from styler import CAPTION_STYLES


def _fmt_srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hrs, rem = divmod(ms, 3_600_000)
    mins, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _fmt_vtt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hrs, rem = divmod(ms, 3_600_000)
    mins, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def _fmt_ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    hrs, rem = divmod(cs, 360_000)
    mins, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hrs:d}:{mins:02d}:{secs:02d}.{centis:02d}"


def _hex_to_ass_bgr(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) < 6:
        return "&H00FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b.upper()}{g.upper()}{r.upper()}"


def export_srt(segments: List[Dict[str, Any]], uppercase: bool = False) -> str:
    lines: List[str] = []
    for i, seg in enumerate(segments, start=1):
        text = seg["text"].upper() if uppercase else seg["text"]
        lines.append(f"{i}")
        lines.append(f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def export_vtt(segments: List[Dict[str, Any]], uppercase: bool = False) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for i, seg in enumerate(segments, start=1):
        text = seg["text"].upper() if uppercase else seg["text"]
        lines.append(f"{i}")
        lines.append(f"{_fmt_vtt_time(seg['start'])} --> {_fmt_vtt_time(seg['end'])}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def export_ass(
    segments: List[Dict[str, Any]],
    style_id: str = "purple_pill",
    aspect_ratio: str = "9:16",
    v_align: str = "bottom",
    h_align: str = "center",
    font_size: int = 72,
) -> str:
    """Export styled Advanced SubStation Alpha (.ass) file for Premiere/DaVinci/Aegisub."""
    width, height = ASPECT_RATIOS.get(aspect_ratio, (1080, 1920))
    style = CAPTION_STYLES.get(style_id, CAPTION_STYLES["purple_pill"])

    # ASS numpad alignment: 1-3 bottom, 4-6 middle, 7-9 top
    v_base = {"bottom": 0, "center": 3, "top": 6}.get(v_align, 0)
    h_off = {"left": 1, "center": 2, "right": 3}.get(h_align, 2)
    alignment = v_base + h_off

    primary = _hex_to_ass_bgr(style["text_color"])
    accent = _hex_to_ass_bgr(style["accent_color"])
    outline = _hex_to_ass_bgr(style["outline_color"])
    margin_v = int(height * 0.10)
    margin_lr = int(width * 0.08)

    header = f"""[Script Info]
Title: Auto Captions Export
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},{primary},{accent},{outline},&H80000000,-1,0,0,0,100,100,0,0,1,{style['outline_width']},2,{alignment},{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: List[str] = []
    for seg in segments:
        words = seg.get("words", [])
        emp = seg.get("emphasis_index", -1)
        styled_parts: List[str] = []
        for idx, w in enumerate(words):
            txt = w["word"].upper() if style["uppercase"] else w["word"]
            if idx == emp and style["accent_color"] != style["text_color"]:
                styled_parts.append(f"{{\\c{accent}}}{txt}{{\\c{primary}}}")
            else:
                styled_parts.append(txt)
        line_text = " ".join(styled_parts) if styled_parts else seg["text"]
        events.append(
            f"Dialogue: 0,{_fmt_ass_time(seg['start'])},{_fmt_ass_time(seg['end'])},Default,,0,0,0,,{line_text}"
        )

    return header + "\n".join(events) + "\n"


def export_json(segments: List[Dict[str, Any]], metadata: Dict[str, Any] | None = None) -> str:
    return json.dumps({"metadata": metadata or {}, "segments": segments}, indent=2)
