"""RGBA Pillow caption renderer + FFmpeg alpha-channel video encoder (.mov, .webm, .mp4)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from config import ASPECT_RATIOS, BACKGROUND_MODES, resolve_font
from styler import CAPTION_STYLES


def _hex_to_rgba(hex_color: str, default_alpha: int = 255) -> Tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), default_alpha)
    if len(h) == 8:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16))
    return (255, 255, 255, default_alpha)


def _load_font(role: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = resolve_font(role)
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap_words_into_lines(
    draw: ImageDraw.ImageDraw,
    words: List[Dict[str, Any]],
    fonts: List[ImageFont.FreeTypeFont | ImageFont.ImageFont],
    texts: List[str],
    max_width: int,
    space_px: int,
) -> List[List[int]]:
    """Return list of lines, where each line is a list of word indices."""
    lines: List[List[int]] = []
    current_line: List[int] = []
    current_w = 0

    for idx, (txt, font) in enumerate(zip(texts, fonts)):
        bbox = draw.textbbox((0, 0), txt, font=font)
        w = bbox[2] - bbox[0]
        added = w if not current_line else (space_px + w)
        if current_line and (current_w + added > max_width):
            lines.append(current_line)
            current_line = [idx]
            current_w = w
        else:
            current_line.append(idx)
            current_w += added

    if current_line:
        lines.append(current_line)
    return lines


def render_caption_frame(
    width: int,
    height: int,
    bg_rgba: Tuple[int, int, int, int],
    segment: Optional[Dict[str, Any]],
    active_word_idx: int,
    style_id: str = "purple_pill",
    v_align: str = "center",
    h_align: str = "center",
    font_size: int = 72,
    offset_x_pct: float = 0.0,
    offset_y_pct: float = 0.0,
) -> Image.Image:
    """Render a single RGBA frame with the active caption segment and style."""
    img = Image.new("RGBA", (width, height), bg_rgba)
    if segment is None or not segment.get("words"):
        return img

    style = CAPTION_STYLES.get(style_id, CAPTION_STYLES["purple_pill"])
    draw = ImageDraw.Draw(img, "RGBA")

    words = segment["words"]
    emp_idx = segment.get("emphasis_index", 0)
    mode_type = style["mode_type"]

    # Determine which word gets accent styling
    highlight_idx = active_word_idx if (mode_type == "word_pill" and active_word_idx >= 0) else emp_idx

    base_font = _load_font(style["font_role"], font_size)
    accent_font = _load_font(style["accent_font_role"], int(font_size * 1.06))

    texts: List[str] = []
    fonts: List[ImageFont.FreeTypeFont | ImageFont.ImageFont] = []
    for i, w in enumerate(words):
        txt = w["word"].upper() if style["uppercase"] else w["word"]
        texts.append(txt)
        fonts.append(accent_font if (i == highlight_idx and style["accent_font_role"] != style["font_role"]) else base_font)

    space_px = max(12, int(font_size * 0.28))
    max_text_width = int(width * 0.84)
    lines = _wrap_words_into_lines(draw, words, fonts, texts, max_text_width, space_px)

    # Measure each line (tracking actual inked bounding-box offsets so pills align symmetrically)
    line_metrics: List[Tuple[int, int, int, int, List[Tuple[int, int, int, int, int, int, int]]]] = []
    line_gap = int(font_size * 0.30)
    total_h = 0

    for line_indices in lines:
        lw = 0
        lh = 0
        min_by0 = 9999
        max_by1 = -9999
        word_boxes: List[Tuple[int, int, int, int, int, int, int]] = []
        for pos, idx in enumerate(line_indices):
            bx0, by0, bx1, by1 = draw.textbbox((0, 0), texts[idx], font=fonts[idx])
            ww = bx1 - bx0
            wh = by1 - by0
            word_boxes.append((idx, ww, wh, bx0, by0, bx1, by1))
            lw += ww + (space_px if pos > 0 else 0)
            lh = max(lh, wh)
            min_by0 = min(min_by0, by0)
            max_by1 = max(max_by1, by1)
        if min_by0 == 9999:
            min_by0, max_by1 = 0, lh
        line_metrics.append((lw, lh, min_by0, max_by1, word_boxes))
        total_h += lh

    total_h += line_gap * max(0, len(lines) - 1)
    max_lw = max((m[0] for m in line_metrics), default=0)
    first_by0 = line_metrics[0][2] if line_metrics else 0
    last_by1 = line_metrics[-1][3] if line_metrics else total_h

    # Anchor Y
    if v_align == "top":
        base_y = int(height * 0.12)
    elif v_align == "center":
        base_y = int((height - total_h) / 2)
    else:  # bottom (above Reels/Shorts UI)
        base_y = int(height * 0.72 - total_h / 2)
    base_y += int((offset_y_pct / 100.0) * height)

    # Anchor X helper per line
    margin_x = int(width * 0.08)
    shift_x = int((offset_x_pct / 100.0) * width)

    def get_line_x(lw: int) -> int:
        if h_align == "left":
            return margin_x + shift_x
        if h_align == "right":
            return width - margin_x - lw + shift_x
        return int((width - lw) / 2) + shift_x

    # Stacked magazine background word
    if mode_type == "stacked_bg" and 0 <= highlight_idx < len(texts):
        bg_word = texts[highlight_idx]
        big_size = int(font_size * 2.15)
        big_font = _load_font(style["accent_font_role"], big_size)
        bb = draw.textbbox((0, 0), bg_word, font=big_font)
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        bx = get_line_x(bw)
        by = base_y - int(bh * 0.35)
        draw.text((bx, by), bg_word, font=big_font, fill=_hex_to_rgba(style["accent_color"], 235))

    # Full-phrase pill/box background (aligned to actual inked text bounds)
    if mode_type == "full_box" and style.get("pill_color"):
        pad_x = int(font_size * 0.42)
        pad_y = int(font_size * 0.28)
        box_x = get_line_x(max_lw) - pad_x
        box_y_top = base_y + first_by0 - pad_y
        # Calculate bottom of last line's inked area
        last_line_cur_y = base_y + sum(m[1] + line_gap for m in line_metrics[:-1])
        box_y_bot = last_line_cur_y + last_by1 + pad_y
        draw.rounded_rectangle(
            [box_x, box_y_top, box_x + max_lw + pad_x * 2, box_y_bot],
            radius=int(font_size * 0.30),
            fill=_hex_to_rgba(style["pill_color"]),
        )

    text_rgba = _hex_to_rgba(style["text_color"])
    accent_rgba = _hex_to_rgba(style["accent_color"])
    outline_rgba = _hex_to_rgba(style["outline_color"])
    outline_w = int(style["outline_width"])

    cur_y = base_y
    for lw, lh, min_by0, max_by1, word_boxes in line_metrics:
        cur_x = get_line_x(lw)
        for idx, ww, wh, bx0, by0, bx1, by1 in word_boxes:
            txt = texts[idx]
            fnt = fonts[idx]
            is_hl = idx == highlight_idx

            # Per-word rounded pill (symmetrical around actual inked glyphs)
            if mode_type == "word_pill" and is_hl and style.get("pill_color"):
                pad_x = int(font_size * 0.24)
                pad_y = int(font_size * 0.18)
                draw.rounded_rectangle(
                    [
                        cur_x + bx0 - pad_x,
                        cur_y + min_by0 - pad_y,
                        cur_x + bx1 + pad_x,
                        cur_y + max_by1 + pad_y,
                    ],
                    radius=int(font_size * 0.28),
                    fill=_hex_to_rgba(style["pill_color"]),
                )

            fill_col = accent_rgba if (is_hl and mode_type != "stacked_bg") else text_rgba

            # Soft drop shadow
            if style.get("shadow"):
                sh_off = max(2, int(font_size * 0.05))
                draw.text(
                    (cur_x + sh_off, cur_y + sh_off),
                    txt,
                    font=fnt,
                    fill=(0, 0, 0, 150),
                )

            # Main word text (with optional stroke)
            draw.text(
                (cur_x, cur_y),
                txt,
                font=fnt,
                fill=fill_col,
                stroke_width=outline_w,
                stroke_fill=outline_rgba if outline_w > 0 else None,
            )
            cur_x += ww + space_px
        cur_y += lh + line_gap

    return img


def _find_state_at_time(
    segments: List[Dict[str, Any]], t: float
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Return (active_segment, active_word_idx) at timestamp t."""
    for seg in segments:
        if seg["start"] <= t <= seg["end"]:
            words = seg.get("words", [])
            active_w = -1
            for idx, w in enumerate(words):
                if w["start"] <= t <= w["end"]:
                    active_w = idx
                    break
            if active_w == -1 and words:
                # Pick nearest word before t
                for idx, w in enumerate(words):
                    if t >= w["start"]:
                        active_w = idx
            return seg, active_w
    return None, -1


def render_caption_video(
    audio_path: Path,
    output_path: Path,
    segments: List[Dict[str, Any]],
    duration: float,
    style_id: str = "purple_pill",
    aspect_ratio: str = "9:16",
    bg_mode: str = "transparent_mov",
    v_align: str = "bottom",
    h_align: str = "center",
    font_size: int = 72,
    offset_x_pct: float = 0.0,
    offset_y_pct: float = 0.0,
    fps: int = 30,
) -> Path:
    """
    Render styled captions into a video file using cached RGBA states piped to FFmpeg.
    Supports true alpha transparency (.mov ProRes 4444 / .webm VP9) as well as green/black .mp4.
    """
    width, height = ASPECT_RATIOS.get(aspect_ratio, (1080, 1920))
    bg_cfg = BACKGROUND_MODES.get(bg_mode, BACKGROUND_MODES["transparent_mov"])
    bg_rgba = bg_cfg["bg_rgba"] if bg_cfg["bg_rgba"] is not None else (0, 0, 0, 0)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(audio_path),
    ]

    if bg_mode == "transparent_mov":
        cmd += [
            "-c:v", "prores_ks",
            "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le",
            "-c:a", "aac",
            "-b:a", "192k",
        ]
    elif bg_mode == "transparent_webm":
        cmd += [
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-b:v", "4M",
            "-c:a", "libopus",
            "-b:a", "128k",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
        ]

    cmd += ["-shortest", str(output_path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    total_frames = max(1, int(round(duration * fps)))
    state_cache: Dict[Tuple[int, int], bytes] = {}

    try:
        for f_idx in range(total_frames):
            t = f_idx / fps
            seg, active_w = _find_state_at_time(segments, t)
            key = (seg["id"], active_w) if seg is not None else (-1, -1)
            if key not in state_cache:
                frame_img = render_caption_frame(
                    width=width,
                    height=height,
                    bg_rgba=bg_rgba,
                    segment=seg,
                    active_word_idx=active_w,
                    style_id=style_id,
                    v_align=v_align,
                    h_align=h_align,
                    font_size=font_size,
                    offset_x_pct=offset_x_pct,
                    offset_y_pct=offset_y_pct,
                )
                state_cache[key] = frame_img.tobytes()
            proc.stdin.write(state_cache[key])
        proc.stdin.close()
    except BrokenPipeError:
        pass

    stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg render failed (code {ret}): {stderr_out[-1200:]}")

    return output_path
