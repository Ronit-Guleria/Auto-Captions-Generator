"""Flask backend for Auto Captions Studio."""

from __future__ import annotations

import base64
import io
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request, send_file

from config import (
    ASPECT_RATIOS,
    BACKGROUND_MODES,
    OUTPUT_DIR,
    UPLOAD_DIR,
    load_api_keys,
    save_api_keys,
)
from exporters import export_ass, export_json, export_srt, export_vtt
from renderer import render_caption_frame, render_caption_video
from styler import (
    CAPTION_STYLES,
    enhance_emphasis_with_gemini,
    group_words_into_segments,
)
from transcriber import extract_wav_16k, transcribe_and_align

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    keys = load_api_keys()
    return jsonify(
        {
            "styles": list(CAPTION_STYLES.values()),
            "aspect_ratios": {k: {"width": v[0], "height": v[1]} for k, v in ASPECT_RATIOS.items()},
            "background_modes": BACKGROUND_MODES,
            "keys_configured": {
                "gemini": bool(keys.get("GEMINI_API_KEY")),
            },
        }
    )


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json(force=True) or {}
    status = save_api_keys(
        gemini_api_key=data.get("gemini_api_key", ""),
    )
    return jsonify({"ok": True, "configured": status})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No audio/video file uploaded."}), 400

    up_file = request.files["file"]
    if not up_file.filename:
        return jsonify({"error": "Empty filename."}), 400

    job_id = uuid.uuid4().hex[:10]
    ext = Path(up_file.filename).suffix or ".mp3"
    raw_path = UPLOAD_DIR / f"{job_id}_raw{ext}"
    wav_path = UPLOAD_DIR / f"{job_id}.wav"
    up_file.save(raw_path)

    script_text = request.form.get("script", "")
    engine = request.form.get("engine", "auto")
    chunk_mode = request.form.get("chunk_mode", "three_word")
    use_gemini = request.form.get("use_gemini", "true").lower() == "true"

    try:
        extract_wav_16k(raw_path, wav_path)
        result = transcribe_and_align(wav_path, engine=engine, script_text=script_text)
        segments = group_words_into_segments(result["words"], chunk_mode=chunk_mode)
        if use_gemini:
            segments = enhance_emphasis_with_gemini(segments)

        return jsonify(
            {
                "job_id": job_id,
                "duration": result["duration"],
                "engine": result["engine"],
                "mode": result["mode"],
                "words": result["words"],
                "segments": segments,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/regroup", methods=["POST"])
def api_regroup():
    data = request.get_json(force=True) or {}
    words = data.get("words", [])
    chunk_mode = data.get("chunk_mode", "three_word")
    use_gemini = bool(data.get("use_gemini", False))

    segments = group_words_into_segments(words, chunk_mode=chunk_mode)
    if use_gemini:
        segments = enhance_emphasis_with_gemini(segments)
    return jsonify({"segments": segments})


@app.route("/api/preview-frame", methods=["POST"])
def api_preview_frame():
    data = request.get_json(force=True) or {}
    segment = data.get("segment")
    if not segment:
        segment = {
            "id": 1,
            "start": 0.0,
            "end": 1.5,
            "text": "Just like this",
            "words": [
                {"word": "Just", "start": 0.0, "end": 0.4},
                {"word": "like", "start": 0.4, "end": 0.8},
                {"word": "this", "start": 0.8, "end": 1.5},
            ],
            "emphasis_index": 2,
        }

    aspect_ratio = data.get("aspect_ratio", "9:16")
    width, height = ASPECT_RATIOS.get(aspect_ratio, (1080, 1920))
    bg_mode = data.get("bg_mode", "transparent_mov")
    bg_cfg = BACKGROUND_MODES.get(bg_mode, BACKGROUND_MODES["transparent_mov"])
    bg_rgba = bg_cfg["bg_rgba"] if bg_cfg["bg_rgba"] is not None else (0, 0, 0, 0)

    img = render_caption_frame(
        width=width,
        height=height,
        bg_rgba=bg_rgba,
        segment=segment,
        active_word_idx=int(data.get("active_word_idx", segment.get("emphasis_index", 0))),
        style_id=data.get("style_id", "purple_pill"),
        v_align=data.get("v_align", "bottom"),
        h_align=data.get("h_align", "center"),
        font_size=int(data.get("font_size", 76)),
        offset_x_pct=float(data.get("offset_x_pct", 0.0)),
        offset_y_pct=float(data.get("offset_y_pct", 0.0)),
    )

    # Downscale slightly for fast UI preview
    img.thumbnail((720, 720))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return jsonify({"image_data_url": f"data:image/png;base64,{b64}"})


def _sanitize_filename(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\- ]+", "", (name or "").strip()).strip().replace(" ", "_")
    return cleaned[:80] if cleaned else fallback


@app.route("/api/export-subtitles", methods=["POST"])
def api_export_subtitles():
    data = request.get_json(force=True) or {}
    fmt = data.get("format", "srt").lower()
    segments = data.get("segments", [])
    style_id = data.get("style_id", "purple_pill")
    style = CAPTION_STYLES.get(style_id, CAPTION_STYLES["purple_pill"])
    job_id = data.get("job_id", uuid.uuid4().hex[:8])

    if fmt == "srt":
        content = export_srt(segments, uppercase=style["uppercase"])
    elif fmt == "vtt":
        content = export_vtt(segments, uppercase=style["uppercase"])
    elif fmt == "ass":
        content = export_ass(
            segments,
            style_id=style_id,
            aspect_ratio=data.get("aspect_ratio", "9:16"),
            v_align=data.get("v_align", "bottom"),
            h_align=data.get("h_align", "center"),
            font_size=int(data.get("font_size", 76)),
        )
    elif fmt == "json":
        content = export_json(segments, {"style_id": style_id})
    else:
        return jsonify({"error": f"Unsupported subtitle format: {fmt}"}), 400

    base_name = _sanitize_filename(data.get("output_name", ""), f"captions_{job_id}")
    filename = f"{base_name}.{fmt}"
    out_path = OUTPUT_DIR / filename
    out_path.write_text(content, encoding="utf-8")
    return jsonify({"filename": filename, "download_url": f"/api/download/{filename}", "preview": content[:2000]})


@app.route("/api/render-video", methods=["POST"])
def api_render_video():
    data: Dict[str, Any] = request.get_json(force=True) or {}
    job_id = data.get("job_id", "")
    wav_path = UPLOAD_DIR / f"{job_id}.wav"
    if not job_id or not wav_path.exists():
        return jsonify({"error": "Voiceover audio not found. Please upload and sync audio first."}), 400

    segments = data.get("segments", [])
    duration = float(data.get("duration", 5.0))
    bg_mode = data.get("bg_mode", "transparent_mov")
    bg_cfg = BACKGROUND_MODES.get(bg_mode, BACKGROUND_MODES["transparent_mov"])
    ext = bg_cfg["ext"]

    base_name = _sanitize_filename(data.get("output_name", ""), f"captions_{job_id}_{bg_mode}")
    filename = f"{base_name}.{ext}"
    out_path = OUTPUT_DIR / filename

    try:
        render_caption_video(
            audio_path=wav_path,
            output_path=out_path,
            segments=segments,
            duration=duration,
            style_id=data.get("style_id", "purple_pill"),
            aspect_ratio=data.get("aspect_ratio", "9:16"),
            bg_mode=bg_mode,
            v_align=data.get("v_align", "bottom"),
            h_align=data.get("h_align", "center"),
            font_size=int(data.get("font_size", 76)),
            offset_x_pct=float(data.get("offset_x_pct", 0.0)),
            offset_y_pct=float(data.get("offset_y_pct", 0.0)),
            fps=int(data.get("fps", 30)),
        )
        return jsonify({"filename": filename, "download_url": f"/api/download/{filename}"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/audio/<job_id>", methods=["GET"])
def api_audio(job_id: str):
    safe_id = Path(job_id).name
    wav_path = UPLOAD_DIR / f"{safe_id}.wav"
    if not wav_path.exists():
        return jsonify({"error": "Audio not found."}), 404
    return send_file(wav_path, mimetype="audio/wav")


@app.route("/api/download/<path:filename>", methods=["GET"])
def api_download(filename: str):
    file_path = OUTPUT_DIR / Path(filename).name
    if not file_path.exists():
        return jsonify({"error": "File not found."}), 404
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    load_api_keys()
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
