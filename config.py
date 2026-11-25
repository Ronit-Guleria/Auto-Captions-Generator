"""Configuration, environment persistence, aspect ratios, and font resolution."""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
FONTS_DIR = BASE_DIR / "fonts"
ENV_FILE = BASE_DIR / ".env"

for _d in (UPLOAD_DIR, OUTPUT_DIR, FONTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}

BACKGROUND_MODES = {
    "transparent_mov": {
        "label": "Transparent (.mov ProRes 4444)",
        "ext": "mov",
        "bg_rgba": None,
        "description": "True alpha channel — Premiere, DaVinci, CapCut Desktop",
    },
    "transparent_webm": {
        "label": "Transparent Light (.webm VP9 Alpha)",
        "ext": "webm",
        "bg_rgba": None,
        "description": "Smaller transparent file — DaVinci, Web, OBS",
    },
    "green_screen": {
        "label": "Green Screen (.mp4)",
        "ext": "mp4",
        "bg_rgba": (0, 255, 0, 255),
        "description": "Chroma-key green (#00FF00) — Mobile & all editors",
    },
    "black_bg": {
        "label": "Black Background (.mp4)",
        "ext": "mp4",
        "bg_rgba": (0, 0, 0, 255),
        "description": "Pure black (#000000) — Screen/Add blend mode",
    },
}

WINDOWS_FONTS_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

FONT_CANDIDATES: Dict[str, list[str]] = {
    "sans_bold": ["Montserrat-ExtraBold.ttf", "Poppins-Bold.ttf", "segoeuib.ttf", "arialbd.ttf"],
    "sans_regular": ["Montserrat-SemiBold.ttf", "segoeui.ttf", "arial.ttf"],
    "condensed": ["Anton-Regular.ttf", "BebasNeue-Regular.ttf", "impact.ttf", "ARIALNB.TTF"],
    "serif_italic": ["PlayfairDisplay-Italic.ttf", "georgiai.ttf", "timesi.ttf"],
    "serif_bold": ["PlayfairDisplay-Bold.ttf", "georgiab.ttf", "timesbd.ttf"],
    "script": ["segoescb.ttf", "segoeprb.ttf", "georgiai.ttf"],
}


LINUX_FALLBACK_FONTS: Dict[str, list[str]] = {
    "sans_bold": ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "sans_regular": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "condensed": ["/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "serif_italic": ["/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf"],
    "serif_bold": ["/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"],
    "script": ["/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"],
}


def resolve_font(role: str) -> str:
    """Return the first existing font path for a logical role."""
    candidates = FONT_CANDIDATES.get(role, FONT_CANDIDATES["sans_bold"])
    for name in candidates:
        local_path = FONTS_DIR / name
        if local_path.exists():
            return str(local_path)
        win_path = WINDOWS_FONTS_DIR / name
        if win_path.exists():
            return str(win_path)
    for linux_path in LINUX_FALLBACK_FONTS.get(role, LINUX_FALLBACK_FONTS["sans_bold"]):
        if Path(linux_path).exists():
            return linux_path
    fallback = WINDOWS_FONTS_DIR / "arial.ttf"
    return str(fallback) if fallback.exists() else ""


def load_api_keys() -> Dict[str, str]:
    """Load API keys from .env file and environment variables."""
    keys = {
        "CF_ACCOUNT_ID": os.environ.get("CF_ACCOUNT_ID", ""),
        "CF_API_TOKEN": os.environ.get("CF_API_TOKEN", ""),
        "CF_ELEVENLABS_TOKEN": os.environ.get("CF_ELEVENLABS_TOKEN", ""),
        "CF_UNIVERSAL_TOKEN": os.environ.get("CF_UNIVERSAL_TOKEN", ""),
        "ELEVENLABS_API_KEY": os.environ.get("ELEVENLABS_API_KEY", ""),
        "UNIVERSAL_3.5_PRO": os.environ.get("UNIVERSAL_3.5_PRO", ""),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
    }
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in keys and v:
                keys[k] = v
                os.environ[k] = v
    return keys


def save_api_keys(gemini_api_key: str = "") -> Dict[str, bool]:
    """Persist API keys to .env and update process environment."""
    current = load_api_keys()
    if gemini_api_key.strip():
        current["GEMINI_API_KEY"] = gemini_api_key.strip()

    lines = [f"{k}={v}" for k, v in current.items() if v]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in current.items():
        if v:
            os.environ[k] = v
    return {k: bool(v) for k, v in current.items()}
