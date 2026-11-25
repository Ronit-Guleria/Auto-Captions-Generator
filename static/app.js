const ASPECT_DIMS = {
  "9:16": [1080, 1920],
  "16:9": [1920, 1080],
  "1:1": [1080, 1080],
  "4:5": [1080, 1350],
};

const DEMO_SEGMENTS = [
  {
    id: 1,
    start: 0.0,
    end: 2.0,
    text: "Just like this",
    words: [
      { word: "Just", start: 0.0, end: 0.5 },
      { word: "like", start: 0.5, end: 1.0 },
      { word: "this", start: 1.0, end: 2.0 },
    ],
    emphasis_index: 2,
  },
];

const state = {
  inputMode: "script",
  jobId: null,
  duration: 2.0,
  words: [],
  segments: [],
  selectedSegIdx: 0,
  styleId: "purple_pill",
  styles: {},
  isPlaying: false,
  currentTime: 0,
};

const els = {
  keyStatusBadge: document.getElementById("keyStatusBadge"),
  openSettingsBtn: document.getElementById("openSettingsBtn"),
  closeSettingsBtn: document.getElementById("closeSettingsBtn"),
  saveSettingsBtn: document.getElementById("saveSettingsBtn"),
  settingsModal: document.getElementById("settingsModal"),
  geminiKeyInput: document.getElementById("geminiKeyInput"),
  checkKeysBtn: document.getElementById("checkKeysBtn"),
  keyCheckResults: document.getElementById("keyCheckResults"),

  toggleInputBtn: document.getElementById("toggleInputBtn"),
  toggleInputIcon: document.getElementById("toggleInputIcon"),
  inputSectionBody: document.getElementById("inputSectionBody"),

  inputModeTabs: document.getElementById("inputModeTabs"),
  scriptBoxGroup: document.getElementById("scriptBoxGroup"),
  audioFile: document.getElementById("audioFile"),
  clearAudioBtn: document.getElementById("clearAudioBtn"),
  scriptInput: document.getElementById("scriptInput"),

  retryModal: document.getElementById("retryModal"),
  closeRetryBtn: document.getElementById("closeRetryBtn"),
  retryErrorMsg: document.getElementById("retryErrorMsg"),
  retryGeminiBtn: document.getElementById("retryGeminiBtn"),
  switchWhisperBtn: document.getElementById("switchWhisperBtn"),
  engineSelect: document.getElementById("engineSelect"),
  chunkModeSelect: document.getElementById("chunkModeSelect"),
  useGeminiCheck: document.getElementById("useGeminiCheck"),
  syncBtn: document.getElementById("syncBtn"),
  syncStatus: document.getElementById("syncStatus"),

  segCount: document.getElementById("segCount"),
  segmentsList: document.getElementById("segmentsList"),

  safeZoneSelect: document.getElementById("safeZoneSelect"),
  aspectSelect: document.getElementById("aspectSelect"),
  bgModeSelect: document.getElementById("bgModeSelect"),
  vAlignSelect: document.getElementById("vAlignSelect"),
  hAlignSelect: document.getElementById("hAlignSelect"),
  fontSizeSlider: document.getElementById("fontSizeSlider"),
  fontSizeVal: document.getElementById("fontSizeVal"),
  offsetXSlider: document.getElementById("offsetXSlider"),
  offsetXVal: document.getElementById("offsetXVal"),
  offsetYSlider: document.getElementById("offsetYSlider"),
  offsetYVal: document.getElementById("offsetYVal"),

  styleGrid: document.getElementById("styleGrid"),
  previewStage: document.querySelector(".preview-stage"),
  previewFrameBox: document.getElementById("previewFrameBox"),
  previewCanvas: document.getElementById("previewCanvas"),
  safeZoneOverlay: document.getElementById("safeZoneOverlay"),

  previewFullscreenContainer: document.getElementById("previewFullscreenContainer"),
  fullscreenBtn: document.getElementById("fullscreenBtn"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  timelineSeek: document.getElementById("timelineSeek"),
  timecodeLabel: document.getElementById("timecodeLabel"),
  previewAudio: document.getElementById("previewAudio"),

  outputFilenameInput: document.getElementById("outputFilenameInput"),
  renderVideoBtn: document.getElementById("renderVideoBtn"),
  exportStatus: document.getElementById("exportStatus"),
};

let rafId = null;

async function init() {
  const res = await fetch("/api/config");
  const cfg = await res.json();
  cfg.styles.forEach((s) => {
    state.styles[s.id] = s;
  });
  updateKeyBadge(cfg.keys_configured);
  if (!cfg.has_local_whisper) {
    Array.from(els.engineSelect.options).forEach((opt) => {
      if (opt.value === "local") opt.remove();
    });
    els.engineSelect.value = "auto";
  }
  renderStyleGrid();
  bindEvents();
  fitPreviewBox();
  drawCanvasAtTime(0);

  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      fitPreviewBox();
      drawCanvasAtTime(state.currentTime);
    }).observe(els.previewStage);
  }
}

function updateKeyBadge(keys) {
  els.keyStatusBadge.textContent = keys.gemini ? "Gemini API OK" : "No Gemini Key";
}

function renderStyleGrid() {
  els.styleGrid.innerHTML = "";
  Object.values(state.styles).forEach((st) => {
    const card = document.createElement("div");
    card.className = `style-card ${st.id === state.styleId ? "active" : ""}`;
    card.innerHTML = `
      <div class="style-card-title">${st.name}</div>
      <div class="style-card-desc">${st.description}</div>
    `;
    card.addEventListener("click", () => {
      state.styleId = st.id;
      renderStyleGrid();
      drawCanvasAtTime(state.currentTime);
    });
    els.styleGrid.appendChild(card);
  });
}

function setInputCollapsed(collapsed) {
  els.inputSectionBody.classList.toggle("hidden", collapsed);
  els.toggleInputBtn.setAttribute("aria-expanded", String(!collapsed));
  els.toggleInputIcon.innerHTML = collapsed ? "&#9654;" : "&#9660;";
}

function bindEvents() {
  window.addEventListener("resize", () => {
    fitPreviewBox();
    drawCanvasAtTime(state.currentTime);
  });

  els.toggleInputBtn.addEventListener("click", () => {
    const isHidden = els.inputSectionBody.classList.contains("hidden");
    setInputCollapsed(!isHidden);
  });

  els.inputModeTabs.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      els.inputModeTabs.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.inputMode = btn.dataset.mode;
      els.scriptBoxGroup.classList.toggle("hidden", state.inputMode !== "script");
    });
  });

  els.openSettingsBtn.addEventListener("click", () => els.settingsModal.classList.remove("hidden"));
  els.closeSettingsBtn.addEventListener("click", () => els.settingsModal.classList.add("hidden"));
  els.saveSettingsBtn.addEventListener("click", async () => {
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gemini_api_key: els.geminiKeyInput.value,
      }),
    });
    const data = await res.json();
    updateKeyBadge({
      gemini: data.configured.GEMINI_API_KEY,
    });
    els.settingsModal.classList.add("hidden");
  });

  els.checkKeysBtn.addEventListener("click", async () => {
    els.checkKeysBtn.disabled = true;
    els.keyCheckResults.innerHTML = '<div class="hint" style="margin-top:10px;">Testing all API keys...</div>';
    try {
      const res = await fetch("/api/check-keys", { method: "POST" });
      const data = await res.json();
      els.keyCheckResults.innerHTML = (data.results || [])
        .map(
          (item) => `
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:7px 9px;margin-top:6px;background:#0d0f14;border:1px solid #252b3b;border-radius:6px;font-size:12px;">
            <span style="color:#e5e7eb;">${item.name}</span>
            <span style="font-weight:700;color:${item.ok ? "#34d399" : "#f87171"};white-space:nowrap;">${item.ok ? "Working" : item.status}</span>
          </div>`
        )
        .join("");
    } catch (err) {
      els.keyCheckResults.innerHTML = `<div class="status-msg">Error: ${err.message}</div>`;
    } finally {
      els.checkKeysBtn.disabled = false;
    }
  });

  els.audioFile.addEventListener("change", () => {
    els.clearAudioBtn.classList.toggle("hidden", !els.audioFile.files.length);
  });

  els.clearAudioBtn.addEventListener("click", () => {
    els.audioFile.value = "";
    els.clearAudioBtn.classList.add("hidden");
    state.isPlaying = false;
    cancelAnimationFrame(rafId);
    els.playPauseBtn.innerHTML = "&#9654; Play";
    els.previewAudio.pause();
    els.previewAudio.removeAttribute("src");
    els.previewAudio.load();
    state.jobId = null;
    state.words = [];
    state.segments = [];
    state.duration = 2.0;
    state.currentTime = 0;
    els.timelineSeek.max = 2.0;
    els.timelineSeek.value = 0;
    updateTimecodeUI(0);
    els.syncStatus.textContent = "";
    renderSegmentsList();
    drawCanvasAtTime(0);
  });

  els.closeRetryBtn.addEventListener("click", () => els.retryModal.classList.add("hidden"));
  els.syncBtn.addEventListener("click", () => handleSync(null, 0));

  els.chunkModeSelect.addEventListener("change", async () => {
    if (!state.words.length) return;
    const res = await fetch("/api/regroup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        words: state.words,
        chunk_mode: els.chunkModeSelect.value,
        use_gemini: els.useGeminiCheck.checked,
      }),
    });
    const data = await res.json();
    state.segments = data.segments || [];
    state.selectedSegIdx = 0;
    renderSegmentsList();
    drawCanvasAtTime(state.currentTime);
  });

  els.aspectSelect.addEventListener("change", () => {
    fitPreviewBox();
    drawCanvasAtTime(state.currentTime);
  });

  els.safeZoneSelect.addEventListener("change", renderSafeZoneOverlay);

  [els.bgModeSelect, els.vAlignSelect, els.hAlignSelect].forEach((el) =>
    el.addEventListener("change", () => drawCanvasAtTime(state.currentTime))
  );

  els.fontSizeSlider.addEventListener("input", () => {
    els.fontSizeVal.textContent = els.fontSizeSlider.value;
    drawCanvasAtTime(state.currentTime);
  });
  els.offsetXSlider.addEventListener("input", () => {
    els.offsetXVal.textContent = els.offsetXSlider.value;
    drawCanvasAtTime(state.currentTime);
  });
  els.offsetYSlider.addEventListener("input", () => {
    els.offsetYVal.textContent = els.offsetYSlider.value;
    drawCanvasAtTime(state.currentTime);
  });

  // Timeline & Audio Playback
  els.playPauseBtn.addEventListener("click", togglePlayPause);
  els.timelineSeek.addEventListener("input", () => {
    const t = Number(els.timelineSeek.value);
    state.currentTime = t;
    if (els.previewAudio.src) {
      els.previewAudio.currentTime = t;
    }
    updateTimecodeUI(t);
    drawCanvasAtTime(t);
  });

  els.previewAudio.addEventListener("ended", () => {
    state.isPlaying = false;
    els.playPauseBtn.innerHTML = "&#9654; Play";
    cancelAnimationFrame(rafId);
  });

  // Fullscreen Preview (native Fullscreen API + CSS fallback for iOS)
  const syncFullscreenUI = () => {
    const active =
      document.fullscreenElement === els.previewFullscreenContainer ||
      els.previewFullscreenContainer.classList.contains("is-fullscreen-fallback");
    els.fullscreenBtn.innerHTML = active ? "&#x2715; Exit" : "&#x26F6; Fullscreen";
    setTimeout(() => {
      fitPreviewBox();
      drawCanvasAtTime(state.currentTime);
    }, 50);
  };

  const toggleFullscreen = async () => {
    els.fullscreenBtn.blur();
    const box = els.previewFullscreenContainer;
    if (document.fullscreenElement === box) {
      await document.exitFullscreen();
    } else if (box.classList.contains("is-fullscreen-fallback")) {
      box.classList.remove("is-fullscreen-fallback");
      syncFullscreenUI();
    } else if (box.requestFullscreen) {
      try {
        await box.requestFullscreen();
      } catch {
        box.classList.add("is-fullscreen-fallback");
        syncFullscreenUI();
      }
    } else {
      box.classList.add("is-fullscreen-fallback");
      syncFullscreenUI();
    }
  };

  els.fullscreenBtn.addEventListener("click", toggleFullscreen);

  document.addEventListener("fullscreenchange", syncFullscreenUI);

  document.addEventListener("keydown", (e) => {
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toUpperCase() : "";
    const isTyping = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    if (isTyping) return;

    if (e.code === "Space") {
      e.preventDefault();
      if (document.activeElement && typeof document.activeElement.blur === "function") {
        document.activeElement.blur();
      }
      togglePlayPause();
    } else if (e.code === "KeyF" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      toggleFullscreen();
    } else if (e.code === "KeyU" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      setInputCollapsed(false);
      els.audioFile.click();
    } else if (e.code === "Escape" && els.previewFullscreenContainer.classList.contains("is-fullscreen-fallback")) {
      els.previewFullscreenContainer.classList.remove("is-fullscreen-fallback");
      syncFullscreenUI();
    }
  });

  document.querySelectorAll(".sub-export-btn").forEach((btn) => {
    btn.addEventListener("click", () => handleSubtitleExport(btn.dataset.fmt));
  });
  els.renderVideoBtn.addEventListener("click", handleRenderVideo);
}

function fitPreviewBox() {
  const [w, h] = ASPECT_DIMS[els.aspectSelect.value] || [1080, 1920];
  const stageRect = els.previewStage.getBoundingClientRect();
  const availW = Math.max(180, stageRect.width - 24);
  const availH = Math.max(180, stageRect.height - 24);
  const scale = Math.min(availW / w, availH / h);

  els.previewFrameBox.style.width = `${Math.floor(w * scale)}px`;
  els.previewFrameBox.style.height = `${Math.floor(h * scale)}px`;
  els.previewCanvas.width = w;
  els.previewCanvas.height = h;
}

function renderSafeZoneOverlay() {
  const mode = els.safeZoneSelect.value;
  const ov = els.safeZoneOverlay;
  if (mode === "none") {
    ov.classList.add("hidden");
    ov.innerHTML = "";
    return;
  }
  ov.classList.remove("hidden");

  if (mode === "reels" || mode === "shorts" || mode === "tiktok") {
    const label =
      mode === "reels" ? "IG REELS SAFE ZONE" : mode === "shorts" ? "YT SHORTS SAFE ZONE" : "TIKTOK SAFE ZONE";
    const user = mode === "shorts" ? "@YourChannel" : "@yourhandle";
    ov.innerHTML = `
      <svg viewBox="0 0 1080 1920" preserveAspectRatio="none" style="width:100%;height:100%;display:block;">
        <!-- Safe zone dashed border -->
        <rect x="48" y="180" width="840" height="1280" rx="24" fill="none" stroke="rgba(239,68,68,0.6)" stroke-width="5" stroke-dasharray="18 14" />
        <!-- Top badge -->
        <rect x="310" y="54" width="460" height="58" rx="29" fill="rgba(239,68,68,0.85)" />
        <text x="540" y="93" text-anchor="middle" fill="#ffffff" font-family="sans-serif" font-size="30" font-weight="700">${label}</text>
        <!-- Right side action buttons -->
        <circle cx="975" cy="1120" r="44" fill="rgba(0,0,0,0.55)" stroke="rgba(255,255,255,0.4)" stroke-width="3" />
        <text x="975" y="1133" text-anchor="middle" fill="#fff" font-size="36">&#9829;</text>
        <circle cx="975" cy="1260" r="44" fill="rgba(0,0,0,0.55)" stroke="rgba(255,255,255,0.4)" stroke-width="3" />
        <text x="975" y="1272" text-anchor="middle" fill="#fff" font-size="32">&#128172;</text>
        <circle cx="975" cy="1400" r="44" fill="rgba(0,0,0,0.55)" stroke="rgba(255,255,255,0.4)" stroke-width="3" />
        <text x="975" y="1413" text-anchor="middle" fill="#fff" font-size="34">&#10148;</text>
        <circle cx="975" cy="1540" r="44" fill="rgba(0,0,0,0.55)" stroke="rgba(255,255,255,0.4)" stroke-width="3" />
        <text x="975" y="1552" text-anchor="middle" fill="#fff" font-size="34">&#8942;</text>
        <!-- Bottom blocked UI area -->
        <rect x="36" y="1520" width="840" height="340" rx="20" fill="rgba(0,0,0,0.62)" stroke="rgba(255,255,255,0.18)" stroke-width="2" />
        <text x="72" y="1605" fill="#ffffff" font-family="sans-serif" font-size="40" font-weight="700">${user}</text>
        <text x="72" y="1675" fill="rgba(255,255,255,0.78)" font-family="sans-serif" font-size="32">Caption &amp; audio UI area</text>
        <text x="72" y="1730" fill="rgba(239,68,68,0.9)" font-family="sans-serif" font-size="28" font-weight="600">Keep captions inside dashed box</text>
      </svg>
    `;
  } else if (mode === "youtube") {
    ov.innerHTML = `
      <svg viewBox="0 0 1920 1080" preserveAspectRatio="none" style="width:100%;height:100%;display:block;">
        <rect x="80" y="60" width="1760" height="780" rx="20" fill="none" stroke="rgba(239,68,68,0.6)" stroke-width="5" stroke-dasharray="20 14" />
        <rect x="740" y="28" width="440" height="54" rx="27" fill="rgba(239,68,68,0.85)" />
        <text x="960" y="65" text-anchor="middle" fill="#ffffff" font-family="sans-serif" font-size="28" font-weight="700">YOUTUBE 16:9 SAFE ZONE</text>
        <rect x="0" y="900" width="1920" height="180" fill="rgba(0,0,0,0.75)" />
        <rect x="48" y="945" width="1824" height="10" rx="5" fill="rgba(255,255,255,0.28)" />
        <rect x="48" y="945" width="720" height="10" rx="5" fill="#ef4444" />
        <text x="56" y="1020" fill="#ffffff" font-family="sans-serif" font-size="36">&#9654;  0:42 / 3:15</text>
        <text x="1860" y="1020" text-anchor="end" fill="#ffffff" font-family="sans-serif" font-size="34">CC   &#9881;   &#9974;</text>
      </svg>
    `;
  }
}

function togglePlayPause() {
  if (!state.segments.length && !els.previewAudio.src) {
    state.duration = 2.0;
    els.timelineSeek.max = 2.0;
  }

  if (state.isPlaying) {
    state.isPlaying = false;
    els.playPauseBtn.innerHTML = "&#9654; Play";
    if (els.previewAudio.src) els.previewAudio.pause();
    cancelAnimationFrame(rafId);
  } else {
    state.isPlaying = true;
    els.playPauseBtn.innerHTML = "&#10074;&#10074; Pause";
    if (els.previewAudio.src) {
      if (els.previewAudio.currentTime >= state.duration) {
        els.previewAudio.currentTime = 0;
      }
      els.previewAudio.play();
      tickPlayback();
    } else {
      let lastTs = performance.now();
      const demoLoop = (now) => {
        if (!state.isPlaying) return;
        const dt = (now - lastTs) / 1000;
        lastTs = now;
        state.currentTime += dt;
        if (state.currentTime > state.duration) state.currentTime = 0;
        els.timelineSeek.value = state.currentTime;
        updateTimecodeUI(state.currentTime);
        drawCanvasAtTime(state.currentTime);
        rafId = requestAnimationFrame(demoLoop);
      };
      rafId = requestAnimationFrame(demoLoop);
    }
  }
}

function tickPlayback() {
  if (!state.isPlaying) return;
  const t = els.previewAudio.currentTime;
  state.currentTime = t;
  els.timelineSeek.value = t;
  updateTimecodeUI(t);
  drawCanvasAtTime(t);
  rafId = requestAnimationFrame(tickPlayback);
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return `${String(m).padStart(2, "0")}:${s}`;
}

function updateTimecodeUI(t) {
  els.timecodeLabel.textContent = `${fmtTime(t)} / ${fmtTime(state.duration)}`;
}

async function handleSync(overrideEngine = null, retryAttempt = 0) {
  const file = els.audioFile.files[0];
  if (!file) {
    els.syncStatus.textContent = "Please choose an audio or video file first.";
    return;
  }

  const chosenEngine = overrideEngine || els.engineSelect.value;
  const form = new FormData();
  form.append("file", file);
  form.append("engine", chosenEngine);
  form.append("chunk_mode", els.chunkModeSelect.value);
  form.append("use_gemini", els.useGeminiCheck.checked ? "true" : "false");
  if (state.inputMode === "script") {
    form.append("script", els.scriptInput.value || "");
  }

  els.syncBtn.disabled = true;
  els.retryModal.classList.add("hidden");
  els.syncStatus.textContent =
    chosenEngine === "local"
      ? "Transcribing locally with Faster-Whisper..."
      : retryAttempt > 0
      ? `Retrying Gemini (${retryAttempt}/3)...`
      : "Transcribing & syncing voiceover...";

  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      if (data.can_retry && (chosenEngine === "auto" || chosenEngine === "gemini")) {
        const nextAttempt = retryAttempt + 1;
        if (chosenEngine === "auto" && nextAttempt >= 3 && data.has_local_whisper) {
          els.syncStatus.textContent = "Gemini failed 3 times. Shifting automatically to Faster-Whisper...";
          els.syncBtn.disabled = false;
          return await handleSync("local", 0);
        }

        els.retryErrorMsg.textContent = data.error || "Gemini is currently facing high demand.";
        els.retryGeminiBtn.textContent = `Retry Gemini (Attempt ${nextAttempt}/3)`;
        els.retryGeminiBtn.onclick = () => handleSync(chosenEngine, nextAttempt);

        els.switchWhisperBtn.classList.toggle("hidden", !data.has_local_whisper);
        els.switchWhisperBtn.onclick = () => {
          els.retryModal.classList.add("hidden");
          handleSync("local", 0);
        };

        els.retryModal.classList.remove("hidden");
      }
      throw new Error(data.error || "Sync failed");
    }

    state.jobId = data.job_id;
    state.duration = data.duration;
    state.words = data.words;
    state.segments = data.segments;
    state.selectedSegIdx = 0;
    state.currentTime = 0;

    els.previewAudio.src = `/api/audio/${data.job_id}`;
    els.timelineSeek.max = data.duration;
    els.timelineSeek.value = 0;
    updateTimecodeUI(0);

    els.syncStatus.innerHTML = `Synced ${data.segments.length} captions (${data.duration}s) &bull; <strong>Model: ${data.engine}</strong>`;
    renderSegmentsList();
    setInputCollapsed(true);
    drawCanvasAtTime(0);
  } catch (err) {
    els.syncStatus.textContent = `Error: ${err.message}`;
  } finally {
    els.syncBtn.disabled = false;
  }
}

function renderSegmentsList() {
  els.segCount.textContent = state.segments.length;
  if (!state.segments.length) {
    els.segmentsList.innerHTML = '<div class="empty-state">No captions yet.</div>';
    return;
  }

  els.segmentsList.innerHTML = "";
  state.segments.forEach((seg, idx) => {
    const div = document.createElement("div");
    div.className = `seg-item ${idx === state.selectedSegIdx ? "active" : ""}`;
    div.innerHTML = `
      <div class="seg-time">#${seg.id} &bull; ${seg.start.toFixed(2)}s &rarr; ${seg.end.toFixed(2)}s</div>
      <input class="seg-text-input" type="text" value="${seg.text.replace(/"/g, "&quot;")}" />
    `;
    div.addEventListener("click", () => {
      state.selectedSegIdx = idx;
      state.currentTime = seg.start + 0.01;
      els.timelineSeek.value = state.currentTime;
      if (els.previewAudio.src) els.previewAudio.currentTime = state.currentTime;
      updateTimecodeUI(state.currentTime);
      document.querySelectorAll(".seg-item").forEach((el, i) => el.classList.toggle("active", i === idx));
      drawCanvasAtTime(state.currentTime);
    });

    const inp = div.querySelector("input");
    inp.addEventListener("change", () => {
      const newTokens = inp.value.trim().split(/\s+/).filter(Boolean);
      if (!newTokens.length) return;
      seg.text = newTokens.join(" ");
      const span = Math.max(0.1, seg.end - seg.start);
      const step = span / newTokens.length;
      seg.words = newTokens.map((tok, i) => ({
        word: tok,
        start: Number((seg.start + i * step).toFixed(3)),
        end: Number((seg.start + (i + 1) * step).toFixed(3)),
      }));
      seg.emphasis_index = Math.min(seg.emphasis_index || 0, newTokens.length - 1);
      drawCanvasAtTime(state.currentTime);
    });

    els.segmentsList.appendChild(div);
  });
}

function getFontCss(role, sizePx) {
  if (role === "condensed") return `900 ${sizePx}px Impact, "Arial Narrow", sans-serif`;
  if (role === "serif_italic") return `italic 700 ${sizePx}px Georgia, "Times New Roman", serif`;
  if (role === "serif_bold") return `700 ${sizePx}px Georgia, "Times New Roman", serif`;
  return `800 ${sizePx}px "Segoe UI", Arial, sans-serif`;
}

function roundRectPath(ctx, x, y, w, h, r) {
  const rad = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function drawCanvasAtTime(t) {
  const canvas = els.previewCanvas;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const bgMode = els.bgModeSelect.value;
  if (bgMode === "green_screen") {
    ctx.fillStyle = "#00FF00";
    ctx.fillRect(0, 0, W, H);
  } else if (bgMode === "black_bg") {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, W, H);
  }

  const segList = state.segments.length ? state.segments : DEMO_SEGMENTS;
  let seg = null;
  let activeWordIdx = -1;
  for (const s of segList) {
    if (t >= s.start && t <= s.end) {
      seg = s;
      const words = s.words || [];
      for (let i = 0; i < words.length; i++) {
        if (t >= words[i].start && t <= words[i].end) {
          activeWordIdx = i;
          break;
        }
      }
      if (activeWordIdx === -1 && words.length) {
        for (let i = 0; i < words.length; i++) {
          if (t >= words[i].start) activeWordIdx = i;
        }
      }
      break;
    }
  }
  if (!seg || !seg.words || !seg.words.length) return;

  const style = state.styles[state.styleId] || Object.values(state.styles)[0];
  if (!style) return;

  const fontSize = Number(els.fontSizeSlider.value);
  const vAlign = els.vAlignSelect.value;
  const hAlign = els.hAlignSelect.value;
  const offX = (Number(els.offsetXSlider.value) / 100) * W;
  const offY = (Number(els.offsetYSlider.value) / 100) * H;
  const empIdx = seg.emphasis_index ?? 0;
  const highlightIdx =
    style.mode_type === "word_pill" && activeWordIdx >= 0 ? activeWordIdx : empIdx;

  const words = seg.words;
  const texts = words.map((w) => (style.uppercase ? w.word.toUpperCase() : w.word));
  const fonts = words.map((_, i) =>
    i === highlightIdx && style.accent_font_role !== style.font_role
      ? getFontCss(style.accent_font_role, Math.round(fontSize * 1.06))
      : getFontCss(style.font_role, fontSize)
  );

  const spacePx = Math.max(12, Math.round(fontSize * 0.28));
  const maxTextW = W * 0.84;

  const lines = [];
  let curLine = [];
  let curLineW = 0;
  texts.forEach((txt, i) => {
    ctx.font = fonts[i];
    const ww = ctx.measureText(txt).width;
    const added = curLine.length ? spacePx + ww : ww;
    if (curLine.length && curLineW + added > maxTextW) {
      lines.push(curLine);
      curLine = [{ i, txt, ww, font: fonts[i] }];
      curLineW = ww;
    } else {
      curLine.push({ i, txt, ww, font: fonts[i] });
      curLineW += added;
    }
  });
  if (curLine.length) lines.push(curLine);

  const lineH = fontSize * 1.05;
  const lineGap = fontSize * 0.28;
  const totalH = lines.length * lineH + Math.max(0, lines.length - 1) * lineGap;
  const lineWidths = lines.map(
    (ln) => ln.reduce((acc, item) => acc + item.ww, 0) + spacePx * Math.max(0, ln.length - 1)
  );
  const maxLW = Math.max(...lineWidths, 0);

  let baseY;
  if (vAlign === "top") baseY = H * 0.12;
  else if (vAlign === "center") baseY = (H - totalH) / 2;
  else baseY = H * 0.72 - totalH / 2;
  baseY += offY;

  const marginX = W * 0.08;
  const getLineX = (lw) => {
    if (hAlign === "left") return marginX + offX;
    if (hAlign === "right") return W - marginX - lw + offX;
    return (W - lw) / 2 + offX;
  };

  if (style.mode_type === "stacked_bg" && highlightIdx >= 0 && highlightIdx < texts.length) {
    const bgWord = texts[highlightIdx];
    const bigSize = Math.round(fontSize * 2.15);
    ctx.font = getFontCss(style.accent_font_role, bigSize);
    const bw = ctx.measureText(bgWord).width;
    ctx.textBaseline = "middle";
    ctx.fillStyle = style.accent_color;
    ctx.globalAlpha = 0.9;
    ctx.fillText(bgWord, getLineX(bw), baseY + lineH * 0.35);
    ctx.globalAlpha = 1.0;
  }

  if (style.mode_type === "full_box" && style.pill_color) {
    const padX = fontSize * 0.42;
    const padY = fontSize * 0.22;
    const bx = getLineX(maxLW) - padX;
    ctx.fillStyle = style.pill_color;
    roundRectPath(ctx, bx, baseY - padY, maxLW + padX * 2, totalH + padY * 2, fontSize * 0.3);
    ctx.fill();
  }

  ctx.textBaseline = "middle";
  let curY = baseY + lineH / 2;

  lines.forEach((ln, lIdx) => {
    let curX = getLineX(lineWidths[lIdx]);
    ln.forEach(({ i, txt, ww, font }) => {
      const isHl = i === highlightIdx;
      ctx.font = font;

      if (style.mode_type === "word_pill" && isHl && style.pill_color) {
        const padX = fontSize * 0.24;
        const padY = fontSize * 0.16;
        ctx.fillStyle = style.pill_color;
        roundRectPath(
          ctx,
          curX - padX,
          curY - lineH / 2 - padY,
          ww + padX * 2,
          lineH + padY * 2,
          fontSize * 0.28
        );
        ctx.fill();
      }

      if (style.shadow) {
        const sh = Math.max(2, fontSize * 0.05);
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillText(txt, curX + sh, curY + sh);
      }

      if (style.outline_width > 0) {
        ctx.lineWidth = style.outline_width * 2;
        ctx.strokeStyle = style.outline_color;
        ctx.lineJoin = "round";
        ctx.strokeText(txt, curX, curY);
      }

      ctx.fillStyle =
        isHl && style.mode_type !== "stacked_bg" ? style.accent_color : style.text_color;
      ctx.fillText(txt, curX, curY);

      curX += ww + spacePx;
    });
    curY += lineH + lineGap;
  });
}

async function handleSubtitleExport(fmt) {
  if (!state.segments.length) {
    els.exportStatus.textContent = "Please sync a voiceover first.";
    return;
  }
  els.exportStatus.textContent = `Generating .${fmt.toUpperCase()}...`;
  const res = await fetch("/api/export-subtitles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job_id: state.jobId || "demo",
      output_name: els.outputFilenameInput.value,
      format: fmt,
      segments: state.segments,
      style_id: state.styleId,
      aspect_ratio: els.aspectSelect.value,
      v_align: els.vAlignSelect.value,
      h_align: els.hAlignSelect.value,
      font_size: Number(els.fontSizeSlider.value),
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    els.exportStatus.textContent = `Error: ${data.error}`;
    return;
  }
  els.exportStatus.innerHTML = `Ready: <a href="${data.download_url}" download="${data.filename}">${data.filename}</a>`;
}

async function handleRenderVideo() {
  if (!state.jobId || !state.segments.length) {
    els.exportStatus.textContent = "Please upload & sync a voiceover first.";
    return;
  }

  els.renderVideoBtn.disabled = true;
  els.exportStatus.textContent = "Rendering transparent/styled video with FFmpeg (this may take a moment)...";

  try {
    const res = await fetch("/api/render-video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: state.jobId,
        output_name: els.outputFilenameInput.value,
        duration: state.duration,
        segments: state.segments,
        style_id: state.styleId,
        aspect_ratio: els.aspectSelect.value,
        bg_mode: els.bgModeSelect.value,
        v_align: els.vAlignSelect.value,
        h_align: els.hAlignSelect.value,
        font_size: Number(els.fontSizeSlider.value),
        offset_x_pct: Number(els.offsetXSlider.value),
        offset_y_pct: Number(els.offsetYSlider.value),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Render failed");

    els.exportStatus.innerHTML = `Video Ready: <a href="${data.download_url}" download="${data.filename}">${data.filename}</a>`;
  } catch (err) {
    els.exportStatus.textContent = `Render Error: ${err.message}`;
  } finally {
    els.renderVideoBtn.disabled = false;
  }
}

init();
