let cases = [];
let filteredCases = [];
let activeCaseId = null;
let activeOverlayMode = "combined";
let streamMode = false;

const els = {
  metricsGrid: document.getElementById("metricsGrid"),
  caseList: document.getElementById("caseList"),
  caseTitle: document.getElementById("caseTitle"),
  caseSubtitle: document.getElementById("caseSubtitle"),
  scoreBadge: document.getElementById("scoreBadge"),
  explanationText: document.getElementById("explanationText"),
  componentBadges: document.getElementById("componentBadges"),
  testClip: document.getElementById("testClip"),
  testClipGif: document.getElementById("testClipGif"),
  overlayFrame: document.getElementById("overlayFrame"),
  overlayModes: document.getElementById("overlayModes"),
  clipMeta: document.getElementById("clipMeta"),
  instrumentMeta: document.getElementById("instrumentMeta"),
  componentMeters: document.getElementById("componentMeters"),
  directionWheel: document.getElementById("directionWheel"),
  speedRays: document.getElementById("speedRays"),
  motionBar: document.getElementById("motionBar"),
  timelineChart: document.getElementById("timelineChart"),
  timelineMeta: document.getElementById("timelineMeta"),
  videoFilter: document.getElementById("videoFilter"),
  reasonFilter: document.getElementById("reasonFilter"),
  sortSelect: document.getElementById("sortSelect"),
  playAllButton: document.getElementById("playAllButton"),
  exportButton: document.getElementById("exportButton"),
  refreshButton: document.getElementById("refreshButton"),
};

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function currentCase() {
  return cases.find((item) => item.case_id === activeCaseId) || filteredCases[0] || cases[0] || null;
}

function renderMetrics(metrics) {
  const frame = metrics.frame_auc || {};
  const spatial = metrics.spatial_mask_auc || {};
  const items = [
    ["Frame AUC", fmt(frame.global_roc_auc || spatial.frame_roc_auc_eval_style)],
    ["Frame AP", fmt(frame.global_average_precision || spatial.frame_average_precision)],
    ["Spatial Pixel AUC", fmt(spatial.pixel_roc_auc_sampled)],
    ["Loaded Cases", String(cases.length)],
  ];
  els.metricsGrid.innerHTML = items.map(([label, value]) => `
    <div class="metric">
      <div class="metric-value">${escapeHtml(value)}</div>
      <div class="metric-label">${escapeHtml(label)}</div>
    </div>
  `).join("");
}

function populateFilters() {
  const current = els.videoFilter.value || "all";
  const videos = [...new Set(cases.map((item) => String(item.video_id).padStart(2, "0")))].sort();
  els.videoFilter.innerHTML = [
    `<option value="all">All videos</option>`,
    ...videos.map((video) => `<option value="${video}">Video ${video}</option>`),
  ].join("");
  els.videoFilter.value = videos.includes(current) ? current : "all";
}

function applyFilters() {
  const video = els.videoFilter.value;
  const reason = els.reasonFilter.value;
  const sort = els.sortSelect.value;

  filteredCases = cases.filter((item) => {
    const itemVideo = String(item.video_id).padStart(2, "0");
    const videoOk = video === "all" || itemVideo === video;
    const reasonOk = reason === "all" || item.main_reason === reason;
    return videoOk && reasonOk;
  });

  filteredCases.sort((a, b) => {
    if (sort === "video_frame") {
      return String(a.video_id).localeCompare(String(b.video_id)) || Number(a.start_frame) - Number(b.start_frame);
    }
    if (sort === "reason") {
      return String(a.main_reason).localeCompare(String(b.main_reason)) || Number(b.anomaly_score) - Number(a.anomaly_score);
    }
    return Number(b.anomaly_score) - Number(a.anomaly_score);
  });

  if (!filteredCases.some((item) => item.case_id === activeCaseId)) {
    activeCaseId = filteredCases[0]?.case_id || cases[0]?.case_id || null;
  }
  renderCaseList();
}

function renderCaseList() {
  if (!filteredCases.length) {
    els.caseList.innerHTML = `<div class="empty-state">No cases match the current filters.</div>`;
    return;
  }

  els.caseList.innerHTML = filteredCases.map((item) => {
    const active = item.case_id === activeCaseId ? "active" : "";
    return `
      <button class="case-item ${active}" data-case-id="${escapeHtml(item.case_id)}" type="button">
        <div class="case-title">
          <span>${escapeHtml(item.case_id)}</span>
          <strong>${fmt(item.anomaly_score)}</strong>
        </div>
        <div class="case-meta">
          video ${escapeHtml(item.video_id)} · frames ${escapeHtml(item.frame_range)} · region ${escapeHtml(item.region_id)}
        </div>
        <div class="reason-pill ${escapeHtml(item.main_reason)}">${escapeHtml(item.main_reason)}</div>
      </button>
    `;
  }).join("");

  for (const button of els.caseList.querySelectorAll(".case-item")) {
    button.addEventListener("click", () => {
      streamMode = false;
      showCase(button.dataset.caseId);
    });
  }
}

function setMedia(el, url) {
  if (!url) {
    el.removeAttribute("src");
    return;
  }
  el.src = `${url}${url.includes("?") ? "&" : "?"}v=${Date.now()}`;
}

function setClip(videoEl, gifEl, videoUrl, gifUrl) {
  setMedia(gifEl, gifUrl);
  setMedia(videoEl, videoUrl);
  if (gifUrl) {
    gifEl.style.display = "block";
    videoEl.classList.add("has-gif-fallback");
  } else {
    gifEl.style.display = "none";
    videoEl.classList.remove("has-gif-fallback");
  }
}

function showCase(caseId) {
  const item = cases.find((entry) => entry.case_id === caseId) || currentCase();
  if (!item) return;
  activeCaseId = item.case_id;

  els.caseTitle.textContent = `Video ${item.video_id} · region ${item.region_id}`;
  els.caseSubtitle.textContent = `frames ${item.frame_range} · ${item.volume_id}`;
  els.scoreBadge.textContent = `score ${fmt(item.anomaly_score)}`;
  els.explanationText.textContent = item.plain_english_explanation;
  els.clipMeta.textContent = `focus frame ${item.frame_focus}, ${item.main_reason} dominant`;
  els.instrumentMeta.textContent = `appearance, direction, speed, and background contributions`;

  renderComponentBadges(item);
  renderComponentMeters(item);
  renderMotionMeters(item);
  setClip(els.testClip, els.testClipGif, item.test_clip_url, item.test_clip_gif_url);
  renderOverlay(item);
  renderCaseList();
  loadTimeline(item.video_id, item.case_id).catch(() => {
    els.timelineChart.innerHTML = `<div class="empty-state">Timeline unavailable for video ${escapeHtml(item.video_id)}.</div>`;
  });
}

function renderOverlay(item) {
  const url = item[`${activeOverlayMode}_overlay_url`] || item.combined_overlay_url;
  setMedia(els.overlayFrame, url);
  for (const button of els.overlayModes.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.mode === activeOverlayMode);
  }
}

function renderComponentBadges(item) {
  const badges = item.component_badges || [];
  els.componentBadges.innerHTML = badges.map((badge) => `
    <span class="component-badge ${escapeHtml(badge.name)} ${escapeHtml(badge.level)}">
      ${escapeHtml(badge.name)} ${fmt(badge.distance, 2)}
    </span>
  `).join("");
}

function renderComponentMeters(item) {
  const badges = item.component_badges || [];
  els.componentMeters.innerHTML = `
    <h4>Distance Breakdown</h4>
    ${badges.map((badge) => `
      <div class="meter-row">
        <span>${escapeHtml(badge.name)}</span>
        <div class="meter-track"><div style="width:${clamp(Number(badge.share) * 100, 3, 100)}%"></div></div>
        <strong>${fmt(badge.distance, 2)}</strong>
      </div>
    `).join("")}
  `;
}

function motionArray(attrs, prefix) {
  return Array.from({ length: 12 }, (_, idx) => Number(attrs?.[`${prefix}_${String(idx).padStart(2, "0")}`] || 0));
}

function polarPoint(cx, cy, radius, angle) {
  return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
}

function renderWheel(values, title, className) {
  const maxValue = Math.max(...values, 0.0001);
  const rays = values.map((value, idx) => {
    const angle = -Math.PI / 2 + idx * (Math.PI * 2 / 12);
    const radius = 18 + 58 * (value / maxValue);
    const [x2, y2] = polarPoint(80, 80, radius, angle);
    const width = 2 + 5 * (value / maxValue);
    return `<line x1="80" y1="80" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke-width="${width.toFixed(1)}" />`;
  }).join("");

  return `
    <h4>${escapeHtml(title)}</h4>
    <svg class="${className}" viewBox="0 0 160 160" role="img">
      <circle cx="80" cy="80" r="64"></circle>
      ${rays}
      <circle cx="80" cy="80" r="4"></circle>
    </svg>
  `;
}

function renderMotionMeters(item) {
  const attrs = item.motion_attributes || {};
  const angles = motionArray(attrs, "motion_angle_hist");
  const speeds = motionArray(attrs, "motion_speed");
  const moving = clamp(Number(attrs.motion_moving_fraction || 0), 0, 1);
  const stationary = clamp(Number(attrs.motion_stationary_fraction || (1 - moving)), 0, 1);

  els.directionWheel.innerHTML = renderWheel(angles, "Direction Wheel", "direction-svg");
  els.speedRays.innerHTML = renderWheel(speeds, "Speed Rays", "speed-svg");
  els.motionBar.innerHTML = `
    <h4>Motion State</h4>
    <div class="motion-stack">
      <div class="moving" style="width:${moving * 100}%">moving ${fmt(moving, 2)}</div>
      <div class="stationary" style="width:${stationary * 100}%">stationary ${fmt(stationary, 2)}</div>
    </div>
    <dl>
      <div><dt>Mean magnitude</dt><dd>${fmt(attrs.motion_mean_magnitude, 2)}</dd></div>
      <div><dt>Max magnitude</dt><dd>${fmt(attrs.motion_max_magnitude, 2)}</dd></div>
      <div><dt>Background class</dt><dd>${Number(attrs.motion_background_cls || 0) ? "yes" : "no"}</dd></div>
    </dl>
  `;
}

async function loadTimeline(videoId, selectedCaseId) {
  const data = await getJson(`/api/videos/${String(videoId).padStart(2, "0")}/timeline`);
  renderTimeline(data, selectedCaseId);
}

function renderTimeline(data, selectedCaseId) {
  const width = 1000;
  const height = 190;
  const pad = 28;
  const frames = data.frames || [];
  const maxFrame = Math.max(...frames.map((f) => Number(f.frame)), 1);
  const minScore = Number(data.score_min || 0);
  const maxScore = Number(data.score_max || 1);
  const scoreRange = Math.max(0.0001, maxScore - minScore);
  const xScale = (frame) => pad + (Number(frame) - 1) / Math.max(1, maxFrame - 1) * (width - pad * 2);
  const yScale = (score) => height - pad - (Number(score) - minScore) / scoreRange * (height - pad * 2);

  const points = frames.map((f) => `${xScale(f.frame).toFixed(1)},${yScale(f.score).toFixed(1)}`).join(" ");
  const gtRects = (data.gt_intervals || []).map((interval) => {
    const x = xScale(interval.start);
    const w = Math.max(2, xScale(interval.end) - x);
    return `<rect class="gt-interval" x="${x.toFixed(1)}" y="${pad}" width="${w.toFixed(1)}" height="${height - pad * 2}" />`;
  }).join("");
  const markers = (data.cases || []).map((item) => {
    const active = item.case_id === selectedCaseId ? "active" : "";
    return `
      <circle class="case-marker ${active}" data-case-id="${escapeHtml(item.case_id)}"
        cx="${xScale(item.frame_focus).toFixed(1)}" cy="${yScale(item.score).toFixed(1)}" r="${active ? 8 : 6}">
      </circle>
    `;
  }).join("");

  els.timelineMeta.textContent = `video ${data.video_id}, ${frames.length} frames, max score ${fmt(data.score_max)}`;
  els.timelineChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img">
      ${gtRects}
      <line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line>
      <line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}"></line>
      <polyline class="score-line" points="${points}"></polyline>
      ${markers}
    </svg>
  `;

  for (const marker of els.timelineChart.querySelectorAll(".case-marker")) {
    marker.addEventListener("click", () => {
      streamMode = false;
      showCase(marker.dataset.caseId);
    });
  }
}

function wireControls() {
  for (const el of [els.videoFilter, els.reasonFilter, els.sortSelect]) {
    el.addEventListener("change", () => {
      streamMode = false;
      applyFilters();
      showCase(activeCaseId);
    });
  }

  for (const button of els.overlayModes.querySelectorAll("button")) {
    button.addEventListener("click", () => {
      activeOverlayMode = button.dataset.mode;
      const item = currentCase();
      if (item) renderOverlay(item);
    });
  }

  els.playAllButton.addEventListener("click", () => {
    if (!filteredCases.length) return;
    streamMode = true;
    showCase(filteredCases[0].case_id);
    startStreamTimer();
  });

  els.exportButton.addEventListener("click", () => {
    const item = currentCase();
    if (!item) return;
    window.open(item.report_url, "_blank", "noopener");
  });

  els.refreshButton.addEventListener("click", () => {
    load();
  });
}

function startStreamTimer() {
  window.clearTimeout(window.__caseStreamTimer);
  if (!streamMode || !filteredCases.length) return;
  window.__caseStreamTimer = window.setTimeout(() => {
    const idx = filteredCases.findIndex((item) => item.case_id === activeCaseId);
    const next = filteredCases[(idx + 1) % filteredCases.length];
    showCase(next.case_id);
    startStreamTimer();
  }, 2600);
}

async function load() {
  const [caseData, metrics] = await Promise.all([
    getJson("/api/cases"),
    getJson("/api/metrics"),
  ]);
  cases = caseData;
  activeCaseId = cases[0]?.case_id || null;
  populateFilters();
  renderMetrics(metrics);
  applyFilters();
  showCase(activeCaseId);
}

wireControls();
load().catch((err) => {
  els.caseTitle.textContent = "Dashboard failed to load";
  els.explanationText.textContent = err.message;
});
