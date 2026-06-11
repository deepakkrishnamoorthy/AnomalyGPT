let cases = [];
let activeIndex = 0;
let streamMode = false;

const els = {
  metricsGrid: document.getElementById("metricsGrid"),
  caseList: document.getElementById("caseList"),
  caseTitle: document.getElementById("caseTitle"),
  caseSubtitle: document.getElementById("caseSubtitle"),
  scoreBadge: document.getElementById("scoreBadge"),
  explanationText: document.getElementById("explanationText"),
  testClip: document.getElementById("testClip"),
  exemplarClip: document.getElementById("exemplarClip"),
  testClipGif: document.getElementById("testClipGif"),
  exemplarClipGif: document.getElementById("exemplarClipGif"),
  testSheet: document.getElementById("testSheet"),
  exemplarSheet: document.getElementById("exemplarSheet"),
  instrumentPanel: document.getElementById("instrumentPanel"),
  playAllButton: document.getElementById("playAllButton"),
  refreshButton: document.getElementById("refreshButton"),
};

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function renderMetrics(metrics) {
  const frame = metrics.frame_auc || {};
  const spatial = metrics.spatial_mask_auc || {};
  const items = [
    ["Frame AUC", fmt(frame.global_roc_auc || spatial.frame_roc_auc_eval_style)],
    ["Frame AP", fmt(frame.global_average_precision || spatial.frame_average_precision)],
    ["Spatial Pixel AUC", fmt(spatial.pixel_roc_auc_sampled)],
    ["Cases", String(cases.length)],
  ];
  els.metricsGrid.innerHTML = items.map(([label, value]) => `
    <div class="metric">
      <div class="metric-value">${value}</div>
      <div class="metric-label">${label}</div>
    </div>
  `).join("");
}

function renderCaseList() {
  els.caseList.innerHTML = cases.map((item, idx) => `
    <button class="case-item ${idx === activeIndex ? "active" : ""}" data-index="${idx}" type="button">
      <div class="case-title">${item.case_id} · score ${fmt(item.anomaly_score)}</div>
      <div class="case-meta">
        video ${item.video_id}, frames ${item.start_frame}-${item.end_frame}<br>
        region ${item.region_id}, reason ${item.main_reason}
      </div>
    </button>
  `).join("");

  for (const button of els.caseList.querySelectorAll(".case-item")) {
    button.addEventListener("click", () => {
      streamMode = false;
      showCase(Number(button.dataset.index), false);
    });
  }
}

function setMedia(el, url) {
  if (!url) {
    el.removeAttribute("src");
    return;
  }
  el.src = url;
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

function showCase(index, autoplay) {
  if (!cases.length) return;
  activeIndex = ((index % cases.length) + cases.length) % cases.length;
  const item = cases[activeIndex];

  els.caseTitle.textContent = `${item.case_id}: ${item.volume_id}`;
  els.caseSubtitle.textContent = `nearest normal ${item.nearest_exemplar_volume_id}`;
  els.scoreBadge.textContent = `score ${fmt(item.anomaly_score)}`;
  els.explanationText.textContent = item.plain_english_explanation;

  setClip(els.testClip, els.testClipGif, item.test_clip_url, item.test_clip_gif_url);
  setClip(els.exemplarClip, els.exemplarClipGif, item.nearest_exemplar_clip_url, item.nearest_exemplar_clip_gif_url);
  setMedia(els.testSheet, item.test_sheet_url);
  setMedia(els.exemplarSheet, item.nearest_exemplar_sheet_url);
  setMedia(els.instrumentPanel, item.instrument_panel_url);

  renderCaseList();

  if (autoplay && item.test_clip_url && !item.test_clip_gif_url) {
    els.testClip.currentTime = 0;
    els.testClip.play().catch(() => {});
  }
}

function wireStreamPlayback() {
  els.testClip.addEventListener("ended", () => {
    if (!streamMode) return;
    showCase(activeIndex + 1, true);
  });

  els.playAllButton.addEventListener("click", () => {
    if (!cases.length) return;
    streamMode = true;
    showCase(0, false);
    startGifStreamTimer();
  });

  els.refreshButton.addEventListener("click", () => {
    load();
  });
}

function startGifStreamTimer() {
  if (!streamMode) return;
  window.clearTimeout(window.__caseStreamTimer);
  window.__caseStreamTimer = window.setTimeout(() => {
    if (!streamMode) return;
    showCase(activeIndex + 1, false);
    startGifStreamTimer();
  }, 2400);
}

async function load() {
  const [caseData, metrics] = await Promise.all([
    getJson("/api/cases"),
    getJson("/api/metrics"),
  ]);
  cases = caseData;
  activeIndex = 0;
  renderMetrics(metrics);
  renderCaseList();
  showCase(0, false);
}

wireStreamPlayback();
load().catch((err) => {
  els.caseTitle.textContent = "Dashboard failed to load";
  els.explanationText.textContent = err.message;
});
