"use strict";

const $ = (selector) => document.querySelector(selector);
const STAGES = [
  { id: "ingestion", name: "Video Ingestion", detail: "Upload and validate the video", color: "#a855f7", inputs: ["Uploaded video file"], outputs: ["Video metadata", "Normalized video", "Extracted audio"] },
  { id: "asr", name: "Audio & ASR", detail: "Extract audio and transcribe speech", color: "#3b82f6", inputs: ["Normalized audio"], outputs: ["Timestamped transcript", "Subtitle segments"] },
  { id: "diarization", name: "Speaker Diarization", detail: "Identify speaker turns", parallelGroup: "audio-scene", parallelWith: "phase4", inputs: ["Audio", "Transcript"], outputs: ["Speaker turns", "RTTM diarization"] },
  { id: "phase4", name: "Scene Detection", detail: "Detect scenes and shot boundaries", parallelGroup: "audio-scene", parallelWith: "diarization", inputs: ["Normalized video", "Metadata"], outputs: ["Shot boundaries", "Candidate frames"] },
  { id: "phase5", name: "Character Identification", detail: "Detect and track characters", parallelGroup: "visual-analysis", parallelWith: "phase6", inputs: ["Candidate frames", "Shot boundaries"], outputs: ["Face observations", "Character tracks"] },
  { id: "phase6", name: "Semantic Alignment", detail: "Match dialogue with visual scenes", parallelGroup: "visual-analysis", parallelWith: "phase5", inputs: ["Transcript", "Candidate frames"], outputs: ["Dialogue-to-frame alignment"] },
  { id: "phase7", name: "Active Speaker Matching", detail: "Match voices to visible speakers", color: "#ec4899", inputs: ["Speaker turns", "Character tracks"], outputs: ["Speaker-to-character map"] },
  { id: "phase8", name: "Keyframe Selection", detail: "Select representative frames", parallelGroup: "story-analysis", parallelWith: "phase9", inputs: ["Visual alignment", "Speaker map"], outputs: ["Selected keyframes", "Selection metrics"] },
  { id: "phase9", name: "Emotion Analysis", detail: "Estimate dialogue emotion", parallelGroup: "story-analysis", parallelWith: "phase8", inputs: ["Transcript", "Audio", "Speaker map"], outputs: ["Emotion annotations"] },
  { id: "phase10", name: "Comic Layout", detail: "Organize panels and story flow", color: "#06b6d4", inputs: ["Keyframes", "Emotion annotations"], outputs: ["Page and panel layout"] },
  { id: "phase11", name: "Visual Rendering", detail: "Style panels and balloons", color: "#22c55e", inputs: ["Page layout", "Dialogue and speakers"], outputs: ["Styled panels", "Speech balloons"] },
  { id: "phase12", name: "Comic Composition", detail: "Add dialogue and export pages", color: "#d946ef", inputs: ["Styled panels", "Page layout"], outputs: ["Page previews", "Comic PDF", "Validation metrics"] },
];
const PIPELINE_GROUPS = [
  { name: "Video input", detail: "Starts the run by validating and preparing the clip.", mode: "sequence", stages: ["ingestion"] },
  { name: "Transcript", detail: "Runs after input. Empty speech selects the visual-only route.", mode: "sequence", stages: ["asr"] },
  { name: "Scene preparation", detail: "Runs after ASR. Diarization joins scene detection when speech exists.", mode: "parallel", stages: ["diarization", "phase4"] },
  { name: "Visual analysis", detail: "Runs after scene detection; tracking and semantic analysis execute together.", mode: "parallel", stages: ["phase5", "phase6"] },
  { name: "Speaker linking", detail: "Waits for diarization and character tracking when dialogue exists.", mode: "sequence", stages: ["phase7"] },
  { name: "Story analysis", detail: "Runs after speaker linking; emotion analysis is skipped when dialogue is absent.", mode: "parallel", stages: ["phase8", "phase9"] },
  { name: "Comic production", detail: "Runs in order: layout, panel styling, then page composition.", mode: "sequence", stages: ["phase10", "phase11", "phase12"] },
];

const state = {
  file: null,
  jobId: localStorage.getItem("cinecomic-job-id"),
  job: null,
  pollHandle: null,
  pollInFlight: false,
  pageNames: [],
  pageIndex: 0,
  previewJobId: "",
  previewRequest: null,
  selectedStage: "",
  serverConfig: null,
  details: null,
  detailsJobId: "",
  analysisTab: "dialogue",
};

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function stageState(stage, job) {
  if (!job) return "queued";
  const reported = job.stage_statuses?.[stage.id]
    || job.stages?.find((item) => item.id === stage.id)?.status;
  if (["completed", "running", "failed", "skipped"].includes(reported)) return reported;
  if ((job.completed_stages || []).includes(stage.id)) return "completed";
  if (job.failed_stage === stage.id) return "failed";
  if ((job.running_stages || []).includes(stage.id)) return "running";
  const apiStage = job.stages?.find((item) => item.id === stage.id);
  if (apiStage?.status === "running") return "running";
  if (job.current_stage === stage.id && ["running", "starting"].includes(job.status)) return "running";
  return "queued";
}

function renderStages(job = state.job) {
  const grid = $("#stage-grid");
  grid.innerHTML = PIPELINE_GROUPS.map((group, groupIndex) => {
    const cards = group.stages.map((stageId) => {
      const stage = STAGES.find((item) => item.id === stageId);
      if (!stage) return "";
      const index = STAGES.indexOf(stage);
      const status = stageState(stage, job);
      const label = status === "completed" ? "Complete" : status === "running" ? "Running" : status === "failed" ? "Failed" : status === "skipped" ? "Skipped" : "Queued";
      return `<button class="stage-card ${status}${state.selectedStage === stage.id ? " selected" : ""}" data-stage="${stage.id}" style="--stage-color:${stage.color}" aria-label="Stage ${index + 1}: ${escapeHTML(stage.name)}, ${label}${group.mode === "parallel" ? ", runs in parallel" : ""}">
        <div class="stage-top"><span class="stage-number">${String(index + 1).padStart(2, "0")}</span><span class="stage-state">${status === "completed" ? "✓ " : status === "running" ? "◌ " : status === "failed" ? "! " : status === "skipped" ? "↷ " : ""}${label}</span></div>
        <div class="stage-title">${escapeHTML(stage.name)}</div><div class="stage-description">${escapeHTML(stage.detail)}</div>
      </button>`;
    }).join("");
    const mode = group.mode === "parallel" ? "Parallel" : group.stages.length > 1 ? "In order" : "Stage";
    const stageLayout = group.mode === "parallel" ? "parallel-stages" : group.stages.length > 1 ? "ordered-chain" : "single-stage";
    return `<section class="pipeline-stage-group ${group.mode === "parallel" ? "parallel" : "sequential"}" aria-label="${escapeHTML(group.name)} group ${mode.toLowerCase()}">
      <div class="pipeline-group-heading"><span class="pipeline-group-number">${String(groupIndex + 1).padStart(2, "0")}</span><div><strong>${escapeHTML(group.name)}</strong><small>${escapeHTML(group.detail)}</small></div><span class="pipeline-group-mode">${mode}</span></div>
      <div class="pipeline-group-stages ${stageLayout}">${cards}</div>
    </section>`;
  }).join("");
  grid.querySelectorAll("[data-stage]").forEach((card) => card.addEventListener("click", () => {
    state.selectedStage = card.dataset.stage;
    renderStages(state.job);
    renderStageDetails(card.dataset.stage);
  }));
}

function renderActiveStageOutputs(job = state.job) {
  const panel = $("#active-stage-outputs");
  if (!panel) return;
  const activeIds = [...new Set(job?.running_stages || [])];
  if (!activeIds.length && ["running", "starting"].includes(job?.status)
    && STAGES.some((stage) => stage.id === job?.current_stage)) {
    activeIds.push(job.current_stage);
  }
  if (!activeIds.length) {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }

  const activeStages = activeIds.map((id) => STAGES.find((stage) => stage.id === id)).filter(Boolean);
  panel.hidden = false;
  panel.innerHTML = `<div class="active-outputs-heading"><div><span class="eyebrow">LIVE WORKERS</span><h3>Output from running stages</h3><p>Each console updates as its stage writes new output.</p></div><span class="live-worker-count">${activeStages.length} active</span></div>
    <div class="active-output-grid">${activeStages.map((stage) => {
      const log = job.stage_logs?.[stage.id] || "Waiting for this stage's first log line…";
      const recentLines = log.split(/\r?\n/).slice(-24).join("\n");
      return `<article class="active-output-card" style="--stage-color:${stage.color}">
        <div class="active-output-title"><span><i></i>${escapeHTML(stage.name)}</span><small>RUNNING</small></div>
        <pre data-active-output="${stage.id}">${escapeHTML(recentLines)}</pre>
      </article>`;
    }).join("")}</div>`;
  panel.querySelectorAll("[data-active-output]").forEach((output) => {
    output.scrollTop = output.scrollHeight;
  });
}

function renderStageDetails(stageId) {
  const stage = STAGES.find((item) => item.id === stageId);
  const panel = $("#stage-detail");
  if (!stage) { panel.hidden = true; return; }
  $("#inspector-empty").hidden = true;
  const status = stageState(stage, state.job);
  const method = state.job?.stages?.find((item) => item.id === stageId)?.method || "Pipeline method details are documented in the project architecture report.";
  const reason = state.job?.stage_reasons?.[stageId] || state.job?.stages?.find((item) => item.id === stageId)?.reason;
  const parallelPeer = stage.parallelWith ? STAGES.find((item) => item.id === stage.parallelWith)?.name : null;
  const outputPaths = stageId === "phase12" ? ["output/<job_id>/page.json", "output/<job_id>/page_*.png", "output/<job_id>/comic.pdf"] : stage.outputs;
  const stageLog = state.job?.stage_logs?.[stageId] || "No output for this stage yet. Live output refreshes every 2 seconds while the job runs.";
  panel.innerHTML = `<h3>Stage ${String(STAGES.indexOf(stage) + 1).padStart(2, "0")} · ${escapeHTML(stage.name)} <span class="detail-status ${status}">${status}</span></h3>
    <p>${escapeHTML(method)}</p>${parallelPeer ? `<p class="stage-execution-note"><strong>Parallel group:</strong> runs alongside ${escapeHTML(parallelPeer)} when both stages are selected.</p>` : ""}${reason ? `<p class="stage-reason"><strong>${status === "skipped" ? "Why it was skipped" : "Stage note"}:</strong> ${escapeHTML(reason)}</p>` : ""}
    <div class="artifact-columns"><div><strong>Inputs</strong><ul>${stage.inputs.map((item) => `<li>${escapeHTML(item)}</li>`).join("")}</ul></div>
    <div><strong>Outputs</strong><ul>${outputPaths.map((item) => `<li>${escapeHTML(item)}</li>`).join("")}</ul></div></div>
    <div class="stage-live-output"><div><strong>Live stage output</strong><small>${status === "running" ? "Running now" : status === "completed" ? "Recent output" : "Stage console"}</small></div><pre>${escapeHTML(stageLog)}</pre></div>`;
  const output = panel.querySelector(".stage-live-output pre");
  output.scrollTop = output.scrollHeight;
  panel.hidden = false;
}

function setMessage(message, kind = "") {
  const element = $("#form-message");
  element.textContent = message;
  element.className = `form-message ${kind}`.trim();
}

function setView(view) {
  const pipelineView = view === "pipeline";
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.classList.toggle("active", link.dataset.view === (pipelineView ? "pipeline" : "home"));
  });
  [".studio-title-row", "#source-clip", ".workspace-grid", "#analysis-panel"].forEach((selector) => {
    const element = $(selector);
    if (element) element.hidden = pipelineView;
  });
  $("#production-timeline").hidden = !pipelineView;
  $(".breadcrumbs strong").textContent = pipelineView ? "Production pipeline" : "Story Studio";
}

function setupNavigation() {
  document.querySelectorAll(".nav-link").forEach((link) => link.addEventListener("click", (event) => {
    const targetView = link.dataset.view;
    if (targetView) {
      event.preventDefault();
      setView(targetView);
      if (targetView === "pipeline") location.hash = "production-timeline";
      else location.hash = "source-clip";
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    setView("home");
    const target = document.querySelector(link.getAttribute("href"));
    if (target) setTimeout(() => target.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
  }));
  setView(location.hash === "#production-timeline" ? "pipeline" : "home");
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function friendlyActivity(job) {
  const log = job.log_tail || "";
  const lastLine = log.split(/\r?\n/).filter(Boolean).at(-1) || "";
  const runningNames = (job.running_stages || [])
    .map((id) => STAGES.find((stage) => stage.id === id)?.name)
    .filter(Boolean);
  if (runningNames.length > 1) return job.dialogue_detected === false
    ? `Running ${runningNames.join(" and ")} in parallel on the visual-only path.`
    : `Running ${runningNames.join(" and ")} in parallel.`;
  if (lastLine.includes("Processing frame")) return "Analyzing sampled video frames.";
  if (lastLine.includes("gate=PASS")) return "A processing stage passed its quality checks.";
  if (/warning/i.test(lastLine)) return "A model warning was recorded; processing continues.";
  if (runningNames.length === 1) return job.dialogue_detected === false
    ? `Running ${runningNames[0].toLowerCase()} on the visual-only path and updating its output.`
    : `Running ${runningNames[0].toLowerCase()} and updating its output.`;
  const name = STAGES.find((stage) => stage.id === job.current_stage)?.name;
  if (job.status === "completed" && job.dialogue_detected === false) return "No dialogue was detected. Your visual-only comic pages are ready to review.";
  if (job.status === "completed_with_warnings") return job.dialogue_detected === false
    ? "Visual-only comic pages are ready. Some optional stages were skipped or failed; open the stage cards for details."
    : "Your comic pages are ready. Some stages failed; open the stage cards for details.";
  if (job.status === "completed") return "Your comic pages are ready to review.";
  if (job.status === "failed") return "Processing stopped after a stage error. Open details for the log.";
  if (["cancelled", "canceled"].includes(job.status)) return "Processing was stopped. Completed artifacts are preserved.";
  return name ? `Working on ${name.toLowerCase()} and preparing its output.` : "Preparing pipeline workers…";
}

function updateJobView(job) {
  state.job = job;
  localStorage.setItem("cinecomic-job-id", job.job_id);
  $("#processing-panel").hidden = false;
  $("#job-id").textContent = job.job_id ? `Job ${job.job_id}` : "";
  $("#job-status").textContent = job.status || "Unknown";
  $("#job-status").parentElement.className = `job-status ${job.status || ""}`;
  $("#progress-percent").textContent = `${Math.round((job.progress || 0) * 100)}%`;
  const processedCount = Object.values(job.stage_statuses || {}).filter((value) => ["completed", "failed", "skipped"].includes(value)).length;
  $("#completed-count").textContent = `${processedCount || (job.completed_stages || []).length} / 12`;
  $("#progress-fill").style.width = `${Math.max(0, Math.min(100, (job.progress || 0) * 100))}%`;
  const runningNames = (job.running_stages || [])
    .map((id) => STAGES.find((stage) => stage.id === id)?.name)
    .filter(Boolean);
  const currentStage = STAGES.find((stage) => stage.id === job.current_stage);
  $("#current-stage").textContent = runningNames.length
    ? runningNames.join(" + ")
    : currentStage?.name || (job.current_stage === "complete" ? "Complete" : "Waiting");
  $("#activity-text").textContent = friendlyActivity(job);
  $("#activity-friendly").textContent = friendlyActivity(job);
  const scanMode = job.settings?.frame_scan || "fast";
  $("#pipeline-mode").textContent = `${scanMode === "detailed" ? "Detailed face scan · 1 sample/sec" : "Fast face scan · 1 sample/2 sec"} · up to 2 stages in parallel`;
  $("#log-output").textContent = job.log_tail || "No processing logs yet.";
  $("#activity-spinner").className = `activity-spinner ${["completed", "completed_with_warnings"].includes(job.status) ? "done" : ["failed", "cancelled", "canceled", "interrupted"].includes(job.status) ? "failed" : ""}`;
  const created = Date.parse(job.created_at || "");
  const elapsed = Number.isFinite(created) ? Math.max(0, (Date.now() - created) / 1000) : NaN;
  $("#elapsed-time").textContent = formatTime(elapsed);
  const eta = job.phase5_progress?.eta || "—";
  $("#eta").textContent = eta;
  $("#stop-button").hidden = !["running", "starting"].includes(job.status);
  const errorBox = $("#error-box");
  errorBox.hidden = !job.error;
  errorBox.textContent = job.error || "";
  $("#progress-subtitle").textContent = job.filename ? `Processing ${job.filename}` : "Processing your video";
  $("#download-run-report").hidden = !job.job_id;
  $("#download-run-report").href = `/api/jobs/${encodeURIComponent(job.job_id)}/report`;
  renderStages(job);
  renderActiveStageOutputs(job);
  const terminal = ["completed", "completed_with_warnings", "failed", "cancelled", "canceled", "interrupted"].includes(job.status);
  $("#analysis-panel").hidden = !terminal;
  if (terminal && state.detailsJobId !== job.job_id) loadAnalysis(job.job_id).catch(() => {});
  const activeStage = (job.running_stages || [])[0] || job.current_stage;
  if (!state.selectedStage && activeStage && activeStage !== "complete") {
    state.selectedStage = activeStage;
    renderStages(job);
  }
  if (state.selectedStage) renderStageDetails(state.selectedStage);
  refreshComic(job.job_id).catch(() => {});
  if (terminal) {
    $("#processing-details").open = false;
  }
}

async function loadAnalysis(jobId) {
  state.details = await api(`/api/jobs/${encodeURIComponent(jobId)}/details`);
  state.detailsJobId = jobId;
  renderAnalysis();
}

function renderAnalysis() {
  const details = state.details || {};
  const content = $("#analysis-content");
  document.querySelectorAll(".analysis-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.analysis === state.analysisTab));
  if (state.analysisTab === "dialogue") {
    const segments = details.transcript?.segments || [];
    content.innerHTML = segments.length ? `<table><thead><tr><th>Time</th><th>Speaker</th><th>Dialogue</th></tr></thead><tbody>${segments.map((item) => `<tr><td>${formatTime(Number(item.start) || 0)}</td><td>${escapeHTML(item.speaker_id || "Unknown")}</td><td>${escapeHTML(item.text || "")}</td></tr>`).join("")}</tbody></table>` : "No transcript artifact was produced for this job.";
  } else if (state.analysisTab === "frames") {
    const panels = details.keyframes?.panels || [];
    content.innerHTML = panels.length ? `<div class="history-list">${panels.map((panel) => `<div class="history-row"><div><strong>Frame ${escapeHTML(panel.frame_id ?? "—")}</strong><small>Timestamp: ${panel.t != null ? `${Number(panel.t).toFixed(2)} s` : "unknown"}</small></div><span class="history-status">${escapeHTML(panel.character_id || panel.speaker_id || "Selected keyframe")}</span></div>`).join("")}</div>` : "No selected-keyframe artifact was produced for this job.";
  } else if (state.analysisTab === "alignment") {
    content.innerHTML = details.alignment ? `<pre>${escapeHTML(JSON.stringify(details.alignment, null, 2))}</pre>` : "No dialogue-to-frame alignment was produced.";
  } else {
    const metrics = details.page?.metrics || {};
    const rows = Object.entries(metrics);
    content.innerHTML = rows.length ? `<div class="progress-metrics">${rows.slice(0, 8).map(([key, value]) => `<div><strong>${escapeHTML(Array.isArray(value) ? value.length : value)}</strong><span>${escapeHTML(key.replaceAll("_", " "))}</span></div>`).join("")}</div><pre>${escapeHTML(JSON.stringify(metrics, null, 2))}</pre>` : "No final validation metrics were produced.";
  }
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok) {
    const error = contentType.includes("application/json") ? await response.json() : { error: await response.text() };
    throw new Error(error.error || `Request failed (${response.status})`);
  }
  return contentType.includes("application/json") ? response.json() : response;
}

async function pollJob() {
  if (!state.jobId || state.pollInFlight) return;
  state.pollInFlight = true;
  const requestedJobId = state.jobId;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(requestedJobId)}`);
    if (state.jobId !== requestedJobId) return;
    const terminalStatuses = ["completed", "completed_with_warnings", "failed", "cancelled", "canceled", "interrupted"];
    if (state.job?.job_id === requestedJobId
      && terminalStatuses.includes(state.job.status)
      && !terminalStatuses.includes(job.status)) return;
    updateJobView(job);
    if (terminalStatuses.includes(job.status) && state.pollHandle) {
      clearInterval(state.pollHandle);
      state.pollHandle = null;
    }
  } catch (error) {
    $("#server-status").textContent = "Could not reach local service";
    setMessage(error.message, "error");
  } finally {
    state.pollInFlight = false;
  }
}

function startPolling() {
  if (state.pollHandle) clearInterval(state.pollHandle);
  pollJob();
  state.pollHandle = setInterval(pollJob, 2000);
}

async function refreshComic(jobId) {
  if (state.previewRequest) return state.previewRequest;
  state.previewRequest = (async () => {
    const pages = await api(`/api/jobs/${encodeURIComponent(jobId)}/pages`);
    if (jobId !== state.jobId) return;
    state.previewJobId = jobId;
    state.pageNames = pages;
    state.pageIndex = Math.min(state.pageIndex, Math.max(0, pages.length - 1));
    const image = $("#comic-page");
    const empty = $(".preview-empty");
    const hasPage = pages.length > 0;
    image.hidden = !hasPage;
    empty.hidden = hasPage;
    $("#page-controls").hidden = !hasPage;
    $("#comic-actions").hidden = !hasPage;
    $("#fullscreen-button").hidden = !hasPage;
    if (!hasPage) return;
    const filename = pages[state.pageIndex];
    const pageUrl = `/api/jobs/${encodeURIComponent(jobId)}/pages/${encodeURIComponent(filename)}`;
    if (image.dataset.pageUrl !== pageUrl) {
      image.src = pageUrl;
      image.dataset.pageUrl = pageUrl;
    }
    image.style.maxWidth = `${Number($("#zoom").value || 100)}%`;
    $("#page-label").textContent = `Page ${state.pageIndex + 1} of ${pages.length}`;
    $("#previous-page").disabled = state.pageIndex <= 0;
    $("#next-page").disabled = state.pageIndex >= pages.length - 1;
    $("#download-pdf").href = `/api/jobs/${encodeURIComponent(jobId)}/comic.pdf`;
    $("#download-pages").href = `/api/jobs/${encodeURIComponent(jobId)}/pages.zip`;
    $("#download-report").href = `/api/jobs/${encodeURIComponent(jobId)}/report`;
    $("#preview-subtitle").textContent = ["completed", "completed_with_warnings"].includes(state.job?.status) ? "Generated comic · real pipeline output" : "Comic preview · pages appear as they render";
  })();
  try {
    await state.previewRequest;
  } finally {
    state.previewRequest = null;
  }
}

async function submitVideo() {
  if (!state.file) return setMessage("Choose a video file first.", "error");
  const form = new FormData();
  form.append("video", state.file);
  form.append("language", $("#language").value.trim() || "auto");
  form.append("model_size", $("#model-size").value);
  form.append("device", $("#device").value);
  form.append("frame_scan", $("#frame-scan").value);
  form.append("panels_per_page", $("#panels").value);
  form.append("allow_model_downloads", String($("#allow-downloads").checked));
  form.append("hf_token", $("#hf-token").value.trim());
  const button = $("#generate-button");
  button.disabled = true;
  button.innerHTML = '<span class="small-spinner"></span> Preparing video…';
  setMessage("Validating the video and starting the processing pipeline…");
  try {
    const job = await api("/api/jobs", { method: "POST", body: form });
    state.jobId = job.job_id;
    state.selectedStage = "";
    state.pageIndex = 0;
    updateJobView(job);
    startPolling();
    setMessage("Video accepted. Your comic is being generated.", "success");
    setView("pipeline");
    location.hash = "production-timeline";
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    button.innerHTML = '<span>✦</span> Generate Comic';
    button.disabled = !state.file;
  }
}

async function stopJob() {
  if (!state.jobId || !window.confirm("Stop processing this video? Completed stages will be kept.")) return;
  const button = $("#stop-button");
  button.disabled = true;
  button.textContent = "Stopping…";
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.jobId)}/stop`, { method: "POST" });
    updateJobView(job);
    if (state.pollHandle) clearInterval(state.pollHandle);
    state.pollHandle = null;
  } catch (error) {
    $("#error-box").hidden = false;
    $("#error-box").textContent = error.message;
  } finally {
    button.disabled = false;
    button.innerHTML = "■ Stop processing";
  }
}

function chooseVideo(file) {
  if (!file) return;
  const allowed = /\.(mp4|mov|mkv|avi|webm)$/i.test(file.name);
  if (!allowed) return setMessage("Use an MP4, MOV, MKV, AVI, or WEBM video.", "error");
  if (file.size > (state.serverConfig?.max_upload_bytes || 512 * 1024 * 1024)) return setMessage("Video exceeds the 512 MB upload limit.", "error");
  state.file = file;
  const preview = $("#video-preview");
  if (preview.dataset.objectUrl) URL.revokeObjectURL(preview.dataset.objectUrl);
  const objectUrl = URL.createObjectURL(file);
  preview.dataset.objectUrl = objectUrl;
  preview.src = objectUrl;
  $("#video-name").textContent = file.name;
  $("#video-info").textContent = `${(file.size / 1048576).toFixed(1)} MB · reading duration…`;
  preview.onloadedmetadata = () => {
    const duration = preview.duration;
    $("#video-info").textContent = `${(file.size / 1048576).toFixed(1)} MB · ${formatTime(duration)}`;
    if (duration > 300) setMessage("Videos must be five minutes or shorter.", "error");
    else setMessage("Video selected and ready to generate.", "success");
  };
  preview.onerror = () => { $("#video-info").textContent = `${(file.size / 1048576).toFixed(1)} MB`; };
  $("#selected-video").hidden = false;
  $("#drop-zone").hidden = true;
  $("#generate-button").disabled = false;
}

function clearVideo() {
  const preview = $("#video-preview");
  if (preview.dataset.objectUrl) URL.revokeObjectURL(preview.dataset.objectUrl);
  preview.removeAttribute("src");
  preview.load();
  state.file = null;
  $("#video-file").value = "";
  $("#selected-video").hidden = true;
  $("#drop-zone").hidden = false;
  $("#generate-button").disabled = true;
  setMessage("Select a video to begin.");
}

function setupUpload() {
  const dropZone = $("#drop-zone");
  $("#browse-video").addEventListener("click", () => $("#video-file").click());
  dropZone.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    $("#video-file").click();
  });
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); $("#video-file").click(); }
  });
  $("#video-file").addEventListener("change", (event) => chooseVideo(event.target.files[0]));
  $("#clear-video").addEventListener("click", clearVideo);
  ["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("drag-over"); }));
  ["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("drag-over"); }));
  dropZone.addEventListener("drop", (event) => chooseVideo(event.dataTransfer.files[0]));
  $("#generate-button").addEventListener("click", submitVideo);
}

function setupComicControls() {
  $("#previous-page").addEventListener("click", () => { state.pageIndex = Math.max(0, state.pageIndex - 1); refreshComic(state.jobId); });
  $("#next-page").addEventListener("click", () => { state.pageIndex = Math.min(state.pageNames.length - 1, state.pageIndex + 1); refreshComic(state.jobId); });
  $("#zoom").addEventListener("input", (event) => { $("#comic-page").style.maxWidth = `${event.target.value}%`; });
  $("#comic-page").addEventListener("click", () => {
    $("#fullscreen-image").src = $("#comic-page").src;
    $("#fullscreen-dialog").showModal();
  });
  $("#fullscreen-button").addEventListener("click", () => {
    $("#fullscreen-image").src = $("#comic-page").src;
    $("#fullscreen-dialog").showModal();
  });
  $("#dialog-close").addEventListener("click", () => $("#fullscreen-dialog").close());
  $("#fullscreen-dialog").addEventListener("click", (event) => { if (event.target === $("#fullscreen-dialog")) $("#fullscreen-dialog").close(); });
  $("#stop-button").addEventListener("click", stopJob);
  $("#download-log").addEventListener("click", () => {
    const blob = new Blob([state.job?.log_tail || "No log excerpt available."], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob); link.download = `cinecomic-${state.jobId}-log-excerpt.txt`; link.click(); URL.revokeObjectURL(link.href);
  });
  document.querySelectorAll(".analysis-tab").forEach((tab) => tab.addEventListener("click", () => {
    state.analysisTab = tab.dataset.analysis;
    renderAnalysis();
  }));
}

async function initialize() {
  setupNavigation();
  renderStages();
  setupUpload();
  setupComicControls();
  try {
    state.serverConfig = await api("/api/config");
    $("#server-status").textContent = "Local studio ready";
    if (state.serverConfig.hf_token_configured) $("#hf-token").placeholder = "Using local environment token";
  } catch (error) {
    $("#server-status").textContent = "Service unavailable";
    setMessage("Start the CineComic web server with `python app.py`.", "error");
  }
  if (state.jobId) startPolling();
}

initialize();
