const state = {
  profile: null,
  report: null,
  candidates: [],
  feedback: [],
  assistantContextItemId: "",
  health: null,
  runEvents: [],
  runEventSource: null,
  assistantEventSource: null,
  assistantAbortController: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || "Request failed");
  }
  return payload;
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderAssistantText(value) {
  const normalized = String(value || "")
    .replace(/\r\n/g, "\n")
    .replace(/[✅🎯⚠️❌🔍➡️]/g, "")
    .trim();
  if (!normalized) return "";

  const blocks = [];
  let listItems = [];
  const flushList = () => {
    if (!listItems.length) return;
    blocks.push(`<ul>${listItems.map((item) => `<li>${formatInlineMarkdown(item)}</li>`).join("")}</ul>`);
    listItems = [];
  };

  normalized.split(/\n+/).forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      return;
    }
    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    const numbered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      listItems.push((bullet || numbered)[1]);
      return;
    }
    flushList();
    const heading = trimmed.replace(/^#{1,6}\s+/, "");
    blocks.push(`<p>${formatInlineMarkdown(heading)}</p>`);
  });
  flushList();
  return blocks.join("");
}

function formatInlineMarkdown(value) {
  let html = escapeHtml(value)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  return html;
}

function splitLines(value) {
  return String(value || "")
    .split(/\n|,/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function joinLines(values) {
  return (values || []).join("\n");
}

async function loadInitial() {
  state.profile = await api("/api/profile?profile=default_user");
  renderProfile();
  try {
    state.report = await api("/api/report?report=latest");
  } catch (error) {
    state.report = null;
  }
  state.candidates = await api("/api/candidates");
  state.feedback = await api("/api/feedback?profile=default_user");
  state.health = await api("/api/health");
  renderReport();
}

function renderProfile() {
  const profile = state.profile;
  if (!profile) return;
  $("#profileName").textContent = profile.display_name || profile.user_id;
  $("#domainChips").innerHTML = (profile.research_domains || [])
    .map((domain) => `<span class="chip">${escapeHtml(domain)}</span>`)
    .join("");
  $("#displayName").value = profile.display_name || "";
  $("#researchDomains").value = joinLines(profile.research_domains);
  $("#methods").value = joinLines(profile.methods);
  $("#applications").value = joinLines(profile.applications);
  $("#preferredContent").value = joinLines(profile.preferred_content);
  $("#excludedTopics").value = joinLines(profile.excluded_topics);
  $("#currentGoals").value = joinLines(profile.current_goals);
  $("#technicalLevel").value = profile.technical_level || "researcher";
}

function renderReport() {
  const report = state.report;
  if (!report) {
    $("#generatedAt").textContent = "No report loaded";
    return;
  }
  $("#generatedAt").textContent = `${report.generated_at || ""} - ${report.source_mode || "unknown"}`;
  const stats = report.filter_stats || {};
  $("#acceptedCount").textContent = (stats.candidate || 0) + (stats.high_priority || 0);
  $("#rejectedCount").textContent = stats.reject || 0;
  $("#candidateCount").textContent = report.candidate_count || 0;
  const allItems = allAnalyses(report);
  $("#topScore").textContent = allItems.length ? Number(allItems[0].score || 0).toFixed(1) : "0.0";

  $("#actionsList").innerHTML = (report.actions || [])
    .map((action) => `<li>${escapeHtml(action)}</li>`)
    .join("");

  renderItems("#topItems", allItems.slice(0, 6), true);
  renderItems("#paperItems", report.top_papers || []);
  renderItems("#repoItems", report.top_repos || []);
  renderTrends(report.trends || []);
  renderFiltered();
  renderSaved();
  renderSourceErrors(report.source_error_count || 0);
  renderSystemStatus();
  renderAssistantContext();
  drawSignalCanvas(allItems);
}

function allAnalyses(report) {
  return [
    ...(report.top_papers || []),
    ...(report.top_repos || []),
    ...(report.top_tools || []),
  ].sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
}

function candidateMap() {
  const map = new Map();
  const candidates = state.report && state.report.candidates && state.report.candidates.length
    ? state.report.candidates
    : state.candidates;
  candidates.forEach((item) => map.set(item.item_id, item));
  return map;
}

function analysisMap() {
  const map = new Map();
  allAnalyses(state.report || {}).forEach((item) => map.set(item.item_id, item));
  return map;
}

function decisionMap() {
  const map = new Map();
  ((state.report && state.report.filter_decisions) || []).forEach((decision) => map.set(decision.item_id, decision));
  return map;
}

function renderItems(selector, items, compact = false) {
  const target = $(selector);
  if (!items.length) {
    target.innerHTML = `<p class="answer">No items selected.</p>`;
    return;
  }
  target.innerHTML = items.map((item) => renderAnalysisCard(item, compact)).join("");
}

function renderAnalysisCard(item, compact = false) {
  const limitations = (item.limitations || []).slice(0, compact ? 1 : 3).join("; ");
  return `
    <article class="item">
      <div class="meta">
        <span>${escapeHtml(item.content_type)}</span>
        <span class="score">${Number(item.score || 0).toFixed(1)}/10</span>
        <span>${escapeHtml(item.confidence || "medium")}</span>
      </div>
      <h4>${escapeHtml(item.title)}</h4>
      <p>${escapeHtml(item.relation_to_user || "")}</p>
      <p>${escapeHtml(item.technical_core || "")}</p>
      <p>${escapeHtml(limitations)}</p>
      <div class="item-actions">
        <button class="micro-btn detail-btn" data-detail="${escapeHtml(item.item_id)}">Details</button>
        <button class="micro-btn" data-assistant-item="${escapeHtml(item.item_id)}">Ask</button>
        <button class="micro-btn" data-feedback="relevant" data-item-id="${escapeHtml(item.item_id)}">Relevant</button>
        <button class="micro-btn" data-feedback="not_relevant" data-item-id="${escapeHtml(item.item_id)}">Not relevant</button>
        <button class="micro-btn" data-feedback="save" data-item-id="${escapeHtml(item.item_id)}">Save</button>
        <button class="micro-btn" data-feedback="deeper" data-item-id="${escapeHtml(item.item_id)}">Deepen</button>
        <a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">Open item</a>
      </div>
    </article>
  `;
}

function renderFiltered() {
  const status = $("#filterStatusSelect").value || "all";
  const candidates = candidateMap();
  const decisions = ((state.report && state.report.filter_decisions) || [])
    .filter((decision) => status === "all" || decision.status === status);

  $("#filteredItems").innerHTML = decisions.length
    ? decisions.map((decision) => {
        const item = candidates.get(decision.item_id) || {};
        return `
          <article class="item">
            <div class="meta">
              <span>${escapeHtml(decision.status)}</span>
              <span>relevance ${Number(decision.relevance_score || 0).toFixed(1)}</span>
              <span>quality ${Number(decision.quality_score || 0).toFixed(1)}</span>
            </div>
            <h4>${escapeHtml(item.title || decision.item_id)}</h4>
            <p>${escapeHtml(item.summary || "")}</p>
            <ul class="reason-list">${(decision.reasons || []).map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>
            <div class="item-actions">
              <button class="micro-btn detail-btn" data-detail="${escapeHtml(decision.item_id)}">Details</button>
              <button class="micro-btn" data-assistant-item="${escapeHtml(decision.item_id)}">Ask</button>
              <button class="micro-btn" data-feedback="skip" data-item-id="${escapeHtml(decision.item_id)}">Skip</button>
              <button class="micro-btn" data-feedback="relevant" data-item-id="${escapeHtml(decision.item_id)}">Recover</button>
            </div>
          </article>
        `;
      }).join("")
    : `<p class="answer">No filtered items for this status.</p>`;
}

function renderSaved() {
  if (!state.feedback.length) {
    $("#savedItems").innerHTML = `<p class="answer">No feedback recorded yet.</p>`;
    return;
  }
  const candidates = candidateMap();
  $("#savedItems").innerHTML = state.feedback
    .slice()
    .reverse()
    .map((event) => {
      const item = candidates.get(event.item_id) || {};
      return `
        <article class="item">
          <div class="meta">
            <span>${escapeHtml(event.action)}</span>
            <span>${escapeHtml(event.created_at)}</span>
          </div>
          <h4>${escapeHtml(item.title || event.item_id)}</h4>
          <p>${escapeHtml(item.summary || event.note || "")}</p>
          <button class="micro-btn detail-btn" data-detail="${escapeHtml(event.item_id)}">Details</button>
          <button class="micro-btn" data-assistant-item="${escapeHtml(event.item_id)}">Ask</button>
        </article>
      `;
    })
    .join("");
}

function renderTrends(trends) {
  $("#trendItems").innerHTML = trends.length
    ? trends
        .map(
          (trend) => `
          <article class="item">
            <div class="meta">
              <span>${trend.window_days}d</span>
              <span>${escapeHtml(trend.confidence)}</span>
            </div>
            <h4>${escapeHtml(trend.topic)}</h4>
            <p>${escapeHtml(trend.summary)}</p>
            <p>${escapeHtml(trend.user_implication)}</p>
            <ul class="evidence-list">${(trend.signals || []).map((signal) => `<li>${escapeHtml(signal)}</li>`).join("")}</ul>
          </article>
        `,
        )
        .join("")
    : `<p class="answer">No trend signal found.</p>`;
}

function renderAssistantContext() {
  const itemId = state.assistantContextItemId;
  if (!itemId) {
    $("#assistantContext").textContent = "Context: current report";
    return;
  }
  const item = analysisMap().get(itemId) || candidateMap().get(itemId);
  $("#assistantContext").textContent = item
    ? `Context: ${item.title}`
    : "Context: selected item";
}

function renderSourceErrors() {
  const panel = $("#sourceErrorsPanel");
  if (panel) {
    panel.classList.add("hidden");
  }
  const sourceErrors = $("#sourceErrors");
  if (sourceErrors) {
    sourceErrors.innerHTML = "";
  }
}

function renderSystemStatus() {
  const health = state.health;
  if (!health) {
    $("#systemStatus").innerHTML = "<div>Status unavailable.</div>";
    return;
  }
  const lines = [];
  const llm = health.llm || {};
  const network = health.network || {};
  const embedding = health.embedding || {};
  const pgvector = health.pgvector || {};
  lines.push(`LLM: ${llm.enabled ? "enabled" : "disabled"} | model=${escapeHtml(llm.model || "")}`);
  if (network.connector_timeout_seconds) {
    lines.push(`Network timeout: ${escapeHtml(network.connector_timeout_seconds)}s`);
  }
  lines.push(`Embedding: ${escapeHtml(embedding.status || "unknown")} | provider=${escapeHtml(embedding.provider || "")} | model=${escapeHtml(embedding.model || "")}`);
  lines.push(`pgvector: ${escapeHtml(pgvector.status || "unknown")}${pgvector.table ? ` | table=${escapeHtml(pgvector.table)}` : ""}`);
  const liveErrorCount = Number(health.latest_live_error_count || 0);
  lines.push(`Live sources: ${liveErrorCount ? "partial" : "ok"}`);
  if (liveErrorCount) {
    lines.push("Source diagnostics are kept in backend artifacts.");
  }
  $("#systemStatus").innerHTML = lines.map((line) => `<div>${line}</div>`).join("");
}

  function shortError(value, limit = 260) {
    const text = String(value || "")
      .replace(/https?:\/\/\S+/g, "[url]")
      .replace(/\s+/g, " ")
      .trim();
    return text.length > limit ? `${text.slice(0, limit - 1).trim()}...` : text;
  }

function closeRunEventSource() {
  if (state.runEventSource) {
    state.runEventSource.close();
    state.runEventSource = null;
  }
}

function closeAssistantEventSource() {
  if (state.assistantEventSource) {
    state.assistantEventSource.close();
    state.assistantEventSource = null;
  }
  if (state.assistantAbortController) {
    state.assistantAbortController.abort();
    state.assistantAbortController = null;
  }
}

function resetRunStream() {
  closeRunEventSource();
  state.runEvents = [];
  const panel = $("#runStreamPanel");
  if (panel) panel.classList.remove("hidden");
  const meta = $("#runStreamMeta");
  if (meta) meta.textContent = "Connecting";
  renderRunTimeline();
}

function parseRunEvent(event) {
  try {
    return JSON.parse(event.data);
  } catch (error) {
    return {
      stage: "stream",
      status: "error",
      message: "Malformed stream event",
      detail: event.data,
    };
  }
}

function appendRunEvent(payload) {
  const event = {
    stage: payload.stage || "run",
    status: payload.status || "running",
    message: payload.message || "",
    timestamp: payload.timestamp || new Date().toISOString(),
    detail: payload.detail || payload.error || "",
    count: payload.count,
    candidate_count: payload.candidate_count,
    analysis_count: payload.analysis_count,
    trend_count: payload.trend_count,
  };
  state.runEvents.push(event);
  if (state.runEvents.length > 80) {
    state.runEvents = state.runEvents.slice(-80);
  }
  $("#runStatus").textContent = event.detail ? `${event.message}: ${event.detail}` : event.message;
  renderRunTimeline();
}

function renderRunTimeline() {
  const panel = $("#runStreamPanel");
  const target = $("#runTimeline");
  if (!panel || !target) return;
  if (!state.runEvents.length) {
    target.innerHTML = `<p class="answer">Waiting for agent events.</p>`;
    return;
  }
  panel.classList.remove("hidden");
  const latest = state.runEvents[state.runEvents.length - 1];
  const meta = $("#runStreamMeta");
  if (meta) {
    meta.textContent = `${stageLabel(latest.stage)} - ${latest.status}`;
  }
  target.innerHTML = state.runEvents
    .slice()
    .reverse()
    .map((event) => {
      const detailParts = [];
      if (event.candidate_count != null) detailParts.push(`${event.candidate_count} candidates`);
      if (event.analysis_count != null) detailParts.push(`${event.analysis_count} analyses`);
      if (event.trend_count != null) detailParts.push(`${event.trend_count} trends`);
      if (event.count != null) detailParts.push(`${event.count} items`);
      if (event.detail) detailParts.push(shortError(event.detail, 180));
      const detail = detailParts.length ? `<p>${escapeHtml(detailParts.join(" | "))}</p>` : "";
      return `
        <div class="run-event ${escapeHtml(statusClass(event.status))}">
          <span class="run-dot"></span>
          <div>
            <strong>${escapeHtml(stageLabel(event.stage))}</strong>
            <span>${escapeHtml(event.message)}</span>
            ${detail}
          </div>
          <time>${escapeHtml(formatStreamTime(event.timestamp))}</time>
        </div>
      `;
    })
    .join("");
}

function statusClass(status) {
  return String(status || "running").replace(/[^a-z0-9_-]/gi, "");
}

function stageLabel(stage) {
  const labels = {
    run: "Run",
    profile: "Profile",
    discovery: "Discovery",
    filtering: "Filtering",
    value_analysis: "Value Analysis",
    evidence: "Evidence",
    trends: "Trends",
    recommendation: "Recommendation",
    storage: "Storage",
    rag: "RAG",
    stream: "Stream",
  };
  return labels[stage] || String(stage || "Run");
}

function formatStreamTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function drawSignalCanvas(items) {
  const canvas = $("#signalCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const labels = ["Relevance", "Depth", "Evidence", "Utility", "Trend"];
  const values = labels.map((label) => {
    const key = {
      Relevance: "relevance",
      Depth: "technical_depth",
      Evidence: "evidence_strength",
      Utility: "practical_utility",
      Trend: "trend_signal",
    }[label];
    if (!items.length) return 0;
    return items.reduce((sum, item) => sum + Number(item[key] || 0), 0) / items.length;
  });

  const maxBar = 340;
  ctx.font = "14px Segoe UI, Arial";
  ctx.textBaseline = "middle";
  labels.forEach((label, index) => {
    const y = 32 + index * 42;
    const width = (values[index] / 10) * maxBar;
    ctx.fillStyle = "#eef3f0";
    ctx.fillRect(120, y - 12, maxBar, 24);
    ctx.fillStyle = index % 2 === 0 ? "#0f766e" : "#2563eb";
    ctx.fillRect(120, y - 12, width, 24);
    ctx.fillStyle = "#1d2420";
    ctx.fillText(label, 12, y);
    ctx.fillText(values[index].toFixed(1), 474, y);
  });
}

function openDetail(itemId) {
  setAssistantContext(itemId);
  const candidates = candidateMap();
  const analyses = analysisMap();
  const decisions = decisionMap();
  const item = candidates.get(itemId) || {};
  const analysis = analyses.get(itemId);
  const decision = decisions.get(itemId);
  $("#drawerTitle").textContent = analysis ? analysis.title : item.title || itemId;
  $("#drawerBody").innerHTML = renderDetailBody(itemId, item, analysis, decision);
  $("#detailDrawer").classList.add("open");
  $("#drawerBackdrop").classList.add("open");
}

function renderDetailBody(itemId, item, analysis, decision) {
  const scoreRows = analysis ? [
    ["Overall", analysis.score],
    ["Relevance", analysis.relevance],
    ["Novelty", analysis.novelty],
    ["Depth", analysis.technical_depth],
    ["Evidence", analysis.evidence_strength],
    ["Reproducibility", analysis.reproducibility],
    ["Utility", analysis.practical_utility],
    ["Trend", analysis.trend_signal],
    ["Opportunity", analysis.research_opportunity],
  ] : [];
  return `
    <div class="meta">
      <span>${escapeHtml(item.content_type || (analysis && analysis.content_type) || "")}</span>
      <span>${escapeHtml(item.source || "")}</span>
      <span>${escapeHtml(decision ? decision.status : "not filtered")}</span>
    </div>
    <p>${escapeHtml(item.summary || "")}</p>
    ${scoreRows.length ? `<div class="score-grid">${scoreRows.map(([label, value]) => `
      <div class="score-row"><span>${escapeHtml(label)}</span><strong>${Number(value || 0).toFixed(1)}</strong></div>
    `).join("")}</div>` : ""}
    ${analysis ? `
      <h4>Why it matters</h4>
      <p>${escapeHtml(analysis.why_it_matters)}</p>
      <h4>Relation to profile</h4>
      <p>${escapeHtml(analysis.relation_to_user)}</p>
      <h4>Strengths</h4>
      <ul class="evidence-list">${(analysis.strengths || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
      <h4>Limitations</h4>
      <ul class="evidence-list">${(analysis.limitations || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
      <h4>Evidence</h4>
      <ul class="evidence-list">${(analysis.evidence || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    ` : ""}
    ${decision ? `
      <h4>Filter reasons</h4>
      <ul class="reason-list">${(decision.reasons || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    ` : ""}
    <div class="item-actions">
      <button class="micro-btn" data-feedback="relevant" data-item-id="${escapeHtml(itemId)}">Relevant</button>
      <button class="micro-btn" data-assistant-item="${escapeHtml(itemId)}">Ask assistant</button>
      <button class="micro-btn" data-feedback="not_relevant" data-item-id="${escapeHtml(itemId)}">Not relevant</button>
      <button class="micro-btn" data-feedback="save" data-item-id="${escapeHtml(itemId)}">Save</button>
      <button class="micro-btn" data-feedback="baseline" data-item-id="${escapeHtml(itemId)}">Baseline</button>
      <a href="${escapeHtml((analysis && analysis.url) || item.url || "")}" target="_blank" rel="noreferrer">Open item</a>
    </div>
  `;
}

function closeDetail() {
  $("#detailDrawer").classList.remove("open");
  $("#drawerBackdrop").classList.remove("open");
}

async function runPipelineFallback(button) {
  appendRunEvent({ stage: "run", status: "running", message: "Running without stream support" });
  try {
    const payload = await api("/api/run", {
      method: "POST",
      body: JSON.stringify({
        profile: "default_user",
        source: $("#sourceMode").value,
        report: "latest",
      }),
    });
    state.report = payload.report;
    state.candidates = await api("/api/candidates");
    state.feedback = await api("/api/feedback?profile=default_user");
    state.health = await api("/api/health");
    renderReport();
    appendRunEvent({ stage: "run", status: "complete", message: "Pipeline complete" });
  } catch (error) {
    appendRunEvent({ stage: "run", status: "error", message: "Run failed", detail: error.message });
  } finally {
    button.disabled = false;
    button.textContent = "Run";
    if (!$("#runStatus").textContent.startsWith("Run failed")) {
      $("#runStatus").textContent = "Run complete.";
    }
  }
}

function runPipeline() {
  const button = $("#runBtn");
  button.disabled = true;
  button.textContent = "Running";
  $("#runStatus").textContent = "Live sources may take a short while. Detailed source diagnostics stay in backend artifacts.";
  resetRunStream();

  if (!window.EventSource) {
    runPipelineFallback(button);
    return;
  }

  const params = new URLSearchParams({
    profile: "default_user",
    source: $("#sourceMode").value,
    report: "latest",
  });
  const source = new EventSource(`/api/run/stream?${params.toString()}`);
  state.runEventSource = source;
  let finished = false;
  let receivedEvent = false;
  let fallbackStarted = false;
  const fallbackTimer = window.setTimeout(() => {
    if (finished || receivedEvent || fallbackStarted) return;
    fallbackStarted = true;
    closeRunEventSource();
    appendRunEvent({ stage: "stream", status: "warning", message: "Stream unavailable, switching to standard run" });
    runPipelineFallback(button);
  }, 2500);

  const handleStreamEvent = (event) => {
    receivedEvent = true;
    window.clearTimeout(fallbackTimer);
    appendRunEvent(parseRunEvent(event));
  };

  source.addEventListener("run.started", handleStreamEvent);
  source.addEventListener("run.progress", handleStreamEvent);
  source.addEventListener("run.completed", (event) => {
    const payload = parseRunEvent(event);
    finished = true;
    receivedEvent = true;
    window.clearTimeout(fallbackTimer);
    appendRunEvent(payload);
    closeRunEventSource();
    state.report = payload.report;
    Promise.all([
      api("/api/candidates"),
      api("/api/feedback?profile=default_user"),
      api("/api/health"),
    ])
      .then(([candidates, feedback, health]) => {
        state.candidates = candidates;
        state.feedback = feedback;
        state.health = health;
        renderReport();
      })
      .catch((error) => {
        appendRunEvent({ stage: "run", status: "warning", message: "Refresh after run failed", detail: error.message });
      })
      .finally(() => {
        button.disabled = false;
        button.textContent = "Run";
        $("#runStatus").textContent = "Run complete.";
      });
  });
  source.addEventListener("run.failed", (event) => {
    const payload = parseRunEvent(event);
    finished = true;
    receivedEvent = true;
    window.clearTimeout(fallbackTimer);
    appendRunEvent(payload);
    closeRunEventSource();
    button.disabled = false;
    button.textContent = "Run";
    $("#runStatus").textContent = `Run failed: ${payload.detail || payload.message}`;
  });
  source.onerror = () => {
    if (finished || fallbackStarted) return;
    if (source.readyState === EventSource.CONNECTING) {
      if (!receivedEvent) {
        $("#runStatus").textContent = "Connecting to stream...";
      }
      return;
    }
    window.clearTimeout(fallbackTimer);
    if (!receivedEvent) {
      fallbackStarted = true;
      closeRunEventSource();
      appendRunEvent({ stage: "stream", status: "warning", message: "Stream unavailable, switching to standard run" });
      runPipelineFallback(button);
      return;
    }
    appendRunEvent({ stage: "stream", status: "warning", message: "Stream interrupted" });
  };
}

async function saveProfile(event) {
  event.preventDefault();
  const payload = {
    user_id: "default_user",
    display_name: $("#displayName").value.trim() || "Researcher",
    research_domains: splitLines($("#researchDomains").value),
    methods: splitLines($("#methods").value),
    applications: splitLines($("#applications").value),
    preferred_content: splitLines($("#preferredContent").value),
    excluded_topics: splitLines($("#excludedTopics").value),
    current_goals: splitLines($("#currentGoals").value),
    technical_level: $("#technicalLevel").value,
  };
  state.profile = await api("/api/profile", { method: "POST", body: JSON.stringify(payload) });
  renderProfile();
}

async function sendFeedback(itemId, action) {
  const payload = await api("/api/feedback", {
    method: "POST",
    body: JSON.stringify({ profile_id: "default_user", item_id: itemId, action }),
  });
  state.feedback.push(payload);
  state.profile = await api("/api/profile?profile=default_user");
  renderProfile();
  renderSaved();
}

function setAssistantContext(itemId) {
  state.assistantContextItemId = itemId || "";
  renderAssistantContext();
}

function openAssistant(itemId = null) {
  if (itemId !== null) {
    setAssistantContext(itemId);
  } else {
    renderAssistantContext();
  }
  $("#assistantDrawer").classList.add("open");
  $("#assistantQuestion").focus();
}

function closeAssistant() {
  closeAssistantEventSource();
  $("#assistantDrawer").classList.remove("open");
}

function appendAssistantMessage(role, text, payload = {}) {
  const container = $("#assistantMessages");
  const node = document.createElement("div");
  node.className = `message ${role}`;
  renderAssistantMessage(node, text, payload);
  container.appendChild(node);
  container.scrollTop = container.scrollHeight;
  return node;
}

function renderAssistantMessage(node, text, payload = {}) {
  const sources = payload.sources || [];
  const mode = payload.mode || "";
  const evaluation = payload.evaluation || null;
  const isAssistant = node.classList.contains("assistant");
  if (!sources.length && !mode && !evaluation) {
    if (isAssistant) {
      node.innerHTML = `<div class="message-text">${renderAssistantText(text)}</div>`;
    } else {
      node.textContent = text;
    }
    return;
  }
  const sourceHtml = sources.length
    ? `
      <div class="assistant-sources">
        <div class="source-title">Sources</div>
        ${sources.slice(0, 4).map((source, index) => `
          <a class="source-pill" href="${escapeHtml(source.url || "#")}" target="_blank" rel="noreferrer">
            <span>${index + 1}</span>
            <strong>${escapeHtml(source.title || source.chunk_id)}</strong>
            <em>${Number(source.score || 0).toFixed(2)}</em>
          </a>
        `).join("")}
      </div>
    `
    : "";
  const warnings = evaluation && evaluation.warnings && evaluation.warnings.length
    ? `<div class="assistant-eval">Check: ${evaluation.warnings.map(escapeHtml).join(", ")}</div>`
    : "";
  node.innerHTML = `
    <div class="message-text">${renderAssistantText(text)}</div>
    ${mode ? `<div class="assistant-mode">${escapeHtml(mode)}</div>` : ""}
    ${sourceHtml}
    ${warnings}
  `;
}

function renderAssistantProgress(node, payload) {
  if (node.dataset.streamingAnswer === "1") {
    return;
  }
  const history = JSON.parse(node.dataset.progressHistory || "[]");
  const label = assistantStageLabel(payload.stage);
  const message = payload.detail ? `${payload.message}: ${payload.detail}` : payload.message;
  const entry = `${label}: ${message || "Working..."}`;
  if (!history.length || history[history.length - 1] !== entry) {
    history.push(entry);
  }
  const visibleHistory = history.slice(-4);
  node.dataset.progressHistory = JSON.stringify(visibleHistory);
  node.innerHTML = `
    <div class="message-text">
      <p>${escapeHtml(message || "Working...")}</p>
      <ul>${visibleHistory.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </div>
    <div class="assistant-mode">${escapeHtml(label)} - ${escapeHtml(payload.status || "running")}</div>
  `;
}

function appendAssistantToken(node, text) {
  const nextText = `${node.dataset.answerText || ""}${text || ""}`;
  node.dataset.answerText = nextText;
  node.dataset.streamingAnswer = "1";
  node.innerHTML = `
    <div class="message-text">${renderAssistantText(nextText)}</div>
    <div class="assistant-mode">Streaming answer</div>
  `;
  const container = $("#assistantMessages");
  container.scrollTop = container.scrollHeight;
}

function assistantStageLabel(stage) {
  const labels = {
    assistant: "Assistant",
    context: "Context",
    rag: "RAG",
    generation: "Generation",
    evaluation: "Evaluation",
    stream: "Stream",
  };
  return labels[stage] || String(stage || "Assistant");
}

async function askAssistantFallback(button, pendingMessage, question, itemId) {
  try {
    const payload = await api("/api/assistant", {
      method: "POST",
      body: JSON.stringify({ question, item_id: itemId }),
    });
    renderAssistantMessage(pendingMessage, payload.answer || "", payload);
  } catch (error) {
    renderAssistantMessage(pendingMessage, error.message);
  } finally {
    button.disabled = false;
  }
}

function parseSseFrame(frame) {
  const lines = frame.split(/\r?\n/);
  let eventName = "message";
  const dataLines = [];
  lines.forEach((line) => {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
      return;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  });
  const data = dataLines.join("\n");
  return { eventName, payload: data ? parseRunEvent({ data }) : {} };
}

function handleAssistantStreamFrame(frame, pendingMessage, button) {
  const { eventName, payload } = parseSseFrame(frame);
  if (eventName === "assistant.started" || eventName === "assistant.progress") {
    renderAssistantProgress(pendingMessage, payload);
    return false;
  }
  if (eventName === "assistant.token") {
    appendAssistantToken(pendingMessage, payload.text || "");
    return false;
  }
  if (eventName === "assistant.completed") {
    renderAssistantMessage(pendingMessage, payload.answer || "", payload);
    button.disabled = false;
    return true;
  }
  if (eventName === "assistant.failed") {
    renderAssistantMessage(pendingMessage, payload.detail || payload.message || "Assistant failed");
    button.disabled = false;
    return true;
  }
  return false;
}

async function askAssistantStream(button, pendingMessage, question, itemId) {
  const params = new URLSearchParams({ question });
  if (itemId) params.set("item_id", itemId);
  const controller = new AbortController();
  state.assistantAbortController = controller;
  renderAssistantProgress(pendingMessage, { stage: "stream", status: "running", message: "Opening assistant stream" });

  const connectTimer = window.setTimeout(() => {
    if (state.assistantAbortController === controller) {
      controller.abort();
    }
  }, 6000);

  let receivedFrame = false;
  try {
    const response = await fetch(`/api/assistant/stream?${params.toString()}`, {
      headers: { Accept: "text/event-stream" },
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error(`Stream request failed: ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() || "";
      for (const frame of frames) {
        if (!frame.trim()) continue;
        receivedFrame = true;
        window.clearTimeout(connectTimer);
        const finished = handleAssistantStreamFrame(frame, pendingMessage, button);
        if (finished) {
          state.assistantAbortController = null;
          return;
        }
      }
    }
    if (buffer.trim()) {
      receivedFrame = true;
      handleAssistantStreamFrame(buffer, pendingMessage, button);
    }
  } catch (error) {
    if (error.name === "AbortError" && state.assistantAbortController !== controller) {
      return;
    }
    if (error.name === "AbortError" && receivedFrame) {
      return;
    }
    await askAssistantFallback(button, pendingMessage, question, itemId);
  } finally {
    window.clearTimeout(connectTimer);
    if (state.assistantAbortController === controller) {
      state.assistantAbortController = null;
    }
  }
}

function askAssistant(event) {
  if (event) event.preventDefault();
  const button = $("#askAssistantBtn");
  const question = $("#assistantQuestion").value.trim();
  const itemId = state.assistantContextItemId;
  if (!question) return;
  closeAssistantEventSource();
  button.disabled = true;
  appendAssistantMessage("user", question);
  $("#assistantQuestion").value = "";
  const pendingMessage = appendAssistantMessage("assistant", "Connecting to assistant...");
  askAssistantStream(button, pendingMessage, question, itemId);
}

function bindTabs() {
  $$(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".nav-item").forEach((item) => item.classList.remove("active"));
      $$(".tab-panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      $(`#${button.dataset.tab}`).classList.add("active");
    });
  });
}

document.body.addEventListener("click", (event) => {
  const detail = event.target.closest("[data-detail]");
  if (detail) {
    openDetail(detail.dataset.detail);
    return;
  }
  const assistantItem = event.target.closest("[data-assistant-item]");
  if (assistantItem) {
    openAssistant(assistantItem.dataset.assistantItem);
    return;
  }
  const feedback = event.target.closest("[data-feedback]");
  if (feedback) {
    sendFeedback(feedback.dataset.itemId, feedback.dataset.feedback).catch((error) => alert(error.message));
  }
});

$("#runBtn").addEventListener("click", runPipeline);
$("#profileForm").addEventListener("submit", saveProfile);
$("#assistantFab").addEventListener("click", () => openAssistant(null));
$("#assistantForm").addEventListener("submit", askAssistant);
$("#closeAssistantBtn").addEventListener("click", closeAssistant);
$$(".quick-prompt").forEach((button) => {
  button.addEventListener("click", () => {
    $("#assistantQuestion").value = button.dataset.prompt || "";
    askAssistant();
  });
});
$("#assistantQuestion").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askAssistant(event);
  }
});
$("#filterStatusSelect").addEventListener("change", renderFiltered);
$("#closeDrawerBtn").addEventListener("click", closeDetail);
$("#drawerBackdrop").addEventListener("click", closeDetail);
bindTabs();
loadInitial().catch((error) => alert(error.message));
