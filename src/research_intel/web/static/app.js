const state = {
  profile: null,
  report: null,
  candidates: [],
  feedback: [],
  assistantContextItemId: "",
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
  $("#generatedAt").textContent = `${report.generated_at || ""} · ${report.source_mode || "unknown"}`;
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
  renderSourceErrors(report.source_errors || []);
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

function renderSourceErrors(errors) {
  const panel = $("#sourceErrorsPanel");
  if (!errors.length) {
    panel.classList.add("hidden");
    $("#sourceErrors").innerHTML = "";
    return;
  }
  panel.classList.remove("hidden");
  $("#sourceErrors").innerHTML = errors
    .slice(0, 8)
    .map((error) => `<div>${escapeHtml(error)}</div>`)
    .join("");
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

async function runPipeline() {
  const button = $("#runBtn");
  button.disabled = true;
  button.textContent = "Running";
  $("#runStatus").textContent = "Live sources may take a short while. Slow sources will be skipped and reported below.";
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
    renderReport();
  } catch (error) {
    $("#runStatus").textContent = `Run failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Run";
    if (!$("#runStatus").textContent.startsWith("Run failed")) {
      $("#runStatus").textContent = "Run complete.";
    }
  }
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
  $("#assistantDrawer").classList.remove("open");
}

function appendAssistantMessage(role, text) {
  const container = $("#assistantMessages");
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  container.appendChild(node);
  container.scrollTop = container.scrollHeight;
}

async function askAssistant(event) {
  if (event) event.preventDefault();
  const button = $("#askAssistantBtn");
  const question = $("#assistantQuestion").value.trim();
  const itemId = state.assistantContextItemId;
  if (!question) return;
  button.disabled = true;
  appendAssistantMessage("user", question);
  $("#assistantQuestion").value = "";
  appendAssistantMessage("assistant", "Thinking...");
  try {
    const payload = await api("/api/assistant", {
      method: "POST",
      body: JSON.stringify({ question, item_id: itemId }),
    });
    const messages = $$("#assistantMessages .message.assistant");
    messages[messages.length - 1].textContent = payload.answer || "";
  } catch (error) {
    const messages = $$("#assistantMessages .message.assistant");
    messages[messages.length - 1].textContent = error.message;
  } finally {
    button.disabled = false;
  }
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
