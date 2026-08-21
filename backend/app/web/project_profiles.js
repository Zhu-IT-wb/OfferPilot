const { escapeHtml, newId, requestJson } = OfferPilotWeb;
const projectsApi = "/api/study/projects";
const sessionsApi = "/api/study/project-training/sessions";
const discoveryApi = "/api/study/project-discovery/jobs";

const sourceCopy = {
  independent: ["为什么想做这个项目", "例如：在准备面试时发现项目表达缺少针对性，因此希望做一个能分析项目并进行追问训练的 Agent。", "说明你的动机、目标用户和要解决的问题。"],
  open_source: ["在原项目基础上做了什么", "说明原项目是什么、你为什么选择它，以及你新增、重构或深入研究了哪一部分。", "重点区分原项目能力和你的实际贡献。"],
  internship: ["业务背景与团队目标", "在不泄露敏感信息的前提下，说明业务场景、团队目标和你接手时的问题。", "不要写公司机密，保留面试所需的业务上下文即可。"],
  course: ["任务要求与评审目标", "说明课程或比赛要求、完成周期、团队规模和最终评审标准。", "突出限制条件和你希望达到的目标。"],
  team: ["团队目标与协作背景", "说明团队为什么做这个项目、成员分工，以及你负责的范围。", "先交代共同目标，个人贡献会在下一步单独确认。"],
};

const responsibilityHints = {
  planning: "负责 [需求/模块] 的方案设计，基于 [约束] 确定 [方案]。",
  agent_workflow: "设计 [Agent/工作流] 的 [节点或协作机制]，解决 [问题]。",
  core_development: "实现 [核心模块]，完成 [关键能力]。",
  integration: "接入 [工具/平台]，处理 [协议、鉴权或异常]。",
  data_storage: "设计 [数据模型/存储方案]，满足 [一致性、检索或性能目标]。",
  quality: "补充 [测试/评估机制]，覆盖 [关键场景]。",
  deployment: "完成 [部署/监控/发布]，处理 [环境或稳定性问题]。",
  collaboration: "负责 [文档/协作流程]，推动 [交付结果]。",
};

const rowState = {
  responsibilities: [],
  "key-decisions": [],
  "technical-challenges": [],
  metrics: [],
  outcomes: [],
};

let projects = [];
let currentStep = 0;
let maxVisitedStep = 0;

function splitValues(value) {
  return String(value || "").split(/[\n,，]/).map((item) => item.trim()).filter(Boolean);
}

async function load() {
  const [projectPayload, historyPayload] = await Promise.all([
    requestJson(projectsApi),
    requestJson(sessionsApi),
  ]);
  projects = projectPayload.projects;
  renderProjects();
  renderHistory(historyPayload.sessions);
  const requestedProjectId = new URLSearchParams(location.search).get("start_project_id");
  if (requestedProjectId && projects.some((item) => item.id === requestedProjectId && item.status === "active")) {
    history.replaceState({}, "", "/study/projects");
    await start(requestedProjectId);
  }
}

function renderProjects() {
  const active = projects.filter((project) => project.status === "active");
  document.getElementById("projects").innerHTML = active.map((project) => `
    <article class="project-card">
      <h3>${escapeHtml(project.name)}</h3>
      <div>${escapeHtml(project.target_role || "尚未填写目标岗位")} · v${project.version}</div>
      <div class="tags">${project.tech_stack.slice(0, 6).map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join("")}</div>
      ${missingFields(project)}
      <div class="actions">
        <button class="primary" data-start="${project.id}">开始训练</button>
        <button class="secondary" data-edit="${project.id}">编辑</button>
        ${project.source_repository_url ? `<button class="secondary" data-reanalyze="${project.id}">重新分析代码</button>` : ""}
        <button class="ghost" data-archive="${project.id}">归档</button>
      </div>
    </article>`).join("") || '<div class="panel empty">还没有项目档案，先创建一个真实项目。</div>';
  document.querySelectorAll("[data-start]").forEach((button) => { button.onclick = () => start(button.dataset.start); });
  document.querySelectorAll("[data-edit]").forEach((button) => { button.onclick = () => openEditor(projects.find((item) => item.id === button.dataset.edit)); });
  document.querySelectorAll("[data-archive]").forEach((button) => { button.onclick = () => archiveProject(button.dataset.archive); });
  document.querySelectorAll("[data-reanalyze]").forEach((button) => { button.onclick = () => openImporter(projects.find((item) => item.id === button.dataset.reanalyze)); });
}

function missingFields(project) {
  if (!project.missing_fields.length) return "";
  const labels = { responsibilities: "个人职责", technical_challenges: "技术难点", metrics: "结果证据" };
  return `<p class="warning">建议补充：${project.missing_fields.map((field) => labels[field] || field).join("、")}</p>`;
}

function renderHistory(items) {
  const projectNames = Object.fromEntries(projects.map((item) => [item.id, item.name]));
  document.getElementById("history").innerHTML = items.map((session) => `
    <div class="history-item">
      <div><strong>${escapeHtml(projectNames[session.project_id] || "历史项目")}</strong><div>${escapeHtml(session.status)} · ${escapeHtml(session.difficulty)} · ${new Date(session.created_at).toLocaleString()}</div></div>
      <button class="secondary" data-session="${session.id}">${session.status === "completed" ? "查看总结" : "继续训练"}</button>
    </div>`).join("") || '<div class="empty">暂无训练记录</div>';
  document.querySelectorAll("[data-session]").forEach((button) => {
    button.onclick = () => { location.href = `/study/projects/training?session_id=${encodeURIComponent(button.dataset.session)}`; };
  });
}

async function start(projectId) {
  const payload = await requestJson(sessionsApi, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: projectId, creation_id: newId("creation"), difficulty: "medium", max_turns: 6 }),
  });
  location.href = `/study/projects/training?session_id=${encodeURIComponent(payload.session.id)}`;
}

function setCheckedValues(selector, values) {
  const selected = new Set(values || []);
  document.querySelectorAll(`${selector} input`).forEach((input) => { input.checked = selected.has(input.value); });
}

function checkedValues(selector) {
  return Array.from(document.querySelectorAll(`${selector} input:checked`)).map((input) => input.value);
}

function setRadioValue(name, value) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((input) => { input.checked = input.value === value; });
}

function radioValue(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value || "";
}

function seedRows(field, values, suggested) {
  rowState[field] = (values || []).map((value) => ({ id: newId("row"), value, suggested }));
  renderRows(field);
}

function collectRows(field) {
  const previous = new Map(rowState[field].map((row) => [row.id, row]));
  rowState[field] = Array.from(document.querySelectorAll(`#${field}-list .editable-row`)).map((wrapper) => {
    const value = wrapper.querySelector("[data-row-input]").value.trim();
    const stored = previous.get(wrapper.dataset.rowId);
    return { id: wrapper.dataset.rowId, value, suggested: stored?.suggested || false };
  }).filter((row) => row.value);
  return rowState[field].map((row) => row.value);
}

function renderRows(field) {
  const container = document.getElementById(`${field}-list`);
  container.replaceChildren();
  if (!rowState[field].length) {
    const empty = document.createElement("p");
    empty.className = "editable-list-empty";
    empty.textContent = "还没有内容，可以从上面的类型开始回忆，或添加一项。";
    container.appendChild(empty);
    return;
  }
  rowState[field].forEach((row) => {
    const wrapper = document.createElement("div");
    wrapper.className = "editable-row";
    wrapper.dataset.rowId = row.id;
    const textarea = document.createElement("textarea");
    textarea.rows = 1;
    textarea.value = row.value;
    textarea.dataset.rowInput = "";
    textarea.setAttribute("aria-label", "可编辑内容");
    textarea.addEventListener("input", () => resizeTextarea(textarea));
    const meta = document.createElement("div");
    meta.className = "row-meta";
    if (row.suggested) {
      const badge = document.createElement("span");
      badge.textContent = "AI 识别候选";
      meta.appendChild(badge);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-button small";
    remove.title = "删除这一项";
    remove.setAttribute("aria-label", "删除这一项");
    remove.textContent = "×";
    remove.onclick = () => {
      collectRows(field);
      rowState[field] = rowState[field].filter((item) => item.id !== row.id);
      renderRows(field);
    };
    meta.appendChild(remove);
    wrapper.append(textarea, meta);
    container.appendChild(wrapper);
    resizeTextarea(textarea);
  });
}

function resizeTextarea(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, 44), 180)}px`;
}

function addRow(field) {
  collectRows(field);
  rowState[field].push({ id: newId("row"), value: "", suggested: false });
  renderRows(field);
  const inputs = document.querySelectorAll(`#${field}-list [data-row-input]`);
  inputs[inputs.length - 1]?.focus();
}

function updateSourceCopy() {
  const copy = sourceCopy[radioValue("project-source")];
  if (!copy) return;
  document.getElementById("background-label").textContent = copy[0];
  document.getElementById("background").placeholder = copy[1];
  document.getElementById("background-help").textContent = copy[2];
}

function updateResponsibilityHint() {
  const example = checkedValues("#responsibility-categories").map((value) => responsibilityHints[value]).find(Boolean);
  document.getElementById("responsibility-hint").textContent = example ? `参考句式：${example}` : "推荐句式：负责 [模块] 的 [设计/实现]，解决 [具体问题]。";
}

function updateMetricsMode() {
  const mode = radioValue("metrics-status");
  document.getElementById("metrics-group").hidden = mode === "none" || !mode;
  document.getElementById("metrics-title").textContent = mode === "validated" ? "测试或验收结果" : "量化指标";
  document.getElementById("metrics-hint").textContent = mode === "validated"
    ? "例如：通过 12 个核心场景测试；完成飞书回调联调。"
    : "写清指标口径和数据来源，不确定的数字不要填写。";
}

function openEditor(project) {
  if (!project || !project.id) return;
  currentStep = 0;
  maxVisitedStep = 0;
  document.getElementById("project-id").value = project.id;
  document.getElementById("name").value = project.name || "";
  document.getElementById("target-role").value = project.target_role || "";
  document.getElementById("background").value = project.background || "";
  document.getElementById("tech-stack").value = (project.tech_stack || []).join(", ");
  document.getElementById("architecture").value = project.architecture || "";
  document.getElementById("resume-description").value = project.resume_description || "";
  document.getElementById("supplemental-text").value = project.supplemental_text || "";
  setRadioValue("project-source", project.project_source || (project.source_repository_url ? "open_source" : ""));
  setCheckedValues("#responsibility-categories", project.responsibility_categories || []);
  setCheckedValues("#outcome-categories", project.outcome_categories || []);
  setRadioValue("metrics-status", project.metrics_status || ((project.metrics || []).length ? "quantified" : ""));
  const suggested = Boolean(project.source_discovery_job_id);
  seedRows("responsibilities", project.responsibilities, suggested);
  seedRows("key-decisions", project.key_decisions, suggested);
  seedRows("technical-challenges", project.technical_challenges, suggested);
  seedRows("metrics", project.metrics, suggested);
  seedRows("outcomes", project.outcomes, suggested);
  updateSourceCopy();
  updateResponsibilityHint();
  updateMetricsMode();
  showStep(0);
  document.getElementById("editor").showModal();
}

function validateStep(step) {
  const error = document.getElementById("form-error");
  error.textContent = "";
  if (step === 0 && !radioValue("project-source")) {
    error.textContent = "请选择项目来源，这会决定后续问题。";
    return false;
  }
  if (step === 0 && !document.getElementById("name").value.trim()) {
    error.textContent = "请填写项目名称。";
    document.getElementById("name").focus();
    return false;
  }
  if (step === 3 && !radioValue("metrics-status")) {
    error.textContent = "请选择当前是否有数据或验收结果；没有也可以如实选择“暂时没有”。";
    return false;
  }
  return true;
}

function showStep(step) {
  currentStep = step;
  maxVisitedStep = Math.max(maxVisitedStep, step);
  document.querySelectorAll("[data-editor-step]").forEach((pane) => { pane.classList.toggle("is-active", Number(pane.dataset.editorStep) === step); });
  document.querySelectorAll("[data-step-target]").forEach((button) => {
    const target = Number(button.dataset.stepTarget);
    button.classList.toggle("is-active", target === step);
    button.classList.toggle("is-complete", target < step);
    button.disabled = target > maxVisitedStep;
  });
  document.getElementById("previous-step").hidden = step === 0;
  document.getElementById("next-step").hidden = step === 3;
  document.getElementById("save-project").hidden = step !== 3;
  document.getElementById("form-error").textContent = "";
  if (step === 3 && !document.getElementById("resume-description").value.trim()) generateResume();
  document.querySelector(".editor-content").scrollTo({ top: 0, behavior: "smooth" });
}

function generateResume() {
  const name = document.getElementById("name").value.trim();
  const responsibilities = collectRows("responsibilities");
  const decisions = collectRows("key-decisions");
  const challenges = collectRows("technical-challenges");
  const metrics = radioValue("metrics-status") === "none" ? [] : collectRows("metrics");
  const outcomes = collectRows("outcomes");
  const lines = [];
  const work = responsibilities.slice(0, 2).join("；");
  const depth = decisions[0] || challenges[0] || "";
  const result = metrics[0] || outcomes[0] || "";
  if (work) lines.push(`• ${name ? `${name}：` : ""}${work}${depth ? `；${depth}` : ""}${result ? `；${result}` : ""}`);
  else if (depth) lines.push(`• ${name ? `${name}：` : ""}${depth}${result ? `；${result}` : ""}`);
  const stack = splitValues(document.getElementById("tech-stack").value).slice(0, 6);
  if (stack.length) lines.push(`• 技术栈：${stack.join("、")}`);
  document.getElementById("resume-description").value = lines.join("\n");
}

function buildPayload() {
  const metricsStatus = radioValue("metrics-status");
  return {
    name: document.getElementById("name").value.trim(),
    target_role: document.getElementById("target-role").value.trim(),
    project_source: radioValue("project-source"),
    background: document.getElementById("background").value.trim(),
    responsibility_categories: checkedValues("#responsibility-categories"),
    responsibilities: collectRows("responsibilities"),
    tech_stack: splitValues(document.getElementById("tech-stack").value),
    architecture: document.getElementById("architecture").value.trim(),
    key_decisions: collectRows("key-decisions"),
    technical_challenges: collectRows("technical-challenges"),
    metrics: metricsStatus === "none" ? [] : collectRows("metrics"),
    metrics_status: metricsStatus,
    outcome_categories: checkedValues("#outcome-categories"),
    outcomes: collectRows("outcomes"),
    resume_description: document.getElementById("resume-description").value.trim(),
    supplemental_text: document.getElementById("supplemental-text").value.trim(),
  };
}

async function save() {
  if (currentStep !== 0) showStep(0);
  if (!validateStep(0)) return;
  showStep(3);
  if (!validateStep(3)) return;
  const button = document.getElementById("save-project");
  button.disabled = true;
  button.textContent = "保存中…";
  try {
    await requestJson(`${projectsApi}/${document.getElementById("project-id").value}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(buildPayload()),
    });
    document.getElementById("editor").close();
    await load();
  } catch (error) {
    document.getElementById("form-error").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "保存项目档案";
  }
}

function openImporter(project = {}) {
  document.getElementById("import-project-id").value = project.id || "";
  document.getElementById("repository-url").value = project.source_repository_url || "";
  document.getElementById("import-error").textContent = "";
  document.getElementById("importer").showModal();
}

async function startImport() {
  const button = document.getElementById("start-import");
  const repositoryUrl = document.getElementById("repository-url").value.trim();
  const projectId = document.getElementById("import-project-id").value || null;
  button.disabled = true;
  try {
    const payload = await requestJson(discoveryApi, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repository_url: repositoryUrl, request_id: newId("discovery"), project_id: projectId }),
    });
    location.href = `/study/projects/discovery?job_id=${encodeURIComponent(payload.job.id)}`;
  } catch (error) {
    document.getElementById("import-error").textContent = error.message;
    button.disabled = false;
  }
}

async function archiveProject(id) {
  if (!confirm("归档后不能开始新的训练，历史记录仍会保留。确定归档？")) return;
  await requestJson(`${projectsApi}/${id}/archive`, { method: "POST" });
  await load();
}

document.getElementById("new-project").onclick = () => openImporter();
document.getElementById("start-import").onclick = startImport;
document.getElementById("project-form").onsubmit = (event) => event.preventDefault();
document.getElementById("close-editor").onclick = () => document.getElementById("editor").close();
document.getElementById("save-project").onclick = save;
document.getElementById("next-step").onclick = () => { if (validateStep(currentStep) && currentStep < 3) showStep(currentStep + 1); };
document.getElementById("previous-step").onclick = () => showStep(Math.max(0, currentStep - 1));
document.getElementById("generate-resume").onclick = generateResume;
document.querySelectorAll("[data-add-row]").forEach((button) => { button.onclick = () => addRow(button.dataset.addRow); });
document.querySelectorAll("[data-step-target]").forEach((button) => { button.onclick = () => showStep(Number(button.dataset.stepTarget)); });
document.querySelectorAll('input[name="project-source"]').forEach((input) => { input.onchange = updateSourceCopy; });
document.querySelectorAll("#responsibility-categories input").forEach((input) => { input.onchange = updateResponsibilityHint; });
document.querySelectorAll('input[name="metrics-status"]').forEach((input) => { input.onchange = updateMetricsMode; });

load().catch((error) => {
  document.getElementById("projects").innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
});
