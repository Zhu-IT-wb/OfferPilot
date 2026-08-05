const { escapeHtml, newId, requestJson } = OfferPilotWeb;
const projectsApi = "/api/study/projects";
const sessionsApi = "/api/study/project-training/sessions";
const discoveryApi = "/api/study/project-discovery/jobs";
let projects = [];

function formLines(id) {
  return document
    .getElementById(id)
    .value.split(/[\n,，]/)
    .map((item) => item.trim())
    .filter(Boolean);
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
  document.getElementById("projects").innerHTML =
    active
      .map(
        (project) => `
          <article class="project-card">
            <h3>${escapeHtml(project.name)}</h3>
            <div>${escapeHtml(project.target_role || "尚未填写目标岗位")} · v${project.version}</div>
            <div class="tags">${project.tech_stack
              .slice(0, 6)
              .map((item) => `<span class="tag">${escapeHtml(item)}</span>`)
              .join("")}</div>
            ${missingFields(project)}
            <div class="actions">
              <button class="primary" data-start="${project.id}">开始训练</button>
              <button class="secondary" data-edit="${project.id}">编辑</button>
              ${project.source_repository_url ? `<button class="secondary" data-reanalyze="${project.id}">重新分析代码</button>` : ""}
              <button class="ghost" data-archive="${project.id}">归档</button>
            </div>
          </article>`,
      )
      .join("") || '<div class="panel empty">还没有项目档案，先创建一个真实项目。</div>';
  document.querySelectorAll("[data-start]").forEach((button) => {
    button.onclick = () => start(button.dataset.start);
  });
  document.querySelectorAll("[data-edit]").forEach((button) => {
    button.onclick = () => openEditor(projects.find((item) => item.id === button.dataset.edit));
  });
  document.querySelectorAll("[data-archive]").forEach((button) => {
    button.onclick = () => archiveProject(button.dataset.archive);
  });
  document.querySelectorAll("[data-reanalyze]").forEach((button) => {
    button.onclick = () => openImporter(projects.find((item) => item.id === button.dataset.reanalyze));
  });
}

function missingFields(project) {
  if (!project.missing_fields.length) return "";
  const labels = {
    responsibilities: "个人职责",
    technical_challenges: "技术难点",
    metrics: "量化指标",
  };
  const fields = project.missing_fields.map((field) => labels[field] || field);
  return `<p class="warning">建议补充：${fields.join("、")}</p>`;
}

function renderHistory(items) {
  const projectNames = Object.fromEntries(projects.map((item) => [item.id, item.name]));
  document.getElementById("history").innerHTML =
    items
      .map(
        (session) => `
          <div class="history-item">
            <div><strong>${escapeHtml(projectNames[session.project_id] || "历史项目")}</strong>
              <div>${escapeHtml(session.status)} · ${escapeHtml(session.difficulty)} · ${new Date(session.created_at).toLocaleString()}</div>
            </div>
            <button class="secondary" data-session="${session.id}">${session.status === "completed" ? "查看总结" : "继续训练"}</button>
          </div>`,
      )
      .join("") || '<div class="empty">暂无训练记录</div>';
  document.querySelectorAll("[data-session]").forEach((button) => {
    button.onclick = () => {
      location.href =
        `/study/projects/training?session_id=${encodeURIComponent(button.dataset.session)}`;
    };
  });
}

async function start(projectId) {
  const payload = await requestJson(sessionsApi, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_id: projectId,
      creation_id: newId("creation"),
      difficulty: "medium",
      max_turns: 6,
    }),
  });
  location.href =
    `/study/projects/training?session_id=${encodeURIComponent(payload.session.id)}`;
}

function openEditor(project) {
  if (!project || !project.id) return;
  document.getElementById("editor-title").textContent = "编辑项目";
  document.getElementById("project-id").value = project.id || "";
  const textFields = [
    ["name", "name"], ["target-role", "target_role"],
    ["background", "background"], ["architecture", "architecture"],
    ["resume-description", "resume_description"], ["supplemental-text", "supplemental_text"],
  ];
  textFields.forEach(([id, key]) => {
    document.getElementById(id).value = project[key] || "";
  });
  const listFields = [
    ["responsibilities", "responsibilities"], ["tech-stack", "tech_stack"],
    ["key-decisions", "key_decisions"], ["technical-challenges", "technical_challenges"],
    ["metrics", "metrics"], ["outcomes", "outcomes"],
  ];
  listFields.forEach(([id, key]) => {
    document.getElementById(id).value = (project[key] || []).join("\n");
  });
  document.getElementById("form-error").textContent = "";
  document.getElementById("editor").showModal();
}

async function save() {
  const id = document.getElementById("project-id").value;
  const value = (field) => document.getElementById(field).value.trim();
  const payload = {
    name: value("name"), target_role: value("target-role"), background: value("background"),
    responsibilities: formLines("responsibilities"), tech_stack: formLines("tech-stack"),
    architecture: value("architecture"), key_decisions: formLines("key-decisions"),
    technical_challenges: formLines("technical-challenges"), metrics: formLines("metrics"),
    outcomes: formLines("outcomes"), resume_description: value("resume-description"),
    supplemental_text: value("supplemental-text"),
  };
  try {
    await requestJson(`${projectsApi}/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    document.getElementById("editor").close();
    await load();
  } catch (error) {
    document.getElementById("form-error").textContent = error.message;
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
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({repository_url: repositoryUrl, request_id: newId("discovery"), project_id: projectId}),
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
document.getElementById("save-project").onclick = save;
load().catch((error) => {
  document.getElementById("projects").innerHTML =
    `<div class="error">${escapeHtml(error.message)}</div>`;
});
