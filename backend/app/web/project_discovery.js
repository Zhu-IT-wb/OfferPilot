const { escapeHtml, newId, requestJson } = OfferPilotWeb;
const params = new URLSearchParams(location.search);
const jobId = params.get("job_id");
const base = `/api/study/project-discovery/jobs/${encodeURIComponent(jobId || "")}`;
let currentJob = null;
let timer = null;

const labels = {
  queued: "等待分析", running: "正在分析", needs_input: "需要你补充信息",
  ready: "可以确认", confirmed: "已创建项目档案", failed: "分析失败", cancelled: "已取消",
};
const stages = {cloning:"读取公开仓库",inventory:"识别仓库结构",selecting:"选择分析文件",analyzing:"分析代码证据",synthesizing:"生成项目草稿",validating:"校验证据"};

async function load() {
  if (!jobId) throw new Error("缺少 job_id，请从项目列表重新开始。");
  const payload = await requestJson(base);
  currentJob = payload.job;
  render(currentJob);
  if (["queued", "running"].includes(currentJob.status)) {
    clearTimeout(timer); timer = setTimeout(() => load().catch(showError), 2000);
  }
}

function render(job) {
  document.getElementById("status-label").textContent = labels[job.status] || job.status;
  document.getElementById("progress-value").textContent = `${job.progress}%`;
  document.getElementById("progress-bar").style.width = `${job.progress}%`;
  document.getElementById("stage-text").textContent = stages[job.stage] || "";
  const stats = job.stats || {};
  document.getElementById("stats").innerHTML = stats.file_count ?
    `<span class="status-pill">${stats.file_count} 个文件</span> <span class="status-pill">分析 ${stats.analyzed_file_count || 0} 个</span> <span class="status-pill">${escapeHtml(Object.keys(stats.languages || {}).join(" / ") || "语言待识别")}</span>` : "";
  document.getElementById("warnings").innerHTML = (job.warnings || []).map((item) => `<p class="warning">${escapeHtml(item)}</p>`).join("");
  document.getElementById("error").textContent = job.error_message || "";
  renderActions(job); renderDraft(job); renderEvidence(job); renderQuestion(job);
  document.getElementById("confirm").hidden = job.status !== "ready";
  if (job.status === "confirmed" && job.confirmed_project_id) location.replace(`/study/projects?start_project_id=${encodeURIComponent(job.confirmed_project_id)}`);
}

function renderActions(job) {
  const container = document.getElementById("task-actions");
  container.innerHTML = "";
  if (job.status === "failed") container.innerHTML = '<button id="retry" class="primary">重新分析</button>';
  if (["queued","running","needs_input","ready"].includes(job.status)) container.innerHTML += '<button id="cancel" class="ghost">取消任务</button>';
  if (document.getElementById("retry")) document.getElementById("retry").onclick = () => mutate(`${base}/retry`);
  if (document.getElementById("cancel")) document.getElementById("cancel").onclick = () => mutate(`${base}/cancel`);
}

function renderDraft(job) {
  const container = document.getElementById("draft");
  if (!job.draft || !Object.keys(job.draft).length) { container.hidden = true; return; }
  const labels = {name:"项目名称",background:"项目背景",tech_stack:"技术栈",architecture:"核心架构",key_decisions:"关键技术选型",technical_challenges:"技术难点",responsibilities:"个人职责",metrics:"指标",outcomes:"项目结果",resume_description:"简历描述",supplemental_text:"测试、部署与补充信息"};
  container.innerHTML = `<h2>AI 生成的待确认草稿</h2>${Object.entries(labels).map(([key,label]) => {
    const raw = job.draft[key]; const value = Array.isArray(raw) ? raw.join("；") : raw;
    return value ? `<div class="draft-section"><strong>${label}</strong><div>${escapeHtml(value)}</div></div>` : "";
  }).join("")}`;
  container.hidden = false;
}

function renderEvidence(job) {
  const container = document.getElementById("evidence");
  if (!(job.evidence || []).length) { container.hidden = true; return; }
  container.innerHTML = `<h2>代码与用户证据</h2><p class="muted">结论只保留能够回到以下原文的内容。</p>${job.evidence.map((item) => `<details><summary>${escapeHtml(item.source_type === "user_statement" ? "你的补充说明" : item.file_path)} ${item.start_line ? `L${item.start_line}–L${item.end_line}` : ""}</summary>${item.claim ? `<p><strong>支持的结论：</strong>${escapeHtml(item.claim)}</p>` : ""}<pre>${escapeHtml(item.excerpt)}</pre></details>`).join("")}`;
  container.hidden = false;
}

function renderQuestion(job) {
  const pending = (job.questions || []).find((item) => !item.answered);
  const container = document.getElementById("question");
  if (!pending || !["needs_input","ready"].includes(job.status)) { container.hidden = true; return; }
  container.innerHTML = `<div class="progress-line"><span>${pending.required ? "必答" : "可选"}</span><span>单题补充</span></div><h1>${escapeHtml(pending.prompt)}</h1><textarea id="answer" class="answer-box" maxlength="5000" placeholder="请基于你的真实经历回答；没有指标或结果可以填写“暂无”。"></textarea><div class="actions"><button id="submit-answer" class="primary">保存并继续</button></div><p id="answer-error" class="error"></p>`;
  container.hidden = false;
  document.getElementById("submit-answer").onclick = async () => {
    const answer = document.getElementById("answer").value.trim();
    if (!answer) { document.getElementById("answer-error").textContent = "请先填写回答。"; return; }
    try { const payload = await requestJson(`${base}/answers`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question_id:pending.id,answer_text:answer,submission_id:newId("discovery-answer")})}); currentJob=payload.job; render(currentJob); }
    catch (error) { document.getElementById("answer-error").textContent = error.message; }
  };
}

async function mutate(url) { const payload = await requestJson(url,{method:"POST"}); currentJob=payload.job; render(currentJob); if (["queued","running"].includes(currentJob.status)) timer=setTimeout(()=>load().catch(showError),2000); }
function showError(error) { document.getElementById("error").textContent = error.message; }
document.getElementById("confirm-project").onclick = async () => { try { const payload=await requestJson(`${base}/confirm`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({confirmation_id:newId("discovery-confirm")})}); location.href=`/study/projects?start_project_id=${encodeURIComponent(payload.project.id)}`; } catch(error){ document.getElementById("confirm-error").textContent=error.message; } };
load().catch(showError);
