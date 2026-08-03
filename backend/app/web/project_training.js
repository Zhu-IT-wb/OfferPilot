const { escapeHtml, newId, requestJson } = OfferPilotWeb;
const sessionId = new URLSearchParams(location.search).get("session_id");
const base = `/api/study/project-training/sessions/${encodeURIComponent(sessionId || "")}`;

let data = null;
let voice = null;
let answerSource = "text";
let submissionId = null;

const dimensionLabels = {
  ownership: "个人贡献",
  technical_depth: "技术深度",
  evidence: "数据证据",
  tradeoff: "方案取舍",
  reliability: "故障边界",
  structure: "表达结构",
  clarity: "表达清晰",
};

async function load() {
  if (!sessionId) throw new Error("缺少训练 Session ID");
  data = await requestJson(`${base}/current`);
  render();
}

function render() {
  const projectName = data.project?.name || "历史项目";
  document.getElementById("session-meta").textContent =
    `${projectName} · ${data.session.target_role || "项目专项"} · ` +
    `${data.session.difficulty}`;
  if (data.session.status === "completed") {
    renderSummary();
    return;
  }
  if (data.session.status === "abandoned") {
    document.getElementById("training").innerHTML =
      '<div class="panel empty">这场训练已经放弃。</div>';
    return;
  }

  const turn = data.current_turn;
  document.getElementById("training").innerHTML = `
    <article class="question-card">
      <div class="progress-line">
        <span>${escapeHtml(projectName)} · ${escapeHtml(turn.theme)} · 第 ${turn.sequence} / ${data.session.max_turns} 题</span>
        <span>剩余 ${data.remaining_turns} 题 · ${turn.hypothetical ? "假设场景" : "项目深挖"}</span>
      </div>
      <h1>${escapeHtml(turn.question_text)}</h1>
      <section class="voice-panel" id="voice-panel">
        <strong><span class="record-dot"></span>用语音回答</strong>
        <div class="voice-status" id="voice-status">说出来，才是真正掌握。录完会转成可编辑文字。</div>
        <div class="actions">
          <button class="secondary" id="voice-start">开始录音</button>
          <button class="secondary" id="voice-stop" hidden>停止录音</button>
          <button class="ghost" id="voice-retry" hidden>重新转写</button>
        </div>
      </section>
      <textarea class="answer-box" id="answer" placeholder="建议回答 1～2 分钟。语音转写后请检查技术术语。"></textarea>
      <p class="error" id="error"></p>
      <div class="actions">
        <button class="primary" id="submit">提交回答</button>
        <button class="ghost" id="finish">提前结束并生成总结</button>
      </div>
      <section id="feedback"></section>
    </article>`;
  setupVoice(turn.id);
  document.getElementById("submit").onclick = submit;
  document.getElementById("finish").onclick = finish;
}

function setupVoice(turnId) {
  voice?.release();
  voice = new OfferPilotVoiceRecorder({
    transcriptionUrl:
      `/api/study/project-training/transcriptions?turn_id=${encodeURIComponent(turnId)}`,
    jsapiConfigUrl: "/api/study/project-training/jsapi-config",
    onTranscript: (text) => {
      document.getElementById("answer").value = text;
      answerSource = "voice_transcript";
    },
    onState: updateVoiceState,
  });
}

function updateVoiceState(mode, detail) {
  const panel = document.getElementById("voice-panel");
  if (!panel) return;
  const status = document.getElementById("voice-status");
  const start = document.getElementById("voice-start");
  const stop = document.getElementById("voice-stop");
  const retry = document.getElementById("voice-retry");
  panel.classList.toggle("recording", mode === "recording");
  start.hidden = !["idle", "ready", "error"].includes(mode);
  stop.hidden = mode !== "recording";
  retry.hidden = !(mode === "error" && detail.retry);
  if (mode === "recording") {
    status.textContent = `正在录音 ${formatTime(detail.durationMs || 0)} / 03:00`;
  } else if (mode === "transcribing") {
    status.textContent = "正在转写，请稍候…";
  } else if (mode === "ready") {
    status.textContent = "转写完成，请检查技术术语后提交。";
  } else {
    status.textContent = detail.message || "录完后会自动转成文字。";
  }
  start.onclick = () => voice.start().catch(() => {});
  stop.onclick = () => voice.stop();
  retry.onclick = () => voice.transcribe().catch(() => {});
}

async function submit() {
  const answer = document.getElementById("answer").value.trim();
  if (!answer) {
    document.getElementById("answer").focus();
    return;
  }
  submissionId = submissionId || newId("submission");
  setBusy(true);
  try {
    const payload = await requestJson(`${base}/answers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        turn_id: data.current_turn.id,
        submission_id: submissionId,
        answer_text: answer,
        answer_source: answerSource,
      }),
    });
    showFeedback(payload);
  } catch (error) {
    document.getElementById("error").textContent =
      `${error.message}，答案仍保留，可以重试。`;
    setBusy(false);
  }
}

function list(items, fallback) {
  const values = items?.length ? items : [fallback];
  return values.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function showFeedback(payload) {
  voice?.release();
  const evaluation = payload.evaluation;
  const dimensions = Object.entries(evaluation.dimension_scores)
    .map(
      ([name, score]) => `
        <div class="dimension">
          <strong>${escapeHtml(dimensionLabels[name] || name)}</strong><div>${score}</div>
        </div>`,
    )
    .join("");
  const answerEvidence = evaluation.answer_evidence
    ?.map(
      (item) =>
        `<li><strong>${escapeHtml(dimensionLabels[item.dimension] || item.dimension)}</strong>：“${escapeHtml(item.quote)}”</li>`,
    )
    .join("");
  const warningSections = `
    ${evaluation.unsupported_claims.length ? `<h3>证据不足的表述</h3><ul>${list(evaluation.unsupported_claims, "")}</ul>` : ""}
    ${evaluation.contradictions.length ? `<h3>与项目档案矛盾</h3><ul>${list(evaluation.contradictions, "")}</ul>` : ""}`;
  document.getElementById("feedback").innerHTML = `
    <section class="feedback-card">
      <div class="score">${payload.answer.overall_score}</div>
      <div class="dimensions">${dimensions}</div>
      <h3>做得好的地方</h3><ul>${list(evaluation.strengths, "继续补充项目细节")}</ul>
      <h3>回答中的证据</h3><ul>${answerEvidence || "<li>本题没有抽取到可引用的原文。</li>"}</ul>
      <h3>需要补充</h3><ul>${list(evaluation.missing_details, "没有明显遗漏")}</ul>
      ${warningSections}
      <h3>AI 建议</h3><p>${escapeHtml(evaluation.feedback)}</p>
      <h3>更好的回答结构</h3><ol>${list(evaluation.improved_outline, "先讲背景，再讲个人行动和结果")}</ol>
      <div class="actions"><button class="primary" id="next">${payload.current_turn ? "下一题" : "查看总结"}</button></div>
    </section>`;
  data = {
    session: payload.session,
    current_turn: payload.current_turn,
    project: payload.project,
    remaining_turns: payload.remaining_turns,
  };
  document.getElementById("next").onclick = () => {
    submissionId = null;
    answerSource = "text";
    render();
  };
}

async function finish() {
  if (!confirm("确定提前结束并生成当前训练总结？")) return;
  await requestJson(`${base}/finish`, { method: "POST" });
  data.session.status = "completed";
  data.current_turn = null;
  renderSummary();
}

async function renderSummary() {
  voice?.release();
  const { summary } = await requestJson(`${base}/summary`);
  const topics = Object.entries(summary.topic_scores)
    .map(
      ([topic, score]) =>
        `<div class="dimension"><strong>${escapeHtml(topic)}</strong><div>${score}</div></div>`,
    )
    .join("");
  document.getElementById("training").innerHTML = `
    <section class="feedback-card">
      <p class="eyebrow">TRAINING SUMMARY</p><h1>项目训练总结</h1>
      <div class="summary-grid"><div class="summary-score">${summary.overall_score}</div>
        <div><h3>主题得分</h3><div class="dimensions">${topics}</div></div>
      </div>
      <h3>优势</h3><ul>${list(summary.strengths, "已完成本次训练")}</ul>
      <h3>下一步改进</h3><ul>${list(summary.improvement_areas, "继续保持结构化表达")}</ul>
      ${summary.unsupported_claims.length ? `<h3>需要补证据的表述</h3><ul>${list(summary.unsupported_claims, "")}</ul>` : ""}
      ${summary.contradictions.length ? `<h3>与项目档案矛盾</h3><ul>${list(summary.contradictions, "")}</ul>` : ""}
      <div class="actions">
        <button class="primary" id="retrain">重练薄弱主题</button>
        <a class="secondary" href="/study/projects">返回项目训练</a>
      </div>
    </section>`;
  document.getElementById("retrain").onclick = async () => {
    const created = await requestJson("/api/study/project-training/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: summary.project_id,
        creation_id: newId("retrain"),
        difficulty: data.session.difficulty,
        max_turns: 3,
        focus_topics: summary.recommended_topics,
      }),
    });
    location.href =
      `/study/projects/training?session_id=${encodeURIComponent(created.session.id)}`;
  };
}

function setBusy(value) {
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = value;
  });
}

function formatTime(milliseconds) {
  const seconds = Math.floor(milliseconds / 1000);
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

document.getElementById("abandon").onclick = async () => {
  if (confirm("放弃后本场不会进入正式总结，确定继续？")) {
    await requestJson(`${base}/abandon`, { method: "POST" });
    location.href = "/study/projects";
  }
};
addEventListener("beforeunload", () => voice?.release(true));
load().catch((error) => {
  document.getElementById("training").innerHTML =
    `<div class="panel error">${escapeHtml(error.message)}</div>`;
});
