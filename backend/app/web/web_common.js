(function exposeOfferPilotWeb(global) {
  function escapeHtml(value) {
    return String(value ?? "").replace(
      /[&<>"']/g,
      (character) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[character],
    );
  }

  function newId(prefix) {
    return (
      global.crypto?.randomUUID?.() ||
      `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
    );
  }

  async function requestJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "请求失败");
    }
    return payload;
  }

  global.OfferPilotWeb = { escapeHtml, newId, requestJson };
})(globalThis);
