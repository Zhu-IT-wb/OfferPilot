(function exposeVoiceRecorder(global) {
  class VoiceRecorder {
    constructor({ transcriptionUrl, jsapiConfigUrl, onState, onTranscript }) {
      this.transcriptionUrl = transcriptionUrl;
      this.jsapiConfigUrl = jsapiConfigUrl;
      this.onState = onState || (() => {});
      this.onTranscript = onTranscript || (() => {});
      this.stream = null;
      this.recorder = null;
      this.manager = null;
      this.blob = null;
      this.mimeType = "";
      this.startedAt = 0;
      this.timer = null;
      this.durationMs = 0;
      this.discardOnStop = false;
      this.onState("idle", { durationMs: 0 });
    }

    async start() {
      this.release(true);
      this.onState("requesting", {});
      try {
        if (navigator.mediaDevices?.getUserMedia && global.MediaRecorder) {
          await this.startBrowser();
          return;
        }
        if (global.tt?.getRecorderManager) {
          await this.startFeishu();
          return;
        }
        throw new Error("当前环境不支持录音，请使用文字回答。");
      } catch (error) {
        const message =
          error.name === "NotAllowedError"
            ? "麦克风权限被拒绝，请允许后重试。"
            : error.message;
        this.onState("error", { message });
        throw error;
      }
    }

    async startBrowser() {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      const types = [
        "audio/webm;codecs=opus",
        "audio/mp4",
        "audio/ogg;codecs=opus",
        "audio/aac",
      ];
      const type = types.find((item) => MediaRecorder.isTypeSupported?.(item)) || "";
      const chunks = [];
      const recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
      this.stream = stream;
      this.recorder = recorder;
      this.mimeType = (recorder.mimeType || type || "audio/webm").split(";", 1)[0];
      this.discardOnStop = false;
      recorder.ondataavailable = (event) => {
        if (event.data?.size) chunks.push(event.data);
      };
      recorder.onstop = async () => {
        const discard = this.discardOnStop;
        this.stopTracks();
        this.stopTimer();
        if (discard) return;
        this.blob = new Blob(chunks, { type: this.mimeType });
        await this.transcribe();
      };
      recorder.onerror = (event) =>
        this.onState("error", { message: event.error?.message || "录音失败" });
      recorder.start(1000);
      this.startTimer();
      this.onState("recording", {});
    }

    async ensureFeishu() {
      if (!global.h5sdk?.config) {
        throw new Error("飞书录音组件不可用，请使用文字回答。");
      }
      const response = await fetch(
        `${this.jsapiConfigUrl}?url=${encodeURIComponent(location.href.split("#", 1)[0])}`,
      );
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "飞书录音鉴权失败");
      }
      await new Promise((resolve, reject) =>
        global.h5sdk.config({
          appId: payload.app_id,
          timestamp: Number(payload.timestamp),
          nonceStr: payload.nonce_str,
          signature: payload.signature,
          jsApiList: ["tt.getRecorderManager"],
          onSuccess: resolve,
          onFail: () => reject(new Error("飞书录音鉴权失败")),
        }),
      );
    }

    async startFeishu() {
      await this.ensureFeishu();
      const manager = global.tt.getRecorderManager();
      this.manager = manager;
      this.mimeType = "audio/aac";
      this.discardOnStop = false;
      manager.onStop(async (result) => {
        const discard = this.discardOnStop;
        this.stopTimer();
        if (discard) return;
        try {
          this.blob = await this.readFeishuFile(result.tempFilePath);
          await this.transcribe();
        } catch (error) {
          this.onState("error", { message: error.message, retry: false });
        }
      });
      manager.onError((error) =>
        this.onState("error", { message: error.errMsg || "飞书录音失败" }),
      );
      manager.start({
        duration: 180000,
        sampleRate: 16000,
        numberOfChannels: 1,
        encodeBitRate: 64000,
        format: "aac",
      });
      this.startTimer();
      this.onState("recording", {});
    }

    async readFeishuFile(tempFilePath) {
      try {
        const response = await fetch(tempFilePath);
        if (response.ok) {
          return new Blob([await response.blob()], { type: "audio/aac" });
        }
      } catch (_error) {
        // Some Feishu mobile builds expose only the file-system JSAPI.
      }
      const fileSystem = global.tt?.getFileSystemManager?.();
      if (!fileSystem) {
        throw new Error("无法读取飞书录音，请重新录制。");
      }
      const data = await new Promise((resolve, reject) =>
        fileSystem.readFile({
          filePath: tempFilePath,
          success: (result) => resolve(result.data),
          fail: () => reject(new Error("无法读取飞书录音，请重新录制。")),
        }),
      );
      return new Blob([data], { type: "audio/aac" });
    }

    stop() {
      this.discardOnStop = false;
      if (this.recorder?.state !== "inactive") this.recorder.stop();
      else if (this.manager) this.manager.stop();
      this.onState("transcribing", {});
    }

    startTimer() {
      this.startedAt = Date.now();
      this.durationMs = 0;
      this.timer = setInterval(() => {
        this.durationMs = Math.min(180000, Date.now() - this.startedAt);
        this.onState("recording", { durationMs: this.durationMs });
        if (this.durationMs >= 180000) this.stop();
      }, 250);
    }

    stopTimer() {
      clearInterval(this.timer);
      this.timer = null;
      if (this.startedAt) {
        this.durationMs = Math.max(1, Math.min(180000, Date.now() - this.startedAt));
      }
    }

    async transcribe() {
      if (!this.blob?.size) {
        this.onState("error", { message: "录音为空，请重新录制。" });
        return;
      }
      this.onState("transcribing", {});
      try {
        const response = await fetch(this.transcriptionUrl, {
          method: "POST",
          headers: {
            "Content-Type": this.mimeType,
            "X-Audio-Duration-Ms": String(this.durationMs),
          },
          body: this.blob,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "语音转写失败");
        this.onTranscript(payload.transcript, payload);
        this.onState("ready", { durationMs: this.durationMs });
      } catch (error) {
        this.onState("error", { message: error.message, retry: true });
        throw error;
      }
    }

    stopTracks() {
      this.stream?.getTracks?.().forEach((track) => track.stop());
      this.stream = null;
    }

    release(discard = true) {
      clearInterval(this.timer);
      this.timer = null;
      this.discardOnStop = discard;
      try {
        if (this.recorder?.state !== "inactive") this.recorder.stop();
      } catch (_error) {}
      try {
        this.manager?.stop?.();
      } catch (_error) {}
      this.stopTracks();
      this.recorder = null;
      this.manager = null;
      if (discard) {
        this.blob = null;
        this.mimeType = "";
      }
    }
  }

  global.OfferPilotVoiceRecorder = VoiceRecorder;
})(globalThis);
