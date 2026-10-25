import { phonemize } from "https://esm.sh/phonemizer@1.2.1";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js";
const MODEL_URL = "voices/en_US-ryan-medium.onnx";
const CONFIG_URL = "voices/en_US-ryan-medium.onnx.json";
const SPEED = 0.8;
const MIN_HOLD = 0.06; // seconds; shorter sounds are merged into the previous shape
const KEEP = new Set([1, 7]); // lip-closing shapes are never merged
const HOP_LENGTH = 256;

const SHAPES = {};
const shapeGroups = {
  1: "mbp", 2: "æɛɪʌəɐɜ", 3: "szdntʃʒ", 4: "ie", 5: "θð", 6: "ouʊw",
  7: "fv", 8: "ɹrɚɝ", 9: "l", 10: "ɑɒɔa", 11: "kɡhŋj", 0: ".,!?;:^$",
};
for (const [shape, chars] of Object.entries(shapeGroups)) {
  for (const ch of chars) SHAPES[ch] = Number(shape);
}

const $ = (id) => document.getElementById(id);
const textInput = $("textInput");
const generateBtn = $("generateBtn");
const playBtn = $("playBtn");
const downloadFaceBtn = $("downloadFaceBtn");
const downloadMouthBtn = $("downloadMouthBtn");
const statusDiv = $("status");
const mouthImg = $("mouth");
const wordDiv = $("word");

let session, config, ort;
let audio = null;
let audioUrl = null;
let audioBlob = null;
let events = [];
let words = [];
let rafId = null;
const mouthImages = {};

const setStatus = (msg, type = "") => {
  statusDiv.textContent = msg;
  statusDiv.className = "status " + type;
};

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error("Failed to load " + src));
    document.head.appendChild(s);
  });
}

async function preloadMouthImages() {
  const promises = [];
  for (let i = 0; i <= 11; i++) {
    const img = new Image();
    img.src = `mouths/${i}.png`;
    promises.push(
      new Promise((resolve) => {
        img.onload = () => { mouthImages[i] = img; resolve(); };
        img.onerror = () => { console.warn(`Failed to load mouth image ${i}`); resolve(); };
      })
    );
  }
  await Promise.all(promises);
}

async function init() {
  setStatus("Loading voice model...", "loading");
  await preloadMouthImages();
  await loadScript(ORT_URL);
  ort = window.ort;
  ort.env.wasm.numThreads = 1; // GitHub Pages cannot send COOP/COEP headers for threads
  config = await (await fetch(CONFIG_URL)).json();
  const buf = await (await fetch(MODEL_URL)).arrayBuffer();
  session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  console.log("model outputs:", session.outputNames);
}

function toIds(ipa) {
  const map = config.phoneme_id_map;
  const ids = [];
  const chars = []; // phoneme character that each id belongs to
  const wordOf = []; // index of the espeak word that each id belongs to
  let word = 0;
  const push = (ch, id) => { ids.push(id); chars.push(ch); wordOf.push(word); };
  push("^", map["^"][0]);
  push("^", map["_"][0]);
  for (const ch of ipa.normalize("NFD")) {
    if (!(ch in map)) { console.warn("unmapped phoneme:", ch, ch.codePointAt(0).toString(16)); continue; }
    for (const id of map[ch]) push(ch, id);
    push(ch, map["_"][0]);
    if (ch === " ") word++;
  }
  push("$", map["$"][0]);
  return { ids, chars, wordOf };
}

function wavBlob(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => {
    const c = Math.max(-1, Math.min(1, s));
    v.setInt16(44 + i * 2, c < 0 ? c * 0x8000 : c * 0x7fff, true);
  });
  return new Blob([buf], { type: "audio/wav" });
}

function buildEvents(chars, durations, wordOf, labels, t0) {
  const out = [];
  const words = [];
  const nIpa = wordOf[wordOf.length - 1] + 1;
  let t = t0;
  chars.forEach((ch, i) => {
    if (!words.length || wordOf[i] !== words[words.length - 1].idx) {
      // Counts can differ for numbers/abbreviations, so scale the index onto the text words.
      const label = labels[Math.min(labels.length - 1, Math.floor((wordOf[i] * labels.length) / nIpa))];
      words.push({ start_time: t, word: label, idx: wordOf[i] });
    }
    const shape = SHAPES[ch];
    const last = out[out.length - 1];
    if (shape !== undefined && (!last || (shape !== last.mouth_shape &&
        (KEEP.has(shape) || t - last.start_time >= MIN_HOLD)))) {
      out.push({ start_time: t, mouth_shape: shape });
    }
    t += durations[i];
  });
  return { events: out, words };
}

async function synthesize(text) {
  // Long inputs degrade the model, so synthesize one sentence at a time like Piper does.
  const sentences = (text.match(/[^.!?]+[.!?]*/g) || []).filter((s) => /[\p{L}\p{N}]/u.test(s));
  if (!sentences.length) throw new Error("No speakable text");
  const chunks = [];
  const events = [];
  const words = [];
  let t0 = 0;
  for (const s of sentences) {
    const c = await synthChunk(s, t0);
    chunks.push(c.samples);
    events.push(...c.events);
    words.push(...c.words);
    t0 += c.samples.length / config.audio.sample_rate;
  }
  const all = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let off = 0;
  for (const c of chunks) { all.set(c, off); off += c.length; }
  return { blob: wavBlob(all, config.audio.sample_rate), events, words };
}

async function synthChunk(text, t0) {
  // phonemizer.js drops punctuation, which Piper keeps, so re-attach it per clause.
  const parts = text.split(/([.,!?;:]+)/).filter((s) => s.trim());
  let ipa = "";
  const labels = [];
  for (const part of parts) {
    if (/^[.,!?;:]+$/.test(part)) {
      ipa += part;
      if (labels.length) labels[labels.length - 1] += part;
    } else {
      const voice = config.espeak?.voice === "en" ? "en-us" : (config.espeak?.voice || "en-us");
      ipa += (ipa ? " " : "") + (await phonemize(part, voice)).join(" ");
      labels.push(...part.trim().split(/\s+/));
    }
  }
  console.log("phonemes:", ipa);
  const { ids, chars, wordOf } = toIds(ipa);
  const sr = config.audio.sample_rate;
  const inf = config.inference;

  const feeds = {
    input: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    input_lengths: new ort.Tensor("int64", BigInt64Array.from([BigInt(ids.length)]), [1]),
    scales: new ort.Tensor("float32", Float32Array.from([inf.noise_scale, 1 / SPEED, inf.noise_w]), [3]),
  };
  const result = await session.run(feeds);
  const samples = result[session.outputNames[0]].data;

  let durations;
  const durName = session.outputNames[1];
  if (durName && result[durName].data.length === ids.length) {
    durations = Array.from(result[durName].data, (d) => (Number(d) * HOP_LENGTH) / sr);
  } else {
    // No duration output in this model: spread the audio evenly across phonemes.
    durations = ids.map(() => samples.length / sr / ids.length);
  }
  return { samples, ...buildEvents(chars, durations, wordOf, labels, t0) };
}

const updateMouth = (shape) => { mouthImg.src = `mouths/${shape}.png`; };

function loop() {
  if (!audio || audio.paused) return;
  const t = audio.currentTime;
  let shape = 0;
  for (const e of events) {
    if (e.start_time <= t) shape = e.mouth_shape; else break;
  }
  updateMouth(shape);
  let w = "";
  for (const e of words) {
    if (e.start_time <= t) w = e.word; else break;
  }
  wordDiv.textContent = w;
  rafId = requestAnimationFrame(loop);
}

function drawMouth(ctx, shape, x, y, w, h) {
  const img = mouthImages[shape] || mouthImages[0];
  if (!img || !img.complete || !img.naturalWidth) return;

  const imgRatio = img.naturalWidth / img.naturalHeight;
  const boxRatio = w / h;

  let drawW, drawH;
  if (imgRatio > boxRatio) {
    drawW = w;
    drawH = w / imgRatio;
  } else {
    drawH = h;
    drawW = h * imgRatio;
  }

  const drawX = x + (w - drawW) / 2;
  const drawY = y + (h - drawH) / 2;

  ctx.drawImage(img, drawX, drawY, drawW, drawH);
}

function drawFrame(ctx, mode, shape) {
  const scale = 4;
  const offsetX = 140;
  const offsetY = 140;
  const mx = offsetX + 50 * scale;
  const my = offsetY + 100 * scale;
  const mw = 100 * scale;
  const mh = 60 * scale;

  if (mode === "face") {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, 1080, 1080);

    const lex = offsetX + 50 * scale;
    const ley = offsetY + 50 * scale;
    const ew = 30 * scale;
    const eh = 40 * scale;

    ctx.fillStyle = "#000000";
    ctx.beginPath();
    ctx.ellipse(lex + ew / 2, ley + eh / 2, ew / 2, eh / 2, 0, 0, 2 * Math.PI);
    ctx.fill();

    const rex = offsetX + 120 * scale;
    const rey = offsetY + 50 * scale;

    ctx.beginPath();
    ctx.ellipse(rex + ew / 2, rey + eh / 2, ew / 2, eh / 2, 0, 0, 2 * Math.PI);
    ctx.fill();

    drawMouth(ctx, shape, mx, my, mw, mh);
  } else {
    ctx.fillStyle = "#00FF00";
    ctx.fillRect(0, 0, 1080, 1080);

    drawMouth(ctx, shape, mx, my, mw, mh);
  }
}

function getMimeType() {
  const types = [
    "video/mp4;codecs=avc1",
    "video/mp4",
    "video/webm;codecs=h264",
    "video/webm;codecs=vp9",
    "video/webm"
  ];
  for (const t of types) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) {
      return t;
    }
  }
  return "";
}

async function exportVideo(mode) {
  if (!audioBlob || !events.length) return;

  generateBtn.disabled = playBtn.disabled = downloadFaceBtn.disabled = downloadMouthBtn.disabled = true;
  setStatus(`Exporting ${mode === "face" ? "face" : "greenscreen mouth"} MP4...`, "loading");

  try {
    const canvas = document.createElement("canvas");
    canvas.width = 1080;
    canvas.height = 1080;
    const ctx = canvas.getContext("2d");

    drawFrame(ctx, mode, 0);

    const arrayBuf = await audioBlob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuf);

    const destNode = audioCtx.createMediaStreamDestination();
    const sourceNode = audioCtx.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(destNode);

    const canvasStream = canvas.captureStream(60);
    const stream = new MediaStream([
      ...canvasStream.getVideoTracks(),
      ...destNode.stream.getAudioTracks()
    ]);

    const mimeType = getMimeType();
    const options = mimeType ? { mimeType } : {};
    let mediaRecorder;
    try {
      mediaRecorder = new MediaRecorder(stream, options);
    } catch (e) {
      mediaRecorder = new MediaRecorder(stream);
    }

    const chunks = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };

    const duration = audioBuffer.duration;

    await new Promise((resolve) => {
      let animId;
      mediaRecorder.onstop = () => {
        const recordedMime = mediaRecorder.mimeType || mimeType || "video/mp4";
        const blob = new Blob(chunks, { type: recordedMime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = mode === "face" ? "talking_face.mp4" : "mouth_greenscreen.mp4";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);

        audioCtx.close();
        resolve();
      };

      mediaRecorder.start();
      sourceNode.start(0);
      const audioStartTime = audioCtx.currentTime;

      function renderLoop() {
        const elapsed = audioCtx.currentTime - audioStartTime;
        if (elapsed >= duration + 0.1) {
          cancelAnimationFrame(animId);
          mediaRecorder.stop();
          sourceNode.stop();
          return;
        }

        let shape = 0;
        for (const e of events) {
          if (e.start_time <= elapsed) shape = e.mouth_shape;
          else break;
        }

        drawFrame(ctx, mode, shape);
        animId = requestAnimationFrame(renderLoop);
      }

      renderLoop();
    });

    setStatus("Export complete!", "success");
  } catch (err) {
    console.error(err);
    setStatus("Export error: " + err.message, "error");
  } finally {
    generateBtn.disabled = playBtn.disabled = downloadFaceBtn.disabled = downloadMouthBtn.disabled = false;
  }
}

generateBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) return setStatus("Please enter some text", "error");
  generateBtn.disabled = playBtn.disabled = downloadFaceBtn.disabled = downloadMouthBtn.disabled = true;
  setStatus("Generating...", "loading");
  await new Promise((r) => setTimeout(r, 50)); // let the browser paint before inference blocks the thread
  try {
    const { blob, events: ev, words: wd } = await synthesize(text);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioBlob = blob;
    audioUrl = URL.createObjectURL(blob);
    events = ev;
    words = wd;
    playBtn.disabled = downloadFaceBtn.disabled = downloadMouthBtn.disabled = false;
    setStatus("Press PLAY or DOWNLOAD", "success");
  } catch (err) {
    console.error(err);
    setStatus("Error: " + err.message, "error");
  } finally {
    generateBtn.disabled = false;
  }
});

playBtn.addEventListener("click", () => {
  if (!audio) {
    audio = new Audio();
    audio.addEventListener("ended", () => {
      playBtn.textContent = "PLAY";
      updateMouth(0);
      wordDiv.textContent = "";
      cancelAnimationFrame(rafId);
    });
  }
  if (audio.paused) {
    audio.src = audioUrl;
    audio.play().then(() => { playBtn.textContent = "STOP"; loop(); });
  } else {
    audio.pause();
    playBtn.textContent = "PLAY";
    updateMouth(0);
    wordDiv.textContent = "";
    cancelAnimationFrame(rafId);
  }
});

downloadFaceBtn.addEventListener("click", () => exportVideo("face"));
downloadMouthBtn.addEventListener("click", () => exportVideo("mouth"));

generateBtn.disabled = downloadFaceBtn.disabled = downloadMouthBtn.disabled = true;
init()
  .then(() => { generateBtn.disabled = false; setStatus("Ready."); })
  .catch((err) => { console.error(err); setStatus("Failed to load: " + err.message, "error"); });
