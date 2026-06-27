// Offscreen document: runs transformers.js in a real browser context.
// Uses AutoTokenizer + AutoModel directly to avoid pipeline()'s blob-worker layer,
// which Chrome MV3 CSP blocks (no blob: in script-src).
import { AutoTokenizer, AutoModel, env } from "./vendor/transformers.min.js";

env.allowLocalModels = false;
env.useBrowserCache = true;
env.backends.onnx.wasm.proxy = false;     // don't spawn ONNX's own proxy blob-worker
env.backends.onnx.wasm.numThreads = 1;   // single-threaded WASM, no threaded blob-workers
// Tell ONNX Runtime where to find the bundled WASM files instead of fetching from CDN
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL("vendor/");
let tokenizer = null;
let model = null;
let clf = null;
let initPromise = null;

/**
 * Apply the SupCon projection head (Linear + L2 normalise) that was exported
 * from train.py into classifier.json.  Falls back to identity if the weights
 * are absent (e.g. when using a baseline-only classifier.json).
 */
function applyProjection(emb, clf) {
  if (!clf.proj_weight) return emb;           // baseline fallback
  const W   = clf.proj_weight;                // (PROJ_DIM, EMBED_DIM)
  const b   = clf.proj_bias;                  // (PROJ_DIM,)
  const dim = b.length;
  const out = new Float32Array(dim);
  for (let j = 0; j < dim; j++) {
    out[j] = b[j];
    for (let i = 0; i < emb.length; i++) out[j] += W[j][i] * emb[i];
  }
  // L2 normalise (mirrors F.normalize in PyTorch)
  let norm = 0;
  for (let j = 0; j < dim; j++) norm += out[j] * out[j];
  norm = Math.sqrt(norm) || 1;
  for (let j = 0; j < dim; j++) out[j] /= norm;
  return out;
}

function meanPool(hiddenState, attMask) {
  const [, seqLen, hiddenSize] = hiddenState.dims;
  const h = hiddenState.data;
  const m = attMask.data;
  const out = new Float32Array(hiddenSize);
  let total = 0;
  for (let i = 0; i < seqLen; i++) {
    if (!Number(m[i])) continue; // handles BigInt64 and regular int
    total++;
    const off = i * hiddenSize;
    for (let j = 0; j < hiddenSize; j++) out[j] += h[off + j];
  }
  for (let j = 0; j < hiddenSize; j++) out[j] /= total || 1;
  return out;
}

function init() {
  if (initPromise) return initPromise;
  initPromise = Promise.all([
    AutoTokenizer.from_pretrained(
      "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
    ),
    AutoModel.from_pretrained("Xenova/paraphrase-multilingual-MiniLM-L12-v2", {
      quantized: true,
    }),
    fetch(chrome.runtime.getURL("classifier.json")).then((r) => r.json()),
  ]).then(([tok, mod, c]) => {
    tokenizer = tok;
    model = mod;
    clf = c;
  });
  return initPromise;
}

async function isJob(text) {
  await init();
  const inputs = tokenizer(text.slice(0, 1500), {
    truncation: true,
    max_length: 512,
    padding: true,
  });
  const output = await model(inputs);
  const emb = meanPool(output.last_hidden_state, inputs.attention_mask);
  const proj = applyProjection(emb, clf);

  // Logistic regression: coef · proj + intercept > 0 → class 1 (job)
  const coef = clf.coef[0];
  const intercept = clf.intercept[0];
  let score = intercept;
  for (let i = 0; i < proj.length; i++) score += coef[i] * proj[i];
  return score > 0;
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type !== 'DO_CLASSIFY') return;
  isJob(msg.text)
    .then(result => chrome.runtime.sendMessage({
      type: 'CLASSIFY_RESULT',
      id: msg.id,
      isJob: result,
    }))
    .catch(() => chrome.runtime.sendMessage({
      type: 'CLASSIFY_RESULT',
      id: msg.id,
      isJob: true, // safe default: show post on error
    }));
});
