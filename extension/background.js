// Service worker: relays CLASSIFY requests to the offscreen document,
// which runs transformers.js in a real browser environment (window, etc.).

const OFFSCREEN_URL = 'offscreen.html';
const pending = new Map();
let nextId = 0;
let offscreenPromise = null;

function ensureOffscreen() {
  if (offscreenPromise) return offscreenPromise;
  offscreenPromise = chrome.offscreen.hasDocument().then(exists => {
    if (exists) return;
    return chrome.offscreen.createDocument({
      url: chrome.runtime.getURL(OFFSCREEN_URL),
      reasons: ['WORKERS'],
      justification: 'Run ONNX model inference for LinkedIn post classification',
    });
  }).catch(err => { offscreenPromise = null; throw err; });
  return offscreenPromise;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  // Result coming back from offscreen.js
  if (msg.type === 'CLASSIFY_RESULT') {
    const cb = pending.get(msg.id);
    if (cb) { cb({ isJob: msg.isJob }); pending.delete(msg.id); }
    return false;
  }

  if (msg.type !== 'CLASSIFY') return false;

  const id = nextId++;
  pending.set(id, sendResponse);

  ensureOffscreen()
    .then(() => chrome.runtime.sendMessage({ type: 'DO_CLASSIFY', text: msg.text, id }))
    .catch(err => {
      console.error('[JobFilter SW]', err);
      sendResponse({ isJob: true });
      pending.delete(id);
    });

  return true; // keep response channel open
});
