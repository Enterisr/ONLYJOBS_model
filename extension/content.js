// Runs on LinkedIn pages. Classifies feed posts and replaces non-job posts with animals.

const seen = new WeakSet();

// ── Animal preload pool (dogs + cats) ────────────────────────────────────────
const POOL_SIZE = 15;
const animalQueue = [];
const preloadImgs = [];

function fetchRandomAnimal() {
  if (Math.random() < 0.5) {
    return fetch('https://dog.ceo/api/breeds/image/random')
      .then(r => r.json())
      .then(d => ({ url: d.message, type: 'dog' }));
  } else {
    return fetch('https://api.thecatapi.com/v1/images/search')
      .then(r => r.json())
      .then(d => ({ url: d[0].url, type: 'cat' }));
  }
}

function preloadOne() {
  fetchRandomAnimal().then(({ url, type }) => {
    const img = new Image();
    img.src = url;
    preloadImgs.push(img);
    animalQueue.push({ url, type });
  }).catch(() => {});
}

function refillPool() {
  const needed = POOL_SIZE - animalQueue.length;
  for (let i = 0; i < needed; i++) preloadOne();
}

refillPool();

window.addEventListener('scroll', () => {
  const distFromBottom = document.documentElement.scrollHeight - window.scrollY - window.innerHeight;
  if (distFromBottom < 1500) refillPool();
}, { passive: true });

// ── Post classification & replacement ────────────────────────────────────────

function getText(el) {
  const textEl = el.querySelector(
    '.update-components-text, .feed-shared-update-v2__description, ' +
    '.feed-shared-text, .attributed-text-segment-list__content, ' +
    'span[dir="ltr"], span[dir="rtl"]'
  );
  return (textEl ? textEl.innerText : el.innerText || '').trim();
}

function replaceWithAnimal(el) {
  const animal = animalQueue.shift();
  refillPool();

  const emoji = animal ? (animal.type === 'dog' ? '🐶' : '🐱') : '🐾';
  const label = animal ? `Here's a ${animal.type} instead` : 'Not a job post';

  // Preserve the element's outer shell (margins, spacing) but replace inner content
  const inner = el.querySelector(':scope > *') || el;
  const target = inner !== el ? inner : el;

  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'text-align:center;padding:20px;background:#f3f2ef;border-radius:8px;';

  if (animal) {
    const img = document.createElement('img');
    img.src = animal.url;
    img.style.cssText = 'max-width:100%;max-height:360px;border-radius:8px;display:block;margin:0 auto;';
    wrapper.appendChild(img);
  }

  const caption = document.createElement('p');
  caption.style.cssText = 'color:#666;font-size:12px;margin-top:8px;';
  caption.textContent = `${emoji} ${label}`;
  wrapper.appendChild(caption);

  target.replaceChildren(wrapper);
}

function processPost(el) {
  if (seen.has(el)) return;
  seen.add(el);

  const text = getText(el).slice(0, 2000);
  if (text.length < 20) return;

  chrome.runtime.sendMessage({ type: 'CLASSIFY', text }, (response) => {
    if (chrome.runtime.lastError) return;
    if (response && response.isJob === false) {
      replaceWithAnimal(el);
    }
  });
}

function scan() {
  document.querySelectorAll('[componentkey], [data-id]').forEach(el => processPost(el));
}

scan();

new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
