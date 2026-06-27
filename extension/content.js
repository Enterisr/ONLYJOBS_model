// Runs on LinkedIn pages. Classifies feed posts and hides non-job ones.

const seen = new WeakSet();

// ── Animal preload pool (dogs + cats) ────────────────────────────────────────
const POOL_SIZE = 15;   // keep this many URLs ready at all times
const animalQueue = []; // { url, type } objects ready to use instantly
const preloadImgs = []; // Image objects keeping browser cache warm

function fetchRandomAnimal() {
  if (Math.random() < 0.5) {
    // Dog from dog.ceo
    return fetch('https://dog.ceo/api/breeds/image/random')
      .then(r => r.json())
      .then(d => ({ url: d.message, type: 'dog' }));
  } else {
    // Cat from The Cat API (no key needed for basic use)
    return fetch('https://api.thecatapi.com/v1/images/search')
      .then(r => r.json())
      .then(d => ({ url: d[0].url, type: 'cat' }));
  }
}

function preloadOne() {
  fetchRandomAnimal().then(({ url, type }) => {
    const img = new Image();
    img.src = url;           // warms browser cache immediately
    preloadImgs.push(img);
    animalQueue.push({ url, type });
  }).catch(() => {});
}

function refillPool() {
  const needed = POOL_SIZE - animalQueue.length;
  for (let i = 0; i < needed; i++) preloadOne();
}

// Initial fill
refillPool();

// Refill aggressively when user nears the bottom of the page
let scrollTimer;
window.addEventListener('scroll', () => {
  clearTimeout(scrollTimer);
  scrollTimer = setTimeout(() => {
    const distFromBottom = document.documentElement.scrollHeight - window.scrollY - window.innerHeight;
    if (distFromBottom < 1500) refillPool();
  }, 100);
}, { passive: true });

fu