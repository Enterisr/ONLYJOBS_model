// Runs on LinkedIn pages. Classifies feed posts and hides non-job ones.

const seen = new WeakSet();

function getText(el) {
  const textEl = el.querySelector(
    '.update-components-text, .feed-shared-update-v2__description, ' +
    '.feed-shared-text, .attributed-text-segment-list__content, ' +
    'span[dir="ltr"], span[dir="rtl"]'
  );
  return (textEl ? textEl.innerText : el.innerText || '').trim();
}

function processPost(el) {
  if (seen.has(el)) return;
  seen.add(el);

  const text = getText(el).slice(0, 2000);
  if (text.length < 20) return;

  chrome.runtime.sendMessage({ type: 'CLASSIFY', text }, (response) => {
    if (chrome.runtime.lastError) return;
    if (response && response.isJob === false) {
      el.style.display = 'none';
    }
  });
}

function scan() {
  document.querySelectorAll('[componentkey], [data-id]').forEach(el => processPost(el));
}

scan();

new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
