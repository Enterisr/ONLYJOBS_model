// Copies the transformers.js bundle from node_modules into vendor/
// Run once after `npm install`: node setup.js
const fs = require('fs');
const path = require('path');

const distDir = path.join(__dirname, 'node_modules', '@xenova', 'transformers', 'dist');
const vendorDir = path.join(__dirname, 'vendor');

fs.mkdirSync(vendorDir, { recursive: true });
for (const file of fs.readdirSync(distDir)) {
  fs.copyFileSync(path.join(distDir, file), path.join(vendorDir, file));
  console.log('  copied', file);
}
console.log('vendor/ ready');
console.log('Load C:\\Users\\drunk\\linkedin-job-filter as an unpacked extension in chrome://extensions');
