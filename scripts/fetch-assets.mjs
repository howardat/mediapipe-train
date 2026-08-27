// Vendors the MediaPipe WASM runtime and the hand-landmarker model into public/,
// so the app runs with no network access and no CDN version drift.
import { cp, mkdir, writeFile, access } from 'node:fs/promises';
import { constants } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const wasmSrc = resolve(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const wasmDest = resolve(root, 'public/mediapipe/wasm');
const modelDest = resolve(root, 'public/mediapipe/hand_landmarker.task');
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const exists = (p) => access(p, constants.F_OK).then(() => true, () => false);

async function main() {
  if (!(await exists(wasmSrc))) {
    console.log('[assets] @mediapipe/tasks-vision not installed yet — skipping.');
    return;
  }

  await mkdir(dirname(modelDest), { recursive: true });
  await cp(wasmSrc, wasmDest, { recursive: true });
  console.log('[assets] WASM runtime vendored to public/mediapipe/wasm');

  if (await exists(modelDest)) {
    console.log('[assets] hand_landmarker.task already present.');
    return;
  }

  console.log('[assets] downloading hand_landmarker.task (~7MB)...');
  try {
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await writeFile(modelDest, Buffer.from(await res.arrayBuffer()));
    console.log('[assets] model saved to public/mediapipe/hand_landmarker.task');
  } catch (err) {
    console.warn(`[assets] download failed (${err.message}).`);
    console.warn('[assets] Fetch it manually, then re-run `npm run fetch-assets`:');
    console.warn(`[assets]   curl -o public/mediapipe/hand_landmarker.task ${MODEL_URL}`);
  }
}

main();
