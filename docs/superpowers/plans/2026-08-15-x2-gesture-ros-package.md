# x2_gesture ROS 2 Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Gesture Atlas export produce a colcon-buildable `x2_gesture` ROS 2 package that publishes detected gestures as JSON on `std_msgs/String` topics, matching the conventions of the existing `x2_yolo_ws` robot workspace.

**Architecture:** The featurizer + classifier Python (currently inlined in `generateInferPy()`) is extracted into a single generated string, `generateFeatureCore()`, consumed by two emitters: the existing standalone `infer.py`, and a new `x2_gesture/gesture.py` inside the ROS package. One generated core means one golden sample and one contract to verify. The ROS node subscribes to a compressed camera topic over DDS (never RAW — see Global Constraints), runs MediaPipe hand landmarking on a worker thread behind a latest-frame-wins slot, and publishes on three topics. The whole package is delivered as a ZIP built by a dependency-free stored-ZIP writer.

**Tech Stack:** TypeScript (Vite/React browser app, source of truth), generated Python 3 (numpy + mediapipe + cv2 + rclpy), ROS 2 Humble, ament_python/colcon.

## Global Constraints

- **Target ROS 2 Humble.** `source /opt/ros/humble/setup.bash` per `scripts/start_yolo_stream.sh:67`.
- **`std_msgs/String` carrying JSON only.** No `vision_msgs`, no custom `.msg` package. Every topic in `x2_yolo_ws` follows this; `x2_greeter` already parses JSON detections.
- **Never subscribe to RAW image topics.** Compressed only. Per `x2_yolo_stream/node.py:9-11`, RAW is 75–90 MB/s and must not cross compute units.
- **JSON payload conventions**, from `x2_yolo_stream/node.py:569-598`: `description` is the first key; `stamp` is `{"sec": int, "nanosec": int}`; `json.dumps(..., ensure_ascii=False)`.
- **`gesture.py` must import cleanly with only `json` + `numpy`.** No cv2, no mediapipe, no rclpy at module scope. This is what lets contract verification exercise it on a dev machine with no ROS.
- **The golden-sample check runs before `rclpy.init()`.** A featurizer mismatch must never reach the DDS graph.
- **Feature spec version is 1** (`src/lib/features.ts:13`). Generated Python must keep the `SPEC_VERSION` guard that exits on mismatch.
- **No new npm dependencies.** The ZIP writer is hand-rolled; the project ships 5 runtime deps and that is deliberate.
- **This repo is not currently a git repository.** Run `git init` before starting if you want the commit steps below to work; otherwise skip them.

---

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `src/lib/zip.ts` | Dependency-free stored (uncompressed) ZIP writer. `zipStore(files) → Blob`. |
| `src/lib/rospkg.ts` | Emits every file of the `x2_gesture` package as `{path, content}[]`. The ROS-specific generator, kept out of `exporter.ts` so that file stays about the model bundle. |
| `scripts/emit-package.ts` | Prints the package manifest as JSON to stdout, so verification can inspect it without a browser. |
| `scripts/verify-package.mjs` | New verification entry point: ZIP round-trip through Python's `zipfile`, core-identity check, `gesture.py` parity, `py_compile` on every emitted `.py`. |
| `scripts/package.py` | Python side of the above — unzips, imports `gesture.py`, replays golden + parity cases. |
| `docs/superpowers/plans/2026-08-15-x2-gesture-ros-package.md` | This plan. |

**Modified:**

| Path | Change |
|---|---|
| `src/lib/exporter.ts:66-294` | Split `generateInferPy()` into `generateFeatureCore()` + `generateInferPy()`. Add ROS section to `generateReadme()`. |
| `src/components/ExportPanel.tsx` | Add `x2_gesture.zip` button + copy. |
| `src/App.tsx:368-379` | Wire `onDownloadPackage`. |
| `package.json` | Add `verify:package` script; make `verify` run both. |
| `README.md` | ROS deployment section. |
| `.gitignore` | Add `x2_yolo_ws/` — the reference copy must not be committed. |

**Emitted by `rospkg.ts`** (the package tree, none of these exist as repo files):

```
x2_gesture/package.xml
x2_gesture/setup.py
x2_gesture/setup.cfg
x2_gesture/resource/x2_gesture          (empty marker file)
x2_gesture/launch/x2_gesture.launch.py
x2_gesture/scripts/start_gesture.sh
x2_gesture/scripts/x2-gesture.service
x2_gesture/x2_gesture/__init__.py
x2_gesture/x2_gesture/gesture.py        GENERATED core — numpy only
x2_gesture/x2_gesture/node.py           the ROS node
x2_gesture/README.md
```

**Explicitly out of scope:** annotated-image republishing and the MJPEG web view that `x2_yolo_stream` provides. Gestures are a label, not a picture; if you want to see hands, `x2_yolo_stream` is already on screen.

---

## Topic Contract

All under `topic_prefix`, default `/gesture`.

### `<prefix>/detections` — every inference cycle

```json
{
  "description": "peace, right hand, 0.98",
  "stamp": {"sec": 1755, "nanosec": 250000000},
  "frame_id": "camera_link",
  "source_topic": "/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed",
  "spec_version": 1,
  "classes": ["fist", "peace", "thumbs_up"],
  "gesture": "peace",
  "confidence": 0.98,
  "handedness": "Right",
  "hands": 1,
  "threshold": 0.75,
  "smoothing": 5,
  "scores": {"peace": 0.98, "fist": 0.01, "thumbs_up": 0.01},
  "instant": {"peace": 0.97, "fist": 0.02, "thumbs_up": 0.01},
  "inference_ms": 12.3
}
```

`gesture` is `null` and `confidence` is `0.0` when no hand is present or the smoothed best is below `threshold`. `scores` is the smoothed distribution, `instant` is this frame alone — the pair is what lets you tune `smoothing` from an echo.

### `<prefix>/summary` — on change, plus a `summary_max_period` heartbeat

```json
{
  "description": "peace, right hand, held 1.2s",
  "stamp": {"sec": 1755, "nanosec": 250000000},
  "gesture": "peace",
  "confidence": 0.98,
  "handedness": "Right",
  "held_seconds": 1.2,
  "changes": 7
}
```

Change key is `(gesture, handedness)`. A confidence wobble is not a change.

### `<prefix>/text` — on change, plain sentence, not JSON

```
peace, right hand, held 1.2s
```

Empty-hand description is `"no hand"`.

---

## Node Parameters

Names and defaults mirror `x2_yolo_stream/node.py:143-217` wherever the concept is shared.

| Param | Default | Notes |
|---|---|---|
| `image_topic` | `""` | `""` = auto-pick from `TOPIC_PREFERENCE` |
| `camera_device` | `-1` | `>= 0` uses local `cv2.VideoCapture(N)` instead of DDS. Dev/test escape hatch. |
| `model_path` | `/home/agi/x2_yolo_ws/models/gesture_model.json` | |
| `landmarker_path` | `/home/agi/x2_yolo_ws/models/hand_landmarker.task` | |
| `topic_prefix` | `/gesture` | |
| `num_hands` | `1` | |
| `min_detection_confidence` | `0.5` | MediaPipe, not the classifier |
| `min_tracking_confidence` | `0.5` | |
| `threshold` | `0.75` | classifier confidence to commit a label |
| `smoothing` | `5` | frames averaged before committing |
| `infer_fps` | `10.0` | |
| `max_width` | `640` | downscale before landmarking |
| `publish_detections` | `true` | |
| `publish_summary` | `true` | |
| `publish_text` | `true` | |
| `pretty_summary` | `true` | |
| `pretty_detections` | `false` | |
| `summary_on_change_only` | `true` | |
| `summary_max_period` | `5.0` | |
| `qos_fallback_seconds` | `12.0` | best-effort ↔ reliable watchdog |

`TOPIC_PREFERENCE` is copied verbatim from `x2_yolo_stream/node.py:44-50`.

---

## Task 1: Dependency-free ZIP writer

**Files:**
- Create: `src/lib/zip.ts`
- Create: `scripts/verify-package.mjs`
- Create: `scripts/package.py`
- Modify: `package.json`

**Interfaces:**
- Consumes: nothing.
- Produces: `zipStore(files: ZipEntry[]): Blob` and `zipStoreBytes(files: ZipEntry[]): Uint8Array`, where `interface ZipEntry { path: string; content: string; mode?: number }`. Later tasks call `zipStoreBytes` in Node and `zipStore` in the browser.

Stored (method 0, no deflate) keeps this ~90 lines with no dependency. The package is ~50 KB of text; compression would save nothing worth a dependency. `mode` exists so `start_gesture.sh` ships executable.

- [ ] **Step 1: Write the failing verification**

Create `scripts/package.py`:

```python
#!/usr/bin/env python3
"""Python side of the package contract check.

Reads a zip produced by src/lib/zip.ts using the stdlib, which is an entirely
independent ZIP implementation -- if our hand-rolled writer is wrong, this is
what says so.

    python3 scripts/package.py zipcheck <zip> <manifest.json>
"""

import json
import sys
import zipfile


def zipcheck(zip_path, manifest_path):
    expected = json.load(open(manifest_path))
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f"corrupt entry in zip: {bad}")
        names = set(zf.namelist())
        want = {e["path"] for e in expected}
        if names != want:
            raise SystemExit(
                f"manifest mismatch\n  missing: {sorted(want - names)}\n"
                f"  extra:   {sorted(names - want)}")
        for entry in expected:
            got = zf.read(entry["path"]).decode("utf-8")
            if got != entry["content"]:
                raise SystemExit(f"content differs for {entry['path']}")
    print(f"zip OK -- {len(expected)} entries, CRCs valid")


if __name__ == "__main__":
    if sys.argv[1] == "zipcheck":
        zipcheck(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown mode {sys.argv[1]}")
```

Create `scripts/verify-package.mjs`:

```js
// Verifies the generated x2_gesture package: the zip is readable by an
// independent implementation, gesture.py's core is byte-identical to
// infer.py's, and every emitted .py compiles.
//
//   npm run verify:package
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const out = mkdtempSync(join(tmpdir(), 'gesture-package-'));
const run = (cmd, args) => execFileSync(cmd, args, { encoding: 'utf8' });

const bundle = (entry, name) => {
  const file = join(out, name);
  run('npx', ['esbuild', '--bundle', '--format=esm', '--platform=node',
              '--log-level=error',
              '--banner:js=import{createRequire}from"module";const require=createRequire(import.meta.url);',
              `--outfile=${file}`, entry]);
  return file;
};

const python = process.env.PYTHON ?? 'python3';

try {
  // Task 1 checks only the zip round-trip. Later tasks extend this file.
  const emitted = run('node', [bundle('scripts/emit-package.ts', 'pkg.mjs'), out]);
  const manifest = join(out, 'manifest.json');
  writeFileSync(manifest, emitted);
  process.stdout.write(run(python, ['scripts/package.py', 'zipcheck',
                                    join(out, 'x2_gesture.zip'), manifest]));
} catch (err) {
  console.error(err.stdout?.toString() ?? '');
  console.error(err.stderr?.toString() ?? err.message);
  process.exit(1);
}
```

Add to `package.json` scripts:

```json
"verify:package": "node scripts/verify-package.mjs",
"verify": "npm run verify:contract && npm run verify:package"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npm run verify:package`
Expected: FAIL — `scripts/emit-package.ts` does not exist yet.

- [ ] **Step 3: Write the ZIP writer**

Create `src/lib/zip.ts`:

```ts
/**
 * A ZIP writer in ~90 lines, storing entries uncompressed.
 *
 * The exported package is about 50 KB of text. Deflate would save maybe 35 KB
 * and cost a dependency, so entries are stored (method 0) and the whole format
 * reduces to: a local header per file, a central directory, and an end record.
 */

export interface ZipEntry {
  path: string;
  content: string;
  /** Unix mode. Defaults to 0o644; pass 0o755 for shell scripts. */
  mode?: number;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** DOS time/date. Fixed at 1980-01-01 so the same input yields the same bytes. */
const DOS_TIME = 0;
const DOS_DATE = 33; // 1980-01-01

export function zipStoreBytes(entries: ZipEntry[]): Uint8Array {
  const enc = new TextEncoder();
  const chunks: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  for (const entry of entries) {
    const name = enc.encode(entry.path);
    const data = enc.encode(entry.content);
    const crc = crc32(data);

    const local = new Uint8Array(30 + name.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);   // local file header signature
    lv.setUint16(4, 20, true);           // version needed
    lv.setUint16(6, 0x0800, true);       // UTF-8 filename flag
    lv.setUint16(8, 0, true);            // method: stored
    lv.setUint16(10, DOS_TIME, true);
    lv.setUint16(12, DOS_DATE, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, data.length, true); // compressed size
    lv.setUint32(22, data.length, true); // uncompressed size
    lv.setUint16(26, name.length, true);
    lv.setUint16(28, 0, true);           // extra length
    local.set(name, 30);

    const dir = new Uint8Array(46 + name.length);
    const dv = new DataView(dir.buffer);
    dv.setUint32(0, 0x02014b50, true);   // central directory signature
    dv.setUint16(4, 0x031e, true);       // version made by: UNIX, 3.0
    dv.setUint16(6, 20, true);
    dv.setUint16(8, 0x0800, true);
    dv.setUint16(10, 0, true);
    dv.setUint16(12, DOS_TIME, true);
    dv.setUint16(14, DOS_DATE, true);
    dv.setUint32(16, crc, true);
    dv.setUint32(20, data.length, true);
    dv.setUint32(24, data.length, true);
    dv.setUint16(28, name.length, true);
    // External attributes carry the Unix mode in the high 16 bits, which is
    // how `unzip` knows start_gesture.sh is executable.
    dv.setUint32(38, ((entry.mode ?? 0o644) & 0xffff) << 16, true);
    dv.setUint32(42, offset, true);
    dir.set(name, 46);

    chunks.push(local, data);
    central.push(dir);
    offset += local.length + data.length;
  }

  const centralSize = central.reduce((n, c) => n + c.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true);     // end of central directory
  ev.setUint16(8, entries.length, true);
  ev.setUint16(10, entries.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);

  const total = offset + centralSize + end.length;
  const out = new Uint8Array(total);
  let at = 0;
  for (const c of [...chunks, ...central, end]) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

export function zipStore(entries: ZipEntry[]): Blob {
  // Copy into a fresh ArrayBuffer: Blob rejects a view with a byteOffset.
  return new Blob([zipStoreBytes(entries).slice().buffer], { type: 'application/zip' });
}
```

- [ ] **Step 4: Write a temporary emitter to exercise it**

Create `scripts/emit-package.ts`. For this task it emits two placeholder entries so the zip path is testable; Task 5 replaces the body with the real manifest.

```ts
// Writes x2_gesture.zip into the directory given as argv[2] and prints the
// manifest as JSON on stdout.
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { zipStoreBytes } from '../src/lib/zip';

const out = process.argv[2];
const entries = [
  { path: 'x2_gesture/README.md', content: '# placeholder\n' },
  { path: 'x2_gesture/scripts/start_gesture.sh', content: '#!/bin/bash\n', mode: 0o755 },
];

writeFileSync(join(out, 'x2_gesture.zip'), zipStoreBytes(entries));
process.stdout.write(JSON.stringify(entries));
```

- [ ] **Step 5: Run verification to confirm it passes**

Run: `npm run verify:package`
Expected: PASS — `zip OK -- 2 entries, CRCs valid`

Also confirm the archive is real to a third implementation:

```bash
python3 -c "import zipfile,glob;z=zipfile.ZipFile(glob.glob('/tmp/gesture-package-*/x2_gesture.zip')[-1]);print(z.namelist())"
```

- [ ] **Step 6: Commit**

```bash
git add src/lib/zip.ts scripts/emit-package.ts scripts/verify-package.mjs scripts/package.py package.json
git commit -m "feat: dependency-free stored-ZIP writer with round-trip verification"
```

---

## Task 2: Extract the shared featurizer core

**Files:**
- Modify: `src/lib/exporter.ts:66-294`

**Interfaces:**
- Consumes: `FEATURE_SPEC`, `FINGER_CHAINS`, `NUM_LANDMARKS`, `SCALE_REF`, `TIP_INDICES` from `./features`.
- Produces: `generateFeatureCore(): string` — Python source defining `NUM_LANDMARKS`, `FINGER_CHAINS`, `TIP_INDICES`, `SCALE_REF`, `MIRROR_LEFT_HAND`, `FEATURE_DIM`, `SPEC_VERSION`, `featurize(world_landmarks, handedness)`, and `class GestureClassifier` with `.classes`, `.predict(feats)`, `.verify()`, `.val_accuracy`. Imports nothing but `json` and `numpy`. `generateInferPy()` keeps its existing signature and output behaviour.

This is a pure refactor. `infer.py`'s bytes should change only where the section banner sits.

- [ ] **Step 1: Pin current behaviour**

Capture the current output as a reference before touching anything:

```bash
npx esbuild --bundle --format=esm --platform=node \
  --banner:js='import{createRequire}from"module";const require=createRequire(import.meta.url);' \
  --outfile=/tmp/emit-before.mjs scripts/emit-infer.ts && node /tmp/emit-before.mjs > /tmp/infer-before.py
```

- [ ] **Step 2: Do the extraction**

In `src/lib/exporter.ts`, add above `generateInferPy()`:

```ts
/**
 * The half of the generated Python that both deployment targets share: the
 * feature constants, the featurizer, and the classifier with its golden check.
 *
 * Imports only json and numpy — deliberately. `x2_gesture/gesture.py` is this
 * text verbatim, and contract verification imports it on a machine with no
 * OpenCV, no MediaPipe and no ROS.
 */
export function generateFeatureCore(): string {
  const py = (v: unknown) => JSON.stringify(v);

  return `# ---------------------------------------------------------------- feature spec
NUM_LANDMARKS = ${NUM_LANDMARKS}
FINGER_CHAINS = ${py(FINGER_CHAINS)}
TIP_INDICES = ${py(TIP_INDICES)}
SCALE_REF = ${py(SCALE_REF)}
MIRROR_LEFT_HAND = ${FEATURE_SPEC.mirrorLeftHand ? 'True' : 'False'}
FEATURE_DIM = ${FEATURE_SPEC.dim}
SPEC_VERSION = ${FEATURE_SPEC.version}
`;
}
```

Then move the existing body of `featurize` and `class GestureClassifier` — lines 106-200 of the current file, unchanged — into that same template literal, keeping the `# ------ classifier` banner between them.

Rewrite `generateInferPy()` so its body is:

```ts
export function generateInferPy(): string {
  return `#!/usr/bin/env python3
"""
${/* the existing docstring, unchanged */ ''}
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

${generateFeatureCore()}
${MAIN_LOOP}
`;
}
```

where `MAIN_LOOP` is the existing `# ---- main loop` banner through `if __name__ == "__main__":`, verbatim.

**Critical:** `scripts/e2e.py:21-23` slices `infer.py` from the literal string `NUM_LANDMARKS =` to the literal string `# ------------------------------------------------------------------- main loop`. Both markers must survive this refactor with identical spelling, including the exact number of dashes.

- [ ] **Step 3: Prove the output is unchanged**

```bash
node /tmp/emit-before.mjs > /dev/null  # sanity: the old bundle still runs
npx esbuild --bundle --format=esm --platform=node \
  --banner:js='import{createRequire}from"module";const require=createRequire(import.meta.url);' \
  --outfile=/tmp/emit-after.mjs scripts/emit-infer.ts && node /tmp/emit-after.mjs > /tmp/infer-after.py
diff /tmp/infer-before.py /tmp/infer-after.py
```

Expected: no output, or differences confined to blank lines around the banners. Any change inside `featurize` or `GestureClassifier` is a bug — revert and redo.

- [ ] **Step 4: Run the full contract**

Run: `npm run verify:contract`
Expected: PASS — `infer.py compiles`, feature parity, `inference parity OK`.

- [ ] **Step 5: Commit**

```bash
git add src/lib/exporter.ts
git commit -m "refactor: extract generateFeatureCore from generateInferPy"
```

---

## Task 3: Emit `gesture.py` and prove it matches `infer.py`

**Files:**
- Create: `src/lib/rospkg.ts`
- Modify: `scripts/package.py`
- Modify: `scripts/verify-package.mjs`

**Interfaces:**
- Consumes: `generateFeatureCore()` from Task 2; `zipStoreBytes`/`ZipEntry` from Task 1.
- Produces: `generateGestureCore(): string` in `rospkg.ts` — the full text of `x2_gesture/x2_gesture/gesture.py`.

- [ ] **Step 1: Write the failing checks**

Add to `scripts/package.py`:

```python
def coreident(pkg_dir, infer_path):
    """gesture.py's core must be byte-identical to infer.py's.

    Two copies of a featurizer that drift apart is the exact failure the golden
    sample exists to catch at runtime; this catches it at build time instead.
    """
    start = "NUM_LANDMARKS ="
    end = "# ------------------------------------------------------------------- main loop"

    infer = open(infer_path).read()
    infer_core = infer[infer.index(start):infer.index(end)].rstrip()

    gesture = open(f"{pkg_dir}/x2_gesture/x2_gesture/gesture.py").read()
    gesture_core = gesture[gesture.index(start):].rstrip()

    if infer_core != gesture_core:
        raise SystemExit(
            "gesture.py and infer.py disagree -- they must come from the same "
            "generateFeatureCore() call")
    print("core identical -- %d chars shared by infer.py and gesture.py"
          % len(infer_core))


def parity(pkg_dir, model_path, cases_path):
    """Import gesture.py with nothing but numpy, then replay the browser."""
    sys.path.insert(0, f"{pkg_dir}/x2_gesture/x2_gesture")
    import gesture as g            # noqa: E402  -- path set above

    clf = g.GestureClassifier(model_path)
    clf.verify()

    import numpy as np
    cases = json.load(open(cases_path))
    worst = 0.0
    for i, case in enumerate(cases):
        probs = clf.predict(g.featurize(case["landmarks"], case["handedness"]))
        expected = np.asarray(case["expectedProbs"], dtype=np.float64)
        err = float(np.abs(probs - expected).max())
        worst = max(worst, err)
        if err > 1e-4:
            raise SystemExit("case %d: probabilities differ by %.3e" % (i, err))
    print("gesture.py parity OK -- %d cases, max error %.2e" % (len(cases), worst))
```

Extend the `__main__` dispatch with `coreident` and `parity` modes taking their arguments positionally.

- [ ] **Step 2: Run to verify it fails**

Run: `npm run verify:package`
Expected: FAIL — `gesture.py` does not exist.

- [ ] **Step 3: Write the emitter**

Create `src/lib/rospkg.ts`:

```ts
/**
 * Emits the x2_gesture ROS 2 package.
 *
 * Shaped after x2_yolo_stream in the robot workspace: std_msgs/String carrying
 * JSON, a compressed camera topic over DDS, a worker thread behind a
 * latest-frame-wins slot, and an ament_python package with a launch file and a
 * systemd --user unit.
 */
import { generateFeatureCore } from './exporter';
import type { ModelBundle } from './exporter';
import type { ZipEntry } from './zip';

export const PACKAGE_NAME = 'x2_gesture';

export function generateGestureCore(): string {
  return `"""Feature extraction and classification for a Gesture Atlas model.

GENERATED -- do not edit. This file is emitted from src/lib/features.ts and is
byte-identical to the corresponding section of infer.py; scripts/package.py
asserts that on every build.

Deliberately imports nothing but json and numpy, so it can be exercised on a
machine with no OpenCV, MediaPipe or ROS.
"""

import json

import numpy as np

${generateFeatureCore()}`;
}
```

- [ ] **Step 4: Wire it into the emitter and the verifier**

In `scripts/emit-package.ts`, replace the placeholder entries with:

```ts
import { generateGestureCore, PACKAGE_NAME } from '../src/lib/rospkg';

const entries: ZipEntry[] = [
  { path: `${PACKAGE_NAME}/${PACKAGE_NAME}/gesture.py`, content: generateGestureCore() },
];
```

In `scripts/verify-package.mjs`, after the zipcheck, unzip and run the new modes. Reuse the fixtures `verify-contract.mjs` already builds by generating them the same way:

```js
const inferPy = join(out, 'infer.py');
writeFileSync(inferPy, run('node', [bundle('scripts/emit-infer.ts', 'emit.mjs')]));

run(python, ['-c',
  `import zipfile;zipfile.ZipFile(${JSON.stringify(join(out, 'x2_gesture.zip'))}).extractall(${JSON.stringify(out)})`]);

process.stdout.write(run(python, ['scripts/package.py', 'coreident', out, inferPy]));

// e2e.ts writes gesture_model.json + predictions.json into `out`.
process.stdout.write(run('node', [bundle('scripts/e2e.ts', 'e2e.mjs'), out]));
process.stdout.write(run(python, ['scripts/package.py', 'parity', out,
                                  join(out, 'gesture_model.json'),
                                  join(out, 'predictions.json')]));
```

- [ ] **Step 5: Run to verify it passes**

Run: `npm run verify:package`
Expected: PASS — `core identical -- N chars shared`, then `gesture.py parity OK -- N cases`.

- [ ] **Step 6: Commit**

```bash
git add src/lib/rospkg.ts scripts/emit-package.ts scripts/verify-package.mjs scripts/package.py
git commit -m "feat: emit x2_gesture/gesture.py from the shared featurizer core"
```

---

## Task 4: The ROS node

**Files:**
- Modify: `src/lib/rospkg.ts`
- Modify: `scripts/verify-package.mjs`

**Interfaces:**
- Consumes: `PACKAGE_NAME` from Task 3.
- Produces: `generateNodePy(): string` — the text of `x2_gesture/x2_gesture/node.py`, defining `class GestureNode(Node)` and `main(args=None)`.

The node is static text (no interpolation from the model), so it lives as one template literal. Structure, mirroring `x2_yolo_stream/node.py`:

1. Module docstring explaining the compressed-only rule, the worker thread, and that the golden check gates `rclpy.init()`.
2. `TOPIC_PREFERENCE` copied verbatim from `x2_yolo_stream/node.py:44-50`.
3. `GestureNode.__init__`: declare every parameter from the table above with a why-comment on the non-obvious ones, `p = self.get_parameter`, load the classifier, **call `clf.verify()`**, build the MediaPipe `HandLandmarker` in `RunningMode.VIDEO`, create the three publishers with depth 10, resolve and subscribe to the image topic, start the worker, `create_timer(2.0, self._watchdog)`.
4. `_resolve_topic`, `_subscribe`, `_watchdog`, `_on_image` — ported from `x2_yolo_stream/node.py:386-433`, unchanged in behaviour.
5. `_infer_loop`: throttle to `infer_fps`, `cv2.imdecode`, downscale to `max_width`, `cv2.cvtColor` to RGB, `mp.Image`, `detect_for_video` with a monotonically increasing millisecond timestamp, featurize, predict, push onto the smoothing deque, publish.
6. `_publish`, `_publish_summary` — the JSON in the Topic Contract section above.
7. `destroy_node` sets `_stop`, wakes the worker, calls `super()`.
8. `main` with `rclpy.init` / `rclpy.spin` / `destroy_node` / `if rclpy.ok(): rclpy.shutdown()`.

Three things that differ from `x2_yolo_stream` and are easy to get wrong:

- **MediaPipe VIDEO mode demands strictly increasing timestamps.** Use a monotonic counter, not the DDS header stamp — frames may arrive with equal or backwards stamps and `detect_for_video` raises on that.
- **Clear the smoothing history when no hand is found**, matching `infer.py:265`. Carrying stale probabilities across an empty gap makes a gesture appear to linger after the hand is gone.
- **`import mediapipe` inside `_infer_loop`,** the way `x2_yolo_stream` imports cv2 at `node.py:438`. It is a multi-second import and keeps the constructor responsive.

- [ ] **Step 1: Write the failing check**

Add to `scripts/verify-package.mjs`, after extraction:

```js
// py_compile parses without importing, so this passes on a machine with no ROS.
for (const rel of ['x2_gesture/x2_gesture/node.py',
                   'x2_gesture/x2_gesture/gesture.py',
                   'x2_gesture/launch/x2_gesture.launch.py',
                   'x2_gesture/setup.py']) {
  run(python, ['-m', 'py_compile', join(out, rel)]);
}
console.log('package python compiles');

// gesture.py must stay importable with only numpy.
const core = readFileSync(join(out, 'x2_gesture/x2_gesture/gesture.py'), 'utf8');
for (const forbidden of ['import cv2', 'import mediapipe', 'import rclpy']) {
  if (core.includes(forbidden)) {
    throw new Error(`gesture.py must not contain "${forbidden}"`);
  }
}
console.log('gesture.py imports only json + numpy');
```

Add `readFileSync` to the `node:fs` import.

- [ ] **Step 2: Run to verify it fails**

Run: `npm run verify:package`
Expected: FAIL — `node.py` does not exist.

- [ ] **Step 3: Write `generateNodePy()`**

Implement the structure above in `src/lib/rospkg.ts`. Add its entry to `scripts/emit-package.ts`:

```ts
{ path: `${PACKAGE_NAME}/${PACKAGE_NAME}/node.py`, content: generateNodePy() },
{ path: `${PACKAGE_NAME}/${PACKAGE_NAME}/__init__.py`, content: '' },
```

- [ ] **Step 4: Run to verify it passes**

Run: `npm run verify:package`
Expected: PASS — `package python compiles`, `gesture.py imports only json + numpy`.

- [ ] **Step 5: Commit**

```bash
git add src/lib/rospkg.ts scripts/emit-package.ts scripts/verify-package.mjs
git commit -m "feat: x2_gesture ROS node publishing detections, summary and text"
```

---

## Task 5: Package scaffolding

**Files:**
- Modify: `src/lib/rospkg.ts`
- Modify: `scripts/emit-package.ts`
- Modify: `scripts/verify-package.mjs`

**Interfaces:**
- Produces: `generateGesturePackage(bundle: ModelBundle): ZipEntry[]` — the complete manifest, the single entry point the UI calls.

Contents follow `x2_yolo_stream` exactly:

- **`package.xml`** — format 3, `ament_python`, `<exec_depend>` on `rclpy`, `sensor_msgs`, `std_msgs`, with the same comment noting that opencv/numpy/mediapipe are deliberately not rosdep keys so nothing tries to upgrade them on the robot image.
- **`setup.py`** — `PACKAGE_NAME = "x2_gesture"`, `data_files` shipping `package.xml`, `launch/x2_gesture.launch.py`, and the two `scripts/` files; `entry_points` console script `gesture_node = x2_gesture.node:main`.
- **`setup.cfg`** — copy from `x2_yolo_stream/setup.cfg`.
- **`resource/x2_gesture`** — empty file, `content: ''`.
- **`launch/x2_gesture.launch.py`** — `DeclareLaunchArgument` for `image_topic`, `model_path`, `landmarker_path`, `threshold`, `smoothing`, `infer_fps`, `summary_on_change_only`; `Node(package="x2_gesture", executable="gesture_node", name="x2_gesture", output="screen")`.
- **`scripts/start_gesture.sh`**, mode `0o755` — modelled on `start_yolo_stream.sh`: `X2_GESTURE_*` env overrides, `flock` single-instance guard, `source /opt/ros/humble/setup.bash`, the `FASTRTPS_DEFAULT_PROFILES_FILE` export (**required** — without it a fresh participant cannot see camera topics from other SoCs, `start_yolo_stream.sh:77`), `cd /tmp`, `exec` under `X2_GESTURE_FOREGROUND`, restart loop otherwise. No `gpu_env.sh` — MediaPipe does not use onnxruntime.
- **`scripts/x2-gesture.service`** — systemd `--user` unit with the lingering instructions, `After=network-online.target`, `Restart=always`, `RestartSec=10`, `StartLimitIntervalSec=0`. `TimeoutStartSec=60` is enough; there is no multi-minute TensorRT build here.
- **`README.md`** — install, build, run, and the topic contract, with the class names from `bundle.classes` filled in.

- [ ] **Step 1: Write the failing manifest check**

Add to `scripts/verify-package.mjs`:

```js
const EXPECTED = [
  'x2_gesture/package.xml',
  'x2_gesture/setup.py',
  'x2_gesture/setup.cfg',
  'x2_gesture/resource/x2_gesture',
  'x2_gesture/launch/x2_gesture.launch.py',
  'x2_gesture/scripts/start_gesture.sh',
  'x2_gesture/scripts/x2-gesture.service',
  'x2_gesture/x2_gesture/__init__.py',
  'x2_gesture/x2_gesture/gesture.py',
  'x2_gesture/x2_gesture/node.py',
  'x2_gesture/README.md',
];
const manifestPaths = JSON.parse(emitted).map((e) => e.path).sort();
if (JSON.stringify(manifestPaths) !== JSON.stringify([...EXPECTED].sort())) {
  throw new Error(`manifest mismatch:\n  got  ${manifestPaths}\n  want ${EXPECTED}`);
}
console.log(`manifest OK -- ${EXPECTED.length} files`);
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm run verify:package`
Expected: FAIL — manifest mismatch, only 3 of 11 files present.

- [ ] **Step 3: Write the scaffolding generators**

Add one function per file to `rospkg.ts`, then the assembler:

```ts
export function generateGesturePackage(bundle: ModelBundle): ZipEntry[] {
  const p = (rel: string) => `${PACKAGE_NAME}/${rel}`;
  return [
    { path: p('package.xml'), content: generatePackageXml() },
    { path: p('setup.py'), content: generateSetupPy() },
    { path: p('setup.cfg'), content: generateSetupCfg() },
    { path: p(`resource/${PACKAGE_NAME}`), content: '' },
    { path: p('launch/x2_gesture.launch.py'), content: generateLaunchPy() },
    { path: p('scripts/start_gesture.sh'), content: generateStartSh(), mode: 0o755 },
    { path: p('scripts/x2-gesture.service'), content: generateServiceUnit() },
    { path: p(`${PACKAGE_NAME}/__init__.py`), content: '' },
    { path: p(`${PACKAGE_NAME}/gesture.py`), content: generateGestureCore() },
    { path: p(`${PACKAGE_NAME}/node.py`), content: generateNodePy() },
    { path: p('README.md'), content: generatePackageReadme(bundle) },
  ];
}
```

Update `scripts/emit-package.ts` to build a bundle via the same helper `scripts/e2e.ts` uses and call `generateGesturePackage(bundle)`.

- [ ] **Step 4: Run to verify it passes**

Run: `npm run verify:package`
Expected: PASS — `manifest OK -- 11 files`, plus every check from Tasks 1, 3, 4.

- [ ] **Step 5: Commit**

```bash
git add src/lib/rospkg.ts scripts/emit-package.ts scripts/verify-package.mjs
git commit -m "feat: colcon scaffolding, launch file and systemd unit for x2_gesture"
```

---

## Task 6: Wire the download into the UI

**Files:**
- Modify: `src/App.tsx:22-29` (imports), `src/App.tsx:368-379` (ExportPanel props)
- Modify: `src/components/ExportPanel.tsx`

**Interfaces:**
- Consumes: `generateGesturePackage` from Task 5, `zipStore` from Task 1.
- Produces: `ExportPanel` prop `onDownloadPackage: () => void`.

- [ ] **Step 1: Add the download helper**

`src/App.tsx` — the existing `download()` takes a string; add a Blob sibling rather than overloading it:

```ts
function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

- [ ] **Step 2: Wire the prop**

```tsx
onDownloadPackage={() =>
  bundle && downloadBlob('x2_gesture.zip', zipStore(generateGesturePackage(bundle)))
}
```

- [ ] **Step 3: Add the button**

In `ExportPanel.tsx`, add to the `Props` interface and the button row, disabled on `!bundle`:

```tsx
<button className="btn" onClick={onDownloadPackage} disabled={!bundle}>
  x2_gesture.zip
</button>
```

- [ ] **Step 4: Verify the build and the browser**

Run: `npm run build`
Expected: PASS — `tsc -b` clean, vite build succeeds.

Then start the dev server and confirm the button downloads a valid archive: train a model on two classes, click `x2_gesture.zip`, and check the file:

```bash
python3 -c "import zipfile;z=zipfile.ZipFile('$HOME/Downloads/x2_gesture.zip');print(z.testzip(), len(z.namelist()))"
```

Expected: `None 11`

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/components/ExportPanel.tsx
git commit -m "feat: export x2_gesture.zip from the export panel"
```

---

## Task 7: Documentation

**Files:**
- Modify: `src/lib/exporter.ts` (`generateReadme`)
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Add the ROS section to the exported README**

In `generateReadme()`, after the existing run instructions:

````
## Run it as a ROS 2 node

    unzip x2_gesture.zip -d ~/x2_yolo_ws/src/
    cp gesture_model.json ~/x2_yolo_ws/models/
    cd ~/x2_yolo_ws && colcon build --packages-select x2_gesture
    source install/setup.bash
    ros2 launch x2_gesture x2_gesture.launch.py

Topics:

    /gesture/detections   every cycle, full JSON
    /gesture/summary      on change, compact
    /gesture/text         on change, one sentence

    ros2 topic echo /gesture/text
````

- [ ] **Step 2: Add the ROS section to the repo README**

Extend the "Deploying to an edge device" section with a subsection covering the package export, the three topics, and this note:

> MediaPipe hand landmarking runs on the CPU via XNNPACK — it does not use CUDA or TensorRT, so on a Jetson it competes with whatever else is on the box. `infer_fps` is the knob that matters. Body-pose keypoints cannot substitute: they have no fingers.

- [ ] **Step 3: Stop the reference workspace being committed**

Add to `.gitignore`:

```
x2_yolo_ws/
```

- [ ] **Step 4: Verify everything together**

Run: `npm run verify`
Expected: PASS — contract checks then package checks, end to end.

- [ ] **Step 5: Commit**

```bash
git add README.md src/lib/exporter.ts .gitignore
git commit -m "docs: ROS 2 deployment instructions for x2_gesture"
```

---

## What this plan does not verify

Stated plainly so nobody reads a green `npm run verify` as more than it is:

- **Nothing here runs ROS.** `py_compile` parses `node.py`; it never imports `rclpy`, never creates a publisher, never puts a message on the wire. The first real test of the topic contract is on the robot.
- **`pip install mediapipe` on aarch64 is unverified.** The `sm87` TensorRT cache in `x2_yolo_ws/models/trt_cache` says Jetson Orin, and MediaPipe has no reliable aarch64 manylinux wheel. This is the single most likely thing to block deployment and it is independent of everything above — worth testing on the Orin before Task 4.
- **Frame rate on the robot is unknown.** The README's 10–20 fps figure is for a Pi 4 with a local USB camera. Decoding a compressed DDS frame and sharing a CPU with `x2_yolo_stream` is a different budget.
- **`x2_greeter` integration is not included.** Making the greeter react to gestures is a separate change to a separate package, and belongs in a plan of its own once `/gesture/summary` is real and observed.
