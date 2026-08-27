// Rewrites the generated, model-independent half of an export directory —
// README.md and infer.py — from the gesture_model.json already in it.
//
// Both bake in EDGE_WORKSPACE, so a change there silently strands them: the
// zip beside them gets regenerated and they do not. Run by scripts/sync-workspace.mjs
// so the whole export directory comes from one set of constants.
//
// It does not touch gesture_model.json or dataset.csv. Those are the trained
// model and the recorded data; only the app can produce them.
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { generateInferPy, generateReadme } from '../src/lib/exporter';
import type { ModelBundle } from '../src/lib/exporter';

const out = process.argv[2];
if (!out) throw new Error('usage: emit-export <export dir>');

const bundle = JSON.parse(
  readFileSync(join(out, 'gesture_model.json'), 'utf8'),
) as ModelBundle;

writeFileSync(join(out, 'README.md'), generateReadme(bundle));
writeFileSync(join(out, 'infer.py'), generateInferPy());
