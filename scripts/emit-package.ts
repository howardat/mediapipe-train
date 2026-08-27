// Writes x2_gesture.zip into the directory given as argv[2] and prints the
// manifest as JSON on stdout, so verification can inspect the package without
// opening a browser.
//
// Reads gesture_model.json from the same directory — run scripts/e2e.ts first.
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { generateGesturePackage } from '../src/lib/rospkg';
import type { ModelBundle } from '../src/lib/exporter';
import { zipStoreBytes } from '../src/lib/zip';

const out = process.argv[2];
if (!out) throw new Error('usage: emit-package <output dir>');

const bundle = JSON.parse(
  readFileSync(join(out, 'gesture_model.json'), 'utf8'),
) as ModelBundle;

const entries = generateGesturePackage(bundle);

writeFileSync(join(out, 'x2_gesture.zip'), zipStoreBytes(entries));
process.stdout.write(JSON.stringify(entries));
