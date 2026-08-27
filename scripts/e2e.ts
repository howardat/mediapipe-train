// End-to-end check: train a real model on synthetic hands, export the bundle,
// and emit fixtures so Python can prove it reproduces the browser's predictions.
// Run via `npm run verify:contract`.
import { writeFileSync } from 'node:fs';
import { NUM_LANDMARKS, type Handedness, type Vec3 } from '../src/lib/features';
import { DEFAULT_CONFIG, trainModel } from '../src/lib/trainer';
import { buildBundle, generateInferPy } from '../src/lib/exporter';
import type { GestureClass, Sample } from '../src/lib/types';

const outDir = process.argv[2];
if (!outDir) throw new Error('usage: e2e.ts <outDir>');

/** Deterministic LCG so the check is reproducible. */
let seed = 20240814;
const rand = () => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
};
const jitter = () => (rand() + rand() + rand() - 1.5) * 0.8;

const FINGERS = [
  { base: -1.05, len: 0.85 },
  { base: -0.35, len: 1.0 },
  { base: 0.0, len: 1.05 },
  { base: 0.33, len: 0.98 },
  { base: 0.66, len: 0.82 },
];

const POSES: Record<string, number[]> = {
  fist: [0.05, 0.05, 0.05, 0.05, 0.05],
  open_palm: [0.95, 1.0, 1.0, 1.0, 0.95],
  peace: [0.1, 1.0, 1.0, 0.1, 0.1],
};

/** A crude but anatomically ordered hand: wrist, then four joints per finger. */
function hand(extensions: number[], scale: number): Vec3[] {
  const pts: Vec3[] = [[0, 0, 0]];
  FINGERS.forEach((f, fi) => {
    const e = Math.min(1, Math.max(0, extensions[fi] + jitter() * 0.04));
    for (let k = 1; k <= 4; k++) {
      const r = 0.25 * k * f.len * (0.4 + 0.6 * e);
      const a = f.base + (1 - e) * k * 0.55;
      pts.push([
        (r * Math.sin(a) + jitter() * 0.012) * scale,
        (-r * Math.cos(a) + jitter() * 0.012) * scale,
        jitter() * 0.01 * scale,
      ]);
    }
  });
  if (pts.length !== NUM_LANDMARKS) throw new Error(`generated ${pts.length} landmarks`);
  return pts;
}

const classes: GestureClass[] = Object.keys(POSES).map((name, i) => ({
  id: `cls-${name}`,
  name,
  createdAt: i,
}));

const samples: Sample[] = [];
Object.entries(POSES).forEach(([name, ext]) => {
  for (let i = 0; i < 60; i++) {
    samples.push({
      id: `s-${name}-${i}`,
      classId: `cls-${name}`,
      worldLandmarks: hand(ext, 0.09),
      screenLandmarks: hand(ext, 0.3),
      handedness: (i % 4 === 0 ? 'Left' : 'Right') as Handedness,
      createdAt: i,
    });
  }
});

const started = Date.now();
const result = await trainModel(samples, classes, { ...DEFAULT_CONFIG, epochs: 60 }, () => {});
const elapsed = Date.now() - started;

const bundle = buildBundle(result, result.meta);
writeFileSync(`${outDir}/gesture_model.json`, JSON.stringify(bundle, null, 2));
writeFileSync(`${outDir}/infer.py`, generateInferPy());

// Predictions Python must reproduce, across every class and both handednesses.
const { featurize } = await import('../src/lib/features');
const { predict } = await import('../src/lib/trainer');
const probeIdx = Array.from({ length: 24 }, (_, i) => (i * 7) % samples.length);
writeFileSync(
  `${outDir}/predictions.json`,
  JSON.stringify(
    probeIdx.map((i) => {
      const s = samples[i];
      return {
        handedness: s.handedness,
        landmarks: s.worldLandmarks,
        expectedProbs: Array.from(
          predict(result.model, featurize(s.worldLandmarks, s.handedness), result.standardization),
        ),
        expectedClass: classes.find((c) => c.id === s.classId)!.name,
      };
    }),
    null,
    2,
  ),
);

console.log(
  `trained ${classes.length} classes / ${samples.length} samples in ${elapsed}ms ` +
    `(${(elapsed / 60).toFixed(1)}ms per epoch), val accuracy ${(result.meta.valAccuracy * 100).toFixed(1)}%`,
);

if (result.meta.valAccuracy < 0.9) {
  console.error(`validation accuracy too low: ${result.meta.valAccuracy}`);
  process.exit(1);
}
