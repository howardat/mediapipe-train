// Emits deterministic test cases for the TS↔Python feature-parity check.
// Paired with scripts/crosscheck.py; run both via `npm run verify:contract`.
import { featurize, NUM_LANDMARKS, type Handedness, type Vec3 } from '../src/lib/features';

/** Deterministic LCG, so the fixtures are identical on every machine. */
function makeHand(seed: number): Vec3[] {
  let s = seed;
  const next = () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648 - 0.5;
  };
  return Array.from({ length: NUM_LANDMARKS }, () => [next(), next(), next() * 0.4] as Vec3);
}

const cases = [];
for (let i = 0; i < 8; i++) {
  const handedness: Handedness = i % 2 === 0 ? 'Right' : 'Left';
  const landmarks = makeHand(i * 7919 + 13);
  cases.push({
    handedness,
    landmarks,
    features: Array.from(featurize(landmarks, handedness)),
  });
}

// A degenerate hand: every point identical, so the scale reference is zero.
// Both implementations must fall back to scale 1 rather than divide by zero.
const flat: Vec3[] = Array.from({ length: NUM_LANDMARKS }, () => [0.5, 0.5, 0] as Vec3);
cases.push({
  handedness: 'Right' as Handedness,
  landmarks: flat,
  features: Array.from(featurize(flat, 'Right')),
});

process.stdout.write(JSON.stringify(cases, null, 2));
