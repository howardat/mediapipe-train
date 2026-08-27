/**
 * The feature contract.
 *
 * Everything in this file defines how 21 raw MediaPipe world landmarks become a
 * fixed-length vector. The exporter generates the Python implementation directly
 * from these constants, so the browser and the edge device cannot drift apart.
 *
 * Change anything here and you must bump FEATURE_SPEC_VERSION. Stored samples
 * keep raw landmarks, never features, so old datasets survive a spec change —
 * you just retrain.
 */

export const FEATURE_SPEC_VERSION = 1;

export const NUM_LANDMARKS = 21;

/** Wrist → each fingertip, following the skeleton. Used for flexion angles. */
export const FINGER_CHAINS: number[][] = [
  [0, 1, 2, 3, 4], // thumb
  [0, 5, 6, 7, 8], // index
  [0, 9, 10, 11, 12], // middle
  [0, 13, 14, 15, 16], // ring
  [0, 17, 18, 19, 20], // pinky
];

export const TIP_INDICES = [4, 8, 12, 16, 20];

/** Wrist → middle-finger MCP. A stable bone we divide by to kill hand size. */
export const SCALE_REF: [number, number] = [0, 9];

const COORD_DIM = NUM_LANDMARKS * 3; // 63
const ANGLE_DIM = FINGER_CHAINS.reduce((n, c) => n + (c.length - 2), 0); // 15
const TIP_PAIR_DIM = (TIP_INDICES.length * (TIP_INDICES.length - 1)) / 2; // 10

/** 63 normalized coordinates + 15 flexion cosines + 10 fingertip distances. */
export const FEATURE_DIM = COORD_DIM + ANGLE_DIM + TIP_PAIR_DIM; // 88

export type Vec3 = [number, number, number];
export type Handedness = 'Left' | 'Right';

export interface FeatureSpec {
  version: number;
  dim: number;
  mirrorLeftHand: boolean;
  scaleRef: [number, number];
  fingerChains: number[][];
  tipIndices: number[];
}

export const FEATURE_SPEC: FeatureSpec = {
  version: FEATURE_SPEC_VERSION,
  dim: FEATURE_DIM,
  mirrorLeftHand: true,
  scaleRef: SCALE_REF,
  fingerChains: FINGER_CHAINS,
  tipIndices: TIP_INDICES,
};

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function norm(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Turn one hand's world landmarks into a feature vector.
 *
 * Normalization, in order:
 *   1. mirror left hands across x, so both hands land in the same space
 *   2. translate so the wrist is the origin  (kills position in frame)
 *   3. divide by the wrist→middle-MCP bone   (kills distance from camera)
 *
 * Global rotation is deliberately preserved: it is the only thing separating
 * thumbs-up from thumbs-down.
 */
export function featurize(
  worldLandmarks: Vec3[],
  handedness: Handedness,
  mirrorLeftHand = FEATURE_SPEC.mirrorLeftHand,
): Float32Array {
  if (worldLandmarks.length !== NUM_LANDMARKS) {
    throw new Error(`expected ${NUM_LANDMARKS} landmarks, got ${worldLandmarks.length}`);
  }

  const flip = mirrorLeftHand && handedness === 'Left' ? -1 : 1;
  const pts: Vec3[] = worldLandmarks.map((p) => [p[0] * flip, p[1], p[2]]);

  const origin = pts[SCALE_REF[0]];
  const centered: Vec3[] = pts.map((p) => sub(p, origin));

  const refLen = norm(centered[SCALE_REF[1]]);
  const scale = refLen > 1e-6 ? refLen : 1;
  const n: Vec3[] = centered.map((p) => [p[0] / scale, p[1] / scale, p[2] / scale]);

  const out = new Float32Array(FEATURE_DIM);
  let k = 0;

  for (const p of n) {
    out[k++] = p[0];
    out[k++] = p[1];
    out[k++] = p[2];
  }

  // Cosine of the angle between consecutive bones along each finger. Gives the
  // net a pre-chewed "how curled is this finger" signal, which matters a lot
  // when there are only a few dozen samples per class.
  for (const chain of FINGER_CHAINS) {
    const bones: Vec3[] = [];
    for (let i = 0; i < chain.length - 1; i++) {
      bones.push(sub(n[chain[i + 1]], n[chain[i]]));
    }
    for (let i = 0; i < bones.length - 1; i++) {
      const la = norm(bones[i]);
      const lb = norm(bones[i + 1]);
      out[k++] = la > 1e-6 && lb > 1e-6 ? dot(bones[i], bones[i + 1]) / (la * lb) : 0;
    }
  }

  // Pairwise fingertip distances — cheap, and what makes pinch and OK-sign
  // separable from an open hand.
  for (let i = 0; i < TIP_INDICES.length; i++) {
    for (let j = i + 1; j < TIP_INDICES.length; j++) {
      out[k++] = norm(sub(n[TIP_INDICES[i]], n[TIP_INDICES[j]]));
    }
  }

  return out;
}
