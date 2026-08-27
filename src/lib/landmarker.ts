import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { Detection, Handedness } from './types';
import type { Vec3 } from './features';

let instance: HandLandmarker | null = null;

/**
 * Loads the pretrained hand landmarker. This model is frozen — we never train
 * it. It turns pixels into 21 joints, and everything downstream works on joints.
 */
export async function createLandmarker(): Promise<HandLandmarker> {
  if (instance) return instance;

  const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  instance = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: '/mediapipe/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  return instance;
}

export function detect(
  landmarker: HandLandmarker,
  video: HTMLVideoElement,
  timestampMs: number,
): Detection | null {
  const result = landmarker.detectForVideo(video, timestampMs);
  if (!result.worldLandmarks?.length || !result.landmarks?.length) return null;

  const world = result.worldLandmarks[0].map((p) => [p.x, p.y, p.z] as Vec3);
  const screen = result.landmarks[0].map((p) => [p.x, p.y, p.z] as Vec3);
  const hand = result.handedness?.[0]?.[0];

  return {
    worldLandmarks: world,
    screenLandmarks: screen,
    handedness: (hand?.categoryName as Handedness) ?? 'Right',
    score: hand?.score ?? 0,
  };
}

/** Bone pairs, for drawing the skeleton. */
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];
