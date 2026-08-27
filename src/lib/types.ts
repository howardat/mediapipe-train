import type { Handedness, Vec3 } from './features';

export type { Handedness, Vec3 };

export interface GestureClass {
  id: string;
  name: string;
  createdAt: number;
}

/**
 * One captured frame. We store raw landmarks rather than features so that the
 * feature spec can change without invalidating everything you have recorded.
 */
export interface Sample {
  id: string;
  classId: string;
  worldLandmarks: Vec3[];
  /** Image-space landmarks, kept only for drawing specimen thumbnails. */
  screenLandmarks: Vec3[];
  handedness: Handedness;
  createdAt: number;
}

export interface Detection {
  worldLandmarks: Vec3[];
  screenLandmarks: Vec3[];
  handedness: Handedness;
  score: number;
}

export interface TrainingEpoch {
  epoch: number;
  loss: number;
  acc: number;
  valLoss: number;
  valAcc: number;
}

export interface TrainedModel {
  classNames: string[];
  /** Row i = true class, column j = predicted class, on the validation split. */
  confusion: number[][];
  valAccuracy: number;
  history: TrainingEpoch[];
  trainedAt: number;
}
