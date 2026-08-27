import * as tf from '@tensorflow/tfjs';
import { FEATURE_DIM, featurize } from './features';
import type { GestureClass, Sample, TrainedModel, TrainingEpoch } from './types';

export interface TrainConfig {
  epochs: number;
  batchSize: number;
  validationSplit: number;
  hiddenUnits: [number, number];
  dropout: number;
  learningRate: number;
}

export const DEFAULT_CONFIG: TrainConfig = {
  epochs: 60,
  batchSize: 16,
  validationSplit: 0.2,
  hiddenUnits: [64, 32],
  dropout: 0.2,
  learningRate: 0.003,
};

/** Per-feature mean/std, computed on the training split and shipped with the model. */
export interface Standardization {
  mean: number[];
  std: number[];
}

export interface TrainResult {
  model: tf.LayersModel;
  standardization: Standardization;
  meta: TrainedModel;
  /** One held-out sample, used to prove the Python port computes identical features. */
  golden: { sample: Sample; features: number[]; probs: number[]; className: string };
}

export const MIN_SAMPLES_PER_CLASS = 10;

function shuffled<T>(items: T[], seed = 42): T[] {
  // Deterministic shuffle so two runs on the same data are comparable.
  const out = items.slice();
  let s = seed;
  for (let i = out.length - 1; i > 0; i--) {
    s = (s * 1664525 + 1013904223) % 4294967296;
    const j = s % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** Splits per class, so every class is represented in validation. */
function stratifiedSplit(samples: Sample[], classes: GestureClass[], valFraction: number) {
  const train: Sample[] = [];
  const val: Sample[] = [];
  for (const cls of classes) {
    const mine = shuffled(samples.filter((s) => s.classId === cls.id));
    const nVal = Math.max(1, Math.round(mine.length * valFraction));
    val.push(...mine.slice(0, nVal));
    train.push(...mine.slice(nVal));
  }
  return { train: shuffled(train, 7), val };
}

function toMatrix(samples: Sample[]): number[][] {
  return samples.map((s) => Array.from(featurize(s.worldLandmarks, s.handedness)));
}

function computeStandardization(rows: number[][]): Standardization {
  const dim = FEATURE_DIM;
  const mean = new Array(dim).fill(0);
  const std = new Array(dim).fill(0);
  for (const row of rows) for (let i = 0; i < dim; i++) mean[i] += row[i];
  for (let i = 0; i < dim; i++) mean[i] /= rows.length;
  for (const row of rows) for (let i = 0; i < dim; i++) std[i] += (row[i] - mean[i]) ** 2;
  for (let i = 0; i < dim; i++) {
    std[i] = Math.sqrt(std[i] / rows.length);
    if (std[i] < 1e-6) std[i] = 1; // constant feature — leave it alone
  }
  return { mean, std };
}

export function standardize(row: number[] | Float32Array, s: Standardization): number[] {
  const out = new Array(s.mean.length);
  for (let i = 0; i < s.mean.length; i++) out[i] = (row[i] - s.mean[i]) / s.std[i];
  return out;
}

function buildModel(numClasses: number, cfg: TrainConfig): tf.LayersModel {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [FEATURE_DIM],
      units: cfg.hiddenUnits[0],
      activation: 'relu',
    }),
  );
  model.add(tf.layers.dropout({ rate: cfg.dropout }));
  model.add(tf.layers.dense({ units: cfg.hiddenUnits[1], activation: 'relu' }));
  model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
  model.compile({
    optimizer: tf.train.adam(cfg.learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

/**
 * Runs training on the CPU backend, deliberately.
 *
 * MediaPipe is already holding a WebGL context to landmark video at 30fps, and
 * tfjs competing for the GPU stalls outright. It would be the wrong trade even
 * if it worked: this model is ~6k parameters, so kernel upload costs more than
 * the arithmetic. On CPU an epoch over a few hundred samples takes milliseconds.
 */
/**
 * Hands control back to the browser so it can paint the curve mid-training.
 *
 * Uses MessageChannel rather than requestAnimationFrame or setTimeout, both of
 * which browsers throttle hard in a background tab — rAF to a full stop, timers
 * to roughly 1Hz. That would turn a two-second train into a minute, or hang it,
 * the moment you switch tabs. MessageChannel is not throttled, which is why
 * React's scheduler uses it for the same job.
 */
function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      channel.port1.close();
      resolve();
    };
    channel.port2.postMessage(null);
  });
}

async function useCpuBackend(): Promise<void> {
  if (tf.getBackend() !== 'cpu') await tf.setBackend('cpu');
  await tf.ready();
}

export async function trainModel(
  samples: Sample[],
  classes: GestureClass[],
  cfg: TrainConfig,
  onEpoch: (e: TrainingEpoch) => void,
): Promise<TrainResult> {
  const usable = classes.filter((c) => samples.some((s) => s.classId === c.id));
  if (usable.length < 2) throw new Error('Need at least two classes with samples.');

  await useCpuBackend();

  const classIndex = new Map(usable.map((c, i) => [c.id, i]));
  const { train, val } = stratifiedSplit(samples, usable, cfg.validationSplit);

  const trainRows = toMatrix(train);
  const stdz = computeStandardization(trainRows);

  const xTrain = tf.tensor2d(trainRows.map((r) => standardize(r, stdz)));
  const yTrain = tf.oneHot(
    tf.tensor1d(train.map((s) => classIndex.get(s.classId)!), 'int32'),
    usable.length,
  );
  const valRows = toMatrix(val);
  const xVal = tf.tensor2d(valRows.map((r) => standardize(r, stdz)));
  const yVal = tf.oneHot(
    tf.tensor1d(val.map((s) => classIndex.get(s.classId)!), 'int32'),
    usable.length,
  );

  const model = buildModel(usable.length, cfg);
  const history: TrainingEpoch[] = [];

  await model.fit(xTrain, yTrain, {
    epochs: cfg.epochs,
    batchSize: cfg.batchSize,
    validationData: [xVal, yVal],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const e: TrainingEpoch = {
          epoch: epoch + 1,
          loss: logs?.loss ?? 0,
          acc: logs?.acc ?? 0,
          valLoss: logs?.val_loss ?? 0,
          valAcc: logs?.val_acc ?? 0,
        };
        history.push(e);
        onEpoch(e);
        await yieldToBrowser();
      },
    },
  });

  // Confusion on the validation split — this is what tells you which two
  // gestures are too similar to each other, which is the usual real problem.
  const predsTensor = model.predict(xVal) as tf.Tensor2D;
  const preds = (await predsTensor.array()) as number[][];
  const confusion = usable.map(() => new Array(usable.length).fill(0));
  let correct = 0;
  preds.forEach((row, i) => {
    const predicted = row.indexOf(Math.max(...row));
    const actual = classIndex.get(val[i].classId)!;
    confusion[actual][predicted]++;
    if (predicted === actual) correct++;
  });

  const goldenIdx = 0;
  const golden = {
    sample: val[goldenIdx],
    features: valRows[goldenIdx],
    probs: preds[goldenIdx],
    className: usable[classIndex.get(val[goldenIdx].classId)!].name,
  };

  tf.dispose([xTrain, yTrain, xVal, yVal, predsTensor]);

  return {
    model,
    standardization: stdz,
    golden,
    meta: {
      classNames: usable.map((c) => c.name),
      confusion,
      valAccuracy: preds.length ? correct / preds.length : 0,
      history,
      trainedAt: Date.now(),
    },
  };
}

/** Runs one live frame through the trained model. */
export function predict(
  model: tf.LayersModel,
  features: Float32Array,
  stdz: Standardization,
): number[] {
  return tf.tidy(() => {
    const x = tf.tensor2d([standardize(features, stdz)]);
    const out = model.predict(x) as tf.Tensor2D;
    return Array.from(out.dataSync());
  });
}
