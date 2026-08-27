import {
  FEATURE_SPEC,
  FINGER_CHAINS,
  NUM_LANDMARKS,
  SCALE_REF,
  TIP_INDICES,
} from './features';
import type { Standardization } from './trainer';
import type { GestureClass, Sample, TrainedModel } from './types';
import type { TrainResult } from './trainer';

/**
 * Where the workspace lands on the robot. Baked into the generated launch
 * defaults, the systemd unit and both READMEs, so it lives in one place rather
 * than six.
 *
 * `x2_yolo_plus_ws`, not `x2_yolo_ws`: the robot already has the latter, and the
 * deployment is "copy the folder over", which would merge two trees whose
 * build/ and install/ directories disagree. The new name lets both sit in the
 * home directory at once.
 *
 * Only the *default* is hardcoded. The workspace's setup.sh resolves the root
 * from its own location and exports X2_YOLO_WS, so a copy that lands somewhere
 * else still builds and runs.
 */
export const EDGE_WORKSPACE = '/home/agi/x2_yolo_plus_ws';

export interface ModelBundle {
  format: 'gesture-atlas/v1';
  createdAt: string;
  classes: string[];
  valAccuracy: number;
  featureSpec: typeof FEATURE_SPEC;
  standardization: Standardization;
  layers: { activation: 'relu' | 'softmax'; w: number[][]; b: number[] }[];
  golden: {
    expectedClass: string;
    handedness: string;
    worldLandmarks: number[][];
    features: number[];
    probs: number[];
  };
}

/** Pulls the dense layers out of the tfjs model as plain arrays. */
export function buildBundle(result: TrainResult, meta: TrainedModel): ModelBundle {
  const layers: ModelBundle['layers'] = [];
  const dense = result.model.layers.filter((l) => l.getWeights().length === 2);

  dense.forEach((layer, i) => {
    const [kernel, bias] = layer.getWeights();
    layers.push({
      activation: i === dense.length - 1 ? 'softmax' : 'relu',
      w: kernel.arraySync() as number[][],
      b: Array.from(bias.dataSync()),
    });
  });

  return {
    format: 'gesture-atlas/v1',
    createdAt: new Date().toISOString(),
    classes: meta.classNames,
    valAccuracy: meta.valAccuracy,
    featureSpec: FEATURE_SPEC,
    standardization: result.standardization,
    layers,
    golden: {
      expectedClass: result.golden.className,
      handedness: result.golden.sample.handedness,
      worldLandmarks: result.golden.sample.worldLandmarks.map((p) => [...p]),
      features: result.golden.features,
      probs: result.golden.probs,
    },
  };
}

/**
 * The half of the generated Python that both deployment targets share: the
 * feature constants, the featurizer, and the classifier with its golden check.
 *
 * Imports nothing but json and numpy — deliberately. `x2_gesture/gesture.py`
 * is this text verbatim, and contract verification imports it on a machine
 * with no OpenCV, no MediaPipe and no ROS.
 *
 * The constants are injected from the TypeScript source of truth. That is the
 * only reason the browser and the edge device can be trusted to agree.
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


def featurize(world_landmarks, handedness):
    """21x3 world landmarks -> feature vector. Mirrors src/lib/features.ts."""
    pts = np.asarray(world_landmarks, dtype=np.float64).reshape(NUM_LANDMARKS, 3)

    if MIRROR_LEFT_HAND and handedness == "Left":
        pts = pts * np.array([-1.0, 1.0, 1.0])

    pts = pts - pts[SCALE_REF[0]]
    ref = np.linalg.norm(pts[SCALE_REF[1]])
    pts = pts / (ref if ref > 1e-6 else 1.0)

    feats = [pts.reshape(-1)]

    angles = []
    for chain in FINGER_CHAINS:
        bones = [pts[chain[i + 1]] - pts[chain[i]] for i in range(len(chain) - 1)]
        for i in range(len(bones) - 1):
            la = np.linalg.norm(bones[i])
            lb = np.linalg.norm(bones[i + 1])
            if la > 1e-6 and lb > 1e-6:
                angles.append(float(np.dot(bones[i], bones[i + 1]) / (la * lb)))
            else:
                angles.append(0.0)
    feats.append(np.asarray(angles))

    dists = []
    for i in range(len(TIP_INDICES)):
        for j in range(i + 1, len(TIP_INDICES)):
            dists.append(float(np.linalg.norm(pts[TIP_INDICES[i]] - pts[TIP_INDICES[j]])))
    feats.append(np.asarray(dists))

    return np.concatenate(feats).astype(np.float32)


# ------------------------------------------------------------------ classifier
class GestureClassifier:
    def __init__(self, path):
        with open(path) as fh:
            bundle = json.load(fh)

        spec = bundle["featureSpec"]
        if spec["version"] != SPEC_VERSION:
            raise SystemExit(
                "Feature spec mismatch: model is v%s, this script is v%s. Re-export."
                % (spec["version"], SPEC_VERSION)
            )

        self.classes = bundle["classes"]
        self.mean = np.asarray(bundle["standardization"]["mean"], dtype=np.float32)
        self.std = np.asarray(bundle["standardization"]["std"], dtype=np.float32)
        self.layers = [
            (
                np.asarray(l["w"], dtype=np.float32),
                np.asarray(l["b"], dtype=np.float32),
                l["activation"],
            )
            for l in bundle["layers"]
        ]
        self.golden = bundle["golden"]
        self.val_accuracy = bundle.get("valAccuracy", float("nan"))

    def predict(self, feats):
        x = (feats - self.mean) / self.std
        for w, b, activation in self.layers:
            x = x @ w + b
            if activation == "relu":
                x = np.maximum(x, 0.0)
            elif activation == "softmax":
                x = np.exp(x - x.max())
                x = x / x.sum()
        return x

    def verify(self):
        """Fails loudly at startup if this featurizer disagrees with the browser.

        Without this, a normalization mismatch produces confident wrong answers
        instead of an error, which is far more expensive to debug in the field.
        """
        g = self.golden
        feats = featurize(g["worldLandmarks"], g["handedness"])
        expected_feats = np.asarray(g["features"], dtype=np.float32)
        feat_err = float(np.abs(feats - expected_feats).max())

        probs = self.predict(feats)
        prob_err = float(np.abs(probs - np.asarray(g["probs"], dtype=np.float32)).max())
        predicted = self.classes[int(np.argmax(probs))]

        ok = feat_err < 1e-3 and prob_err < 1e-3 and predicted == g["expectedClass"]
        status = "PASS" if ok else "FAIL"
        print(
            "[golden] %s  feature_err=%.2e  prob_err=%.2e  predicted=%s expected=%s"
            % (status, feat_err, prob_err, predicted, g["expectedClass"])
        )
        if not ok:
            raise SystemExit("Golden-sample check failed — features do not match the browser.")
`;
}

/**
 * The standalone edge-device script: the shared core plus a camera loop.
 *
 * scripts/e2e.py slices this file between "NUM_LANDMARKS =" and the main-loop
 * banner below, and execs the result without cv2 or mediapipe. Both markers
 * must keep their exact spelling.
 */
export function generateInferPy(): string {
  return `#!/usr/bin/env python3
"""
Runs a gesture model exported from Gesture Atlas.

  pip install mediapipe numpy opencv-python-headless

  python infer.py --model gesture_model.json --landmarker hand_landmarker.task

No TensorFlow, no TFLite, no ONNX. The classifier is a two-layer MLP, so
inference is a couple of numpy matmuls. The expensive part is MediaPipe's
hand landmarker, which is pretrained and ships inside the mediapipe wheel.

That landmarker is what --delegate points at a GPU. The default, "auto", tries
MediaPipe's GPU delegate and drops back to the CPU when it cannot be used,
saying which it got; --delegate cpu pins the CPU. Expect a few times faster,
not an order of magnitude: the model is small and every frame is uploaded from
CPU memory.

The feature code below is generated from the same constants the browser used.
Do not edit it by hand — re-export instead.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

${generateFeatureCore()}

# ------------------------------------------------------------------- main loop
def to_image(frame_bgr, delegate):
    """An OpenCV BGR frame -> the mp.Image this delegate can actually take.

    The GPU path only accepts 4 channels. Handing it 3 does not raise, it
    aborts the process with "unsupported ImageFrame format", so the channel
    count is not a detail to leave to the caller.
    """
    if delegate == "gpu":
        return mp.Image(image_format=mp.ImageFormat.SRGBA,
                        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA))
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def build_landmarker(model_path, delegate):
    """One HandLandmarker on one delegate. May abort the process -- see below."""
    kinds = mp_python.BaseOptions.Delegate
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path,
                delegate=kinds.GPU if delegate == "gpu" else kinds.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
    )


def run_blank_frames(landmarker, delegate, runs=3, start=0):
    """Push empty frames through the graph. Returns the last timestamp used."""
    if delegate == "gpu":
        blank = np.zeros((256, 256, 4), dtype=np.uint8)
        blank[:, :, 3] = 255
        image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=blank)
    else:
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.zeros((256, 256, 3), dtype=np.uint8))

    stamp = start
    for _ in range(runs):
        stamp += 1
        landmarker.detect_for_video(image, stamp)
    return stamp


def open_landmarker(model_path, delegate="auto"):
    """Open the landmarker on the GPU if the GPU works, else the CPU.

    Returns (landmarker, delegate, last timestamp used), where the delegate is
    what was actually built rather than what was asked for.

    The awkward part is that MediaPipe does not raise when a delegate cannot
    run. It fails as a CHECK failure inside C++, which calls abort(): there is
    no exception, nothing for try/except to catch, and the script dies instead
    of falling back. So the GPU is tried in a CHILD PROCESS first -- this same
    file, run with --probe -- and built here only once the child has survived
    it. The crash, if there is one, lands on the child.

    The ROS package does the same thing with more care in
    x2_gesture/runtime.py; if you are deploying rather than trying this out,
    read that one.
    """
    wanted = ["gpu", "cpu"] if delegate == "auto" else [delegate]
    if "cpu" not in wanted:
        wanted.append("cpu")

    for i, kind in enumerate(wanted):
        # The last rung is not probed: there is nothing left to fall back to,
        # so a probe could only make the same failure slower to arrive.
        if i < len(wanted) - 1:
            done = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--probe",
                 model_path, kind],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if done.returncode != 0:
                why = ("crashed with signal %d" % -done.returncode
                       if done.returncode < 0
                       else "exited %d" % done.returncode)
                print("[landmarker] %s delegate unusable (%s)" % (kind, why))
                continue

        try:
            landmarker = build_landmarker(model_path, kind)
            return landmarker, kind, run_blank_frames(landmarker, kind)
        except Exception as exc:                            # noqa: BLE001
            print("[landmarker] %s delegate failed to open (%s)"
                  % (kind, " ".join(str(exc).split())[:200]))

    raise SystemExit("could not open %s on any delegate" % model_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gesture_model.json")
    ap.add_argument("--landmarker", default="hand_landmarker.task")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--smoothing", type=int, default=5,
                    help="frames to average before committing to a gesture")
    ap.add_argument("--show", action="store_true", help="open a preview window")
    ap.add_argument("--delegate", default="auto", choices=("auto", "gpu", "cpu"),
                    help="auto tries MediaPipe's GPU delegate and falls back")
    args = ap.parse_args()

    clf = GestureClassifier(args.model)
    print("[model] %d classes: %s" % (len(clf.classes), ", ".join(clf.classes)))
    print("[model] validation accuracy at export: %.3f" % clf.val_accuracy)
    clf.verify()

    landmarker, delegate, stamp = open_landmarker(args.landmarker,
                                                  args.delegate)
    print("[landmarker] running on the %s" % delegate.upper())

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Could not open camera %d" % args.camera)

    history = []
    current = None
    start = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            image = to_image(frame, delegate)
            # VIDEO mode rejects a timestamp that is not strictly greater than
            # the last one it saw, and opening the landmarker already spent a
            # few. Carrying on from that count rather than from zero is what
            # keeps the first real frame from raising.
            stamp = max(stamp + 1, int((time.time() - start) * 1000))
            result = landmarker.detect_for_video(image, stamp)

            label, confidence = None, 0.0
            if result.hand_world_landmarks:
                lm = [[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]]
                handedness = result.handedness[0][0].category_name
                probs = clf.predict(featurize(lm, handedness))
                history.append(probs)
                history = history[-args.smoothing:]

                averaged = np.mean(history, axis=0)
                best = int(np.argmax(averaged))
                if averaged[best] >= args.threshold:
                    label, confidence = clf.classes[best], float(averaged[best])
            else:
                history.clear()

            if label != current:
                current = label
                if label:
                    print("gesture: %-16s %.2f" % (label, confidence), flush=True)
                else:
                    print("gesture: -", flush=True)

            if args.show:
                text = "%s %.2f" % (label, confidence) if label else "-"
                cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 200), 2)
                cv2.imshow("gesture", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if frames % 120 == 0:
                fps = frames / (time.time() - start)
                print("[perf] %.1f fps" % fps, file=sys.stderr, flush=True)
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # The child half of open_landmarker's delegate check. Its whole job is to
    # do the thing that might abort, so that the abort happens here instead of
    # in the run the user is watching. Exit status is the answer; the noise
    # MediaPipe writes on the way is captured and thrown away.
    if len(sys.argv) > 3 and sys.argv[1] == "--probe":
        probe_landmarker = build_landmarker(sys.argv[2], sys.argv[3])
        run_blank_frames(probe_landmarker, sys.argv[3], runs=2)
        probe_landmarker.close()
        raise SystemExit(0)
    main()
`;
}

/** Raw landmarks as CSV — the escape hatch to MediaPipe Model Maker or pandas. */
export function generateCsv(samples: Sample[], classes: GestureClass[]): string {
  const nameById = new Map(classes.map((c) => [c.id, c.name]));
  const header = ['class', 'handedness'];
  for (let i = 0; i < NUM_LANDMARKS; i++) header.push(`lm${i}_x`, `lm${i}_y`, `lm${i}_z`);

  const rows = [header.join(',')];
  for (const s of samples) {
    const row: (string | number)[] = [nameById.get(s.classId) ?? 'unknown', s.handedness];
    for (const p of s.worldLandmarks) row.push(p[0].toFixed(6), p[1].toFixed(6), p[2].toFixed(6));
    rows.push(row.join(','));
  }
  return rows.join('\n');
}

export function generateReadme(bundle: ModelBundle): string {
  return `# ${bundle.classes.length}-gesture model

Exported ${bundle.createdAt} · validation accuracy ${(bundle.valAccuracy * 100).toFixed(1)}%
Classes: ${bundle.classes.join(', ')}

## Run it on Ubuntu

    pip install mediapipe numpy opencv-python-headless
    python infer.py --model gesture_model.json --landmarker hand_landmarker.task

\`hand_landmarker.task\` is the pretrained MediaPipe model. Grab it with:

    curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Landmarking uses MediaPipe's GPU delegate when it is available and the CPU
when it is not — the startup line says which. \`--delegate cpu\` pins the CPU;
on a Jetson, see \`x2_gesture/scripts/gpu_env.sh\` for what has to be on the
library path before python starts.

## Run it as a ROS 2 node

\`x2_gesture\` is one package in \`x2_yolo_plus_ws\`, next to \`x2_yolo_stream\` and
a dashboard that shows both on one page. The whole workspace is copied to the
robot and brought up with one script:

    scp -r x2_yolo_plus_ws agi@<robot>:~/
    ssh agi@<robot> 'cd ${EDGE_WORKSPACE} && ./setup.sh'

That builds every package, fetches \`hand_landmarker.task\`, and launches
detection, gesture and proximity — with everything on one page at
**http://<robot>:8080/**.

To update only this package in a workspace that is already there:

    unzip -o x2_gesture.zip -d ${EDGE_WORKSPACE}/src/
    cp gesture_model.json ${EDGE_WORKSPACE}/models/
    cd ${EDGE_WORKSPACE} && colcon build --packages-select x2_gesture
    source install/setup.bash
    ros2 launch x2_gesture x2_gesture.launch.py

\`--packages-select\` matters: a bare \`colcon build\` rebuilds \`x2_yolo_stream\`
too, which is not what you came for.

Three topics, all \`std_msgs/String\` carrying JSON:

    /gesture/detections   every cycle, all ${bundle.classes.length} class scores
    /gesture/summary      on change + 5s heartbeat, the committed label
    /gesture/text         on change, one sentence

    ros2 topic echo /gesture/text

A web view starts with the node on port 8082 (8081 is x2_yolo_stream's):

    http://localhost:8082/

It shows the annotated camera plus a bar per class, which is how you tell
"the model is unsure" apart from "the threshold is too high".

Subscribe to \`/gesture/summary\` from a behaviour node. \`/gesture/detections\` is
the full distribution at the cycle rate and will wake you ten times a second.

See \`x2_gesture/README.md\` inside the zip for parameters and deployment.

## What's in the box

- \`gesture_model.json\` — class names, feature spec, standardization, and ${bundle.layers.length} dense layers (${bundle.featureSpec.dim} inputs)
- \`infer.py\` — the featurizer and MLP in numpy, generated from the same constants the browser used
- \`x2_gesture.zip\` — the same classifier as a colcon-buildable ROS 2 package
- \`dataset.csv\` — raw world landmarks, if you later want MediaPipe Model Maker to build a .task bundle

## Golden-sample check

\`infer.py\` verifies one held-out sample at startup and exits if the features it
computes differ from the browser's. A silent normalization mismatch is the most
likely way this deployment breaks, so it is checked rather than assumed.
`;
}
