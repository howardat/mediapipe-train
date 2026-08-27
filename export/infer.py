#!/usr/bin/env python3
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

# ---------------------------------------------------------------- feature spec
NUM_LANDMARKS = 21
FINGER_CHAINS = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
TIP_INDICES = [4,8,12,16,20]
SCALE_REF = [0,9]
MIRROR_LEFT_HAND = True
FEATURE_DIM = 88
SPEC_VERSION = 1


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
