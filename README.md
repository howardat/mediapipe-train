# Gesture Atlas

A local web app for teaching MediaPipe new hand gestures. Record poses from your
webcam, train a classifier in the browser, verify it live, and export something
that runs on an Ubuntu edge device with three pip packages and no ML runtime.

```bash
npm install
npm run dev
```

Then open http://localhost:5173. Camera access needs a secure context, so use
`localhost` — a LAN IP will be refused by the browser.

## How it works

MediaPipe's own gesture training (Model Maker) is Python-only and cannot run in a
browser. So this app splits the problem the same way MediaPipe does internally,
and trains only the last stage:

```
camera frame
   ↓
Palm detector + hand landmark model    ← pretrained, frozen, ships with the app
   ↓  21 landmarks (x, y, z) + handedness
Normalize + featurize                  ← 88 features, see src/lib/features.ts
   ↓
Dense 64 → dropout → dense 32 → softmax  ← THIS is what you train
   ↓
"peace" 0.98
```

The expensive model — the hand landmarker, ~7MB of real CNN — is never trained
and never converted. Your classifier is about 7,900 parameters for three classes,
which is why training takes roughly two seconds and the export is a JSON file.

### Features

Landmarks are normalized before they reach the model:

1. left hands are mirrored, so both hands land in the same space
2. the wrist becomes the origin, killing position in frame
3. everything is divided by the wrist→middle-knuckle bone, killing distance from camera

Global rotation is deliberately kept. It is the only thing that separates 👍 from
👎, which is why this uses normalized coordinates rather than the joint angles
people often reach for — angles are rotation-invariant, and that invariance
destroys the distinction.

The 88 features are 63 normalized coordinates, 15 finger-flexion cosines, and 10
pairwise fingertip distances. The angles and distances are redundant with the
coordinates in principle, but they measurably help when you only have a few dozen
samples per class.

## Recording good data

- Hold <kbd>space</kbd> or the record button to capture. Every frame with a hand
  in it becomes a sample, so three seconds is roughly 90 samples.
- Aim for 40+ samples per class, and at least two classes.
- Vary angle, distance, and hand position while recording. A class recorded from
  one fixed pose will look perfect in training and fall apart in the real world.
- Watch the confusion matrix after training. Off-diagonal cells tell you which
  two gestures are too similar — that is nearly always the real problem, not the
  model size or the epoch count.
- The specimen strip along the bottom draws the most representative sample of
  each class. If one looks wrong, the data is wrong.

## Deploying to an edge device

```bash
pip install mediapipe numpy opencv-python-headless
python infer.py --model gesture_model.json --landmarker hand_landmarker.task
```

Use `opencv-python-headless` on a headless box — the full build drags in GTK and
gives you `ImportError: libGL.so.1`.

Get the landmarker model with:

```bash
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

There is no TensorFlow, TFLite, or ONNX on the device. A two-layer MLP is two
matrix multiplies, so `infer.py` does it in numpy and skips the entire
tfjs → Keras → TFLite conversion chain, which would drag several hundred
megabytes of tooling in to package ~31KB of weights.

Expect roughly 10–20 fps on a Raspberry Pi 4, better on a Pi 5 or an x86 mini
PC. The cost is hand landmarking, not your classifier, which takes microseconds.
If it is slow, lower the capture resolution.

### Publishing to ROS 2

`x2_gesture.zip` is the same classifier as a colcon-buildable ROS 2 package,
generated from the same feature constants as `infer.py`. It ships as one package
inside **`x2_yolo_plus_ws/`** — the robot workspace in this repo — which is
copied to the robot whole:

```bash
scp -r x2_yolo_plus_ws agi@<robot>:~/
ssh agi@<robot> 'cd ~/x2_yolo_plus_ws && ./setup.sh'
```

One script builds every package, fetches the MediaPipe landmarker, and launches
detection, gesture, proximity and a dashboard that shows all of them on
**http://&lt;robot&gt;:8080/**. See [x2_yolo_plus_ws/README.md](x2_yolo_plus_ws/README.md).

The name is `x2_yolo_plus_ws`, not `x2_yolo_ws`, because the robot already has
the latter — copying over it would merge two trees whose `build/` and `install/`
disagree.

The workspace path is `EDGE_WORKSPACE` in `src/lib/exporter.ts` — change it there
and every generated default, launch file, start script and systemd unit follows.

Only `x2_yolo_plus_ws/src/x2_gesture/` is generated. Refresh it after touching
the generator:

```bash
npm run sync:ws
```

It writes `src/x2_gesture/` and `models/gesture_model.json`, rewrites
`export/x2_gesture.zip`, `README.md` and `infer.py` from the same generator call,
and refuses to clobber any file it did not emit unless you pass `--force`. Its
prune scan is confined to `src/x2_gesture/` — the other four packages and the
74 MB of ONNX are hand-maintained and are never deletion candidates.

Never edit `src/x2_gesture/` directly; `src/lib/rospkg.ts` is the source.

It subscribes to a **compressed** camera topic — a raw stream is 75–90 MB/s and
must not be crossed between compute units — and publishes three
`std_msgs/String` topics carrying JSON:

| Topic | Rate | What |
|---|---|---|
| `/gesture/detections` | every cycle | all class scores, smoothed and instantaneous |
| `/gesture/summary` | on change + heartbeat | the committed label and how long it's been held |
| `/gesture/text` | on change | one sentence, for `ros2 topic echo` |

Behaviour nodes should sit on `/gesture/summary`. `/gesture/detections` is the
full distribution at the cycle rate and will wake a subscriber ten times a
second to say the same thing.

A web view comes up with the node on `http://localhost:8082/` — annotated
camera, current gesture, and a bar per class. The bars separate "the model is
unsure" from "the threshold is too high", which the topics alone don't show.
Port 8081 is deliberately avoided; that's `x2_yolo_stream`'s.

The node runs the golden-sample check *before* creating any publisher, so a
featurizer mismatch never reaches the DDS graph.

Two things worth knowing before you deploy:

- MediaPipe hand landmarking is **CPU-only** (XNNPACK). It does not use CUDA or
  TensorRT, so on a Jetson it competes with whatever else is on the box rather
  than offloading. `infer_fps` is the knob.
- Body-pose keypoints cannot substitute. They have 17 joints and no fingers, so
  this is a second model, not a reuse of one you're already running.

### The failure this guards against

The browser and the device must compute *identical* features. If they disagree —
one normalizes by bone length, the other by bounding box — the model returns
confident nonsense instead of an error, which is miserable to debug in the field.

So it is checked rather than assumed:

- `infer.py` is **generated** from the same constants the browser uses, not
  written by hand alongside them
- the export embeds a **golden sample**: one held-out frame, its expected feature
  vector, and its expected output. `infer.py` verifies it at startup and exits if
  anything differs.

```bash
npm run verify:contract
```

trains a real model on synthetic hands, exports it, and has Python reproduce the
browser's predictions from the actual bundle. Run it after touching
`src/lib/features.ts` or the generator in `src/lib/exporter.ts`.

## Exports

| File | What it is |
|---|---|
| `gesture_model.json` | class names, feature spec, standardization, dense layers, golden sample |
| `infer.py` | featurizer + MLP in numpy, generated from the TypeScript source of truth |
| `x2_gesture.zip` | the same classifier as a ROS 2 package — node, launch file, systemd unit |
| `README.md` | run instructions specific to your export |
| `dataset.csv` | raw world landmarks — the escape hatch to MediaPipe Model Maker |

Samples are stored as **raw landmarks**, never as features, so changing the
feature spec does not invalidate anything you have recorded. You just retrain.

`dataset.csv` is the exit path: if you later want a real `.task` bundle, feed it
to Model Maker in Python. You are not locked into this app's format.

## When to outgrow this

This trains static poses — hand *shapes*. Motion gestures (swipe, wave, circle)
need a sequence of frames and a temporal model, which is a real addition rather
than a tweak: same landmarks, same normalization, a 1D-CNN or GRU over a ~30
frame buffer instead of an MLP over one. At that size a hand-rolled numpy port
stops being pleasant and LiteRT (`pip install ai-edge-litert`, ~5MB) earns its
place.

## Layout

```
src/lib/features.ts    the feature contract — change this and bump the version
src/lib/landmarker.ts  MediaPipe setup and per-frame detection
src/lib/dataset.ts     IndexedDB storage for classes and samples
src/lib/trainer.ts     model construction, training, confusion matrix
src/lib/exporter.ts    bundle format, the shared feature core, infer.py
src/lib/rospkg.ts      the x2_gesture ROS 2 package generator
src/lib/zip.ts         a stored-ZIP writer, so the package needs no dependency
scripts/               the contract verification described above
x2_yolo_plus_ws/       the robot workspace — scp it over, run ./setup.sh
x2_yolo_plus_ws/           the robot workspace: scp it over, run ./setup.sh
x2_yolo_plus_ws/src/x2_gesture/   generated by `npm run sync:ws` — never edit
```
