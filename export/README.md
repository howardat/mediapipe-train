# 4-gesture model

Exported 2026-08-15T05:33:57.359Z · validation accuracy 100.0%
Classes: okay, phone, untitled, korean_love

## Run it on Ubuntu

    pip install mediapipe numpy opencv-python-headless
    python infer.py --model gesture_model.json --landmarker hand_landmarker.task

`hand_landmarker.task` is the pretrained MediaPipe model. Grab it with:

    curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Landmarking uses MediaPipe's GPU delegate when it is available and the CPU
when it is not — the startup line says which. `--delegate cpu` pins the CPU;
on a Jetson, see `x2_gesture/scripts/gpu_env.sh` for what has to be on the
library path before python starts.

## Run it as a ROS 2 node

`x2_gesture` is one package in `x2_yolo_plus_ws`, next to `x2_yolo_stream` and
a dashboard that shows both on one page. The whole workspace is copied to the
robot and brought up with one script:

    scp -r x2_yolo_plus_ws agi@<robot>:~/
    ssh agi@<robot> 'cd /home/agi/x2_yolo_plus_ws && ./setup.sh'

That builds every package, fetches `hand_landmarker.task`, and launches
detection, gesture and proximity — with everything on one page at
**http://<robot>:8080/**.

To update only this package in a workspace that is already there:

    unzip -o x2_gesture.zip -d /home/agi/x2_yolo_plus_ws/src/
    cp gesture_model.json /home/agi/x2_yolo_plus_ws/models/
    cd /home/agi/x2_yolo_plus_ws && colcon build --packages-select x2_gesture
    source install/setup.bash
    ros2 launch x2_gesture x2_gesture.launch.py

`--packages-select` matters: a bare `colcon build` rebuilds `x2_yolo_stream`
too, which is not what you came for.

Three topics, all `std_msgs/String` carrying JSON:

    /gesture/detections   every cycle, all 4 class scores
    /gesture/summary      on change + 5s heartbeat, the committed label
    /gesture/text         on change, one sentence

    ros2 topic echo /gesture/text

A web view starts with the node on port 8082 (8081 is x2_yolo_stream's):

    http://localhost:8082/

It shows the annotated camera plus a bar per class, which is how you tell
"the model is unsure" apart from "the threshold is too high".

Subscribe to `/gesture/summary` from a behaviour node. `/gesture/detections` is
the full distribution at the cycle rate and will wake you ten times a second.

See `x2_gesture/README.md` inside the zip for parameters and deployment.

## What's in the box

- `gesture_model.json` — class names, feature spec, standardization, and 3 dense layers (88 inputs)
- `infer.py` — the featurizer and MLP in numpy, generated from the same constants the browser used
- `x2_gesture.zip` — the same classifier as a colcon-buildable ROS 2 package
- `dataset.csv` — raw world landmarks, if you later want MediaPipe Model Maker to build a .task bundle

## Golden-sample check

`infer.py` verifies one held-out sample at startup and exits if the features it
computes differ from the browser's. A silent normalization mismatch is the most
likely way this deployment breaks, so it is checked rather than assumed.
