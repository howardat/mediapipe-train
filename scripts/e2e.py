#!/usr/bin/env python3
"""Proves the exported bundle predicts identically in Python and in the browser.

Loads the real generated infer.py and the real exported gesture_model.json, runs
the golden-sample check, then replays predictions the browser already computed.

    python3 scripts/e2e.py <dir containing infer.py, gesture_model.json, predictions.json>
"""

import json
import sys

import numpy as np

TOLERANCE = 1e-4


def load_module(infer_path):
    """Executes infer.py's feature + classifier sections without cv2/mediapipe."""
    src = open(infer_path).read()
    start = src.index("NUM_LANDMARKS =")
    end = src.index("# ------------------------------------------------------------------- main loop")
    namespace = {}
    exec(compile("import json\nimport numpy as np\n" + src[start:end], infer_path, "exec"), namespace)
    return namespace


def main():
    out = sys.argv[1]
    ns = load_module(f"{out}/infer.py")
    clf = ns["GestureClassifier"](f"{out}/gesture_model.json")
    featurize = ns["featurize"]

    # The check that ships to the device.
    clf.verify()

    cases = json.load(open(f"{out}/predictions.json"))
    worst = 0.0
    mismatches = 0
    for i, case in enumerate(cases):
        probs = clf.predict(featurize(case["landmarks"], case["handedness"]))
        expected = np.asarray(case["expectedProbs"], dtype=np.float64)

        err = float(np.abs(probs - expected).max())
        worst = max(worst, err)
        if err > TOLERANCE:
            raise SystemExit(
                "case %d (%s): probabilities differ by %.3e\n  browser=%s\n  python =%s"
                % (i, case["expectedClass"], err, np.round(expected, 6), np.round(probs, 6))
            )
        if clf.classes[int(np.argmax(probs))] != case["expectedClass"]:
            mismatches += 1

    if mismatches:
        raise SystemExit("%d of %d cases predicted the wrong class" % (mismatches, len(cases)))

    print("inference parity OK — %d cases, max probability error %.2e" % (len(cases), worst))


if __name__ == "__main__":
    main()
