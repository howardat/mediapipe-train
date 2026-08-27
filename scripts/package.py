#!/usr/bin/env python3
"""Python side of the x2_gesture package contract check.

Reads a zip produced by src/lib/zip.ts using the stdlib, which is an entirely
independent ZIP implementation -- if our hand-rolled writer is wrong, this is
what says so.

    python3 scripts/package.py zipcheck <zip> <manifest.json>
    python3 scripts/package.py coreident <pkg dir> <infer.py>
    python3 scripts/package.py parity <pkg dir> <gesture_model.json> <predictions.json>
"""

import json
import sys
import zipfile

# The markers scripts/e2e.py slices infer.py on. Duplicated here deliberately:
# if either script's idea of where the core starts drifts, the tests that
# depend on it should fail loudly rather than silently compare nothing.
CORE_START = "NUM_LANDMARKS ="
CORE_END = "# ------------------------------------------------------------------- main loop"

TOLERANCE = 1e-4


def zipcheck(zip_path, manifest_path):
    expected = json.load(open(manifest_path))
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit("corrupt entry in zip: %s" % bad)
        names = set(zf.namelist())
        want = {e["path"] for e in expected}
        if names != want:
            raise SystemExit(
                "manifest mismatch\n  missing: %s\n  extra:   %s"
                % (sorted(want - names), sorted(names - want)))
        for entry in expected:
            got = zf.read(entry["path"]).decode("utf-8")
            if got != entry["content"]:
                raise SystemExit("content differs for %s" % entry["path"])
    print("zip OK -- %d entries, CRCs valid" % len(expected))


def coreident(pkg_dir, infer_path):
    """gesture.py's core must be byte-identical to infer.py's.

    Two copies of a featurizer that drift apart is the exact failure the golden
    sample exists to catch at runtime; this catches it at build time instead.
    """
    infer = open(infer_path).read()
    infer_core = infer[infer.index(CORE_START):infer.index(CORE_END)].rstrip()

    gesture = open("%s/x2_gesture/x2_gesture/gesture.py" % pkg_dir).read()
    gesture_core = gesture[gesture.index(CORE_START):].rstrip()

    if infer_core != gesture_core:
        raise SystemExit(
            "gesture.py and infer.py disagree -- they must come from the same "
            "generateFeatureCore() call")
    print("core identical -- %d chars shared by infer.py and gesture.py"
          % len(infer_core))


def parity(pkg_dir, model_path, cases_path):
    """Import gesture.py with nothing but numpy, then replay the browser."""
    sys.path.insert(0, "%s/x2_gesture/x2_gesture" % pkg_dir)
    import gesture as g                                      # noqa: E402

    import numpy as np                                       # noqa: E402

    clf = g.GestureClassifier(model_path)
    clf.verify()

    cases = json.load(open(cases_path))
    worst = 0.0
    for i, case in enumerate(cases):
        probs = clf.predict(g.featurize(case["landmarks"], case["handedness"]))
        expected = np.asarray(case["expectedProbs"], dtype=np.float64)
        err = float(np.abs(probs - expected).max())
        worst = max(worst, err)
        if err > TOLERANCE:
            raise SystemExit(
                "case %d (%s): probabilities differ by %.3e"
                % (i, case["expectedClass"], err))
    print("gesture.py parity OK -- %d cases, max error %.2e"
          % (len(cases), worst))


MODES = {"zipcheck": zipcheck, "coreident": coreident, "parity": parity}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        raise SystemExit("usage: package.py {%s} ..." % "|".join(MODES))
    MODES[sys.argv[1]](*sys.argv[2:])
