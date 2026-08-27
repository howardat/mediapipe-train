#!/usr/bin/env python3
"""Proves the generated Python featurizer matches the TypeScript one.

The whole export design rests on these two implementations agreeing. If they
drift, models trained in the browser produce confident nonsense on the device,
which is close to impossible to debug from the symptoms. So it is tested.

    python3 scripts/crosscheck.py <infer.py> <cases.json>
"""

import json
import sys

import numpy as np

TOLERANCE = 1e-5


def load_featurize(infer_path):
    """Pulls featurize() out of infer.py without importing cv2 or mediapipe."""
    src = open(infer_path).read()
    start = src.index("NUM_LANDMARKS =")
    end = src.index("# ------------------------------------------------------------------ classifier")
    namespace = {"np": np}
    exec(compile("import numpy as np\n" + src[start:end], infer_path, "exec"), namespace)
    return namespace["featurize"], namespace["FEATURE_DIM"]


def main():
    infer_path, cases_path = sys.argv[1], sys.argv[2]
    featurize, feature_dim = load_featurize(infer_path)
    cases = json.load(open(cases_path))

    worst = 0.0
    for i, case in enumerate(cases):
        expected = np.asarray(case["features"], dtype=np.float64)
        actual = featurize(case["landmarks"], case["handedness"]).astype(np.float64)

        if actual.shape[0] != feature_dim:
            raise SystemExit("case %d: expected %d features, got %d"
                             % (i, feature_dim, actual.shape[0]))
        if not np.all(np.isfinite(actual)):
            raise SystemExit("case %d: produced NaN or inf" % i)

        err = float(np.abs(actual - expected).max())
        worst = max(worst, err)
        if err > TOLERANCE:
            bad = int(np.argmax(np.abs(actual - expected)))
            raise SystemExit(
                "case %d (%s): feature %d differs by %.3e — ts=%.8f py=%.8f"
                % (i, case["handedness"], bad, err, expected[bad], actual[bad])
            )

    print("feature parity OK — %d cases, %d dims, max error %.2e"
          % (len(cases), feature_dim, worst))


if __name__ == "__main__":
    main()
