import type { ModelBundle } from '../lib/exporter';

interface Props {
  bundle: ModelBundle | null;
  onDownloadModel: () => void;
  onDownloadInfer: () => void;
  onDownloadCsv: () => void;
  onDownloadReadme: () => void;
  onDownloadPackage: () => void;
  sampleCount: number;
}

export default function ExportPanel({
  bundle,
  onDownloadModel,
  onDownloadInfer,
  onDownloadCsv,
  onDownloadReadme,
  onDownloadPackage,
  sampleCount,
}: Props) {
  return (
    <div className="panel">
      <header>
        <span className="stage">04</span>
        <h2>Export</h2>
        <span className="note">
          {bundle ? `${bundle.featureSpec.dim} features → ${bundle.classes.length} classes` : 'train first'}
        </span>
      </header>

      <div className="body">
        <div className="row">
          <button className="btn primary" onClick={onDownloadModel} disabled={!bundle}>
            gesture_model.json
          </button>
          <button className="btn" onClick={onDownloadInfer} disabled={!bundle}>
            infer.py
          </button>
          <button className="btn" onClick={onDownloadPackage} disabled={!bundle}>
            x2_gesture.zip
          </button>
          <button className="btn" onClick={onDownloadReadme} disabled={!bundle}>
            README.md
          </button>
          <button className="btn ghost" onClick={onDownloadCsv} disabled={!sampleCount}>
            dataset.csv
          </button>
        </div>

        <p className="note-block" style={{ marginTop: 12 }}>
          Run it on Ubuntu with three packages and no ML runtime — the classifier is two
          matmuls in numpy:
        </p>

        <pre
          style={{
            background: 'var(--film)',
            border: '1px solid var(--edge)',
            borderRadius: 'var(--radius)',
            padding: '10px 12px',
            fontSize: 11,
            color: 'var(--emulsion)',
            overflowX: 'auto',
            margin: '8px 0 0',
          }}
        >
{`pip install mediapipe numpy opencv-python-headless
python infer.py --model gesture_model.json \\
  --landmarker hand_landmarker.task`}
        </pre>

        <p className="note-block" style={{ marginTop: 12 }}>
          <code>x2_gesture.zip</code> is the same classifier as a ROS 2 package — unzip into a
          workspace, <code>colcon build</code>, and it publishes on{' '}
          <code>/gesture/detections</code>, <code>/gesture/summary</code> and{' '}
          <code>/gesture/text</code>. It reads a compressed camera topic, never a raw one.
        </p>

        <p className="note-block">
          <code>infer.py</code> is generated from the same constants this page uses, and checks
          one held-out sample at startup. If the two featurizers ever disagree it exits instead
          of quietly returning confident nonsense.
        </p>

        <p className="note-block">
          <code>dataset.csv</code> holds the raw world landmarks — the way out to MediaPipe
          Model Maker if you later want a real <code>.task</code> bundle.
        </p>
      </div>
    </div>
  );
}
