import { DEFAULT_CONFIG, MIN_SAMPLES_PER_CLASS, type TrainConfig } from '../lib/trainer';
import type { GestureClass, TrainedModel, TrainingEpoch } from '../lib/types';

interface Props {
  classes: GestureClass[];
  counts: Record<string, number>;
  config: TrainConfig;
  onConfigChange: (c: TrainConfig) => void;
  epochTarget: number;
  training: boolean;
  history: TrainingEpoch[];
  trained: TrainedModel | null;
  error: string | null;
  onTrain: () => void;
}

/** Accuracy over epochs. Fixed 0–1 range, so the shape is comparable run to run. */
function AccuracyCurve({ history }: { history: TrainingEpoch[] }) {
  const W = 320;
  const H = 96;
  const pad = 4;

  const path = (pick: (e: TrainingEpoch) => number) => {
    if (history.length < 2) return '';
    return history
      .map((e, i) => {
        const x = pad + (i / (history.length - 1)) * (W - pad * 2);
        const y = H - pad - pick(e) * (H - pad * 2);
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  };

  return (
    <>
      <svg className="curve" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
           aria-label="Training and validation accuracy per epoch">
        {[0.25, 0.5, 0.75].map((g) => (
          <line key={g} x1={0} x2={W} y1={H - pad - g * (H - pad * 2)}
                y2={H - pad - g * (H - pad * 2)} stroke="#1e2f39" strokeWidth={1} />
        ))}
        <path d={path((e) => e.acc)} fill="none" stroke="#8faebc" strokeWidth={1.5} />
        <path d={path((e) => e.valAcc)} fill="none" stroke="#58d6e8" strokeWidth={2} />
      </svg>
      <div className="legend">
        <span className="train">train</span>
        <span className="val">validation</span>
      </div>
    </>
  );
}

function Confusion({ model }: { model: TrainedModel }) {
  return (
    <table className="confusion">
      <caption className="legend" style={{ captionSide: 'top', marginBottom: 4 }}>
        <span style={{ color: 'var(--dim)' }}>rows = actual · columns = predicted</span>
      </caption>
      <thead>
        <tr>
          <th />
          {model.classNames.map((n) => (
            <th key={n}>{n.slice(0, 4)}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {model.confusion.map((row, i) => (
          <tr key={model.classNames[i]}>
            <th className="row-head">{model.classNames[i]}</th>
            {row.map((v, j) => (
              <td
                key={j}
                data-hit={i === j && v > 0}
                data-miss={i !== j && v > 0}
                title={`${v} ${model.classNames[i]} predicted as ${model.classNames[j]}`}
              >
                {v || '·'}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function TrainPanel({
  classes,
  counts,
  config,
  onConfigChange,
  epochTarget,
  training,
  history,
  trained,
  error,
  onTrain,
}: Props) {
  const populated = classes.filter((c) => (counts[c.id] ?? 0) > 0);
  const thin = populated.filter((c) => (counts[c.id] ?? 0) < MIN_SAMPLES_PER_CLASS);
  const total = Object.values(counts).reduce((a, b) => a + b, 0);

  // Says exactly what is missing and how much, so the disabled button is never
  // a mystery. Ordered by what you have to fix first.
  let blocker: string | null = null;
  if (classes.length === 0) {
    blocker = 'Add a gesture above, then hold space to record it.';
  } else if (populated.length === 0) {
    blocker = `Select ${classes.length === 1 ? classes[0].name : 'a gesture'} and hold space to record samples.`;
  } else if (populated.length < 2) {
    blocker = `Only ${populated[0].name} has samples. Record a second gesture — a classifier needs something to choose between.`;
  } else if (thin.length) {
    blocker = thin
      .map((c) => `${c.name} needs ${MIN_SAMPLES_PER_CLASS - (counts[c.id] ?? 0)} more (${counts[c.id] ?? 0}/${MIN_SAMPLES_PER_CLASS})`)
      .join(' · ');
  }

  const last = history[history.length - 1];

  return (
    <div className="panel">
      <header>
        <span className="stage">02</span>
        <h2>Train</h2>
        <span className="note">{total} samples</span>
      </header>

      <div className="body">
        <div className="readout">
          <div>
            <div className="label">Epoch</div>
            <div className="value">
              {last ? `${last.epoch}/${epochTarget}` : '—'}
            </div>
          </div>
          <div>
            <div className="label">Val acc</div>
            <div className="value" style={{ color: trained ? 'var(--live)' : undefined }}>
              {last ? (last.valAcc * 100).toFixed(1) : '—'}
            </div>
          </div>
          <div>
            <div className="label">Loss</div>
            <div className="value">{last ? last.loss.toFixed(3) : '—'}</div>
          </div>
        </div>

        <AccuracyCurve history={history} />

        <div className="stack" style={{ marginTop: 12 }}>
          <div className="field">
            <label htmlFor="epochs">Epochs</label>
            <input
              id="epochs"
              type="range"
              min={20}
              max={200}
              step={10}
              value={config.epochs}
              disabled={training}
              onChange={(e) => onConfigChange({ ...config, epochs: Number(e.target.value) })}
            />
            <span className="val">{config.epochs}</span>
          </div>
          <div className="field">
            <label htmlFor="split">Validation split</label>
            <input
              id="split"
              type="range"
              min={0.1}
              max={0.4}
              step={0.05}
              value={config.validationSplit}
              disabled={training}
              onChange={(e) =>
                onConfigChange({ ...config, validationSplit: Number(e.target.value) })
              }
            />
            <span className="val">{config.validationSplit.toFixed(2)}</span>
          </div>
        </div>

        {error && <p className="alert">{error}</p>}
        {blocker && !training && <p className="requirement">{blocker}</p>}

        <div className="row" style={{ marginTop: 12 }}>
          <button className="btn primary" onClick={onTrain} disabled={training || !!blocker}>
            {training ? 'Training…' : trained ? 'Retrain' : 'Train model'}
          </button>
          {!training && config !== DEFAULT_CONFIG && (
            <button className="btn ghost" onClick={() => onConfigChange(DEFAULT_CONFIG)}>
              Reset
            </button>
          )}
        </div>

        {trained && <Confusion model={trained} />}
      </div>
    </div>
  );
}
