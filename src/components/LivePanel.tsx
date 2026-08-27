interface Props {
  classNames: string[];
  probs: number[] | null;
  threshold: number;
  onThresholdChange: (v: number) => void;
  handPresent: boolean;
}

export default function LivePanel({
  classNames,
  probs,
  threshold,
  onThresholdChange,
  handPresent,
}: Props) {
  const topIndex = probs ? probs.indexOf(Math.max(...probs)) : -1;
  const confident = probs && topIndex >= 0 && probs[topIndex] >= threshold;

  return (
    <div className="panel">
      <header>
        <span className="stage">03</span>
        <h2>Verify</h2>
        <span className="note">smoothed over 5 frames</span>
      </header>

      <div className="body">
        <div className="verdict" data-idle={!confident}>
          {confident ? classNames[topIndex] : handPresent ? 'uncertain' : 'no hand'}
        </div>

        <div className="probs">
          {classNames.map((name, i) => {
            const p = probs?.[i] ?? 0;
            return (
              <div className="prob" key={name} data-top={i === topIndex && !!confident}>
                <div className="bar">
                  <div className="fill" style={{ width: `${(p * 100).toFixed(1)}%` }} />
                  <span>{name}</span>
                </div>
                <div className="pct">{(p * 100).toFixed(0)}%</div>
              </div>
            );
          })}
        </div>

        <div className="field" style={{ marginTop: 12 }}>
          <label htmlFor="threshold">Confidence threshold</label>
          <input
            id="threshold"
            type="range"
            min={0.3}
            max={0.99}
            step={0.01}
            value={threshold}
            onChange={(e) => onThresholdChange(Number(e.target.value))}
          />
          <span className="val">{threshold.toFixed(2)}</span>
        </div>

        <p className="note-block">
          Try each gesture at different angles and distances. Whatever fails here will fail on
          the device — record more samples of the poses that wobble, then retrain.
        </p>
      </div>
    </div>
  );
}
