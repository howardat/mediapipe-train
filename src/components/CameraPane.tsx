import type { RefObject } from 'react';
import { MIN_SAMPLES_PER_CLASS } from '../lib/trainer';

interface Props {
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  status: 'loading' | 'ready' | 'error';
  handPresent: boolean;
  handedness: string | null;
  fps: number;
  recording: boolean;
  burstCount: number;
  selectedName: string | null;
  selectedCount: number;
  onRecordStart: () => void;
  onRecordStop: () => void;
}

export default function CameraPane({
  videoRef,
  canvasRef,
  status,
  handPresent,
  handedness,
  fps,
  recording,
  burstCount,
  selectedName,
  selectedCount,
  onRecordStart,
  onRecordStop,
}: Props) {
  const canRecord = status === 'ready' && !!selectedName;
  const short = Math.max(0, MIN_SAMPLES_PER_CLASS - selectedCount);

  return (
    <div className="panel" style={{ flex: 1 }}>
      <header>
        <span className="stage">01</span>
        <h2>Record</h2>
        <span className="note">
          {selectedName ? `capturing to “${selectedName}”` : 'select a gesture on the right'}
        </span>
      </header>

      <div className="plate" data-recording={recording}>
        <video ref={videoRef} playsInline muted />
        <canvas ref={canvasRef} />

        {status === 'ready' && !handPresent && !recording && (
          <div className="placeholder">No hand in frame</div>
        )}
        {status === 'loading' && <div className="placeholder">Starting landmarker…</div>}

        <div className="corner tl">
          {handPresent ? `${handedness ?? '—'} hand · 21 joints` : '—'}
        </div>
        <div className="corner tr">
          {recording ? `● REC ${burstCount}` : `${fps.toFixed(0)} fps`}
        </div>
        <div className="corner br">world landmarks · metric</div>
      </div>

      <div className="controls">
        <button
          className="btn record"
          data-active={recording}
          disabled={!canRecord}
          onPointerDown={onRecordStart}
          onPointerUp={onRecordStop}
          onPointerLeave={onRecordStop}
        >
          {recording ? `Recording — ${burstCount} frames` : 'Hold to record'}
        </button>
        <div className="hint">
          hold <kbd>space</kbd>
          {selectedName && short > 0 && (
            <>
              {' '}
              · {short} more needed
            </>
          )}
        </div>
      </div>
    </div>
  );
}
