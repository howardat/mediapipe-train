import { useCallback, useEffect, useRef, useState } from 'react';
import type { HandLandmarker } from '@mediapipe/tasks-vision';

import CameraPane from './components/CameraPane';
import ClassRail from './components/ClassRail';
import TrainPanel from './components/TrainPanel';
import LivePanel from './components/LivePanel';
import ExportPanel from './components/ExportPanel';
import SpecimenStrip from './components/SpecimenStrip';

import { createLandmarker, detect } from './lib/landmarker';
import { drawSkeleton, LIVE_STYLE, RECORDING_STYLE } from './lib/draw';
import { featurize } from './lib/features';
import * as store from './lib/dataset';
import {
  DEFAULT_CONFIG,
  predict,
  trainModel,
  type TrainConfig,
  type TrainResult,
} from './lib/trainer';
import {
  buildBundle,
  generateCsv,
  generateInferPy,
  generateReadme,
  type ModelBundle,
} from './lib/exporter';
import { generateGesturePackage } from './lib/rospkg';
import { zipStore } from './lib/zip';
import type { Detection, GestureClass, Sample, TrainedModel, TrainingEpoch } from './lib/types';

const SMOOTHING_FRAMES = 5;
const PREDICT_INTERVAL_MS = 66;

function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function download(filename: string, content: string, type: string) {
  downloadBlob(filename, new Blob([content], { type }));
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [fatal, setFatal] = useState<string | null>(null);

  const [classes, setClasses] = useState<GestureClass[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [recording, setRecording] = useState(false);
  const [burstCount, setBurstCount] = useState(0);
  const [handPresent, setHandPresent] = useState(false);
  const [handedness, setHandedness] = useState<string | null>(null);
  const [fps, setFps] = useState(0);

  const [config, setConfig] = useState<TrainConfig>(DEFAULT_CONFIG);
  const [training, setTraining] = useState(false);
  /** Epoch count the running job started with — the slider can move underneath it. */
  const [epochTarget, setEpochTarget] = useState(DEFAULT_CONFIG.epochs);
  const [history, setHistory] = useState<TrainingEpoch[]>([]);
  const [trained, setTrained] = useState<TrainedModel | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<ModelBundle | null>(null);

  const [probs, setProbs] = useState<number[] | null>(null);
  const [threshold, setThreshold] = useState(0.75);

  // Refs the render loop reads without forcing re-renders.
  const recordingRef = useRef(false);
  const selectedRef = useRef<string | null>(null);
  const bufferRef = useRef<Omit<Sample, 'id' | 'createdAt'>[]>([]);
  const trainRef = useRef<TrainResult | null>(null);
  const smoothRef = useRef<number[][]>([]);

  selectedRef.current = selectedId;

  const counts = samples.reduce<Record<string, number>>((acc, s) => {
    acc[s.classId] = (acc[s.classId] ?? 0) + 1;
    return acc;
  }, {});

  const refresh = useCallback(async () => {
    const [cls, smp] = await Promise.all([store.listClasses(), store.listSamples()]);
    setClasses(cls);
    setSamples(smp);
    setSelectedId((current) => current ?? cls[0]?.id ?? null);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // ---------------------------------------------------------------- camera
  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    let stream: MediaStream | null = null;
    let landmarker: HandLandmarker | null = null;
    let lastVideoTime = -1;
    let lastPredict = 0;
    let lastFpsPush = 0;
    let frameTimes: number[] = [];

    const run = async () => {
      try {
        landmarker = await createLandmarker();
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 960, height: 720, facingMode: 'user' },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();
        if (video.readyState < 1) {
          await new Promise<void>((r) =>
            video.addEventListener('loadedmetadata', () => r(), { once: true }),
          );
        }

        const canvas = canvasRef.current!;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        setStatus('ready');

        const ctx = canvas.getContext('2d')!;

        const loop = () => {
          raf = requestAnimationFrame(loop);
          if (!landmarker || video.readyState < 2) return;

          const now = performance.now();
          frameTimes.push(now);
          frameTimes = frameTimes.filter((t) => now - t < 1000);
          // Once a second — pushing this every frame would re-render the whole
          // tree at 60fps for a number nobody reads that fast.
          if (now - lastFpsPush > 1000) {
            lastFpsPush = now;
            setFps(frameTimes.length);
          }

          if (video.currentTime === lastVideoTime) return;
          lastVideoTime = video.currentTime;

          let det: Detection | null = null;
          try {
            det = detect(landmarker, video, now);
          } catch {
            return; // transient WASM hiccup; next frame usually recovers
          }

          ctx.clearRect(0, 0, canvas.width, canvas.height);
          setHandPresent(!!det);

          if (!det) {
            smoothRef.current = [];
            setProbs(null);
            setHandedness(null);
            return;
          }

          setHandedness(det.handedness);
          drawSkeleton(
            ctx,
            det.screenLandmarks.map((p) => [p[0] * canvas.width, p[1] * canvas.height]),
            recordingRef.current ? RECORDING_STYLE : LIVE_STYLE,
          );

          if (recordingRef.current && selectedRef.current) {
            bufferRef.current.push({
              classId: selectedRef.current,
              worldLandmarks: det.worldLandmarks,
              screenLandmarks: det.screenLandmarks,
              handedness: det.handedness,
            });
            setBurstCount(bufferRef.current.length);
          }

          const model = trainRef.current;
          if (model && now - lastPredict > PREDICT_INTERVAL_MS) {
            lastPredict = now;
            const raw = predict(
              model.model,
              featurize(det.worldLandmarks, det.handedness),
              model.standardization,
            );
            smoothRef.current.push(raw);
            if (smoothRef.current.length > SMOOTHING_FRAMES) smoothRef.current.shift();

            const avg = raw.map(
              (_, i) =>
                smoothRef.current.reduce((sum, r) => sum + r[i], 0) / smoothRef.current.length,
            );
            setProbs(avg);
          }
        };
        loop();
      } catch (err) {
        if (cancelled) return;
        setStatus('error');
        setFatal(
          err instanceof Error
            ? `${err.message}. Camera access needs a secure context — use localhost, not a LAN IP.`
            : String(err),
        );
      }
    };

    run();
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // ---------------------------------------------------------------- record
  const startRecording = useCallback(() => {
    if (!selectedRef.current || recordingRef.current) return;
    bufferRef.current = [];
    recordingRef.current = true;
    setBurstCount(0);
    setRecording(true);
  }, []);

  const stopRecording = useCallback(async () => {
    if (!recordingRef.current) return;
    recordingRef.current = false;
    setRecording(false);
    const captured = bufferRef.current;
    bufferRef.current = [];
    if (!captured.length) return;
    await store.addSamples(captured);
    await refresh();
  }, [refresh]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLInputElement;
      if (typing) return;

      if (e.code === 'Space' && !e.repeat) {
        e.preventDefault();
        startRecording();
      }
      const n = Number(e.key);
      if (n >= 1 && n <= 9 && classes[n - 1]) setSelectedId(classes[n - 1].id);
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault();
        stopRecording();
      }
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, [classes, startRecording, stopRecording]);

  // ---------------------------------------------------------------- train
  const handleTrain = useCallback(async () => {
    setTraining(true);
    setTrainError(null);
    setHistory([]);
    setEpochTarget(config.epochs);
    try {
      const result = await trainModel(samples, classes, config, (e) =>
        setHistory((h) => [...h, e]),
      );
      trainRef.current?.model.dispose();
      trainRef.current = result;
      setTrained(result.meta);
      setBundle(buildBundle(result, result.meta));
      smoothRef.current = [];
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : String(err));
    } finally {
      setTraining(false);
    }
  }, [samples, classes, config]);

  // ---------------------------------------------------------------- classes
  const handleCreate = async (name: string) => {
    const cls = await store.addClass(name);
    await refresh();
    setSelectedId(cls.id);
  };

  const handleDelete = async (id: string) => {
    await store.deleteClass(id);
    if (selectedId === id) setSelectedId(null);
    await refresh();
  };

  const selected = classes.find((c) => c.id === selectedId) ?? null;

  return (
    <div className="app">
      <header className="masthead">
        <h1>Gesture Atlas</h1>
        <span className="sub">MediaPipe landmarks → your classifier → the edge</span>
        <span className="spacer" />
        <span className="status" data-state={status}>
          <span className="lamp" />
          {status === 'ready' ? 'landmarker live' : status === 'error' ? 'camera failed' : 'loading'}
        </span>
      </header>

      {fatal && <p className="alert">{fatal}</p>}

      <div className="workspace">
        <div className="column">
          <CameraPane
            videoRef={videoRef}
            canvasRef={canvasRef}
            status={status}
            handPresent={handPresent}
            handedness={handedness}
            fps={fps}
            recording={recording}
            burstCount={burstCount}
            selectedName={selected?.name ?? null}
            selectedCount={selected ? (counts[selected.id] ?? 0) : 0}
            onRecordStart={startRecording}
            onRecordStop={stopRecording}
          />
        </div>

        <div className="column rail">
          <ClassRail
            classes={classes}
            counts={counts}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onCreate={handleCreate}
            onDelete={handleDelete}
          />
          <TrainPanel
            classes={classes}
            counts={counts}
            config={config}
            onConfigChange={setConfig}
            epochTarget={epochTarget}
            training={training}
            history={history}
            trained={trained}
            error={trainError}
            onTrain={handleTrain}
          />
          {trained && (
            <LivePanel
              classNames={trained.classNames}
              probs={probs}
              threshold={threshold}
              onThresholdChange={setThreshold}
              handPresent={handPresent}
            />
          )}
          <ExportPanel
            bundle={bundle}
            sampleCount={samples.length}
            onDownloadModel={() =>
              bundle && download('gesture_model.json', JSON.stringify(bundle, null, 2), 'application/json')
            }
            onDownloadInfer={() => download('infer.py', generateInferPy(), 'text/x-python')}
            onDownloadPackage={() =>
              bundle && downloadBlob('x2_gesture.zip', zipStore(generateGesturePackage(bundle)))
            }
            onDownloadReadme={() =>
              bundle && download('README.md', generateReadme(bundle), 'text/markdown')
            }
            onDownloadCsv={() => download('dataset.csv', generateCsv(samples, classes), 'text/csv')}
          />
        </div>
      </div>

      <SpecimenStrip classes={classes} samples={samples} />
    </div>
  );
}
