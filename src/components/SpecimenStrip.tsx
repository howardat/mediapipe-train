import { useEffect, useRef } from 'react';
import { featurize } from '../lib/features';
import { drawSkeleton, fitToBox, SPECIMEN_STYLE } from '../lib/draw';
import type { GestureClass, Sample } from '../lib/types';

/**
 * The most representative sample in a class: the one whose feature vector sits
 * closest to the class mean. Shows you what the model actually learned, and
 * makes a wildly off pose obvious at a glance.
 */
function representative(samples: Sample[]): Sample | null {
  if (!samples.length) return null;
  const feats = samples.map((s) => featurize(s.worldLandmarks, s.handedness));
  const dim = feats[0].length;
  const mean = new Float32Array(dim);
  for (const f of feats) for (let i = 0; i < dim; i++) mean[i] += f[i] / feats.length;

  let best = 0;
  let bestDist = Infinity;
  feats.forEach((f, idx) => {
    let d = 0;
    for (let i = 0; i < dim; i++) d += (f[i] - mean[i]) ** 2;
    if (d < bestDist) {
      bestDist = d;
      best = idx;
    }
  });
  return samples[best];
}

function Specimen({ cls, samples }: { cls: GestureClass; samples: Sample[] }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const sample = representative(samples);
    if (!sample) return;
    drawSkeleton(ctx, fitToBox(sample.screenLandmarks, w, h), SPECIMEN_STYLE);
  }, [samples]);

  return (
    <div className="specimen">
      <canvas ref={ref} aria-label={`Representative pose for ${cls.name}`} />
      <div className="name">{cls.name}</div>
      <div className="n">{samples.length}</div>
    </div>
  );
}

export default function SpecimenStrip({
  classes,
  samples,
}: {
  classes: GestureClass[];
  samples: Sample[];
}) {
  const populated = classes.filter((c) => samples.some((s) => s.classId === c.id));
  if (!populated.length) return null;

  return (
    <div className="strip">
      <div className="caption">Specimens</div>
      {populated.map((cls) => (
        <Specimen
          key={cls.id}
          cls={cls}
          samples={samples.filter((s) => s.classId === cls.id)}
        />
      ))}
    </div>
  );
}
