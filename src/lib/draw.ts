import { HAND_CONNECTIONS } from './landmarker';
import type { Vec3 } from './features';

interface SkeletonStyle {
  color: string;
  glow: number;
  lineWidth: number;
  jointRadius: number;
  alpha?: number;
}

export const LIVE_STYLE: SkeletonStyle = {
  color: '#e6f2f7',
  glow: 12,
  lineWidth: 2,
  jointRadius: 3,
};

export const RECORDING_STYLE: SkeletonStyle = {
  color: '#f2a93b',
  glow: 16,
  lineWidth: 2.5,
  jointRadius: 3.5,
};

export const SPECIMEN_STYLE: SkeletonStyle = {
  color: '#58d6e8',
  glow: 6,
  lineWidth: 1.4,
  jointRadius: 1.6,
  alpha: 0.9,
};

/** Draws the hand as luminous linework. Points are pixel coordinates. */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  points: [number, number][],
  style: SkeletonStyle,
): void {
  ctx.save();
  ctx.globalAlpha = style.alpha ?? 1;
  ctx.strokeStyle = style.color;
  ctx.fillStyle = style.color;
  ctx.lineWidth = style.lineWidth;
  ctx.lineCap = 'round';
  ctx.shadowColor = style.color;
  ctx.shadowBlur = style.glow;

  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(points[a][0], points[a][1]);
    ctx.lineTo(points[b][0], points[b][1]);
    ctx.stroke();
  }

  points.forEach(([x, y], i) => {
    ctx.beginPath();
    // The wrist anchors the whole normalization, so it reads heavier.
    ctx.arc(x, y, i === 0 ? style.jointRadius * 1.9 : style.jointRadius, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

/** Scales landmarks to fill a box, preserving aspect. Used for thumbnails. */
export function fitToBox(
  landmarks: Vec3[],
  width: number,
  height: number,
  padding = 8,
): [number, number][] {
  const xs = landmarks.map((p) => p[0]);
  const ys = landmarks.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);

  const offsetX = (width - spanX * scale) / 2;
  const offsetY = (height - spanY * scale) / 2;

  return landmarks.map((p) => [
    offsetX + (p[0] - minX) * scale,
    offsetY + (p[1] - minY) * scale,
  ]);
}
