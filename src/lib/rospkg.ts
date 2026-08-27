/**
 * Emits the x2_gesture ROS 2 package.
 *
 * Shaped after x2_yolo_stream rather than after generic ROS tutorials, because
 * it ships into the same workspace and sits next to it: std_msgs/String
 * carrying JSON (x2_yolo_ws has no custom messages and every consumer already
 * parses JSON), a compressed camera topic over DDS, a worker thread behind a
 * latest-frame-wins slot, and an ament_python package with a launch file and a
 * systemd --user unit.
 *
 * It installs into x2_yolo_plus_ws — x2_yolo_ws plus this package — because the
 * deployment is "copy the folder to the robot and run one script", and the robot
 * already has an x2_yolo_ws whose build/ and install/ would collide with the
 * copy. Same workspace as its three siblings, different directory name.
 */
import { EDGE_WORKSPACE, generateFeatureCore } from './exporter';
import type { ModelBundle } from './exporter';
import type { ZipEntry } from './zip';

export const PACKAGE_NAME = 'x2_gesture';

const WORKSPACE = EDGE_WORKSPACE;

/**
 * x2_gesture/x2_gesture/gesture.py — the shared core, verbatim.
 *
 * scripts/package.py asserts this is byte-identical to the corresponding
 * region of infer.py. Two copies of a featurizer that drift apart is exactly
 * what the golden sample catches at runtime; that check catches it at build
 * time instead.
 */
export function generateGestureCore(): string {
  return `"""Feature extraction and classification for a Gesture Atlas model.

GENERATED — do not edit. Emitted from src/lib/features.ts, and byte-identical
to the corresponding section of infer.py.

Imports nothing but json and numpy on purpose, so it can be exercised on a
machine with no OpenCV, MediaPipe or ROS.
"""

import json

import numpy as np

${generateFeatureCore()}`;
}

/**
 * Hand skeleton edges, for the overlay only.
 *
 * Cosmetic — nothing downstream of the drawing uses this, so unlike the
 * feature constants it is not part of the browser/device contract. It mirrors
 * HAND_CONNECTIONS in src/lib/landmarker.ts; drift would make the overlay look
 * wrong, which is immediately visible.
 */
const HAND_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/** x2_gesture/x2_gesture/web.py — MJPEG + JSON front end. */
export function generateWebPy(): string {
  return `"""MJPEG + JSON HTTP front end for the gesture node.

Standard-library ThreadingHTTPServer on purpose, the same choice
x2_yolo_stream makes: it avoids putting a second event loop (gevent/flask)
next to the rclpy executor.

The page shows the annotated camera, the committed gesture, and a bar per
class — the bars are what tell you whether a gesture is failing to trigger
because the model is unsure or because the threshold is too high.
"""

import json
import select
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<title>X2 gesture</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0e1116; color: #e6edf3;
         font: 14px/1.5 system-ui, sans-serif; }
  header { padding: 10px 16px; border-bottom: 1px solid #30363d;
           display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
  h1 { font-size: 15px; margin: 0; font-weight: 600; }
  .meta { color: #8b949e; font-size: 12px; }
  main { display: flex; gap: 16px; padding: 16px; flex-wrap: wrap; }
  img { max-width: 100%; border-radius: 8px; border: 1px solid #30363d;
        background: #000; }
  .panel { min-width: 280px; }
  .gesture { font-size: 34px; font-weight: 600; letter-spacing: -0.5px;
             margin: 0 0 2px; }
  .gesture.none { color: #6e7681; font-weight: 400; }
  .sub { color: #8b949e; font-size: 12px; margin-bottom: 18px; }
  .row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .name { width: 110px; font-size: 13px; }
  .bar { display: block; flex: 1; height: 8px; background: #21262d;
         border-radius: 4px; overflow: hidden; }
  /* display:block matters: these are spans, and an inline element ignores
     width, which silently leaves every bar empty. */
  .fill { display: block; height: 100%; background: #3fb950; width: 0;
          transition: width .15s; }
  .fill.under { background: #484f58; }
  .val { width: 46px; text-align: right; font-size: 12px;
         font-variant-numeric: tabular-nums; color: #8b949e; }
  .mark { color: #6e7681; font-size: 11px; margin-top: 10px; }
</style>
<header>
  <h1>X2 gesture</h1>
  <span class="meta" id="meta">connecting...</span>
</header>
<main>
  <img src="/stream" alt="annotated camera stream">
  <div class="panel">
    <p class="gesture none" id="gesture">waiting</p>
    <div class="sub" id="sub">&nbsp;</div>
    <div id="scores"></div>
    <div class="mark" id="mark"></div>
  </div>
</main>
<script>
async function poll() {
  try {
    const r = await fetch('/state', {cache: 'no-store'});
    const d = await r.json();
    document.getElementById('meta').textContent =
      \`\${d.topic} | \${d.width}x\${d.height} | \` +
      // The delegate belongs next to the milliseconds, not in a corner: the
      // same 45 ms is healthy on the CPU and a sign of a broken GPU path.
      \`infer \${d.inference_ms} ms on \${(d.delegate || '?').toUpperCase()} | \` +
      \`\${d.stream_fps} fps in, \${d.infer_fps} fps analysed | spec v\${d.spec_version}\`;

    const g = document.getElementById('gesture');
    if (d.gesture) {
      g.textContent = d.gesture;
      g.className = 'gesture';
      document.getElementById('sub').textContent =
        \`\${d.handedness || ''} hand · \${d.confidence.toFixed(2)} · held \${d.held_seconds}s\`;
    } else {
      g.textContent = d.hands ? 'unrecognised' : 'no hand';
      g.className = 'gesture none';
      document.getElementById('sub').textContent =
        d.hands ? 'hand visible, nothing over threshold' : '\\u00a0';
    }

    const scores = d.scores || {};
    document.getElementById('scores').innerHTML =
      (d.classes || []).map(c => {
        const v = scores[c] || 0;
        const under = v < d.threshold ? ' under' : '';
        return \`<div class="row"><span class="name">\${c}</span>\` +
               \`<span class="bar"><span class="fill\${under}" style="width:\${(v * 100).toFixed(1)}%"></span></span>\` +
               \`<span class="val">\${v.toFixed(2)}</span></div>\`;
      }).join('');

    document.getElementById('mark').textContent =
      \`threshold \${d.threshold} · smoothing \${d.smoothing} frames · \` +
      \`\${d.changes} changes · \${d.viewers} viewer(s)\`;
  } catch (e) {
    document.getElementById('meta').textContent = 'disconnected';
  }
}
poll(); setInterval(poll, 400);
</script>
"""


def make_handler(state):
    """Build a request handler bound to a SharedState instance."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "x2_gesture"

        def log_message(self, fmt, *args):
            pass                      # the ROS log is the log

        def _send(self, code, body, ctype="text/plain; charset=utf-8"):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _client_gone(self):
            """True once the browser has closed the connection."""
            try:
                rlist, _, _ = select.select([self.connection], [], [], 0)
                if not rlist:
                    return False
                return self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif path == "/state":
                self._send(200, json.dumps(state.snapshot()),
                           "application/json")
            elif path == "/healthz":
                ok = state.frames_in > 0
                self._send(200 if ok else 503,
                           json.dumps({"ok": ok, "frames": state.frames_in}),
                           "application/json")
            elif path == "/snapshot":
                jpeg = state.latest_jpeg()
                if jpeg is None:
                    self._send(503, "no frame yet")
                else:
                    self._send(200, jpeg, "image/jpeg")
            elif path == "/stream":
                self.route_stream()
            else:
                self._send(404, "not found")

        def route_stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control",
                             "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            state.add_viewer(+1)
            last_seq = -1
            try:
                while True:
                    if self._client_gone():
                        break
                    jpeg, last_seq = state.wait_for_frame(last_seq, timeout=5.0)
                    if jpeg is None:
                        continue          # nothing published yet; keep waiting
                    self.wfile.write(b"--frame\\r\\n")
                    self.wfile.write(b"Content-Type: image/jpeg\\r\\n")
                    self.wfile.write(b"Content-Length: " +
                                     str(len(jpeg)).encode() + b"\\r\\n\\r\\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\\r\\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                state.add_viewer(-1)

    return Handler


def serve(state, host, port):
    """Start the HTTP server. Returns the server; caller runs/stops it."""
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    httpd.daemon_threads = True
    return httpd


def local_ips():
    """Best-effort list of addresses the browser could use."""
    ips = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ips.append(sock.getsockname()[0])
        sock.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass
    return ips or ["<robot-ip>"]
`;
}

/** x2_gesture/x2_gesture/node.py — the ROS node. */
/**
 * x2_gesture/x2_gesture/runtime.py — which delegate the landmarker runs on.
 *
 * The sibling of x2_yolo_stream/runtime.py, which does the same job for
 * onnxruntime. Same contract: try the fast thing, fall back rather than fail,
 * and report what actually happened instead of what was asked for.
 */function generateRuntimePy(): string {
  return `"""One place that decides whether the hand landmarker runs on the GPU.

Runnable on its own, which is also how it tests a delegate:

    python3 runtime.py --probe /path/to/hand_landmarker.task gpu

Shaped after x2_yolo_stream/runtime.py, which does this for onnxruntime, and
written for the same reason: before it, every model in this workspace opened
with a hard-coded CPU backend, so on a Jetson Orin NX the GPU sat idle while
six ARM cores did convolutions.

MediaPipe has no execution providers. It has one delegate field on BaseOptions
and the TFLite GPU delegate behind it -- OpenGL ES compute shaders, not CUDA
and not TensorRT, so none of x2_yolo_stream's TensorRT machinery applies. Two
things about it do not survive being copied from that file, and both were
found the expensive way:

* **MediaPipe does not raise, it aborts.** A delegate that cannot run fails
  inside C++ as a CHECK failure, which calls abort(). There is no Python
  exception, so there is nothing for try/except to catch: the node does not
  fall back, it dies. That is why every delegate above the last one is tried
  in a SEPARATE PROCESS first (see probe) and only built in this one after a
  child has survived it. A robot whose GPU stack is broken has to come up
  slowly, not blind.
* **The GPU path only accepts 4-channel images.** Handing SRGB to a GPU
  landmarker aborts with "unsupported ImageFrame format", so the conversion
  differs per delegate: SRGBA for the GPU, SRGB for the CPU. Landmarker owns
  that conversion. A caller that builds its own mp.Image has to remember which
  delegate it got, and will eventually get it wrong.

Expect roughly 2-3x from the GPU, not the 10x it gives onnxruntime. The
landmarker is a small model and every frame is uploaded from CPU memory. Once
inference is cheap the JPEG decode and encode around it are the budget.
"""

import os
import subprocess
import sys
import time

# Preference order for "auto". Only two rungs, unlike onnxruntime's three.
DELEGATE_ORDER = ("gpu", "cpu")

# A human types "gpu" on the command line; something scripted passes back
# whatever this module reported, which is also "gpu"/"cpu". Both have to land
# in the same place -- the equivalent lookup in x2_yolo_stream was
# case-sensitive once and quietly benchmarked the CPU for an evening.
ALIASES = {
    "gpu": "gpu",
    "cuda": "gpu",
    "opengl": "gpu",
    "gl": "gpu",
    "cpu": "cpu",
    "xnnpack": "cpu",
}

LABELS = {
    "gpu": "GPU (TFLite delegate, OpenGL ES)",
    "cpu": "CPU (XNNPACK)",
}

# How long a probe child gets before it is called a failure. Generous: it pays
# for a cold mediapipe import and a 7 MB model load on a busy Orin, and the
# only thing a too-short timeout buys is a robot that runs on the CPU forever.
PROBE_TIMEOUT = 120.0


def resolve_chain(requested):
    """'auto' | 'gpu' | 'cpu' | 'gpu,cpu' -> list of delegates to try.

    Unknown names are dropped rather than raised on, and "cpu" is always
    appended, because a node that refuses to start is worse than a node that
    runs slowly: this one is a robot's eyes.
    """
    requested = (requested or "auto").strip().lower()

    if requested in ("auto", "", "best"):
        wanted = list(DELEGATE_ORDER)
    else:
        wanted = []
        for part in requested.replace(" ", "").split(","):
            name = ALIASES.get(part)
            if name is not None and name not in wanted:
                wanted.append(name)

    if "cpu" not in wanted:
        wanted.append("cpu")
    return wanted


def _one_line(text):
    """MediaPipe errors arrive as paragraphs of C++ log. Keep it to a line."""
    return " ".join(str(text).split())[:200]


def _interesting(stderr):
    """The line of a probe's output worth putting in the log.

    MediaPipe is loud on stderr even when everything works, so the last line
    is usually a stack frame and the first is usually a banner. A CHECK
    failure is the line that starts with F<digits>, and it is the one that
    says what actually went wrong.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in lines:
        if line[:1] == "F" and "Check failed" in line:
            return _one_line(line)
    for line in reversed(lines):
        if not line.startswith(("I0", "W0", "INFO:", "@", "***")):
            return _one_line(line)
    return _one_line(lines[-1]) if lines else "no output"


def _base_options(model_path, delegate):
    from mediapipe.tasks import python as mp_python

    kinds = mp_python.BaseOptions.Delegate
    return mp_python.BaseOptions(
        model_asset_path=model_path,
        delegate=kinds.GPU if delegate == "gpu" else kinds.CPU,
    )


def open_raw(model_path, delegate, num_hands=1, min_detection_confidence=0.5,
             min_tracking_confidence=0.5):
    """A bare HandLandmarker on one delegate. No probe, no fallback.

    Used by make_landmarker once a delegate is known good, and by the probe
    child, which is what makes it known good. Nothing else should call it:
    on its own it is the abort-instead-of-falling-back behaviour this module
    exists to prevent.
    """
    from mediapipe.tasks.python import vision

    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=_base_options(model_path, delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    )


def to_image(frame_bgr, delegate):
    """An OpenCV BGR frame -> the mp.Image this delegate can actually take.

    The GPU path rejects 3-channel data outright -- not with an exception,
    with an abort -- so the channel count is not a detail to leave to the
    caller.
    """
    import cv2
    import mediapipe as mp

    if delegate == "gpu":
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        return mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def blank_image(delegate, size=256):
    """A frame of nothing, in this delegate's format. numpy only, no cv2."""
    import mediapipe as mp
    import numpy as np

    if delegate == "gpu":
        data = np.zeros((size, size, 4), dtype=np.uint8)
        data[:, :, 3] = 255
        return mp.Image(image_format=mp.ImageFormat.SRGBA, data=data)
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.zeros((size, size, 3), dtype=np.uint8))


def warmup(landmarker, delegate, runs=3, start_timestamp_ms=0):
    """Push blank frames through the graph. Returns (ms, last timestamp).

    The milliseconds are a floor, not the per-frame cost. A blank frame has no
    hand in it, so the palm detector runs and the landmark model does not. The
    honest number is the node's own inference_ms, measured on real frames and
    published every cycle.
    """
    image = blank_image(delegate)

    timestamp = int(start_timestamp_ms)
    elapsed = 0.0
    for _ in range(max(1, runs)):
        timestamp += 1
        started = time.monotonic()
        landmarker.detect_for_video(image, timestamp)
        elapsed = (time.monotonic() - started) * 1000.0
    return elapsed, timestamp


class Landmarker:
    """A HandLandmarker that knows what it runs on and owns its clock.

    The clock is not a convenience. VIDEO mode rejects a timestamp that is not
    strictly greater than the last one it saw, and "the last one it saw"
    includes the frames warmup() pushed through. Keeping the counter here
    means one place can get it wrong instead of one per caller, and it also
    survives a monotonic clock that returns the same millisecond twice at a
    high infer_fps.
    """

    def __init__(self, landmarker, delegate, warmup_ms, timestamp_ms=0):
        self._landmarker = landmarker
        self.delegate = delegate
        self.label = LABELS.get(delegate, delegate)
        self.warmup_ms = warmup_ms
        self._last_ts = int(timestamp_ms)
        self._base = int(timestamp_ms) + 1
        self._started = time.monotonic()

    def detect(self, frame_bgr):
        """One OpenCV BGR frame -> a HandLandmarkerResult."""
        image = to_image(frame_bgr, self.delegate)
        stamp = self._base + int((time.monotonic() - self._started) * 1000.0)
        if stamp <= self._last_ts:
            stamp = self._last_ts + 1
        self._last_ts = stamp
        return self._landmarker.detect_for_video(image, stamp)

    def close(self):
        """Release the graph, and with it the GL context on the GPU path."""
        try:
            self._landmarker.close()
        except Exception:                                   # noqa: BLE001
            pass


def probe(model_path, delegate, timeout=PROBE_TIMEOUT):
    """Does this delegate work? Answered in a child process. -> (ok, detail).

    The child is this file, run as a script. It builds the landmarker and
    pushes frames through it, which is the whole of what can abort, and the
    abort lands on the child instead of on the node.

    The environment is inherited deliberately: PYTHONPATH and LD_LIBRARY_PATH
    from scripts/gpu_env.sh are most of what decides the answer, so the child
    has to be looking at the same libraries the parent will.
    """
    command = [sys.executable, os.path.abspath(__file__), "--probe",
               model_path, delegate]
    try:
        done = subprocess.run(command, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "probe timed out after %.0fs" % timeout
    except OSError as exc:
        # No interpreter, no file -- do not let a broken probe be read as a
        # broken delegate. Say so and let the caller decide.
        return False, "could not run the probe: %s" % _one_line(exc)

    if done.returncode == 0:
        return True, ""

    stderr = done.stderr.decode("utf-8", "replace")
    if done.returncode < 0:
        # Negative means a signal. This is the case the whole probe exists
        # for: SIGABRT from a CHECK failure inside MediaPipe.
        return False, ("crashed with signal %d: %s"
                       % (-done.returncode, _interesting(stderr)))
    return False, _interesting(stderr)


def make_landmarker(model_path, delegate="auto", num_hands=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5,
                    warmup_runs=3, probe_first=True, logger=None):
    """Open a hand landmarker on the fastest delegate that actually works.

    Returns a Landmarker whose .delegate is what was built, not what was asked
    for -- "GPU requested, running on CPU" is exactly the failure this module
    exists to make visible.

    probe_first=False skips the subprocess check and saves the few seconds it
    costs. Only do that where a crash is cheap: on the robot the node is the
    eyes, and a delegate that aborts takes the whole node with it.
    """
    def say(message):
        if logger is not None:
            logger(message)

    name = os.path.basename(model_path)
    chain = resolve_chain(delegate)
    last_error = "no delegate was tried"

    for i, kind in enumerate(chain):
        # The last rung is not probed. There is nothing to fall back TO, so a
        # probe could only turn a slow start into a slow start plus a wrong
        # error message.
        if probe_first and i < len(chain) - 1:
            started = time.monotonic()
            ok, detail = probe(model_path, kind)
            if not ok:
                last_error = detail
                say("%s: %s delegate unusable (%s); trying next"
                    % (name, kind, detail))
                continue
            say("%s: %s delegate verified in a child process (%.1fs)"
                % (name, kind, time.monotonic() - started))

        try:
            landmarker = open_raw(model_path, kind, num_hands,
                                  min_detection_confidence,
                                  min_tracking_confidence)
            elapsed_ms, timestamp = warmup(landmarker, kind, warmup_runs)
        except Exception as exc:                            # noqa: BLE001
            # Reachable for the ordinary failures -- a missing file, a corrupt
            # model. The violent ones were caught by the probe above.
            last_error = _one_line(exc)
            say("%s: %s delegate failed to open (%s); trying next"
                % (name, kind, last_error))
            continue

        say("%s: running on the %s" % (name, LABELS.get(kind, kind)))
        return Landmarker(landmarker, kind, elapsed_ms, timestamp)

    raise RuntimeError("no delegate could load %s: %s"
                       % (model_path, last_error))


def _loadable(soname):
    """Can the dynamic loader find this library right now?

    ctypes.util.find_library is the obvious call and the wrong one: it shells
    out to ldconfig, which does not know about LD_LIBRARY_PATH. On this robot
    the Tegra GL libraries are reached through exactly that variable, set by
    scripts/gpu_env.sh, so the obvious call reports them missing on a machine
    where they work.
    """
    import ctypes

    try:
        ctypes.CDLL(soname)
        return True
    except OSError:
        return False


def describe_environment(check_gl=True):
    """One line for the log: what mediapipe is this, and what can it see.

    check_gl=False reports the version and stops there. Worth passing when the
    CPU was chosen on purpose: the GL check has to dlopen libEGL to be honest
    about it, and neither loading a graphics library nor a paragraph about a
    delegate nobody asked for belongs in the boot log of a node that is doing
    exactly what it was told.

    A hint, deliberately, and never a verdict. probe() is the authority on
    whether a delegate works, and the two disagree on a Mac, where there is no
    OpenGL ES anywhere and the GPU delegate runs perfectly well on Metal. A
    line that announced "will fall back to the CPU" there would be a confident
    lie in the log, sitting directly above the line saying it got the GPU.
    """
    try:
        import mediapipe as mp
    except Exception as exc:                                # noqa: BLE001
        return "mediapipe not importable: %s" % _one_line(exc)

    version = getattr(mp, "__version__", "unknown")
    if not check_gl:
        return "mediapipe %s" % version

    missing = [name for name, soname in
               (("libEGL", "libEGL.so.1"), ("libGLESv2", "libGLESv2.so.2"))
               if not _loadable(soname)]
    if missing:
        return ("mediapipe %s, %s not loadable -- on a Jetson that is the "
                "usual reason the GPU delegate is refused, and what "
                "scripts/gpu_env.sh is for. Harmless on a Mac, which reaches "
                "its GPU through Metal instead."
                % (version, " and ".join(missing)))
    return "mediapipe %s, OpenGL ES present" % version


def _probe_main(argv):
    """The child half of probe(). Exits 0 only if the delegate really ran.

    Deliberately does everything the node does to a frame, in the same order.
    A probe that skips a step is a probe that passes a delegate the node then
    aborts on.
    """
    if len(argv) < 2:
        sys.stderr.write("usage: runtime.py --probe MODEL gpu|cpu\\n")
        return 2

    model_path, delegate = argv[0], ALIASES.get(argv[1].lower(), argv[1])
    landmarker = open_raw(model_path, delegate)
    try:
        elapsed_ms, _ = warmup(landmarker, delegate, runs=2)
    finally:
        try:
            landmarker.close()
        except Exception:                                   # noqa: BLE001
            pass
    sys.stdout.write("%s ok, blank frame %.1f ms\\n" % (delegate, elapsed_ms))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        raise SystemExit(_probe_main(sys.argv[2:]))
    print(describe_environment())
    print("delegates in preference order: %s"
          % ", ".join(resolve_chain("auto")))
    print("usage: runtime.py --probe MODEL gpu|cpu")
`;
}

export function generateNodePy(): string {
  return `"""ROS 2 node: camera -> hand landmarks -> gesture label -> topics.

Consumes a compressed camera topic and publishes what the hand is doing. The
classifier is a two-layer MLP exported from Gesture Atlas; the expensive part
is MediaPipe's hand landmarker.

Design notes that matter on this robot:

* Only /compressed camera topics are subscribed. A RAW stream is 75-90 MB/s
  and must not be crossed between compute units; compressed is a few hundred
  kB/s. Set camera_device >= 0 to read a local V4L2 camera instead, which is
  for a dev machine, not for the robot.
* Landmarking runs in its own thread on a latest-frame-wins slot. The DDS
  callback only stores bytes, so a slow frame never stalls the executor and
  never builds a backlog -- old frames are dropped, not queued.
* The golden-sample check runs before any publisher is created. If the
  featurizer here disagrees with the browser that trained the model, the node
  exits instead of publishing confident nonsense onto the graph.
* Landmarking is on the CPU (XNNPACK) by default, and that is a choice rather
  than a limitation. The Orin has one GPU, x2_yolo_stream gets about 10x out
  of it, and MediaPipe's GPU delegate is worth 2-3x here -- so the two nodes
  divide the machine instead of fighting over it. delegate:=gpu moves this one
  across when that trade changes; runtime.py owns the mechanics and proves a
  delegate works in a child process before trusting it, because MediaPipe
  aborts rather than raising when it cannot. Whichever it ends up on is
  logged, shown on the web view, and published as "delegate" on every
  /gesture/detections message. Body-pose keypoints cannot substitute for this:
  they have no fingers.
* The subscription starts best-effort and falls back to reliable if no frame
  arrives, because large fragmented samples from another SoC do not always
  come through on the first QoS choice.
* The web view is served from this process on port 8082 (8081 is
  x2_yolo_stream's). It is the fastest way to see why a gesture is not
  triggering: the per-class bars separate "the model is unsure" from "the
  threshold is too high", which the topics alone make you squint at.
"""

import json
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .gesture import SPEC_VERSION, GestureClassifier, featurize
from .runtime import describe_environment, make_landmarker
from .web import local_ips, serve

# Hand skeleton, for the overlay only. Cosmetic: nothing downstream reads it.
HAND_EDGES = ${JSON.stringify(HAND_EDGES)}

# Ceiling on the num_hands parameter. MediaPipe has no hard limit of its own,
# so this one is ours: every extra hand in frame is another featurize plus
# another MLP forward pass inside one infer_fps budget, and a request for
# twenty hands is a typo, not a configuration.
MAX_HANDS = 4

# First existing topic wins when image_topic is left empty. Same list as
# x2_yolo_stream, so both nodes land on the same camera by default.
TOPIC_PREFERENCE = [
    "/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed",
    "/aima/hal/sensor/rgbd_head_front/rgb_image/compressed",
    "/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed",
    "/aima/hal/sensor/rgb_head_rear/rgb_image/compressed",
    "/camera/color/image_raw/compressed",
]


def draw_overlay(frame, tracks, threshold):
    """A skeleton per hand plus a label chip.

    tracks is one dict per hand, as _infer_loop assembles them: pixel
    "points", the committed "label" (or None), its "confidence" and
    "handedness". An empty list means the frame had no hand.
    """
    for track in tracks:
        points = track["points"]
        for a, b in HAND_EDGES:
            cv2.line(frame, points[a], points[b], (0, 255, 200), 2,
                     cv2.LINE_AA)
        for i, pt in enumerate(points):
            # The wrist anchors the whole normalization, so it reads heavier.
            cv2.circle(frame, pt, 5 if i == 0 else 3, (255, 255, 255), -1,
                       cv2.LINE_AA)
        if len(tracks) > 1:
            # Only worth the ink with more than one hand up: which label
            # belongs to which hand is unreadable from a single chip, and
            # drawing it when there is nothing to disambiguate is clutter.
            tag = ("%s  %.2f" % (track["label"], track["confidence"])
                   if track["label"] else "?  %.2f" % track["confidence"])
            cv2.putText(frame, tag, (points[0][0] + 10, points[0][1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 200) if track["label"] else (120, 120, 120),
                        2, cv2.LINE_AA)

    if not tracks:
        text = "no hand"
        colour = (120, 120, 120)
    elif len(tracks) == 1 and tracks[0]["label"]:
        text = "%s  %.2f" % (tracks[0]["label"], tracks[0]["confidence"])
        colour = (0, 255, 200)
    elif len(tracks) == 1:
        text = "unrecognised  %.2f / %.2f" % (tracks[0]["confidence"],
                                              threshold)
        colour = (120, 120, 120)
    else:
        # Every hand on one line, in detection order, so the chip still says
        # what the whole frame is doing.
        text = "   ".join(
            "%s %.2f" % (t["label"], t["confidence"]) if t["label"]
            else "unrecognised %.2f" % t["confidence"] for t in tracks)
        colour = ((0, 255, 200) if any(t["label"] for t in tracks)
                  else (120, 120, 120))

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (10, 10), (22 + tw, 30 + th), (14, 17, 22), -1)
    cv2.putText(frame, text, (16, 24 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                colour, 2, cv2.LINE_AA)
    return frame


class SharedState:
    """Latest annotated frame + gesture, shared with the HTTP threads."""

    def __init__(self):
        self.lock = threading.Condition()
        self.jpeg = None
        self.seq = 0
        self.snapshot_data = {}
        self.viewers = 0
        self.frames_in = 0
        self.frames_inferred = 0
        self._in_times = []
        self._infer_times = []

    def publish(self, jpeg, data):
        with self.lock:
            self.jpeg = jpeg
            self.seq += 1
            self.snapshot_data = data
            self.lock.notify_all()

    def note_frame_in(self):
        with self.lock:
            self.frames_in += 1
            self._in_times.append(time.monotonic())
            self._in_times = self._in_times[-30:]

    def note_inferred(self):
        """One cycle finished. Counted here rather than in publish().

        publish() only runs when the web view is on, so counting there made
        infer_fps read 0.0 under publish_web:=false -- on the dashboard that
        is indistinguishable from a node that has stopped analysing, which is
        the one thing the number exists to tell you.
        """
        with self.lock:
            self.frames_inferred += 1
            self._infer_times.append(time.monotonic())
            self._infer_times = self._infer_times[-30:]

    @staticmethod
    def _rate(times):
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return round((len(times) - 1) / span, 1) if span > 0 else 0.0

    def rates(self):
        """(frames arriving, frames analysed) per second."""
        with self.lock:
            return self._rate(self._in_times), self._rate(self._infer_times)

    def add_viewer(self, delta):
        with self.lock:
            self.viewers = max(0, self.viewers + delta)

    def latest_jpeg(self):
        with self.lock:
            return self.jpeg

    def wait_for_frame(self, last_seq, timeout=5.0):
        with self.lock:
            if self.seq == last_seq:
                self.lock.wait(timeout)
            return self.jpeg, self.seq

    def snapshot(self):
        with self.lock:
            out = dict(self.snapshot_data)
            out.update({
                "stream_fps": self._rate(self._in_times),
                "infer_fps": self._rate(self._infer_times),
                "frames_in": self.frames_in,
                "frames_inferred": self.frames_inferred,
                "viewers": self.viewers,
            })
            return out


class GestureNode(Node):

    def __init__(self):
        super().__init__("x2_gesture")

        # -- input ---------------------------------------------------------- #
        self.declare_parameter("image_topic", "")
        # >= 0 opens a local V4L2 camera instead of subscribing. The escape
        # hatch for testing on a laptop; on the robot the cameras live on other
        # SoCs and are only reachable over DDS.
        self.declare_parameter("camera_device", -1)
        self.declare_parameter("model_path",
                               "${WORKSPACE}/models/gesture_model.json")
        self.declare_parameter("landmarker_path",
                               "${WORKSPACE}/models/hand_landmarker.task")

        # -- detection ------------------------------------------------------ #
        # "cpu" and not "auto", on purpose. The Orin's GPU already belongs to
        # x2_yolo_stream, which gets 10x from it on a model far bigger than
        # this one; MediaPipe's GPU delegate is worth 2-3x on a landmarker
        # small enough that the frame upload is most of the cost. Two
        # processes contending for one GPU to speed up the cheaper of them is
        # a bad trade, so the split is deliberate: YOLO on the GPU, hands on
        # the CPU.
        #
        # "gpu" or "auto" switches it, and everything needed for that is in
        # place and safe -- see runtime.py and the README. Measure both before
        # deciding; the node reports inference_ms either way. Note that "gpu"
        # does NOT make the node fail when there is no GPU: runtime.py always
        # keeps the CPU behind it.
        self.declare_parameter("delegate", "cpu")
        # The GPU delegate aborts the process rather than raising when it
        # cannot run, so it is verified in a child process first. Costs a few
        # seconds at startup and is skipped entirely on the CPU path, where
        # there is nothing to fall back to. false only if you like crashes.
        self.declare_parameter("probe_delegate", True)
        # The one knob for multi-hand: how many hands the landmarker looks for
        # and this node classifies. 1 is single-hand behaviour; up to MAX_HANDS
        # each get their own smoothing window, their own label, and an entry in
        # "per_hand" on the JSON topics. It is a launch argument too --
        #     ros2 launch x2_gesture x2_gesture.launch.py num_hands:=2
        # -- because "how many hands" is a thing you retune per scene, not a
        # thing you should have to reach for --ros-args to change.
        self.declare_parameter("num_hands", 4)
        # MediaPipe's own gates, not the classifier's: how sure it must be that
        # a hand is a hand before landmarks come out at all.
        self.declare_parameter("min_detection_confidence", 0.5)
        self.declare_parameter("min_tracking_confidence", 0.5)
        # The classifier's gate: smoothed confidence needed to commit a label.
        # Below it the topic reports the scores but no gesture.
        self.declare_parameter("threshold", 0.75)
        # Frames averaged before committing. Raising it makes the label steadier
        # and slower to react; the /detections topic carries both the smoothed
        # and the instantaneous scores so this can be tuned from an echo.
        self.declare_parameter("smoothing", 5)
        self.declare_parameter("infer_fps", 10.0)
        # Landmarking cost scales with pixels and the hand is a small part of
        # the frame; 640 is plenty and keeps a cycle affordable.
        self.declare_parameter("max_width", 640)

        # -- output --------------------------------------------------------- #
        # Namespace for this node's topics, so a second instance watching
        # another camera does not publish over the first one's.
        self.declare_parameter("topic_prefix", "/gesture")
        self.declare_parameter("publish_detections", True)
        self.declare_parameter("publish_summary", True)
        # A plain sentence on <prefix>/text. \`ros2 topic echo\` on the JSON
        # topic is a wall of scores; this is the line you actually wanted.
        self.declare_parameter("publish_text", True)
        self.declare_parameter("pretty_summary", True)
        self.declare_parameter("pretty_detections", False)
        # Publish the summary only when the gesture actually changes, so a
        # behaviour node can sit on it without a 10 Hz firehose. The heartbeat
        # still goes out at summary_max_period.
        self.declare_parameter("summary_on_change_only", True)
        self.declare_parameter("summary_max_period", 5.0)
        self.declare_parameter("qos_fallback_seconds", 12.0)

        # -- overlay base --------------------------------------------------- #
        # Draw the hand overlay onto the frame x2_yolo_stream has already drawn
        # boxes on, so one stream carries both layers instead of the dashboard
        # showing two videos of the same camera. Empty disables it.
        #
        # Classification still runs on the RAW camera. Landmarking a frame with
        # detection boxes drawn across it would corrupt the landmarks, so this
        # image is a canvas and nothing else.
        self.declare_parameter("overlay_base_topic",
                               "/yolo/image_annotated/compressed")
        # Past this age the base frame is ignored and the raw camera is drawn
        # on instead. Without it, a dead x2_yolo_stream leaves one frozen frame
        # under a live skeleton, which looks like the camera has stopped.
        self.declare_parameter("overlay_base_max_age", 2.0)

        # -- web view ------------------------------------------------------- #
        self.declare_parameter("publish_web", True)
        self.declare_parameter("host", "0.0.0.0")
        # 8081 is x2_yolo_stream's. Two nodes on one port is a silent failure
        # where whichever started second serves nothing.
        self.declare_parameter("port", 8082)
        self.declare_parameter("jpeg_quality", 80)

        p = self.get_parameter
        self.camera_device = int(p("camera_device").value)
        requested_hands = int(p("num_hands").value)
        self.num_hands = max(1, min(MAX_HANDS, requested_hands))
        if requested_hands != self.num_hands:
            # Clamped, not ignored, and said out loud: silently running with a
            # different hand budget than the one asked for is the kind of thing
            # you only discover by wondering why the fourth hand never appears.
            self.get_logger().warning(
                "num_hands:=%d is out of range -- using %d (1..%d)"
                % (requested_hands, self.num_hands, MAX_HANDS))
        self.threshold = float(p("threshold").value)
        self.smoothing = max(1, int(p("smoothing").value))
        self.infer_fps = max(0.5, float(p("infer_fps").value))
        self.max_width = int(p("max_width").value)
        self.publish_detections = bool(p("publish_detections").value)
        self.publish_summary = bool(p("publish_summary").value)
        self.publish_text = bool(p("publish_text").value)
        self.pretty_summary = bool(p("pretty_summary").value)
        self.pretty_detections = bool(p("pretty_detections").value)
        self.summary_on_change_only = bool(p("summary_on_change_only").value)
        self.summary_max_period = float(p("summary_max_period").value)
        self.qos_fallback_seconds = float(p("qos_fallback_seconds").value)
        self.publish_web = bool(p("publish_web").value)
        self.jpeg_quality = int(p("jpeg_quality").value)
        self.overlay_base_topic = str(p("overlay_base_topic").value).strip()
        self.overlay_base_max_age = float(p("overlay_base_max_age").value)

        # Latest annotated frame from x2_yolo_stream, still compressed. Decoded
        # in the worker, never in the DDS callback -- the callback's only job is
        # to be fast enough not to drop the executor behind.
        self._base_slot = None
        self._base_lock = threading.Lock()

        # -- model ---------------------------------------------------------- #
        model_path = str(p("model_path").value)
        self.get_logger().info("loading %s" % model_path)
        self.clf = GestureClassifier(model_path)
        self.get_logger().info(
            "%d classes: %s" % (len(self.clf.classes), ", ".join(self.clf.classes)))
        self.get_logger().info(
            "validation accuracy at export: %.3f" % self.clf.val_accuracy)
        # Before any publisher exists. A featurizer that disagrees with the
        # browser returns confident wrong answers rather than an error, and a
        # behaviour node downstream cannot tell the difference -- so this node
        # refuses to join the graph at all rather than lie to it.
        self.clf.verify()

        landmarker_path = str(p("landmarker_path").value)
        self.get_logger().info("loading %s" % landmarker_path)
        requested_delegate = str(p("delegate").value).strip().lower()
        # Report the runtime before loading anything: when the GPU is wanted
        # and missing, this line is what explains every number below it. On
        # the plain CPU default there is nothing to explain, so it stays short.
        self.get_logger().info(
            describe_environment(check_gl=requested_delegate != "cpu"))
        self.landmarker = make_landmarker(
            landmarker_path,
            delegate=requested_delegate,
            num_hands=self.num_hands,
            min_detection_confidence=float(
                p("min_detection_confidence").value),
            min_tracking_confidence=float(
                p("min_tracking_confidence").value),
            probe_first=bool(p("probe_delegate").value),
            logger=self.get_logger().info,
        )
        # What it is ACTUALLY on, not what was asked for. Kept on the node so
        # the topics and the web view report the same thing the log did.
        self.delegate = self.landmarker.delegate
        self.get_logger().info(
            "hand landmarker: %s, blank-frame pass %.0f ms"
            % (self.landmarker.label, self.landmarker.warmup_ms))
        # Only when the GPU was asked for and not given. The plain CPU default
        # is the intended configuration on this robot, not a degraded one, and
        # a warning on the happy path teaches people to ignore warnings.
        if requested_delegate != "cpu" and self.delegate == "cpu":
            self.get_logger().warning(
                "delegate:=%s was requested but landmarking fell back to the "
                "CPU -- the lines above say which part failed, and the README "
                "section 'If the GPU is refused' says what to do about each "
                "one" % requested_delegate)

        # -- state ---------------------------------------------------------- #
        # One smoothing window per hand, not one per frame: with num_hands > 1
        # a shared window would average a left fist and a right peace into a
        # distribution neither hand is making. Keyed by (handedness, nth of
        # that handedness) -- the landmarker does not track hands between
        # frames, so that is the closest thing to an identity it gives us.
        self.histories = {}
        self._current_key = None
        self._since = time.monotonic()
        self._changes = 0
        self._last_summary_key = None
        self._last_summary_time = 0.0
        self._frames = 0
        self.state = SharedState()
        self.httpd = None

        # -- topics --------------------------------------------------------- #
        prefix = str(p("topic_prefix").value).rstrip("/") or "/gesture"
        self.pub_dets = self.create_publisher(String, prefix + "/detections", 10)
        self.pub_summary = self.create_publisher(String, prefix + "/summary", 10)
        self.pub_text = self.create_publisher(String, prefix + "/text", 10)

        # latest-frame-wins slot between the DDS callback and the worker
        self._slot = None
        self._slot_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()

        self._sub = None
        self._cap = None
        self.topic = ""
        self._last_frame_time = time.monotonic()

        if self.camera_device >= 0:
            self.topic = "camera:%d" % self.camera_device
            self._cap = cv2.VideoCapture(self.camera_device)
            if not self._cap.isOpened():
                raise SystemExit("could not open camera %d" % self.camera_device)
            self.get_logger().warning(
                "reading local camera %d -- this is the dev path, not the robot "
                "path" % self.camera_device)
        else:
            self.topic = self._resolve_topic(str(p("image_topic").value))
            self._use_best_effort = True
            self._subscribe()
            self.create_timer(2.0, self._watchdog)

        # Independent of the camera subscription above, and of camera_device:
        # even reading a local webcam, the annotated frames still arrive over
        # DDS if x2_yolo_stream is watching the same scene.
        if self.overlay_base_topic:
            base_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                  reliability=ReliabilityPolicy.BEST_EFFORT,
                                  durability=DurabilityPolicy.VOLATILE)
            self.create_subscription(CompressedImage, self.overlay_base_topic,
                                     self._on_base_image, base_qos)
            self.get_logger().info(
                "overlay base: %s (falls back to the raw camera if it goes "
                "quiet for %.1fs)"
                % (self.overlay_base_topic, self.overlay_base_max_age))

        self.worker = threading.Thread(target=self._infer_loop, daemon=True)
        self.worker.start()

        if self.publish_web:
            host, port = str(p("host").value), int(p("port").value)
            try:
                self.httpd = serve(self.state, host, port)
            except OSError as exc:
                # Almost always "address already in use" -- a second instance,
                # or someone put this on 8081 next to x2_yolo_stream. Losing
                # the web view is not worth killing gesture publishing over.
                self.get_logger().error(
                    "web view disabled: cannot bind %s:%d (%s)"
                    % (host, port, exc))
            else:
                threading.Thread(target=self.httpd.serve_forever,
                                 daemon=True).start()
                for ip in ["localhost"] + local_ips():
                    self.get_logger().info("viewer: http://%s:%d/" % (ip, port))

        self.get_logger().info(
            "gesture up: %s -> %s/{detections,summary,text}, threshold %.2f, "
            "smoothing %d frames, up to %d hand%s, %.0f fps cap, "
            "feature spec v%d"
            % (self.topic, prefix, self.threshold, self.smoothing,
               self.num_hands, "" if self.num_hands == 1 else "s",
               self.infer_fps, SPEC_VERSION))

    # -- subscription ------------------------------------------------------- #

    def _resolve_topic(self, requested):
        if requested:
            return requested
        # Discovery needs a moment before the graph is complete.
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            available = {name for name, _ in self.get_topic_names_and_types()}
            for candidate in TOPIC_PREFERENCE:
                if candidate in available:
                    return candidate
            time.sleep(0.5)
        fallback = TOPIC_PREFERENCE[0]
        self.get_logger().warning(
            "no camera topic discovered; will wait on %s" % fallback)
        return fallback

    def _subscribe(self):
        if self._sub is not None:
            self.destroy_subscription(self._sub)
        reliability = (ReliabilityPolicy.BEST_EFFORT if self._use_best_effort
                       else ReliabilityPolicy.RELIABLE)
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=reliability,
                         durability=DurabilityPolicy.VOLATILE)
        self._sub = self.create_subscription(
            CompressedImage, self.topic, self._on_image, qos)
        self.get_logger().info(
            "subscribed to %s (%s)" % (self.topic, reliability.name))

    def _watchdog(self):
        """If nothing arrives, flip reliability and resubscribe."""
        idle = time.monotonic() - self._last_frame_time
        if idle < self.qos_fallback_seconds:
            return
        self._use_best_effort = not self._use_best_effort
        self.get_logger().warning(
            "no frames for %.0fs, retrying with %s"
            % (idle, "BEST_EFFORT" if self._use_best_effort else "RELIABLE"))
        self._last_frame_time = time.monotonic()
        self._subscribe()

    def _on_base_image(self, msg):
        """Annotated frame from x2_yolo_stream. Stored compressed, decoded later."""
        with self._base_lock:
            self._base_slot = (bytes(msg.data), time.monotonic())

    def _base_fresh(self):
        """Is there a recent annotated frame to draw on? Timestamp only.

        Cheap on purpose -- no decode -- because the detections topic reports
        it every cycle whether or not the web view is running. It is what lets
        a consumer say "boxes + skeleton" without guessing from x2_yolo_stream
        being alive, which is a different question: the node can be publishing
        detections happily with publish_annotated:=false.
        """
        if not self.overlay_base_topic:
            return False
        with self._base_lock:
            item = self._base_slot
        return (item is not None
                and time.monotonic() - item[1] <= self.overlay_base_max_age)

    def _overlay_base(self, width, height):
        """The freshest annotated frame, resized to (width, height), or None.

        Returned at the gesture frame's size so the caller can draw landmarks
        without rescaling them: the two nodes size their frames independently
        (each has its own max_width), and matching the canvas to the points is
        simpler than matching every point to the canvas.
        """
        with self._base_lock:
            item = self._base_slot
        if item is None:
            return None
        data, stamped = item
        if time.monotonic() - stamped > self.overlay_base_max_age:
            return None                # x2_yolo_stream has gone quiet
        try:
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        except Exception:                                   # noqa: BLE001
            return None
        if frame is None:
            return None
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height),
                               interpolation=cv2.INTER_AREA)
        return frame

    def _on_image(self, msg):
        """Keep this cheap: store the bytes, let the worker do the work."""
        self._last_frame_time = time.monotonic()
        self.state.note_frame_in()
        with self._slot_lock:
            self._slot = (bytes(msg.data), msg.header.stamp.sec,
                          msg.header.stamp.nanosec, msg.header.frame_id)
        self._wake.set()

    # -- inference ---------------------------------------------------------- #

    def _next_frame(self, min_interval):
        """One decoded frame plus its stamp, or None to skip this cycle."""
        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                return None
            self.state.note_frame_in()
            now = self.get_clock().now().to_msg()
            return frame, now.sec, now.nanosec, "camera"

        if not self._wake.wait(timeout=0.5):
            return None
        self._wake.clear()
        with self._slot_lock:
            item, self._slot = self._slot, None
        if item is None:
            return None

        data, sec, nanosec, frame_id = item
        try:
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        except Exception as exc:                            # noqa: BLE001
            self.get_logger().warning("decode failed: %s" % exc)
            return None
        if frame is None:
            return None
        return frame, sec, nanosec, frame_id

    def _infer_loop(self):
        min_interval = 1.0 / self.infer_fps
        next_allowed = 0.0
        # The frame timestamps MediaPipe's VIDEO mode insists on are the
        # Landmarker's business, not this loop's -- see runtime.py. The DDS
        # header stamp could never have been used for them anyway: frames from
        # another SoC arrive with equal or backwards stamps.

        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_allowed:
                time.sleep(min(0.01, next_allowed - now))
                continue

            item = self._next_frame(min_interval)
            if item is None:
                continue
            next_allowed = time.monotonic() + min_interval

            frame, sec, nanosec, frame_id = item
            if self.max_width and frame.shape[1] > self.max_width:
                scale = self.max_width / frame.shape[1]
                frame = cv2.resize(
                    frame, (self.max_width, int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA)

            began = time.monotonic()
            try:
                # BGR straight in. The colour conversion belongs to the
                # Landmarker because it depends on the delegate -- the GPU
                # path takes 4 channels and aborts the process on 3.
                result = self.landmarker.detect(frame)
            except Exception as exc:                        # noqa: BLE001
                self.get_logger().error("landmarking failed: %s" % exc,
                                        throttle_duration_sec=5.0)
                continue

            # Every hand the landmarker returned, not just the first. The slice
            # is the num_hands budget enforced a second time: the landmarker is
            # already configured with it, and this makes a landmarker that ever
            # returns more than it was asked for cost nothing extra here.
            worlds = (result.hand_world_landmarks or [])[:self.num_hands]
            hands = len(worlds)
            height, width = frame.shape[0], frame.shape[1]
            tracks = []
            seen = {}
            for i, world in enumerate(worlds):
                try:
                    lm = [[p.x, p.y, p.z] for p in world]
                    handedness = result.handedness[i][0].category_name
                    instant = self.clf.predict(featurize(lm, handedness))
                    # Screen landmarks are normalised to the frame and are only
                    # used for drawing -- the classifier reads world landmarks.
                    points = [(int(q.x * width), int(q.y * height))
                              for q in result.hand_landmarks[i]]
                except Exception as exc:                    # noqa: BLE001
                    # One bad hand loses that hand, not the frame: the others
                    # are still worth publishing.
                    self.get_logger().error("classification failed: %s" % exc,
                                            throttle_duration_sec=5.0)
                    continue

                seen[handedness] = seen.get(handedness, 0) + 1
                key = (handedness, seen[handedness])
                history = self.histories.get(key)
                if history is None:
                    history = self.histories[key] = deque(
                        maxlen=self.smoothing)
                history.append(instant)
                label, confidence, smoothed = self._smooth(history)
                tracks.append({
                    "handedness": handedness,
                    "instant": instant,
                    "points": points,
                    "label": label,
                    "confidence": confidence,
                    "smoothed": smoothed,
                })

            elapsed_ms = (time.monotonic() - began) * 1000.0
            self._frames += 1
            # A cycle is analysed here, whether or not anyone is watching.
            self.state.note_inferred()

            # Carrying stale probabilities across an empty gap makes a gesture
            # appear to linger after the hand is gone -- and a window kept for
            # a hand that left the frame is that same bug one hand deeper.
            for key in list(self.histories):
                if seen.get(key[0], 0) < key[1]:
                    del self.histories[key]

            # One hand still leads the summary topics and the top-level fields:
            # subscribers act on "what is the hand doing", and a list is the
            # wrong shape for that. The strongest committed hand wins, so a
            # second hand resting in frame cannot outvote the gesturing one.
            primary = None
            if tracks:
                committed = [t for t in tracks if t["label"]]
                primary = max(committed or tracks, key=lambda t: t["confidence"])

            if primary is None:
                instant, handedness = None, None
                label, confidence, smoothed = self._smooth(None)
            else:
                instant = primary["instant"]
                handedness = primary["handedness"]
                label = primary["label"]
                confidence = primary["confidence"]
                smoothed = primary["smoothed"]

            try:
                self._publish(instant, smoothed, label, confidence, handedness,
                              hands, sec, nanosec, frame_id, elapsed_ms, tracks)
            except Exception as exc:                        # noqa: BLE001
                self.get_logger().error("publish failed: %s" % exc)

            if self.publish_web:
                try:
                    self._serve_frame(frame, tracks, label, confidence,
                                      handedness, hands, smoothed, elapsed_ms)
                except Exception as exc:                    # noqa: BLE001
                    # A broken overlay must not stop the topics. They are what
                    # the robot acts on; the web view is for a human.
                    self.get_logger().error("web frame failed: %s" % exc,
                                            throttle_duration_sec=5.0)

    def _serve_frame(self, frame, tracks, label, confidence, handedness, hands,
                     smoothed, elapsed_ms):
        # Prefer x2_yolo_stream's annotated frame as the canvas, so this one
        # stream carries boxes and skeleton together. The landmarks were
        # computed from the raw camera either way -- only what they are drawn
        # on changes.
        canvas = self._overlay_base(frame.shape[1], frame.shape[0])
        base_live = canvas is not None
        if not base_live:
            canvas = frame

        annotated = draw_overlay(canvas, tracks, self.threshold)
        ok, buf = cv2.imencode(".jpg", annotated,
                               [int(cv2.IMWRITE_JPEG_QUALITY),
                                self.jpeg_quality])
        if not ok:
            return
        self.state.publish(buf.tobytes(), {
            "topic": self.topic,
            # False means this frame is the raw camera: either the overlay base
            # is switched off, or x2_yolo_stream has stopped publishing.
            "overlay_base": base_live,
            "width": annotated.shape[1],
            "height": annotated.shape[0],
            "spec_version": SPEC_VERSION,
            "classes": list(self.clf.classes),
            "gesture": label,
            "confidence": round(confidence, 4) if label else 0.0,
            "handedness": handedness,
            "hands": hands,
            "num_hands": self.num_hands,
            "threshold": self.threshold,
            "smoothing": self.smoothing,
            "held_seconds": round(time.monotonic() - self._since, 1),
            "changes": self._changes,
            "scores": {c: round(float(s), 4)
                       for c, s in zip(self.clf.classes, smoothed)},
            "per_hand": self._per_hand(tracks),
            "inference_ms": round(elapsed_ms, 1),
            "delegate": self.delegate,
        })

    # -- publishing --------------------------------------------------------- #

    @staticmethod
    def _dump(payload, pretty):
        # ensure_ascii=False so non-Latin class names stay legible rather than
        # arriving as escapes in a terminal.
        if pretty:
            return json.dumps(payload, indent=2, ensure_ascii=False)
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _describe(label, handedness, confidence, hands, held=None):
        if not hands:
            return "no hand"
        hand = "%s hand" % handedness.lower() if handedness else "hand"
        if label is None:
            return "unrecognised, %s, best %.2f" % (hand, confidence)
        if held is None:
            return "%s, %s, %.2f" % (label, hand, confidence)
        return "%s, %s, held %.1fs" % (label, hand, held)

    @classmethod
    def _describe_hands(cls, tracks, hands, held=None):
        """One line for the whole frame -- every hand it found.

        A single hand reads exactly as it did before this took a list; the
        joined form only appears when there is a second hand to join.
        """
        if not tracks:
            # A detected hand the classifier could not use is not "no hand":
            # the error log says why, and this must not claim the frame was
            # empty when it was not.
            return "no hand" if not hands else "hand, unclassified"
        if len(tracks) == 1:
            t = tracks[0]
            return cls._describe(t["label"], t["handedness"], t["confidence"],
                                 1, held)
        text = " + ".join(cls._describe(t["label"], t["handedness"],
                                        t["confidence"], 1) for t in tracks)
        return text if held is None else "%s, held %.1fs" % (text, held)

    def _per_hand(self, tracks):
        """Per-hand block for the JSON topics, in detection order."""
        classes = self.clf.classes
        return [{
            "handedness": t["handedness"],
            "gesture": t["label"],
            "confidence": round(t["confidence"], 4) if t["label"] else 0.0,
            "scores": {c: round(float(s), 4)
                       for c, s in zip(classes, t["smoothed"])},
        } for t in tracks]

    def _smooth(self, history):
        """Committed label, its confidence, and the averaged distribution.

        Over one hand's window -- the caller owns the windows, because which
        hand a frame's landmarks belong to is decided in _infer_loop.
        """
        if not history:
            return None, 0.0, np.zeros(len(self.clf.classes), dtype=np.float32)
        smoothed = np.mean(history, axis=0)
        best = int(np.argmax(smoothed))
        best_score = float(smoothed[best])
        label = self.clf.classes[best] if best_score >= self.threshold else None
        return label, best_score, smoothed

    def _publish(self, instant, smoothed, label, best_score, handedness, hands,
                 sec, nanosec, frame_id, elapsed_ms, tracks):
        classes = self.clf.classes
        description = self._describe_hands(tracks, hands)
        stream_fps, infer_fps = self.state.rates()

        if self.publish_detections:
            msg = String()
            msg.data = self._dump({
                # First key, so the head of an echo is the answer.
                "description": description,
                "stamp": {"sec": int(sec), "nanosec": int(nanosec)},
                "frame_id": frame_id,
                "source_topic": self.topic,
                # True when the annotated stream on this node's web view is
                # x2_yolo_stream's frame with the hand drawn over it, so a
                # viewer can label the layers honestly rather than inferring
                # them from whether YOLO happens to be alive.
                "overlay_base": self._base_fresh(),
                "spec_version": SPEC_VERSION,
                "classes": list(classes),
                "gesture": label,
                "confidence": round(best_score, 4) if label else 0.0,
                "handedness": handedness,
                # How many hands this frame had, and the ceiling it was allowed
                # -- "hands": 4, "num_hands": 4 is the case where a fifth hand
                # would have been dropped, which the count alone cannot say.
                "hands": hands,
                "num_hands": self.num_hands,
                "threshold": self.threshold,
                "smoothing": self.smoothing,
                "scores": {c: round(float(s), 4)
                           for c, s in zip(classes, smoothed)},
                "instant": ({c: round(float(s), 4)
                             for c, s in zip(classes, instant)}
                            if instant is not None else None),
                # Every hand, in detection order. The keys above are the one
                # answer a subscriber acts on -- the leading hand -- and stay
                # where they were; this is for the ones that want both hands.
                "per_hand": self._per_hand(tracks),
                "inference_ms": round(elapsed_ms, 1),
                # "gpu" or "cpu" -- what the landmarker is really on. Next to
                # inference_ms because the two only make sense together: 40 ms
                # is good on the CPU and bad on the GPU.
                "delegate": self.delegate,
                # Rate as well as cost, and both rates: frames analysed against
                # frames arriving is what separates "the model is slow" from
                # "infer_fps is capping it" from "the camera stopped".
                "stream_fps": stream_fps,
                "infer_fps": infer_fps,
            }, self.pretty_detections)
            self.pub_dets.publish(msg)

        if self.publish_summary or self.publish_text:
            self._publish_summary(label, handedness, best_score, hands, sec,
                                  nanosec, tracks)

    def _publish_summary(self, label, handedness, confidence, hands, sec,
                         nanosec, tracks):
        """Compact "what is the hand doing" topic, rate limited to changes.

        A confidence wobble is not a change: the key is the committed label and
        which hand it is on -- every hand, so a second hand starting a gesture
        counts -- and a subscriber is not woken ten times a second by a frame
        that says the same thing as the last one.
        """
        key = tuple((t["label"], t["handedness"]) for t in tracks)
        now = time.monotonic()
        if key != self._current_key:
            self._current_key = key
            self._since = now
            self._changes += 1

        held = now - self._since

        if (self.summary_on_change_only and key == self._last_summary_key and
                now - self._last_summary_time < self.summary_max_period):
            return
        self._last_summary_key = key
        self._last_summary_time = now

        committed = any(t["label"] for t in tracks)
        description = self._describe_hands(tracks, hands,
                                           held=held if committed else None)

        if self.publish_text:
            # Plain text, not JSON: an echo of <prefix>/text should read like a
            # sentence, and it does not need quoting to.
            text = String()
            text.data = description
            self.pub_text.publish(text)

        if not self.publish_summary:
            return

        msg = String()
        msg.data = self._dump({
            "description": description,
            "stamp": {"sec": int(sec), "nanosec": int(nanosec)},
            "gesture": label,
            "confidence": round(confidence, 4) if label else 0.0,
            "handedness": handedness,
            "hands": hands,
            "per_hand": self._per_hand(tracks),
            "held_seconds": round(held, 2),
            "changes": self._changes,
        }, self.pretty_summary)
        self.pub_summary.publish(msg)

    def destroy_node(self):
        self._stop.set()
        self._wake.set()
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
            except Exception:                               # noqa: BLE001
                pass
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:                               # noqa: BLE001
                pass
        # Only after the worker has stopped: closing the graph out from under
        # an in-flight detect() tears down the GL context mid-frame, which
        # ends the process with a native stack trace instead of a clean exit.
        # If the worker will not come back, leave it to the OS -- a hung
        # shutdown is worse than a leaked context on a process that is going
        # away anyway.
        self.worker.join(timeout=2.0)
        if not self.worker.is_alive():
            self.landmarker.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
`;
}

function generatePackageXml(): string {
  return `<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>${PACKAGE_NAME}</name>
  <version>1.0.0</version>
  <description>Hand-gesture classification on an X2 camera topic, published as
  ROS topics. The classifier is exported from Gesture Atlas.</description>
  <maintainer email="dev@kazbot.kz">Kazbot</maintainer>
  <license>Proprietary</license>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>std_msgs</exec_depend>

  <!-- python3-opencv and python3-numpy are already present on the robot image,
       and mediapipe is installed with pip. They are deliberately not declared
       as rosdep keys so that nothing tries to install or upgrade them. -->

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
`;
}

function generateSetupPy(): string {
  return `"""ament_python package definition.

Neither the gesture model nor the hand landmarker is shipped inside the
package: they live in the workspace's models/ directory and are passed in as
the model_path and landmarker_path parameters. Keeping them out of the install
space means colcon builds stay fast and the repository stays free of binaries.
"""

from setuptools import find_packages, setup

PACKAGE_NAME = "${PACKAGE_NAME}"

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + PACKAGE_NAME]),
        ("share/" + PACKAGE_NAME, ["package.xml"]),
        ("share/" + PACKAGE_NAME + "/launch",
         ["launch/${PACKAGE_NAME}.launch.py"]),
        # Run from src/ on the robot rather than the install space, but ship
        # them too so a fresh checkout has them somewhere findable.
        ("share/" + PACKAGE_NAME + "/scripts",
         ["scripts/start_gesture.sh",
          "scripts/gpu_env.sh",
          "scripts/x2-gesture.service"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kazbot",
    maintainer_email="dev@kazbot.kz",
    description="Hand-gesture classification on an X2 camera topic, as ROS "
                "topics.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "gesture_node = ${PACKAGE_NAME}.node:main",
        ],
    },
)
`;
}

function generateSetupCfg(): string {
  return `[develop]
script_dir=$base/lib/${PACKAGE_NAME}
[install]
install_scripts=$base/lib/${PACKAGE_NAME}
`;
}

function generateLaunchPy(): string {
  return `"""Launch the gesture node with the robot's defaults.

Landmarking is on the CPU here, deliberately: the Orin's GPU is
x2_yolo_stream's, which gets far more out of it. delegate:=gpu moves it, and
if you do that, note that the GPU delegate needs LD_LIBRARY_PATH set before
the process starts -- no launch file can arrange that from the inside. Either
start through scripts/start_gesture.sh, which sources scripts/gpu_env.sh for
you, or source it yourself first:

    . src/${PACKAGE_NAME}/scripts/gpu_env.sh
    ros2 launch ${PACKAGE_NAME} ${PACKAGE_NAME}.launch.py delegate:=gpu

Asking for the GPU without it is the quiet failure worth knowing about: the
node comes up, publishes correctly, and runs on the CPU. It logs a warning
when that happens, and the startup log says which delegate it got.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("image_topic", default_value="",
                              description="camera topic ('' = auto-pick)"),
        DeclareLaunchArgument(
            "model_path",
            default_value="${WORKSPACE}/models/gesture_model.json"),
        DeclareLaunchArgument(
            "landmarker_path",
            default_value="${WORKSPACE}/models/hand_landmarker.task"),
        DeclareLaunchArgument(
            "delegate", default_value="cpu",
            description="cpu | gpu | auto -- the GPU belongs to "
                        "x2_yolo_stream by default; see the README"),
        DeclareLaunchArgument("threshold", default_value="0.75",
                              description="confidence to commit a gesture"),
        DeclareLaunchArgument("smoothing", default_value="5",
                              description="frames averaged before committing"),
        DeclareLaunchArgument(
            "num_hands", default_value="4",
            description="hands to detect and classify, 1..4 -- each gets its "
                        "own smoothing window and an entry in per_hand"),
        DeclareLaunchArgument("infer_fps", default_value="10.0"),
        DeclareLaunchArgument(
            "summary_on_change_only", default_value="true",
            description="false = /gesture/summary every cycle"),
        DeclareLaunchArgument("port", default_value="8082",
                              description="web view; 8081 is x2_yolo_stream"),
        DeclareLaunchArgument("publish_web", default_value="true"),
    ]
    node = Node(
        package="${PACKAGE_NAME}",
        executable="gesture_node",
        name="${PACKAGE_NAME}",
        output="screen",
        parameters=[{
            "image_topic": LaunchConfiguration("image_topic"),
            "model_path": LaunchConfiguration("model_path"),
            "landmarker_path": LaunchConfiguration("landmarker_path"),
            "delegate": LaunchConfiguration("delegate"),
            "threshold": LaunchConfiguration("threshold"),
            "smoothing": LaunchConfiguration("smoothing"),
            "num_hands": LaunchConfiguration("num_hands"),
            "infer_fps": LaunchConfiguration("infer_fps"),
            "summary_on_change_only":
                LaunchConfiguration("summary_on_change_only"),
            "port": LaunchConfiguration("port"),
            "publish_web": LaunchConfiguration("publish_web"),
        }],
    )
    return LaunchDescription(args + [node])
`;
}

/**
 * scripts/gpu_env.sh — the environment the GPU delegate needs, in one file.
 *
 * The counterpart of x2_yolo_stream/scripts/gpu_env.sh, which does the same
 * for onnxruntime-gpu. Sourced, never executed: everything it does is an
 * export, and a subshell would throw all of it away.
 */
function generateGpuEnvSh(): string {
  return `# Sourced, not executed. Puts the GPU-capable mediapipe and the GL/CUDA
# libraries where a process started by systemd --user (or by hand) will find
# them.
#
#   . "$(dirname "$0")/gpu_env.sh"
#
# It is its own file, not lines inside start_gesture.sh, because 'ros2 launch'
# and a bare 'ros2 run' need it too, and the equivalent in x2_yolo_stream was
# inline once: the second node to be written was launched a different way and
# quietly got the CPU build for weeks.
#
# Everything here is an export that must be set BEFORE python starts. The
# delegate parameter cannot compensate for a missing libEGL from inside the
# process -- that is why this is a shell file and not more Python.

# The stock PyPI mediapipe has no aarch64 build at all, so on the robot the
# wheel comes from elsewhere and this account has no sudo. Unpack rather than
# install, exactly as the yolo package does with onnxruntime-gpu: PYTHONPATH
# precedes the site directories, so this shadows any CPU mediapipe for this
# process and nothing else. Undo by deleting the directory.
#
#   mkdir -p ~/mediapipe-gpu && cd ~/mediapipe-gpu
#   pip3 download mediapipe --no-deps -d . \\
#       --index-url https://pypi.jetson-ai-lab.io/jp6/cu129
#   python3 -m zipfile -e mediapipe-*.whl pkg
#
# Leave X2_GESTURE_MEDIAPIPE unset and nothing is shadowed: whatever
# 'pip show mediapipe' points at is what runs.
MP_GPU="\${X2_GESTURE_MEDIAPIPE:-}"
if [ -n "$MP_GPU" ] && [ -d "$MP_GPU" ]; then
  export PYTHONPATH="$MP_GPU\${PYTHONPATH:+:$PYTHONPATH}"
  echo "mediapipe from $MP_GPU"
elif [ -n "$MP_GPU" ]; then
  echo "X2_GESTURE_MEDIAPIPE=$MP_GPU does not exist -- using the installed mediapipe"
fi

# Neither systemd --user nor cron reads ~/.bashrc, so a node that finds libEGL
# in an interactive shell can fail to at boot and fall back to the CPU with one
# line of log to say so. The tegra directories are where JetPack puts the real
# EGL/GLES implementation; adding them costs nothing when they are already on
# the loader path.
for _d in /usr/lib/aarch64-linux-gnu/tegra-egl \\
          /usr/lib/aarch64-linux-gnu/tegra \\
          /usr/local/cuda/lib64; do
  [ -d "$_d" ] && case ":\${LD_LIBRARY_PATH:-}:" in
    *":$_d:"*) ;;
    *) export LD_LIBRARY_PATH="$_d\${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
done
unset _d

# The delegate opens an EGL display, and on a headless box that means the
# device platform rather than an X server. Naming the vendor library keeps a
# stray Mesa ICD from being picked ahead of NVIDIA's, which fails in a way that
# reads as "no GPU" rather than as a configuration mistake.
if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="\${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
fi

# Check by hand, when the node says it is on the CPU and you expected otherwise:
#
#   python3 -c "import ctypes; ctypes.CDLL('libEGL.so.1'); print('EGL ok')"
#   python3 -c "import mediapipe; print(mediapipe.__version__, mediapipe.__file__)"
`;
}

function generateStartSh(): string {
  return `#!/bin/bash
# Boot entry point for the gesture node.
#
# Run by the systemd --user unit scripts/x2-gesture.service, and by hand.
#
# Either way the process gets almost no environment -- systemd --user does not
# read ~/.bashrc -- so everything the node needs is set up explicitly here
# rather than inherited from a shell that will not be there. Runs from /tmp
# because the robot's FastDDS profile writes fastdds.log into the working
# directory.
#
# Under the unit:  systemctl --user status x2-gesture
#                  journalctl --user -u x2-gesture -f
# By hand:         tail -f /tmp/x2_gesture.log

# No \`set -u\`: ROS's own setup.bash reads unbound variables and would abort.

# X2_YOLO_WS, the same variable start_yolo_stream.sh and start_greeter.sh read:
# the workspace setup.sh exports it once from its own location, and all four
# packages follow. The default below is only for running this script bare.
WS="\${X2_YOLO_WS:-${WORKSPACE}}"
LOG="\${X2_GESTURE_LOG:-/tmp/x2_gesture.log}"
MODEL="\${X2_GESTURE_MODEL:-\$WS/models/gesture_model.json}"
LANDMARKER="\${X2_GESTURE_LANDMARKER:-\$WS/models/hand_landmarker.task}"
# Deliberately "cpu" and not "auto". The GPU on this box is x2_yolo_stream's:
# it gets ~10x there on a much larger model, where MediaPipe's delegate is
# worth 2-3x on a landmarker whose cost is mostly the frame upload. Two nodes
# contending for one GPU to accelerate the cheaper of them is the wrong trade.
# X2_GESTURE_DELEGATE=gpu moves it across, and gpu_env.sh below is exactly
# what makes that work.
DELEGATE="\${X2_GESTURE_DELEGATE:-cpu}"
# 10 fps is comfortable on the CPU alongside x2_yolo_stream. If you do move
# this node to the GPU, raise it -- a cap chosen for the CPU throws most of
# the speedup away.
INFER_FPS="\${X2_GESTURE_INFER_FPS:-10.0}"
THRESHOLD="\${X2_GESTURE_THRESHOLD:-0.75}"
# Hands detected and classified, 1..4. Each one in frame costs another
# featurize plus another forward pass inside the INFER_FPS budget, so if
# inference_ms climbs on a crowded frame, this is the knob to bring down.
NUM_HANDS="\${X2_GESTURE_NUM_HANDS:-4}"
# Web view. 8081 belongs to x2_yolo_stream.
PORT="\${X2_GESTURE_PORT:-8082}"

HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

if [ -z "\$X2_GESTURE_FOREGROUND" ]; then
  exec >>"\$LOG" 2>&1
fi

# One instance per port: a boot entry and a manual start must not both publish
# contradictory gestures onto the same topic. Keyed by port so a second
# instance watching another camera on another port is still possible.
LOCK="/tmp/x2_gesture.\$PORT.lock"
exec 9>"\$LOCK"
if ! flock -n 9; then
  [ -n "\$X2_GESTURE_QUIET" ] || \\
    echo "=== \$(date -Is) already running, not starting a second copy ==="
  exit 0
fi

echo "=== start_gesture \$(date -Is) ==="

# GPU mediapipe + the GL and CUDA library paths. None of this can be set from
# inside the process, so it has to happen before python starts. Sourced after
# the redirect above, unlike x2_yolo_stream's copy, so that what it reports
# lands in this session's log rather than on whatever terminal ran the script.
# shellcheck source=gpu_env.sh
. "\$HERE/gpu_env.sh"

source /opt/ros/humble/setup.bash
if [ -f "\$WS/install/setup.bash" ]; then
  source "\$WS/install/setup.bash"
else
  echo "workspace not built at \$WS/install -- run:"
  echo "  cd \$WS && colcon build --packages-select ${PACKAGE_NAME}"
  exit 1
fi

if [ ! -f "\$MODEL" ]; then
  echo "no gesture model at \$MODEL -- copy gesture_model.json from the export"
  exit 1
fi
if [ ! -f "\$LANDMARKER" ]; then
  echo "no hand landmarker at \$LANDMARKER -- fetch it with:"
  echo "  curl -o \$LANDMARKER https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  exit 1
fi

# Without this profile a fresh participant cannot see topics published from the
# other SoCs, which includes every camera.
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/data/home/agi/.aima/env/ros_dds_configuration.xml
export ROS_DOMAIN_ID="\${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=0

cd /tmp || exit 1

echo "--- launching \$(date -Is) ---"

# Under systemd: exec, so the unit tracks the real process and Restart=always
# does the supervising. A wrapper that loops internally looks permanently
# healthy to systemd even while the node crashes every ten seconds.
if [ -n "\$X2_GESTURE_FOREGROUND" ]; then
  exec ros2 run ${PACKAGE_NAME} gesture_node \\
    --ros-args \\
    -p "model_path:=\$MODEL" \\
    -p "landmarker_path:=\$LANDMARKER" \\
    -p "delegate:=\$DELEGATE" \\
    -p "infer_fps:=\$INFER_FPS" \\
    -p "threshold:=\$THRESHOLD" \\
    -p "num_hands:=\$NUM_HANDS" \\
    -p "port:=\$PORT"
fi

# Standalone: at boot the camera stack may still be coming up; the node waits
# for its topic on its own, but if it exits for any reason keep bringing it
# back.
while true; do
  ros2 run ${PACKAGE_NAME} gesture_node \\
    --ros-args \\
    -p "model_path:=\$MODEL" \\
    -p "landmarker_path:=\$LANDMARKER" \\
    -p "delegate:=\$DELEGATE" \\
    -p "infer_fps:=\$INFER_FPS" \\
    -p "threshold:=\$THRESHOLD" \\
    -p "num_hands:=\$NUM_HANDS" \\
    -p "port:=\$PORT"
  echo "node exited with \$? -- restarting in 10s"
  sleep 10
  echo "--- launching \$(date -Is) ---"
done
`;
}

function generateServiceUnit(): string {
  return `# systemd --user unit for the gesture node.
#
# Install (no root needed, provided lingering is on for the account):
#
#   loginctl enable-linger agi
#   mkdir -p ~/.config/systemd/user
#   cp ${WORKSPACE}/src/${PACKAGE_NAME}/scripts/x2-gesture.service \\
#      ~/.config/systemd/user/
#   systemctl --user daemon-reload
#   systemctl --user enable --now x2-gesture
#
#   systemctl --user status x2-gesture
#   journalctl --user -u x2-gesture -f
#
# Lingering is what makes this start at boot rather than at login. Without it
# the unit still works, but only while a session is open.

[Unit]
Description=X2 hand-gesture classification
# The cameras live on other SoCs and are reached over DDS, so there is nothing
# useful to do before the network is up.
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# FOREGROUND hands restarts and logging to systemd: the wrapper execs the node
# instead of looping, and writes to the journal instead of /tmp.
Environment=X2_GESTURE_FOREGROUND=1
ExecStart=${WORKSPACE}/src/${PACKAGE_NAME}/scripts/start_gesture.sh
Restart=always
RestartSec=10
# No TensorRT engine build here -- MediaPipe loads in a couple of seconds.
TimeoutStartSec=60
# The camera stack may not be up when this first runs. The node waits for its
# topic and then the unit restarts it if it gives up, so early failures are
# expected and must not trip the start-limit burst logic.
StartLimitIntervalSec=0

[Install]
WantedBy=default.target
`;
}

function generatePackageReadme(bundle: ModelBundle): string {
  const classes = bundle.classes.join(', ');
  return `# ${PACKAGE_NAME}

Publishes hand gestures seen on an X2 camera topic. Exported from Gesture Atlas
${bundle.createdAt}, ${bundle.classes.length} classes: ${classes}.
Validation accuracy at export ${(bundle.valAccuracy * 100).toFixed(1)}%.

## Install

One of four packages in \`x2_yolo_plus_ws\`, next to \`x2_yolo_stream\`,
\`x2_greeter\`, \`x2_lidar_proximity\` and the \`x2_dashboard\` that renders all of
them on one page. **The normal path is the workspace's \`setup.sh\`, not these
commands** — copy the folder to the robot and run it:

    cd ~/x2_yolo_plus_ws && ./setup.sh

That builds every package, fetches the landmarker, and brings the nodes up. What
follows is the same thing by hand, for when something in the middle has gone
wrong:

    cp gesture_model.json ~/x2_yolo_plus_ws/models/
    curl -o ~/x2_yolo_plus_ws/models/hand_landmarker.task \\
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

    pip install mediapipe

    cd ~/x2_yolo_plus_ws && colcon build --packages-select ${PACKAGE_NAME}
    source install/setup.bash

\`pip install mediapipe\` is the step most likely to fail on a Jetson: there is no
reliable aarch64 manylinux wheel. Check it before anything else.

To drop this package into an existing workspace instead — \`x2_yolo_ws\`, say —
unzip it there and build the one package. Nothing in it depends on a workspace
of its own:

    unzip -o x2_gesture.zip -d <workspace>/src/
    cd <workspace> && colcon build --packages-select ${PACKAGE_NAME}

Everything under \`src/${PACKAGE_NAME}/\` is generated by Gesture Atlas. Editing it
in place works right up until the next export overwrites it — change
\`src/lib/rospkg.ts\` in the web app instead.

Re-exporting a retrained model means replacing both \`gesture_model.json\` and this
package together — the golden-sample check exists precisely to stop you running
one from a different export than the other.

## Run

    ros2 launch ${PACKAGE_NAME} ${PACKAGE_NAME}.launch.py

## Web view

Starts with the node, in the same process, on **port 8082** — 8081 belongs to
\`x2_yolo_stream\`.

    http://localhost:8082/

It shows the annotated camera, the committed gesture, and a bar per class. The
bars are the point: they separate "the model is unsure" from "the threshold is
too high", which the topics alone make you squint at. A bar that sits at 0.6
while \`threshold\` is 0.75 means lower the threshold or record more samples —
the topic just says nothing at all.

| Route | |
|---|---|
| \`/\` | the page |
| \`/stream\` | MJPEG, annotated |
| \`/snapshot\` | one JPEG |
| \`/state\` | the same numbers as JSON |
| \`/healthz\` | 200 once frames are arriving, 503 before |

\`publish_web:=false\` turns it off; \`port:=NNNN\` moves it. If the port is already
taken the node logs the error and keeps publishing — losing the view is not
worth killing gesture detection over.

Or under systemd, which is how it should come up on the robot — see the
instructions at the top of \`scripts/x2-gesture.service\`.

On a dev machine with a local webcam and no robot:

    ros2 run ${PACKAGE_NAME} gesture_node --ros-args -p camera_device:=0 \\
      -p model_path:=./gesture_model.json \\
      -p landmarker_path:=./hand_landmarker.task

## Topics

| Topic | Rate | What |
|---|---|---|
| \`/gesture/detections\` | every cycle | full JSON: all class scores, smoothed and instantaneous |
| \`/gesture/summary\` | on change + 5s heartbeat | committed label, confidence, how long it has been held |
| \`/gesture/text\` | on change | one sentence |

All three are \`std_msgs/String\`. The JSON topics put \`description\` first, so
the head of an echo is the answer:

    ros2 topic echo /gesture/text
    ---
    data: peace, right hand, held 1.2s

To act on gestures from a behaviour node, subscribe to \`/gesture/summary\` and
read \`gesture\`. It is null when no hand is visible or nothing clears the
threshold. Use \`/gesture/detections\` only if you want the raw distribution —
it is published at the full cycle rate and will wake you ten times a second.

## More than one hand

\`num_hands\` is the one knob: how many hands the landmarker looks for and this
node classifies, \`1\` to \`4\`. It is a launch argument, so it is retunable per
scene without touching code:

    ros2 launch ${PACKAGE_NAME} ${PACKAGE_NAME}.launch.py num_hands:=2

Every hand is classified, and each one gets its own smoothing window — a
shared window would average a left fist and a right peace into a distribution
neither hand is making. A hand that leaves the frame drops its window.

\`gesture\`, \`handedness\` and \`confidence\` stay what they always were: the
**leading hand**, meaning the strongest one that cleared the threshold. So a
one-hand subscriber needs no changes, and a second hand resting in frame
cannot outvote the one making a gesture. Alongside them, \`per_hand\` on
\`/gesture/detections\` and \`/gesture/summary\` is the full list in detection
order, each entry with its own \`handedness\`, \`gesture\` and \`scores\`:

    ros2 topic echo /gesture/text
    ---
    data: fist, left hand, 0.91 + peace, right hand, 0.88, held 1.2s

\`hands\` is how many were found this cycle, \`num_hands\` the ceiling they were
allowed — equal values mean a further hand would have been dropped. At
\`num_hands:=1\` all of this collapses to the old behaviour, with \`per_hand\` a
one-element list.

Cost: hands are classified one at a time inside a single \`infer_fps\` budget,
so watch \`inference_ms\` when raising this — on the CPU path each extra hand
in frame is another featurize plus another forward pass.

## Key parameters

| Param | Default | Notes |
|---|---|---|
| \`image_topic\` | \`""\` | auto-picks the first available X2 camera |
| \`camera_device\` | \`-1\` | \`>= 0\` reads a local webcam instead — dev only |
| \`delegate\` | \`"cpu"\` | \`gpu\` / \`auto\` move landmarking to the GPU, see below |
| \`threshold\` | \`0.75\` | smoothed confidence needed to commit a label |
| \`smoothing\` | \`5\` | frames averaged per hand; higher is steadier and slower |
| \`num_hands\` | \`4\` | hands detected and classified, \`1\`..\`4\` — see above |
| \`infer_fps\` | \`10.0\` | the knob that matters, see below |
| \`port\` | \`8082\` | web view; 8081 is \`x2_yolo_stream\` |
| \`publish_web\` | \`true\` | \`false\` skips the HTTP server and JPEG encoding |

## Which processor this runs on

Landmarking runs on the CPU, via XNNPACK. That is the default and it is a
decision, not a missing feature: the Orin has one GPU, \`x2_yolo_stream\` gets
about 10x out of it on models many times this size, and MediaPipe's GPU
delegate is worth roughly 2-3x on a hand landmarker whose cost is largely the
per-frame upload from CPU memory. Two processes contending for one GPU in
order to speed up the cheaper of them is the wrong trade. So the machine is
divided: **YOLO on the GPU, hands on the CPU.**

\`infer_fps\` is what you tune if the box gets tight.

### Moving it to the GPU anyway

Everything needed is in place, and one parameter switches it:

    ros2 launch x2_gesture x2_gesture.launch.py delegate:=gpu
    X2_GESTURE_DELEGATE=gpu ./scripts/start_gesture.sh

Worth doing if \`x2_yolo_stream\` is not running, if this node has the box to
itself, or simply to measure. Two things to know first.

**The environment has to be right before python starts**, which no ROS
parameter can arrange from the inside:

    . src/x2_gesture/scripts/gpu_env.sh
    ros2 launch x2_gesture x2_gesture.launch.py delegate:=gpu

\`scripts/start_gesture.sh\` and the systemd unit already source it; a bare
\`ros2 run\` or \`ros2 launch\` in a fresh shell does not.

**MediaPipe aborts instead of raising.** A GPU delegate that cannot run fails
as a CHECK failure inside C++, which calls \`abort()\` — no Python exception,
nothing to catch, and without care the node would die at startup rather than
fall back. So \`runtime.py\` verifies a delegate in a child process before
building it here, and the crash lands on the child:

    hand_landmarker.task: gpu delegate unusable (crashed with signal 6:
      F0000 ... Check failed: ...); trying next

That check costs a few seconds of startup, only on the GPU path, and
\`probe_delegate:=false\` skips it if you prefer the crash.

Whichever delegate it ends up on is stated in the log, in the web view's header
line, and as \`delegate\` on every \`/gesture/detections\` message — a silent
fallback is the whole problem, so it is reported three times. Asking for the
GPU and getting the CPU also logs a warning.

To compare honestly, run it both ways and read \`inference_ms\` off the web
view. Those are real frames with a real hand in them, which no synthetic
benchmark here would be:

    X2_GESTURE_DELEGATE=cpu X2_GESTURE_FOREGROUND=1 ./scripts/start_gesture.sh
    X2_GESTURE_DELEGATE=gpu X2_GESTURE_FOREGROUND=1 ./scripts/start_gesture.sh

Then raise \`infer_fps\` if you keep the GPU: 10 was chosen for the CPU cost, and
leaving it there throws most of the speedup away. Past that point the JPEG
decode at the front and the encode for the web view are the budget —
\`publish_web:=false\` removes the encode entirely when a behaviour node rather
than a person is the consumer.

\`runtime.py\` also runs on its own, which is the quickest way to find out
whether a robot could do this at all:

    python3 src/x2_gesture/x2_gesture/runtime.py
    python3 src/x2_gesture/x2_gesture/runtime.py --probe models/hand_landmarker.task gpu

### If the GPU is refused

The log says which part failed. In order of likelihood:

* **\`mediapipe not importable\`** — there is no aarch64 wheel on PyPI. On the
  robot it comes from \`pypi.jetson-ai-lab.io\`; the header of
  \`scripts/gpu_env.sh\` has the unpack-don't-install recipe, which needs no
  sudo and is undone by deleting a directory.
* **\`libEGL … cannot be loaded\`** — the GL libraries are not on the loader
  path. That is what \`gpu_env.sh\` fixes; if it did not, check that
  \`/usr/lib/aarch64-linux-gnu/tegra-egl\` exists.
* **\`crashed with signal 6\`** with \`unsupported ImageFrame format\` — should not
  happen, since the GPU path is fed 4-channel SRGBA on purpose, but it is what
  3-channel frames do to a GPU landmarker. Worth knowing if you call
  \`runtime.to_image\` yourself.
* **\`crashed with signal 6\`** otherwise — mediapipe and GL are both there but
  no EGL display could be created, usually a headless session with no NVIDIA
  EGL vendor file. \`gpu_env.sh\` names one explicitly.
* **\`gpu delegate unusable\`** with a message about a missing op — this wheel
  was built without GPU support. Only a different wheel fixes that.

Body-pose keypoints cannot substitute for any of this: they have 17 joints and
no fingers.

## If it exits at startup

The node runs a golden-sample check before creating any publisher. If the
featurizer here disagrees with the browser that trained the model, it exits
rather than joining the graph — a mismatch produces confident wrong answers,
not errors, and a behaviour node downstream cannot tell the difference.

    Golden-sample check failed — features do not match the browser.

means \`gesture_model.json\` and this package came from different exports. Re-export
both together.
`;
}

/** The complete package, ready to zip. */
export function generateGesturePackage(bundle: ModelBundle): ZipEntry[] {
  const p = (rel: string) => `${PACKAGE_NAME}/${rel}`;
  return [
    { path: p('package.xml'), content: generatePackageXml() },
    { path: p('setup.py'), content: generateSetupPy() },
    { path: p('setup.cfg'), content: generateSetupCfg() },
    { path: p(`resource/${PACKAGE_NAME}`), content: '' },
    { path: p(`launch/${PACKAGE_NAME}.launch.py`), content: generateLaunchPy() },
    { path: p('scripts/start_gesture.sh'), content: generateStartSh(), mode: 0o755 },
    // Not 0o755: sourced, never executed. An executable bit here invites
    // someone to run it in a subshell, where every export is discarded and
    // the node comes up on the CPU with nothing to explain why.
    { path: p('scripts/gpu_env.sh'), content: generateGpuEnvSh() },
    { path: p('scripts/x2-gesture.service'), content: generateServiceUnit() },
    { path: p(`${PACKAGE_NAME}/__init__.py`), content: '' },
    { path: p(`${PACKAGE_NAME}/gesture.py`), content: generateGestureCore() },
    { path: p(`${PACKAGE_NAME}/runtime.py`), content: generateRuntimePy() },
    { path: p(`${PACKAGE_NAME}/node.py`), content: generateNodePy() },
    { path: p(`${PACKAGE_NAME}/web.py`), content: generateWebPy() },
    { path: p('README.md'), content: generatePackageReadme(bundle) },
  ];
}
