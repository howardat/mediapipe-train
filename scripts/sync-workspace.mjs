// Refreshes the generated x2_gesture package inside x2_yolo_plus_ws/, the folder
// you copy to the robot, and drops the trained model into its models/.
//
//   npm run sync:ws                  # from export/gesture_model.json
//   npm run sync:ws -- --force       # delete files the generator did not emit
//
// Scope matters here. x2_yolo_plus_ws/ is YOUR workspace -- x2_yolo_stream,
// x2_greeter, x2_lidar_proximity, x2_dashboard and 74 MB of ONNX are hand-built
// and none of this script's business. Only two paths are written:
//
//   src/x2_gesture/           entirely generated; --force may prune it
//   models/gesture_model.json overwritten, never pruned
//
// The stale-file scan is deliberately confined to src/x2_gesture/. Widening it
// to the workspace root would put --force in a position to delete the other
// three packages, which is a mistake worth making structurally impossible.
//
// src/x2_gesture/ is build output, not source. Edit src/lib/rospkg.ts and re-run
// this; anything typed into the package copy is lost on the next sync, which is
// what --force refuses to do quietly.
//
// hand_landmarker.task is NOT synced -- a 7 MB binary that setup.sh fetches on
// the robot, one hop from the internet rather than two.
//
// The rest of export/ is rewritten alongside it -- x2_gesture.zip, README.md and
// infer.py all come from one set of constants, so the zip a user downloads, the
// instructions beside it and the tree committed here cannot disagree.
// gesture_model.json and dataset.csv are left alone: only the app produces those.
import { execFileSync } from 'node:child_process';
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';

const PACKAGE_NAME = 'x2_gesture';
const EXPORT_DIR = 'export';
const WORKSPACE_DIR = 'x2_yolo_plus_ws';
const PACKAGE_DIR = join(WORKSPACE_DIR, 'src', PACKAGE_NAME);
const MODEL_NAME = 'gesture_model.json';

const force = process.argv.includes('--force');

if (!existsSync(join(EXPORT_DIR, 'gesture_model.json'))) {
  console.error(
    `no ${EXPORT_DIR}/gesture_model.json -- export a trained model from the app first`,
  );
  process.exit(1);
}

// Bundling matches scripts/verify-package.mjs: the generator is TypeScript and
// imports the same feature constants the browser uses, which is the only reason
// the workspace copy can be trusted to agree with a model trained in the app.
const tmp = mkdtempSync(join(tmpdir(), 'gesture-sync-'));
const bundleTs = (src, name) => {
  const file = join(tmp, name);
  execFileSync('npx', ['esbuild', '--bundle', '--format=esm', '--platform=node',
                       '--log-level=error', `--outfile=${file}`, src],
               { encoding: 'utf8' });
  return file;
};

// Paths are relative to a workspace's src/ -- that is what x2_gesture.zip
// unpacks into -- so they need no rebasing to land in x2_yolo_plus_ws/src/.
const manifest = JSON.parse(
  execFileSync('node', [bundleTs('scripts/emit-package.ts', 'pkg.mjs'), EXPORT_DIR],
               { encoding: 'utf8' }),
);

execFileSync('node', [bundleTs('scripts/emit-export.ts', 'exp.mjs'), EXPORT_DIR],
             { encoding: 'utf8' });

const emitted = new Set(manifest.map((e) => e.path));

// Files the generator no longer emits have to go, or a renamed module lingers
// and colcon installs both. Everything under src/x2_gesture/ is generated, so
// anything unaccounted for is either stale or a hand edit -- and the second case
// is worth stopping for.
//
// This walks PACKAGE_DIR, never WORKSPACE_DIR. The other three packages and the
// ONNX models are not in the manifest and must never be candidates for deletion.
const walk = (dir) =>
  existsSync(dir)
    ? readdirSync(dir).flatMap((name) => {
        const full = join(dir, name);
        return statSync(full).isDirectory() ? walk(full) : [full];
      })
    : [];

const IGNORED = /(^|\/)\.DS_Store$|(^|\/)__pycache__\//;

const existing = walk(PACKAGE_DIR)
  .map((f) => relative(join(WORKSPACE_DIR, 'src'), f))
  .filter((f) => !IGNORED.test(f));

const unknown = existing.filter((f) => !emitted.has(f));
if (unknown.length && !force) {
  console.error(
    `${PACKAGE_DIR} holds ${unknown.length} file(s) this generator does not emit:\n` +
      unknown.map((f) => `  ${f}`).join('\n') +
      '\n\nThey are stale output or hand edits. Re-run with --force to delete them.',
  );
  process.exit(1);
}

for (const rel of unknown) rmSync(join(WORKSPACE_DIR, 'src', rel));

// The classifier travels with the folder: "copy it to the robot and run
// setup.sh" has to include the one file that cannot be regenerated there. Not
// part of the manifest, so the prune above can never consider it.
mkdirSync(join(WORKSPACE_DIR, 'models'), { recursive: true });
writeFileSync(join(WORKSPACE_DIR, 'models', MODEL_NAME),
              readFileSync(join(EXPORT_DIR, MODEL_NAME)));

for (const { path, content, mode } of manifest) {
  const full = join(WORKSPACE_DIR, 'src', path);
  mkdirSync(dirname(full), { recursive: true });
  writeFileSync(full, content);
  chmodSync(full, mode ?? 0o644);
}

// __pycache__ from a previous colcon build shadows a module that has since been
// regenerated; Python will happily import the stale .pyc. PACKAGE_DIR only --
// the other packages' caches are theirs.
for (const file of walk(PACKAGE_DIR)) {
  if (file.includes('__pycache__')) rmSync(dirname(file), { recursive: true, force: true });
}

const bundle = JSON.parse(readFileSync(join(EXPORT_DIR, MODEL_NAME), 'utf8'));
console.log(
  `${PACKAGE_DIR}/: ${manifest.length} files` +
    (unknown.length ? `, ${unknown.length} removed` : ''),
);
console.log(`${WORKSPACE_DIR}/models/${MODEL_NAME}: updated`);
console.log(`${EXPORT_DIR}/: x2_gesture.zip, README.md, infer.py regenerated`);
console.log(
  `model: ${bundle.classes.length} classes (${bundle.classes.join(', ')}), ` +
    `feature spec v${bundle.featureSpec.version}, exported ${bundle.createdAt}`,
);
console.log(`\nready to deploy:\n  scp -r ${WORKSPACE_DIR} agi@<robot>:~/`);
console.log(`  ssh agi@<robot> '~/${WORKSPACE_DIR}/setup.sh'`);
