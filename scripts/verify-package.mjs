// Verifies the generated x2_gesture ROS package.
//
//   npm run verify:package
//
// What this proves: the zip is readable by an independent implementation, the
// manifest is complete, gesture.py's featurizer is byte-identical to the one
// that ships in infer.py, gesture.py still imports with nothing but numpy, and
// every emitted .py parses.
//
// What it does NOT prove: anything about ROS. py_compile parses node.py without
// importing rclpy — it never creates a publisher or puts a message on the wire.
// The topic contract is first tested on the robot.
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const PACKAGE_NAME = 'x2_gesture';

const EXPECTED = [
  `${PACKAGE_NAME}/package.xml`,
  `${PACKAGE_NAME}/setup.py`,
  `${PACKAGE_NAME}/setup.cfg`,
  `${PACKAGE_NAME}/resource/${PACKAGE_NAME}`,
  `${PACKAGE_NAME}/launch/${PACKAGE_NAME}.launch.py`,
  `${PACKAGE_NAME}/scripts/start_gesture.sh`,
  `${PACKAGE_NAME}/scripts/gpu_env.sh`,
  `${PACKAGE_NAME}/scripts/x2-gesture.service`,
  `${PACKAGE_NAME}/${PACKAGE_NAME}/__init__.py`,
  `${PACKAGE_NAME}/${PACKAGE_NAME}/gesture.py`,
  `${PACKAGE_NAME}/${PACKAGE_NAME}/node.py`,
  `${PACKAGE_NAME}/${PACKAGE_NAME}/runtime.py`,
  `${PACKAGE_NAME}/${PACKAGE_NAME}/web.py`,
  `${PACKAGE_NAME}/README.md`,
];

const out = mkdtempSync(join(tmpdir(), 'gesture-package-'));
const run = (cmd, args) => execFileSync(cmd, args, { encoding: 'utf8' });

const bundle = (entry, name) => {
  const file = join(out, name);
  run('npx', ['esbuild', '--bundle', '--format=esm', '--platform=node',
              '--log-level=error',
              // tfjs reaches for node builtins via require(); ESM output needs the shim.
              '--banner:js=import{createRequire}from"module";const require=createRequire(import.meta.url);',
              `--outfile=${file}`, entry]);
  return file;
};

const python = process.env.PYTHON ?? 'python3';

try {
  // Trains a real model on synthetic hands and writes gesture_model.json,
  // infer.py and predictions.json into `out`.
  process.stdout.write(run('node', [bundle('scripts/e2e.ts', 'e2e.mjs'), out]));

  const emitted = run('node', [bundle('scripts/emit-package.ts', 'pkg.mjs'), out]);
  const manifest = join(out, 'manifest.json');
  writeFileSync(manifest, emitted);

  const got = JSON.parse(emitted).map((e) => e.path).sort();
  const want = [...EXPECTED].sort();
  if (JSON.stringify(got) !== JSON.stringify(want)) {
    throw new Error(`manifest mismatch:\n  got  ${got}\n  want ${want}`);
  }
  console.log(`manifest OK -- ${EXPECTED.length} files`);

  process.stdout.write(run(python, ['scripts/package.py', 'zipcheck',
                                    join(out, 'x2_gesture.zip'), manifest]));

  const zip = join(out, 'x2_gesture.zip');
  run(python, ['-c',
               `import zipfile;zipfile.ZipFile(${JSON.stringify(zip)})`
               + `.extractall(${JSON.stringify(out)})`]);

  // py_compile parses without importing, so this passes with no ROS installed.
  for (const rel of [`${PACKAGE_NAME}/${PACKAGE_NAME}/node.py`,
                     `${PACKAGE_NAME}/${PACKAGE_NAME}/web.py`,
                     `${PACKAGE_NAME}/${PACKAGE_NAME}/gesture.py`,
                     `${PACKAGE_NAME}/${PACKAGE_NAME}/runtime.py`,
                     `${PACKAGE_NAME}/launch/${PACKAGE_NAME}.launch.py`,
                     `${PACKAGE_NAME}/setup.py`]) {
    run(python, ['-m', 'py_compile', join(out, rel)]);
  }
  console.log('package python compiles');

  // start_gesture.sh is emitted from a TS template literal, so every $ and \ in
  // it is escaped by hand. bash -n is what catches an escape that got away.
  // Same for gpu_env.sh, which is worse: it is sourced, so a syntax error in
  // it takes down start_gesture.sh rather than failing on its own.
  for (const rel of [`${PACKAGE_NAME}/scripts/start_gesture.sh`,
                     `${PACKAGE_NAME}/scripts/gpu_env.sh`]) {
    run('bash', ['-n', join(out, rel)]);
  }
  console.log('start_gesture.sh and gpu_env.sh parse');

  // gesture.py must stay importable with only numpy, or the parity check below
  // stops being runnable anywhere but the robot.
  const core = readFileSync(
    join(out, `${PACKAGE_NAME}/${PACKAGE_NAME}/gesture.py`), 'utf8');
  for (const forbidden of ['import cv2', 'import mediapipe', 'import rclpy']) {
    if (core.includes(forbidden)) {
      throw new Error(`gesture.py must not contain "${forbidden}"`);
    }
  }
  console.log('gesture.py imports only json + numpy');

  // The core in the package must match the core that ships standalone.
  process.stdout.write(run(python, ['scripts/package.py', 'coreident', out,
                                    join(out, 'infer.py')]));

  process.stdout.write(run(python, ['scripts/package.py', 'parity', out,
                                    join(out, 'gesture_model.json'),
                                    join(out, 'predictions.json')]));
} catch (err) {
  console.error(err.stdout?.toString() ?? '');
  console.error(err.stderr?.toString() ?? err.message);
  process.exit(1);
}
