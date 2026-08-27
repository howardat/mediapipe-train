// Generates infer.py and the fixture cases, then checks that the Python
// featurizer reproduces the TypeScript one exactly.
//
//   npm run verify:contract
//
// Run this after touching src/lib/features.ts or the generator in exporter.ts.
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const out = mkdtempSync(join(tmpdir(), 'gesture-contract-'));
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
  const inferPy = join(out, 'infer.py');
  writeFileSync(inferPy, run('node', [bundle('scripts/emit-infer.ts', 'emit.mjs')]));

  const cases = join(out, 'cases.json');
  writeFileSync(cases, run('node', [bundle('scripts/crosscheck.ts', 'cc.mjs')]));

  run(python, ['-m', 'py_compile', inferPy]);
  console.log('infer.py compiles');

  process.stdout.write(run(python, ['scripts/crosscheck.py', inferPy, cases]));

  // Full path: train a real model, export it, and have Python reproduce the
  // browser's predictions from the actual bundle.
  process.stdout.write(run('node', [bundle('scripts/e2e.ts', 'e2e.mjs'), out]));
  process.stdout.write(run(python, ['scripts/e2e.py', out]));
} catch (err) {
  console.error(err.stdout?.toString() ?? '');
  console.error(err.stderr?.toString() ?? err.message);
  process.exit(1);
}
