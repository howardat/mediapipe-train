// Prints the generated infer.py to stdout, so CI (or you) can syntax-check the
// Python that ships to the edge device without opening a browser.
import { generateInferPy } from '../src/lib/exporter';

process.stdout.write(generateInferPy());
