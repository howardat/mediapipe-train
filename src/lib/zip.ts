/**
 * A ZIP writer in ~90 lines, storing entries uncompressed.
 *
 * The exported package is about 50 KB of text. Deflate would save maybe 35 KB
 * and cost a dependency, so entries are stored (method 0) and the whole format
 * reduces to: a local header per file, a central directory, and an end record.
 *
 * Verified against Python's zipfile in scripts/package.py — an entirely
 * independent implementation, which is the point.
 */

export interface ZipEntry {
  path: string;
  content: string;
  /** Unix mode. Defaults to 0o644; pass 0o755 for shell scripts. */
  mode?: number;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// Fixed at 1980-01-01 so identical input yields identical bytes. A timestamp
// here would make every export differ from the last for no reason.
const DOS_TIME = 0;
const DOS_DATE = 33;

export function zipStoreBytes(entries: ZipEntry[]): Uint8Array {
  const enc = new TextEncoder();
  const chunks: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  for (const entry of entries) {
    const name = enc.encode(entry.path);
    const data = enc.encode(entry.content);
    const crc = crc32(data);

    const local = new Uint8Array(30 + name.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true); // local file header signature
    lv.setUint16(4, 20, true); // version needed to extract
    lv.setUint16(6, 0x0800, true); // UTF-8 filename flag
    lv.setUint16(8, 0, true); // method: stored
    lv.setUint16(10, DOS_TIME, true);
    lv.setUint16(12, DOS_DATE, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, data.length, true); // compressed size
    lv.setUint32(22, data.length, true); // uncompressed size
    lv.setUint16(26, name.length, true);
    lv.setUint16(28, 0, true); // extra field length
    local.set(name, 30);

    const dir = new Uint8Array(46 + name.length);
    const dv = new DataView(dir.buffer);
    dv.setUint32(0, 0x02014b50, true); // central directory signature
    dv.setUint16(4, 0x031e, true); // version made by: UNIX, 3.0
    dv.setUint16(6, 20, true);
    dv.setUint16(8, 0x0800, true);
    dv.setUint16(10, 0, true);
    dv.setUint16(12, DOS_TIME, true);
    dv.setUint16(14, DOS_DATE, true);
    dv.setUint32(16, crc, true);
    dv.setUint32(20, data.length, true);
    dv.setUint32(24, data.length, true);
    dv.setUint16(28, name.length, true);
    // External attributes carry the Unix mode in the high 16 bits, which is how
    // `unzip` knows start_gesture.sh is executable.
    dv.setUint32(38, ((entry.mode ?? 0o644) & 0xffff) << 16, true);
    dv.setUint32(42, offset, true);
    dir.set(name, 46);

    chunks.push(local, data);
    central.push(dir);
    offset += local.length + data.length;
  }

  const centralSize = central.reduce((n, c) => n + c.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true); // end of central directory
  ev.setUint16(8, entries.length, true);
  ev.setUint16(10, entries.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);

  const out = new Uint8Array(offset + centralSize + end.length);
  let at = 0;
  for (const c of [...chunks, ...central, end]) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

export function zipStore(entries: ZipEntry[]): Blob {
  // slice() copies into a fresh ArrayBuffer: Blob rejects a view with a byteOffset.
  return new Blob([zipStoreBytes(entries).slice().buffer], { type: 'application/zip' });
}
