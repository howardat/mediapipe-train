import { openDB, type IDBPDatabase } from 'idb';
import type { GestureClass, Sample } from './types';

const DB_NAME = 'gesture-atlas';
const DB_VERSION = 1;

let dbPromise: Promise<IDBPDatabase> | null = null;

function db() {
  if (!dbPromise) {
    dbPromise = openDB(DB_NAME, DB_VERSION, {
      upgrade(database) {
        database.createObjectStore('classes', { keyPath: 'id' });
        const samples = database.createObjectStore('samples', { keyPath: 'id' });
        samples.createIndex('classId', 'classId');
      },
    });
  }
  return dbPromise;
}

const uid = () => crypto.randomUUID();

export async function listClasses(): Promise<GestureClass[]> {
  const all = (await (await db()).getAll('classes')) as GestureClass[];
  return all.sort((a, b) => a.createdAt - b.createdAt);
}

export async function addClass(name: string): Promise<GestureClass> {
  const cls: GestureClass = { id: uid(), name, createdAt: Date.now() };
  await (await db()).put('classes', cls);
  return cls;
}

export async function renameClass(id: string, name: string): Promise<void> {
  const conn = await db();
  const cls = (await conn.get('classes', id)) as GestureClass | undefined;
  if (cls) await conn.put('classes', { ...cls, name });
}

export async function deleteClass(id: string): Promise<void> {
  const conn = await db();
  const tx = conn.transaction(['classes', 'samples'], 'readwrite');
  await tx.objectStore('classes').delete(id);
  const index = tx.objectStore('samples').index('classId');
  for (const key of await index.getAllKeys(id)) {
    await tx.objectStore('samples').delete(key);
  }
  await tx.done;
}

export async function addSamples(samples: Omit<Sample, 'id' | 'createdAt'>[]): Promise<void> {
  const conn = await db();
  const tx = conn.transaction('samples', 'readwrite');
  const now = Date.now();
  for (const s of samples) {
    await tx.store.put({ ...s, id: uid(), createdAt: now });
  }
  await tx.done;
}

export async function listSamples(): Promise<Sample[]> {
  return (await (await db()).getAll('samples')) as Sample[];
}

export async function samplesForClass(classId: string): Promise<Sample[]> {
  return (await (await db()).getAllFromIndex('samples', 'classId', classId)) as Sample[];
}

export async function countsByClass(): Promise<Record<string, number>> {
  const counts: Record<string, number> = {};
  for (const s of await listSamples()) {
    counts[s.classId] = (counts[s.classId] ?? 0) + 1;
  }
  return counts;
}

export async function clearSamplesForClass(classId: string): Promise<void> {
  const conn = await db();
  const tx = conn.transaction('samples', 'readwrite');
  for (const key of await tx.store.index('classId').getAllKeys(classId)) {
    await tx.store.delete(key);
  }
  await tx.done;
}
