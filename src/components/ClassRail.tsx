import { useState, type FormEvent } from 'react';
import { MIN_SAMPLES_PER_CLASS } from '../lib/trainer';
import type { GestureClass } from '../lib/types';

interface Props {
  classes: GestureClass[];
  counts: Record<string, number>;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCreate: (name: string) => void;
  onDelete: (id: string) => void;
}

export default function ClassRail({
  classes,
  counts,
  selectedId,
  onSelect,
  onCreate,
  onDelete,
}: Props) {
  const [draft, setDraft] = useState('');

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const name = draft.trim();
    if (!name) return;
    onCreate(name);
    setDraft('');
  };

  const max = Math.max(1, ...Object.values(counts));

  return (
    <div className="panel">
      <header>
        <h2>Gestures</h2>
        <span className="note">{classes.length} classes</span>
      </header>

      <div className="body">
        {classes.length === 0 ? (
          <p className="empty">
            Name a gesture to start. Two classes minimum — a classifier needs something to
            choose between.
          </p>
        ) : (
          <div className="classes">
            {classes.map((cls, i) => {
              const n = counts[cls.id] ?? 0;
              return (
                <div key={cls.id} className="class-row" data-selected={cls.id === selectedId}>
                  {/* Two sibling buttons, never nested — a button inside a
                      role="button" wipes out the row's accessible name. */}
                  <button
                    className="pick"
                    aria-pressed={cls.id === selectedId}
                    onClick={() => onSelect(cls.id)}
                  >
                    <span className="key">{i < 9 ? i + 1 : '·'}</span>
                    <span className="name">{cls.name}</span>
                    <span
                      className="count"
                      data-thin={n < MIN_SAMPLES_PER_CLASS}
                      style={{ opacity: 0.35 + 0.65 * (n / max) }}
                    >
                      {n} {n === 1 ? 'sample' : 'samples'}
                    </span>
                  </button>
                  <button
                    className="kill"
                    aria-label={`Delete ${cls.name}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm(`Delete “${cls.name}” and its ${n} samples?`)) onDelete(cls.id);
                    }}
                  >
                    ✕
                  </button>
                </div>
              );
            })}
          </div>
        )}

        <form className="newclass" onSubmit={submit}>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="new gesture name"
            aria-label="New gesture name"
            maxLength={32}
          />
          <button className="btn" type="submit" disabled={!draft.trim()}>
            Add
          </button>
        </form>
      </div>
    </div>
  );
}
