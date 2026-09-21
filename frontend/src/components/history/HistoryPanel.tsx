import type { HistoryEntry } from '../../hooks/useHistory';
import { seconds } from '../../utils/format';

/** The kinds a run installed, or why it has none. */
function tagFor(entry: HistoryEntry): string {
  const behaviors = entry.program?.behaviors;
  if (!behaviors?.length) return entry.ok ? 'reply' : 'error';
  const kinds = [...new Set(behaviors.map((b) => b.kind))];
  return kinds.length === 1 ? kinds[0] : `${behaviors.length} behaviours`;
}

export function HistoryPanel({ entries, onRecall }: {
  entries: HistoryEntry[];
  onRecall: (entry: HistoryEntry) => void;
}) {
  return (
    <ul className="history panel-scroll">
      {!entries.length && <li className="empty-row">Nothing sent yet.</li>}
      {entries.map((entry, i) => (
        <li
          // Two identical instructions can settle at the same time, so the
          // index is the only stable key here.
          // eslint-disable-next-line react/no-array-index-key
          key={`${entry.text}-${i}`}
          className={`item ${entry.ok ? '' : 'bad'}`}
          onClick={() => onRecall(entry)}
        >
          <span className="h-time">{seconds(entry.seconds)}s</span>
          <span className="h-text">{entry.text}</span>
          <span className="h-tag">{tagFor(entry)}</span>
        </li>
      ))}
    </ul>
  );
}
