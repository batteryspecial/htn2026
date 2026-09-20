import type { Notice } from '../../hooks/useRetaskRun';

export function Notices({ notices }: { notices: Notice[] }) {
  if (!notices.length) return null;

  return (
    <div className="notices">
      {notices.map((n, i) => (
        // eslint-disable-next-line react/no-array-index-key -- append-only log
        <div key={i} className={`notice ${n.kind}`}>
          {n.path && <span className="path">{n.path}</span>}
          {n.text}
        </div>
      ))}
    </div>
  );
}
