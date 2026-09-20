import { ALERT_EVENTS } from '../../config/constants';
import type { PerceptionEvent } from '../../contracts/behavior';
import { clockTime } from '../../utils/format';
import { Panel } from '../common/Panel';

/**
 * Everything the pipeline reports, in its own vocabulary.
 *
 * The stage strip only knows the three events that have an old equivalent;
 * `count_changed`, `crossed`, `armed`, `near` and the rest are shown here in
 * full rather than being flattened into a stage name that does not fit.
 */
export function EventsPanel({ events, snapshotUrl, onClear }: {
  events: PerceptionEvent[];
  snapshotUrl: (eventId: string) => string;
  onClear: () => void;
}) {
  return (
    <Panel
      className="panel-live"
      title="Events"
      aside={
        <>
          <span className="count">{events.length}</span>
          <button type="button" className="title-btn" onClick={onClear}>CLEAR</button>
        </>
      }
    >
      <div className="panel-scroll">
        <ul className="rows">
          {!events.length && <li className="empty-row">Nothing yet.</li>}
          {events.map((ev) => (
            <li key={ev.id} className={ALERT_EVENTS.includes(ev.type) ? 'alert' : ''}>
              <span className="row-time">{clockTime(ev.ts)}</span>
              <span className="row-kind">{ev.type}</span>
              <span className="row-main" title={ev.detail}>{ev.detail || '—'}</span>
              {ev.snapshot_url && (
                <img className="row-shot" src={snapshotUrl(ev.id)} alt="" loading="lazy" />
              )}
            </li>
          ))}
        </ul>
      </div>
    </Panel>
  );
}
