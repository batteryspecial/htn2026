import { stateClass } from '../../config/constants';
import type { BehaviorView } from '../../contracts/behavior';
import { Badge } from '../common/Badge';

/**
 * What the pipeline is actually running, straight off the per-frame StateView.
 *
 * This is the panel the legacy `/spec` route cannot serve: it maps everything
 * to one `track` behaviour, so `watch`, `count_line`, `privacy`, `pan_to` and
 * `pose_trigger` never appear. On the native path they do.
 */
export function BehaviorsPanel({ behaviors, onDrop }: {
  behaviors: BehaviorView[];
  onDrop: (id: string) => void;
}) {
  return (
    <div className="panel-scroll">
      <ul className="rows">
        {!behaviors.length && <li className="empty-row">Nothing running.</li>}
        {behaviors.map((b) => (
          <li key={b.id}>
            <Badge kind={stateClass(b.state)} title={b.detail ?? undefined}>{b.state}</Badge>
            <span className="row-kind">{b.kind}</span>
            <span className="row-main" title={b.detail ?? b.spec}>
              {b.label || b.spec}
              {b.matches > 0 && <span className="row-sub">{`  ·  ${b.matches} match`}</span>}
            </span>
            <button
              type="button"
              className="row-drop"
              onClick={() => onDrop(b.id)}
              title={`DELETE /behaviors/${b.id}`}
            >
              DROP
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
