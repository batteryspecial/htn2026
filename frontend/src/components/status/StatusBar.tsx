import { stateClass } from '../../config/constants';
import type { BehaviorView, Health, TrackView } from '../../contracts/behavior';
import { joinParts, percent } from '../../utils/format';
import { Badge } from '../common/Badge';

/**
 * What the pipeline is doing, in one strip.
 *
 * It sits with the instruction rather than under the video, because it is
 * the answer to "did that work?" — and that question is asked of the thing
 * you just typed, not of the picture.
 *
 * **A behaviour that is ACTIVE with zero matches is not working.** That is
 * why the match count is here and not buried in a panel.
 */
export function StatusBar({ behavior, track, health, behaviorCount }: {
  behavior: BehaviorView | null;
  track: TrackView | null;
  health: Health | null;
  behaviorCount: number;
}) {
  const phase = behavior?.state ?? (health ? health.status.toUpperCase() : null);

  const pipeline = health
    ? joinParts([
      `${Math.round(health.fps)} fps`,
      health.model,
      health.device,
      behaviorCount ? `${behaviorCount} running` : null,
      health.camera_ok === false ? 'NO CAMERA' : null,
    ])
    : 'pipeline unreachable';

  const readout = track
    ? joinParts([
      track.label,
      `cx ${track.cx.toFixed(2)}`,
      `area ${percent(track.area)}`,
      `conf ${track.conf.toFixed(2)}`,
    ])
    : behavior?.matches
      ? `${behavior.matches} match${behavior.matches === 1 ? '' : 'es'}, no track yet`
      : behavior
        ? 'matching nothing — the wording may be wrong'
        : null;

  const warn = !!behavior && !behavior.matches && !track;

  return (
    <div className="statusbar">
      <div className="statusbar-top">
        <Badge kind={stateClass(phase)} title={behavior?.detail ?? undefined}>
          {phase ?? '—'}
        </Badge>
        <span className="statusbar-pipeline">{pipeline}</span>
      </div>
      <div className={`statusbar-readout ${readout ? '' : 'none'} ${warn ? 'warn' : ''}`}>
        {readout || 'no target state'}
      </div>
    </div>
  );
}
