import type { TimerView } from '../../hooks/useRetaskRun';
import { seconds } from '../../utils/format';

/**
 * The next-train display. The big number is instruction sent → the objective
 * being live, which is the metric the project is built around.
 *
 * `compact` fits it into a panel's title band instead of giving it a block of
 * its own — the same number, sized for a header rather than a projector.
 */
export function RetaskTimer({ timer, compact = false }: {
  timer: TimerView;
  compact?: boolean;
}) {
  const classes = [
    'timer',
    timer.running ? 'running' : '',
    timer.failed ? 'failed' : '',
  ].filter(Boolean).join(' ');

  if (compact) {
    return (
      <span className="timer-inline">
        <span className={classes}>{seconds(timer.value)}<small>s</small></span>
        <span className="timer-label">{timer.label}</span>
      </span>
    );
  }

  return (
    <div className="timer-wrap">
      <div className={classes}>
        {seconds(timer.value)}
        <small>s</small>
      </div>
      <div className="timer-meta">
        <div className="timer-label">{timer.label}</div>
        <div className="timer-sub">{timer.sub}</div>
      </div>
    </div>
  );
}
