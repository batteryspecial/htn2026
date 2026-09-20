import type { TimerView } from '../../hooks/useRetaskRun';
import { seconds } from '../../utils/format';

/**
 * The next-train display. The big number is instruction sent -> acquired,
 * which is the metric the project is built around; compile time is the small
 * line underneath.
 */
export function RetaskTimer({ timer }: { timer: TimerView }) {
  return (
    <div className="timer-wrap">
      <div className={`timer ${timer.running ? 'running' : ''} ${timer.failed ? 'failed' : ''}`}>
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
