import { stateClass } from '../../config/constants';
import type { BehaviorView, Health, TrackView } from '../../contracts/behavior';
import type { VideoStream } from '../../hooks/useVideoStream';
import { joinParts, percent } from '../../utils/format';
import { Badge } from '../common/Badge';
import { Panel } from '../common/Panel';

export interface CameraPanelProps {
  video: VideoStream;
  /** Summary of the current objective, shown as a platform sign. */
  objective: string | null;
  behavior: BehaviorView | null;
  track: TrackView | null;
  health: Health | null;
}

export function CameraPanel({ video, objective, behavior, track, health }: CameraPanelProps) {
  const phase = behavior?.state ?? (health ? health.status.toUpperCase() : null);

  const readout = track
    ? joinParts([
      track.label,
      `cx ${track.cx.toFixed(2)}`,
      `area ${percent(track.area)}`,
      `conf ${track.conf.toFixed(2)}`,
    ])
    : behavior?.matches
      ? `${behavior.matches} match(es), no track yet`
      : null;

  const pipeline = health && joinParts([
    `${Math.round(health.fps)} fps`,
    health.model,
    health.device,
    health.behaviors ? `${health.behaviors} behaviour${health.behaviors === 1 ? '' : 's'}` : null,
    health.camera_ok === false ? 'NO CAMERA' : null,
  ]);

  return (
    <Panel
      className="panel-video"
      title="Camera"
      aside={
        <>
          <Badge kind={video.live ? 'pass' : 'fail'}>{video.live ? 'LIVE' : 'OFFLINE'}</Badge>
          <Badge kind={stateClass(phase)} title={behavior?.detail ?? undefined}>
            {phase ?? '—'}
          </Badge>
          <span className="title-note">{pipeline || ''}</span>
          <button type="button" className="title-btn" onClick={video.reconnect}>
            RECONNECT
          </button>
        </>
      }
    >
      <div className={`video-frame ${video.live ? 'live' : ''}`}>
        {video.src && (
          <img
            className="video-feed"
            src={video.src}
            alt=""
            onLoad={video.onLoad}
            onError={video.onError}
          />
        )}
        {!video.live && (
          <div className="video-placeholder">
            <strong>NO STREAM</strong>
            <span>{video.note}</span>
          </div>
        )}
        <div className={`video-objective ${objective ? '' : 'none'}`}>
          {objective || 'No objective'}
        </div>
      </div>

      <div className={`target-readout ${readout ? '' : 'none'}`}>
        {readout || 'no target state'}
      </div>
    </Panel>
  );
}
