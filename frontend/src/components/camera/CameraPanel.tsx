import type { VideoStream } from '../../hooks/useVideoStream';
import { Badge } from '../common/Badge';
import { Panel } from '../common/Panel';

export interface CameraPanelProps {
  video: VideoStream;
  /** Summary of the current objective, shown as a platform sign. */
  objective: string | null;
}

/**
 * The projected centrepiece: the pipeline's enriched MJPEG, and nothing this
 * app has drawn on top of it.
 *
 * Status used to live here. It moved next to the instruction, because "did
 * that work?" is a question about what you just typed.
 */
export function CameraPanel({ video, objective }: CameraPanelProps) {
  return (
    <Panel
      className="panel-video"
      title="Camera"
      aside={
        <>
          <Badge kind={video.live ? 'pass' : 'fail'}>
            {video.live ? 'LIVE' : 'OFFLINE'}
          </Badge>
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
    </Panel>
  );
}
