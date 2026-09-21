import { useEffect, useRef } from 'react';
import type { ChatMessage } from '../../hooks/useRetaskRun';
import { seconds } from '../../utils/format';

export function Transcript({ messages }: { messages: ChatMessage[] }) {
  const box = useRef<HTMLDivElement>(null);

  // scrollTop on the box, not scrollIntoView on a tail element: scrollIntoView
  // scrolls every scrollable ancestor, and this one sits inside a column that
  // has its own. Pinning the transcript should not move anything else.
  useEffect(() => {
    const el = box.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length]);

  if (!messages.length) {
    return (
      <div className="transcript empty">
        <p className="empty-row">
          Say what the camera should do. One sentence is enough.
        </p>
      </div>
    );
  }

  return (
    <div className="transcript" ref={box}>
      {messages.map((message) => (
        <div key={message.id} className={`bubble ${message.role}`}>
          {message.images?.map((src, i) => (
            // eslint-disable-next-line react/no-array-index-key -- fixed per message
            <img key={i} className="bubble-image" src={src} alt="" />
          ))}
          <span className="bubble-text">{message.text}</span>
          {message.seconds !== undefined && (
            <span className="bubble-time">{seconds(message.seconds)}s</span>
          )}
        </div>
      ))}
    </div>
  );
}
