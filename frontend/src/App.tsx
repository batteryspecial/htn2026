import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CameraPanel } from './components/camera/CameraPanel';
import { ChatPanel } from './components/chat/ChatPanel';
import { Accordion } from './components/common/Accordion';
import { HistoryPanel } from './components/history/HistoryPanel';
import { nextMode, TopBar } from './components/layout/TopBar';
import { SpecAside, SpecPanel } from './components/output/SpecPanel';
import { BehaviorsPanel } from './components/pipeline/BehaviorsPanel';
import { EventsPanel } from './components/pipeline/EventsPanel';
import { SettingsDrawer } from './components/settings/SettingsDrawer';
import { StatusBar } from './components/status/StatusBar';
import { TraceAside, TracePanel } from './components/trace/TracePanel';
import { EVENT_STAGE } from './config/constants';
import { liveObjective } from './contracts/liveRun';
import { useHistory } from './hooks/useHistory';
import { useAgentRun } from './hooks/useAgentRun';
import { useHotkeys } from './hooks/useHotkeys';
import {
  healthOf, leadBehavior, leadTrack, useBehaviorKinds, useCameras, useModels,
  usePerceptionEvents, usePerceptionHealth, usePerceptionState,
} from './hooks/usePerception';
import { useRetaskRun } from './hooks/useRetaskRun';
import { useSettings } from './hooks/useSettings';
import { useSpeech } from './hooks/useSpeech';
import { useSpeechRecognition } from './hooks/useSpeechRecognition';
import { useTrace } from './hooks/useTrace';
import { useVideoStream } from './hooks/useVideoStream';
import type { Reachability } from './services/perception';
import { ReconnectingSocket } from './services/socket';

export default function App() {
  const { settings, update, compiler, perception, orchestrator } = useSettings();
  const [text, setText] = useState('');
  const [interim, setInterim] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const live = settings.mode === 'live';

  /* ------------------------------------------------------------ pipeline */

  const pipelineBase = settings.pipelineBase;
  const state = usePerceptionState(perception, pipelineBase);
  const pipelineHealth = usePerceptionHealth(perception, pipelineBase);
  const { events, clear: clearEvents } = usePerceptionEvents(perception, pipelineBase);
  const { models, refresh: refreshModels } = useModels(perception, pipelineBase);
  const { cameras, refresh: refreshCameras } = useCameras(perception, pipelineBase);
  const availableKinds = useBehaviorKinds(perception, pipelineBase);

  const behavior = leadBehavior(state);
  const track = leadTrack(state, behavior);
  const video = useVideoStream(settings.videoUrl);
  const cameraOk = pipelineHealth?.up === true && pipelineHealth.health?.camera_ok === true;
  const cameraVideo = {
    ...video,
    live: video.live && cameraOk,
    note: !cameraOk ? (pipelineHealth?.detail ?? 'camera health unknown') : video.note,
  };

  /* --------------------------------------------------------------- voice */

  const speak = useSpeech(settings.tts, settings.lang);
  const { entries, push: pushHistory } = useHistory();

  const mockRun = useRetaskRun({
    settings, compiler, perception, availableKinds, speak, pushHistory,
  });
  const agentRun = useAgentRun({ orchestrator, perception, speak, pushHistory, pipelineState: state });
  const run = live ? agentRun : mockRun;

  // An alert the agent speaks between turns arrives on the trace socket, not
  // as a reply — it belongs in the transcript and out of the speakers.
  const trace = useTrace(orchestrator, settings.apiBase, live, {
    onSay: agentRun.alert, onEntry: agentRun.handleTrace,
  });

  const textRef = useRef(text);
  textRef.current = text;

  const send = useCallback((value: string, images: File[] = []) => {
    if (run.view.busy) return;
    setInterim(false);
    setText('');
    run.compile(value, images);
  }, [run]);

  const speech = useSpeechRecognition({
    lang: settings.lang,
    currentText: () => textRef.current,
    onTranscript: (transcript, stillChanging) => {
      setText(transcript);
      setInterim(stillChanging);
    },
    onSilence: settings.autoSend ? (t) => send(t) : undefined,
  });

  /* ------------------------------------------------------- stage sources */

  const handleStage = mockRun.handleStage;

  // Pipeline events drive the stage strip. Only the three with a stage
  // equivalent are translated; the rest belong in the events panel, where
  // they are shown in full rather than flattened into a name that does not fit.
  useEffect(() => {
    if (live) return;
    const socket = new ReconnectingSocket<Record<string, unknown>>(
      perception.eventsSocketUrl(),
      {
        onMessage: (msg) => {
          const stage = EVENT_STAGE[String(msg.type)];
          if (!stage) return;
          handleStage({
            instruction_id: '',
            stage,
            ts: Number(msg.ts) || Date.now() / 1000,
            detail: typeof msg.detail === 'string' ? msg.detail : undefined,
          });
        },
      },
    );
    socket.connect();
    return () => socket.close();
  }, [handleStage, perception, pipelineBase, live]);

  /* -------------------------------------------------------- compiler dot */

  const [compilerReach, setCompilerReach] = useState<Reachability | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      const r = await compiler.health();
      if (alive) setCompilerReach(r);
    };
    poll();
    const timer = setInterval(poll, 2000);
    return () => { alive = false; clearInterval(timer); };
  }, [compiler]);

  /* ------------------------------------------------------------ hotkeys */

  const hotkeys = useMemo(() => ({
    onEscape: () => {
      if (speech.listening) speech.stop();
      setDrawerOpen(false);
    },
    onToggleMic: speech.toggle,
    onToggleVoice: () => update({ tts: !settings.tts }),
  }), [settings.tts, speech, update]);

  useHotkeys(hotkeys);

  /* ------------------------------------------------------------ actions */

  const dropBehavior = useCallback((id: string) => {
    perception.removeBehavior(id).catch((error: Error) => run.alert(`Could not remove behavior: ${error.message}`));
  }, [perception, run.alert]);

  const selectModel = useCallback((name: string) => {
    perception.setModel(name).then(refreshModels)
      .catch((error: Error) => run.alert(`Could not switch model: ${error.message}`));
  }, [perception, refreshModels, run.alert]);

  // The MJPEG <img> does not recover on its own from a stream that stopped
  // mid-frame, and a camera switch stops it for a second or two. So reopen it
  // rather than leaving the operator looking at a dead feed and a green dot.
  const selectCamera = useCallback((name: string) => {
    perception.setCamera({ name })
      .then(() => { video.reconnect(); return refreshCameras(); })
      .catch((error: Error) => run.alert(`Could not switch camera: ${error.message}`));
  }, [perception, refreshCameras, run.alert, video]);

  const traceProps = {
    entries: trace.entries,
    connected: trace.connected,
    onClear: trace.clear,
    live,
  };

  const behaviors = state?.behaviors ?? [];

  return (
    <>
      <TopBar
        mode={settings.mode}
        compilerLabel={live ? settings.apiBase.replace(/^https?:\/\//, '') : 'mock adapter'}
        compiler={compilerReach}
        camera={{ live: cameraVideo.live, note: cameraVideo.note }}
        tts={settings.tts}
        onCycleMode={() => { if (!run.view.busy) update({ mode: nextMode(settings.mode) }); }}
        onToggleTts={() => update({ tts: !settings.tts })}
        onOpenSettings={() => { refreshModels(); refreshCameras(); setDrawerOpen(true); }}
      />

      <main className="grid">
        {/* The camera is the thing the room is looking at. It gets the half. */}
        <div className="col col-camera">
          <CameraPanel
            video={cameraVideo}
            objective={liveObjective(state)}
          />
        </div>

        <div className="col col-side">
          <ChatPanel
            messages={run.view.messages}
            text={text}
            onTextChange={setText}
            interim={interim}
            busy={run.view.busy}
            timer={run.view.timer}
            speech={speech}
            status={
              <StatusBar
                behavior={behavior}
                track={track}
                health={healthOf(pipelineHealth)}
                behaviorCount={behaviors.length}
              />
            }
            onSend={send}
            onStop={run.stop}
          />

          <Accordion
            className="col-logs"
            initial={['trace', 'behaviors']}
            sections={[
              {
                id: 'trace',
                label: 'Agent trace',
                count: trace.entries.length,
                aside: <TraceAside {...traceProps} />,
                render: () => <TracePanel {...traceProps} />,
              },
              {
                id: 'events',
                label: 'Events',
                count: events.length,
                aside: (
                  <button type="button" className="title-btn" onClick={clearEvents}>
                    CLEAR
                  </button>
                ),
                render: () => (
                  <EventsPanel
                    events={events}
                    snapshotUrl={(id) => perception.eventSnapshotUrl(id)}
                  />
                ),
              },
              {
                id: 'behaviors',
                label: 'Behaviours',
                count: behaviors.length,
                render: () => (
                  <BehaviorsPanel behaviors={behaviors} onDrop={dropBehavior} />
                ),
              },
              {
                id: 'spec',
                label: 'Spec',
                aside: <SpecAside run={run.view} />,
                render: () => <SpecPanel run={run.view} />,
              },
              {
                id: 'history',
                label: 'History',
                count: entries.length,
                render: () => <HistoryPanel entries={entries} onRecall={run.recall} />,
              },
            ]}
          />
        </div>
      </main>

      <SettingsDrawer
        open={drawerOpen}
        settings={settings}
        models={models}
        cameras={cameras}
        onSave={(prefs) => update(run.view.busy ? { ...prefs, mode: settings.mode } : prefs)}
        onSelectModel={selectModel}
        onSelectCamera={selectCamera}
        onRescanCameras={() => refreshCameras(true)}
        onClose={() => setDrawerOpen(false)}
      />
    </>
  );
}
