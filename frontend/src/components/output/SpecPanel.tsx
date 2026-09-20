import { useCallback, useState } from 'react';
import { reorder } from '../../contracts/format';
import type { TaskSpec } from '../../contracts/taskspec';
import type { RunView } from '../../hooks/useRetaskRun';
import { Badge } from '../common/Badge';
import { Panel } from '../common/Panel';
import { JsonView } from './JsonView';
import { Notices } from './Notices';
import { RetaskTimer } from './RetaskTimer';
import { StageStrip } from './StageStrip';

const placeholderFor = (run: RunView): string => {
  if (run.busy) return '// compiling…';
  if (run.badge.text === 'AWAITING SPEC') return '// accepted — waiting for the compiled event';
  return '// awaiting an instruction';
};

export function SpecPanel({ run }: { run: RunView }) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(() => {
    if (!run.spec) return;
    navigator.clipboard.writeText(serialize(run.spec)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  }, [run.spec]);

  const download = useCallback(() => {
    if (!run.spec) return;
    const blob = new Blob([serialize(run.spec)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${run.spec.spec_id || 'taskspec'}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, [run.spec]);

  return (
    <Panel
      className="panel-output"
      title="TaskSpec"
      aside={<Badge kind={run.badge.kind}>{run.badge.text}</Badge>}
    >
      <RetaskTimer timer={run.timer} />
      <StageStrip reached={run.reached} />
      <JsonView value={run.spec} placeholder={placeholderFor(run)} />
      <Notices notices={run.notices} />

      <div className="actions">
        <button type="button" className="btn ghost" disabled={!run.spec} onClick={copy}>
          {copied ? 'Copied' : 'Copy JSON'}
        </button>
        <button type="button" className="btn ghost" disabled={!run.spec} onClick={download}>
          Download
        </button>
      </div>
    </Panel>
  );
}

const serialize = (spec: TaskSpec) => JSON.stringify(reorder(spec), null, 2);
