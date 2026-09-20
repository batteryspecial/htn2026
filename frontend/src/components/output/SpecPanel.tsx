import { useCallback, useState } from 'react';
import { reorder, type Program } from '../../contracts/program';
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

/** What is posted to POST /behaviors, one object per behaviour. */
const serialize = (program: Program) => JSON.stringify(reorder(program), null, 2);

export function SpecPanel({ run }: { run: RunView }) {
  const [copied, setCopied] = useState(false);
  const { program } = run;

  const copy = useCallback(() => {
    if (!program) return;
    navigator.clipboard.writeText(serialize(program)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  }, [program]);

  const download = useCallback(() => {
    if (!program) return;
    const blob = new Blob([serialize(program)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'behaviors.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, [program]);

  const count = program?.behaviors.length ?? 0;

  return (
    <Panel
      className="panel-output"
      title="Behaviours"
      aside={
        <>
          {count > 0 && <span className="count">{count}</span>}
          <Badge kind={run.badge.kind}>{run.badge.text}</Badge>
        </>
      }
    >
      <RetaskTimer timer={run.timer} />
      <StageStrip reached={run.reached} />
      <JsonView
        value={program ? reorder(program) : null}
        placeholder={placeholderFor(run)}
      />
      <Notices notices={run.notices} />

      <div className="actions">
        <button type="button" className="btn ghost" disabled={!program} onClick={copy}>
          {copied ? 'Copied' : 'Copy JSON'}
        </button>
        <button type="button" className="btn ghost" disabled={!program} onClick={download}>
          Download
        </button>
      </div>
    </Panel>
  );
}
