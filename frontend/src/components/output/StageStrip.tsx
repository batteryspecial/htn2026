import { BAD_STAGES, OPTIONAL_STAGES, STAGES } from '../../config/constants';

/**
 * A line diagram of the run.
 *
 * Only stages the pipeline can actually reach are drawn. Anything unexpected
 * that is also a failure gets appended in red; anything else is dropped,
 * because an unknown name on the strip reads as a broken UI.
 */
export function StageStrip({ reached }: { reached: string[] }) {
  const extras = reached.filter(
    (s) => !STAGES.includes(s as never)
      && (BAD_STAGES.includes(s) || s === 'clarify' || OPTIONAL_STAGES.includes(s)),
  );

  return (
    <ol className="stages">
      {STAGES.map((stage) => (
        <li
          key={stage}
          data-stage={stage}
          title={stage}
          className={reached.includes(stage) ? 'done' : ''}
        >
          {stage}
        </li>
      ))}
      {extras.map((stage) => (
        <li
          key={stage}
          data-stage={stage}
          title={stage}
          className={OPTIONAL_STAGES.includes(stage) ? 'optional done' : 'bad'}
        >
          {stage.replace('model_', '')}
        </li>
      ))}
    </ol>
  );
}
