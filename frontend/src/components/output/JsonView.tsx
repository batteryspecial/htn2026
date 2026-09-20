import { useMemo } from 'react';
import { tokenizeJson } from '../../contracts/format';

/** Syntax-coloured JSON, tokenised rather than built as an HTML string. */
export function JsonView({ value, placeholder }: { value: unknown; placeholder: string }) {
  const tokens = useMemo(() => (value ? tokenizeJson(value) : null), [value]);

  return (
    <div className="json-wrap">
      <pre className="json">
        {tokens
          ? tokens.map((t, i) => (
            // eslint-disable-next-line react/no-array-index-key -- positional tokens
            <span key={i} className={t.kind === 'plain' ? undefined : `j-${t.kind}`}>
              {t.text}
            </span>
          ))
          : <span className="j-dim">{placeholder}</span>}
      </pre>
    </div>
  );
}
