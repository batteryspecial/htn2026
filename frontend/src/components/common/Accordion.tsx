import { useCallback, useState, type ReactNode } from 'react';
import { readJson, writeJson } from '../../utils/storage';

export interface Section {
  id: string;
  label: string;
  /** Shown as a chip next to the label. */
  count?: number;
  /** Controls for the header, e.g. a CLEAR button. Clicks do not toggle. */
  aside?: ReactNode;
  render: () => ReactNode;
}

const STORE_KEY = 'retask.accordion.v1';

/**
 * Stacked sections, any number open at once.
 *
 * Tabs were wrong here: during a demo you want the trace *and* the events
 * visible together, and to fold away the ones you are not using. Open
 * sections share the height; closed ones cost a header.
 *
 * Which are open is remembered, so the layout survives a reload mid-demo.
 */
export function Accordion({ sections, className = '', initial }: {
  sections: Section[];
  className?: string;
  /** Open on a first visit, before anything has been remembered. */
  initial?: string[];
}) {
  const [open, setOpen] = useState<string[]>(
    () => readJson<string[]>(STORE_KEY, initial ?? [sections[0]?.id].filter(Boolean)),
  );

  const toggle = useCallback((id: string) => {
    setOpen((prev) => {
      const next = prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id];
      writeJson(STORE_KEY, next);
      return next;
    });
  }, []);

  return (
    <div className={`accordion ${className}`}>
      {sections.map((section) => {
        const isOpen = open.includes(section.id);
        return (
          <section
            key={section.id}
            className={`fold ${isOpen ? 'open' : ''}`}
          >
            <h2 className="panel-title fold-head">
              <button
                type="button"
                className="fold-toggle"
                aria-expanded={isOpen}
                onClick={() => toggle(section.id)}
              >
                <span className="fold-mark" aria-hidden="true">{isOpen ? '▾' : '▸'}</span>
                {section.label}
                {section.count !== undefined && (
                  <span className="count">{section.count}</span>
                )}
              </button>
              {isOpen && section.aside}
            </h2>
            {isOpen && <div className="fold-body">{section.render()}</div>}
          </section>
        );
      })}
    </div>
  );
}
