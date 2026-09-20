import type { ReactNode } from 'react';

/** A white plate with the black signage band across its head. */
export function Panel({ className = '', title, aside, children }: {
  className?: string;
  title: ReactNode;
  /** Badges, readouts and buttons that sit in the title band. */
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className={`panel ${className}`}>
      <h2 className="panel-title">
        {title}
        {aside}
      </h2>
      {children}
    </section>
  );
}
