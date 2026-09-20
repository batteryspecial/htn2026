import type { ReactNode } from 'react';
import type { BadgeKind } from '../../hooks/useRetaskRun';

export function Badge({ kind, children, title }: {
  kind: BadgeKind | string;
  children: ReactNode;
  title?: string;
}) {
  return <span className={`badge ${kind}`} title={title}>{children}</span>;
}
