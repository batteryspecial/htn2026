/** One interface, two adapters: live (the orchestrator) and mock (offline). */

import type { Program } from '../../contracts/program';
import type { Reachability } from '../perception';

/** A stage the adapter reached, shaped like the pipeline's own status message. */
export interface StageEvent {
  instruction_id: string;
  stage: string;
  ts: number;
  detail?: string;
  data?: unknown;
}

export interface CompileResult {
  instruction_id: string | null;
  /**
   * The behaviours to install. Null in live mode: the agent installs them
   * itself as part of its loop, so there is nothing for this side to post.
   */
  program: Program | null;
  /** What the agent said back. Live mode only. */
  reply?: string | null;
  ok?: boolean;
  raw: unknown;
}

export type StageSink = (event: StageEvent) => void;

export interface Compiler {
  readonly name: string;
  /** `images` reach the agent as references; the mock ignores them. */
  compile(text: string, onStage?: StageSink, images?: File[], turn?: string): Promise<CompileResult>;
  /** Whether the compiler is reachable. Never throws. */
  health(): Promise<Reachability>;
}
