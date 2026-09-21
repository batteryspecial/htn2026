/**
 * Mode in, compiler out.
 *
 * There used to be a third adapter that called OpenAI from the browser. It is
 * gone: the LLM and its key live in the orchestrator now, which is where an
 * agent loop can probe, retry and ask follow-up questions that one
 * structured-output call never could.
 */

import type { Settings } from '../../config/settings';
import { LiveCompiler } from './live';
import { MockCompiler } from './mock';
import type { Compiler } from './types';

export function createCompiler(settings: Settings): Compiler {
  return settings.mode === 'mock' ? new MockCompiler() : new LiveCompiler(settings);
}

export type { CompileResult, Compiler, StageEvent, StageSink } from './types';
