/** The orchestrator compiles. This adapter only carries the request. */

import { REQUEST_TIMEOUT_MS, type Settings } from '../../config/settings';
import { OrchestratorClient } from '../orchestrator';
import type { Reachability } from '../perception';
import type { CompileResult, Compiler } from './types';

export class LiveCompiler implements Compiler {
  readonly name = 'live';
  private readonly client: OrchestratorClient;

  constructor(settings: Settings) {
    this.client = new OrchestratorClient(settings.apiBase, REQUEST_TIMEOUT_MS);
  }

  async compile(text: string, _onStage?: unknown,
                images: File[] = []): Promise<CompileResult> {
    // /chat rather than /instruction: it carries attachments, and the agent
    // installs the behaviours itself. Nothing comes back to post.
    const { turn, reply } = await this.client.chat(text, images);
    return { instruction_id: turn, program: null, reply, raw: null };
  }

  health(): Promise<Reachability> {
    return this.client.health();
  }
}
