
import { SessionManager } from '../SessionManager.js';
import { SessionEventBroadcaster } from '../events/SessionEventBroadcaster.js';
import { DatabaseManager } from '../DatabaseManager.js';
import { logger } from '../../../utils/logger.js';
import { ModeManager } from '../../domain/ModeManager.js';
import { SettingsDefaultsManager } from '../../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../../shared/paths.js';
import { isObserverCaptureEnabled } from '../../../shared/observer-capture.js';
import { auditObserverSession } from '../observer-audit.js';

/** One prompt through a fresh observer agent (ClaudeProvider.runObserverBackfillPrompt). */
export type ObserverBackfill = (
  prompt: string,
  context: { sessionDbId: number; contentSessionId: string; project: string },
) => Promise<string | null>;

export class SessionCompletionHandler {
  private readonly auditsInFlight = new Set<Promise<void>>();

  constructor(
    private sessionManager: SessionManager,
    private eventBroadcaster: SessionEventBroadcaster,
    private dbManager: DatabaseManager,
    private backfill?: ObserverBackfill,
  ) {}

  async finalizeSession(sessionDbId: number): Promise<void> {
    const sessionStore = this.dbManager.getSessionStore();

    const row = sessionStore.getSessionById(sessionDbId);
    if (!row) {
      logger.debug('SESSION', 'finalizeSession: session not found, skipping', { sessionId: sessionDbId });
      return;
    }
    if (row.status === 'completed') {
      logger.debug('SESSION', 'finalizeSession: already completed, skipping', { sessionId: sessionDbId });
      return;
    }

    sessionStore.markSessionCompleted(sessionDbId);

    // The in-RAM message buffer is dropped when the session is removed
    // (SessionManager.removeSessionImmediate/deleteSession → buffer.dispose),
    // so there is nothing durable to clear here.

    this.eventBroadcaster.broadcastSessionCompleted(sessionDbId);

    logger.info('SESSION', 'Session finalized', { sessionId: sessionDbId });

    const audit = this.auditAfterSession(sessionDbId, row.content_session_id, row.project);
    this.auditsInFlight.add(audit);
    void audit.finally(() => this.auditsInFlight.delete(audit));
  }

  /** Resolves once every post-session audit started so far has finished. */
  async auditsSettled(): Promise<void> {
    await Promise.all([...this.auditsInFlight]);
  }

  /**
   * List queued tool events that never got an observation or skip, and
   * backfill them with a fresh observer agent. Report-only when capture is
   * off, the session stopped on quota or auth (backfilling would spend the
   * exhausted allowance), or Claude is not the configured provider.
   */
  private async auditAfterSession(sessionDbId: number, contentSessionId: string, project: string): Promise<void> {
    try {
      const sessionStore = this.dbManager.getSessionStore();
      if (!sessionStore?.db) return;
      const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
      const abortCategory = (this.sessionManager.getSession(sessionDbId)?.abortReason ?? '').split(':')[0];
      const canBackfill = !!this.backfill
        && isObserverCaptureEnabled(settings)
        && abortCategory !== 'quota' && abortCategory !== 'auth'
        && (settings.CLAUDE_MEM_PROVIDER || 'claude') === 'claude';
      const backfill = this.backfill;
      await auditObserverSession(sessionStore, ModeManager.getInstance().getActiveMode(), sessionDbId, {
        complete: canBackfill && backfill ? prompt => backfill(prompt, { sessionDbId, contentSessionId, project }) : undefined,
      });
    } catch (error) {
      logger.warn('LEDGER', 'Post-session observer audit failed', { sessionId: sessionDbId },
        error instanceof Error ? error : new Error(String(error)));
    }
  }
}
