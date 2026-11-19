
import { SessionManager } from '../SessionManager.js';
import { SessionEventBroadcaster } from '../events/SessionEventBroadcaster.js';
import { DatabaseManager } from '../DatabaseManager.js';
import { logger } from '../../../utils/logger.js';
import { ModeManager } from '../../domain/ModeManager.js';
import { SettingsDefaultsManager } from '../../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../../shared/paths.js';
import { isObserverCaptureEnabled } from '../../../shared/observer-capture.js';
import { isObserverUnhealthy, readObserverHealth } from '../../../shared/observer-health.js';
import { ObserverEventLedger } from '../../sqlite/ObserverEventLedger.js';
import { auditObserverSession } from '../observer-audit.js';

/** One prompt through a fresh observer agent (ClaudeProvider.runObserverBackfillPrompt). */
export type ObserverBackfill = (
  prompt: string,
  context: { sessionDbId: number; contentSessionId: string; project: string },
) => Promise<string | null>;

/** How far back the startup sweep looks for events an interrupted audit left open. */
export const RESUME_AUDIT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;

export class SessionCompletionHandler {
  private readonly auditsInFlight = new Set<Promise<void>>();
  /** The latest audit scheduled for each session; the next one for that session waits for it. */
  private readonly latestAuditBySession = new Map<number, Promise<void>>();

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

    void this.scheduleAudit(sessionDbId, row.content_session_id, row.project);
  }

  /** Resolves once every post-session audit started so far has finished. */
  async auditsSettled(): Promise<void> {
    await Promise.all([...this.auditsInFlight]);
  }

  /**
   * Finish post-session audits that a worker shutdown cut off.
   *
   * finalizeSession marks a session completed and then audits it in the
   * background. A shutdown during that audit (a reboot, a worker restart)
   * leaves a completed session with events still queued or dropped, and a
   * completed session is never finalized again, so nothing else would retry
   * them. Called once after startup. Sessions are audited one at a time, and
   * only those with open events created within `windowMs`.
   *
   * Does nothing while the observer is in a quota cooldown or failing. A
   * session's own quota or auth stop, which makes its audit report-only, is
   * not kept across a restart, so the observer's health is checked instead.
   * Returns the sessions audited.
   */
  async resumeInterruptedAudits(options: { now?: number; windowMs?: number } = {}): Promise<number[]> {
    const sessionStore = this.dbManager.getSessionStore();
    if (!sessionStore?.db) return [];
    const now = options.now ?? Date.now();
    let sessionIds: number[];
    try {
      sessionIds = new ObserverEventLedger(sessionStore.db)
        .listCompletedSessionsWithUnaccounted(now - (options.windowMs ?? RESUME_AUDIT_WINDOW_MS));
    } catch (error) {
      logger.warn('LEDGER', 'Could not list completed sessions with open events', {},
        error instanceof Error ? error : new Error(String(error)));
      return [];
    }
    if (sessionIds.length === 0) return [];

    const health = readObserverHealth();
    const coolingDown = !!health?.quotaCooldown?.active && health.quotaCooldown.until > now;
    if (coolingDown || isObserverUnhealthy(health)) {
      logger.info('LEDGER', `AUDIT_RESUME deferred: the observer is ${coolingDown ? 'in a quota cooldown' : 'failing'} | sessions=[${sessionIds.join(',')}]`);
      return [];
    }

    logger.info('LEDGER', `AUDIT_RESUME | sessions=[${sessionIds.join(',')}]`);
    const audited: number[] = [];
    for (const sessionDbId of sessionIds) {
      const row = sessionStore.getSessionById(sessionDbId);
      // Reopened since the listing: its own finalize will audit it.
      if (!row || row.status !== 'completed') continue;
      audited.push(sessionDbId);
      await this.scheduleAudit(sessionDbId, row.content_session_id, row.project);
    }
    return audited;
  }

  /**
   * Run a session's audit after any earlier audit of the same session. Two at
   * once would backfill the same open events twice and store duplicate
   * observations; this happens when a session is reopened and finalized again
   * while its first audit is still running. The later audit lists the open
   * events only when it starts, so it sees what the earlier one left.
   */
  private scheduleAudit(sessionDbId: number, contentSessionId: string, project: string): Promise<void> {
    let abortCategory: string;
    try {
      // Read now: the session may have left SessionManager by the time a queued audit starts.
      abortCategory = (this.sessionManager.getSession(sessionDbId)?.abortReason ?? '').split(':')[0];
    } catch (error) {
      // Never fails the caller (finalizeSession), as before the audit was queued.
      logger.warn('LEDGER', 'Post-session observer audit failed', { sessionId: sessionDbId },
        error instanceof Error ? error : new Error(String(error)));
      return Promise.resolve();
    }
    const previous = this.latestAuditBySession.get(sessionDbId) ?? Promise.resolve();
    const audit = previous.then(() => this.auditAfterSession(sessionDbId, contentSessionId, project, abortCategory));
    this.latestAuditBySession.set(sessionDbId, audit);
    this.auditsInFlight.add(audit);
    void audit.finally(() => {
      this.auditsInFlight.delete(audit);
      if (this.latestAuditBySession.get(sessionDbId) === audit) this.latestAuditBySession.delete(sessionDbId);
    });
    return audit;
  }

  /**
   * List queued tool events that never got an observation or skip, and
   * backfill them with a fresh observer agent. Report-only when capture is
   * off, the session stopped on quota or auth (backfilling would spend the
   * exhausted allowance), or Claude is not the configured provider.
   */
  private async auditAfterSession(sessionDbId: number, contentSessionId: string, project: string, abortCategory: string): Promise<void> {
    try {
      const sessionStore = this.dbManager.getSessionStore();
      if (!sessionStore?.db) return;
      const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
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
