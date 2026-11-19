import type { Database } from 'bun:sqlite';
import { logger } from '../../utils/logger.js';

/**
 * What became of every tool event queued for the observer.
 *
 * The event id is the durable `tool_uses.id` row of the event, so it survives
 * the in-RAM queue (SessionMessageBuffer), which drops work on purpose: after
 * LIVE_OBSERVER_MAX_AGE_MS unclaimed, past its per-session cap, or after a
 * batch is rejected twice. A row stays unaccounted ('queued' or 'dropped')
 * until the observer stored an observation citing it, skipped it, or the
 * post-session audit backfilled it. The tool payload is read from `tool_uses`.
 *
 * `committed_observation_tool_uses` answers "was this replayed tool use already
 * answered"; this ledger answers "which observation recorded this event".
 */

export const OBSERVER_EVENT_LEDGER_SCHEMA_VERSION = 76;

export type LedgerStatus = 'queued' | 'observed' | 'skipped' | 'backfilled' | 'dropped';

export const UNACCOUNTED_STATUSES: readonly LedgerStatus[] = ['queued', 'dropped'];

export interface LedgerRow {
  event_id: number;
  session_db_id: number;
  status: LedgerStatus;
  observation_ids: string | null;
  note: string | null;
  created_at_epoch: number;
  accounted_at_epoch: number | null;
}

/** A ledger row joined with the tool event it tracks. */
export interface LedgerEvent extends LedgerRow {
  tool_use_id: string;
  tool_name: string;
  tool_input: string | null;
  tool_response: string | null;
  cwd: string | null;
  prompt_number: number | null;
  tool_created_at_epoch: number;
}

export interface LedgerSummary {
  total: number;
  queued: number;
  observed: number;
  skipped: number;
  backfilled: number;
  dropped: number;
}

/** Idempotent DDL, run by the SessionStore migration chain. */
export function ensureObserverEventLedgerSchema(db: Database): void {
  db.run(`
    CREATE TABLE IF NOT EXISTS observer_event_ledger (
      event_id INTEGER PRIMARY KEY,
      session_db_id INTEGER NOT NULL,
      status TEXT NOT NULL DEFAULT 'queued'
        CHECK(status IN ('queued', 'observed', 'skipped', 'backfilled', 'dropped')),
      observation_ids TEXT,
      note TEXT,
      created_at_epoch INTEGER NOT NULL,
      accounted_at_epoch INTEGER
    )
  `);
  db.run('CREATE INDEX IF NOT EXISTS idx_observer_event_ledger_session_status ON observer_event_ledger(session_db_id, status)');
  db.prepare('INSERT OR IGNORE INTO schema_versions (version, applied_at) VALUES (?, ?)')
    .run(OBSERVER_EVENT_LEDGER_SCHEMA_VERSION, new Date().toISOString());
}

export class ObserverEventLedger {
  constructor(private readonly db: Database) {}

  /** The event was handed to the observer queue. A replay keeps the first outcome. */
  recordQueued(eventId: number, sessionDbId: number): void {
    this.db.prepare(`
      INSERT INTO observer_event_ledger (event_id, session_db_id, status, created_at_epoch)
      VALUES (?, ?, 'queued', ?)
      ON CONFLICT(event_id) DO NOTHING
    `).run(eventId, sessionDbId, Date.now());
  }

  /**
   * The queue refused the event (live buffer limit). Only a first sighting is
   * recorded: a row that already exists means the refusal was a duplicate.
   */
  recordRefused(eventId: number, sessionDbId: number, note: string): void {
    this.db.prepare(`
      INSERT INTO observer_event_ledger (event_id, session_db_id, status, note, created_at_epoch)
      VALUES (?, ?, 'dropped', ?, ?)
      ON CONFLICT(event_id) DO NOTHING
    `).run(eventId, sessionDbId, note, Date.now());
  }

  /**
   * Record an observation against the events it cites. Observation ids merge,
   * so an event covered by several observations lists them all.
   */
  markObserved(eventIds: readonly number[], observationId: number, status: 'observed' | 'backfilled' = 'observed'): void {
    const select = this.db.prepare('SELECT observation_ids FROM observer_event_ledger WHERE event_id = ?');
    const update = this.db.prepare(`
      UPDATE observer_event_ledger
         SET status = CASE WHEN status IN ('queued', 'dropped', 'skipped') THEN ? ELSE status END,
             observation_ids = ?,
             note = NULL,
             accounted_at_epoch = COALESCE(accounted_at_epoch, ?)
       WHERE event_id = ?
    `);
    const now = Date.now();
    for (const eventId of eventIds) {
      const row = select.get(eventId) as { observation_ids: string | null } | null;
      if (!row) continue;
      const ids = new Set<number>(row.observation_ids ? JSON.parse(row.observation_ids) : []);
      ids.add(observationId);
      update.run(status, JSON.stringify([...ids]), now, eventId);
    }
  }

  markSkipped(eventIds: readonly number[], reason: string | null): void {
    const stmt = this.db.prepare(`
      UPDATE observer_event_ledger
         SET status = 'skipped', note = ?, accounted_at_epoch = ?
       WHERE event_id = ? AND status IN ('queued', 'dropped')
    `);
    const now = Date.now();
    for (const eventId of eventIds) stmt.run(reason, now, eventId);
  }

  /** The live observer gave up on these events; the audit picks them up. */
  markDropped(eventIds: readonly number[], reason: string): void {
    const stmt = this.db.prepare(`
      UPDATE observer_event_ledger SET status = 'dropped', note = ? WHERE event_id = ? AND status = 'queued'
    `);
    for (const eventId of eventIds) stmt.run(reason, eventId);
  }

  isUnaccounted(eventId: number): boolean {
    const row = this.db.prepare('SELECT status FROM observer_event_ledger WHERE event_id = ?').get(eventId) as { status: LedgerStatus } | null;
    return !!row && UNACCOUNTED_STATUSES.includes(row.status);
  }

  /** Unaccounted events of a session, with the tool payload needed to re-prompt them. */
  listUnaccounted(sessionDbId: number): LedgerEvent[] {
    return this.db.prepare(`
      SELECT l.*, t.tool_use_id, t.tool_name, t.tool_input, t.tool_response, t.cwd, t.prompt_number,
             t.created_at_epoch AS tool_created_at_epoch
        FROM observer_event_ledger l
        JOIN tool_uses t ON t.id = l.event_id
       WHERE l.session_db_id = ? AND l.status IN ('queued', 'dropped')
       ORDER BY l.event_id
    `).all(sessionDbId) as LedgerEvent[];
  }

  summarize(sessionDbId: number): LedgerSummary {
    const rows = this.db.prepare(`
      SELECT status, COUNT(*) AS n FROM observer_event_ledger WHERE session_db_id = ? GROUP BY status
    `).all(sessionDbId) as Array<{ status: LedgerStatus; n: number }>;
    const summary: LedgerSummary = { total: 0, queued: 0, observed: 0, skipped: 0, backfilled: 0, dropped: 0 };
    for (const { status, n } of rows) {
      summary[status] = n;
      summary.total += n;
    }
    return summary;
  }

  logOutcome(sessionDbId: number, observed: Map<number, number[]>, skipped: readonly number[], source: string): void {
    const observedText = [...observed.entries()].map(([eventId, obsIds]) => `evt_${eventId}->#${obsIds.join('+#')}`).join(',');
    logger.info('LEDGER', `EVENT_OUTCOME | sessionDbId=${sessionDbId} | source=${source} | observed=[${observedText}] | skipped=[${skipped.map(id => `evt_${id}`).join(',')}]`, {
      sessionId: sessionDbId,
    });
  }
}
