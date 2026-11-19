/**
 * Post-session audits that a worker shutdown cut off are finished when the
 * next worker starts (SessionCompletionHandler.resumeInterruptedAudits), and
 * two audits of one session never run at once, so no event is backfilled
 * twice. A fake backfill agent stands in for the model.
 */
import { afterEach, beforeAll, describe, expect, it } from 'bun:test';
import { Database } from 'bun:sqlite';
import { rmSync } from 'fs';
import { join } from 'path';
import { SessionStore } from '../../src/services/sqlite/SessionStore.js';
import { ObserverEventLedger } from '../../src/services/sqlite/ObserverEventLedger.js';
import { ModeManager } from '../../src/services/domain/ModeManager.js';
import { SessionCompletionHandler, RESUME_AUDIT_WINDOW_MS } from '../../src/services/worker/session/SessionCompletionHandler.js';
import { OBSERVER_HEALTH_FILENAME, recordObserverQuotaCooldown } from '../../src/shared/observer-health.js';
import { paths } from '../../src/shared/paths.js';

const DAY_MS = 24 * 60 * 60 * 1000;

function eventIdsInPrompt(prompt: string): number[] {
  return [...prompt.matchAll(/<observed_from_primary_session>\s*<event_id>evt_(\d+)<\/event_id>/g)].map(m => Number(m[1]));
}

function observationFor(id: number): string {
  return `<observation><type>discovery</type><title>Backfilled tool event evt_${id}</title><subtitle>Recorded by the backfill observer</subtitle><facts><fact>Event evt_${id} was backfilled</fact></facts><narrative>The backfill observer recorded tool event evt_${id}.</narrative><concepts><concept>how-it-works</concept></concepts><event_ids><event_id>evt_${id}</event_id></event_ids></observation>`;
}

/** Answers every event it is sent with one clean observation; records what it was sent. */
class FakeBackfill {
  readonly sent: number[] = [];
  gate: Promise<void> | null = null;
  readonly complete = async (prompt: string): Promise<string> => {
    const ids = eventIdsInPrompt(prompt);
    this.sent.push(...ids);
    if (this.gate) await this.gate;
    return ids.map(observationFor).join('\n');
  };
}

interface Fixture {
  store: SessionStore;
  ledger: ObserverEventLedger;
  fake: FakeBackfill;
  handler: SessionCompletionHandler;
  session(name: string, events: number, opts?: { status?: 'active' | 'completed'; dropped?: number; createdAtEpoch?: number }): { id: number; eventIds: number[] };
}

function fixture(): Fixture {
  const store = new SessionStore(new Database(':memory:'));
  const ledger = new ObserverEventLedger(store.db);
  const fake = new FakeBackfill();
  const dbManager = { getSessionStore: () => store };
  const sessionManager = { getSession: () => undefined };
  const handler = new SessionCompletionHandler(sessionManager as never, { broadcastSessionCompleted: () => {} } as never, dbManager as never,
    prompt => fake.complete(prompt));
  let nextToolUse = 0;
  return {
    store, ledger, fake, handler,
    session(name, events, opts = {}) {
      const id = store.createSDKSession(`content-${name}`, 'resume-project', `Prompt for ${name}`, undefined, 'claude');
      store.ensureMemorySessionIdRegistered(id, `memory-${name}`);
      const eventIds: number[] = [];
      for (let i = 0; i < events; i++) {
        const eventId = store.upsertToolUse({
          toolUseId: `toolu_${name}_${nextToolUse++}`, contentSessionId: `content-${name}`, sessionDbId: id, memorySessionId: `memory-${name}`,
          project: 'resume-project', platformSource: 'claude', toolName: 'Bash', toolInput: `{"command":"echo ${i}"}`, toolResponse: `{"stdout":"${i}"}`,
          cwd: '/tmp', promptNumber: 1,
        })!;
        if (i < (opts.dropped ?? 0)) ledger.recordRefused(eventId, id, 'live buffer refused the event');
        else ledger.recordQueued(eventId, id);
        eventIds.push(eventId);
      }
      if (opts.createdAtEpoch !== undefined) {
        store.db.prepare('UPDATE observer_event_ledger SET created_at_epoch = ? WHERE session_db_id = ?').run(opts.createdAtEpoch, id);
      }
      if ((opts.status ?? 'completed') === 'completed') store.markSessionCompleted(id);
      return { id, eventIds };
    },
  };
}

function statuses(f: Fixture, sessionDbId: number): Record<string, number> {
  const { total: _total, ...byStatus } = f.ledger.summarize(sessionDbId);
  return Object.fromEntries(Object.entries(byStatus).filter(([, n]) => n > 0));
}

const healthFile = () => join(paths.dataDir(), OBSERVER_HEALTH_FILENAME);

beforeAll(() => {
  ModeManager.getInstance().loadMode('code');
});

afterEach(() => {
  rmSync(healthFile(), { force: true });
});

describe('resumeInterruptedAudits', () => {
  it('backfills the open events of a completed session whose audit was cut off', async () => {
    const f = fixture();
    const cutOff = f.session('cut-off', 4, { dropped: 1 });
    expect(statuses(f, cutOff.id)).toEqual({ queued: 3, dropped: 1 });

    const audited = await f.handler.resumeInterruptedAudits();

    expect(audited).toEqual([cutOff.id]);
    expect(f.fake.sent.sort((a, b) => a - b)).toEqual(cutOff.eventIds);
    expect(statuses(f, cutOff.id)).toEqual({ backfilled: 4 });
    expect(f.ledger.listUnaccounted(cutOff.id)).toHaveLength(0);
  });

  it('leaves active sessions, fully accounted sessions and events older than the window alone', async () => {
    const f = fixture();
    const now = Date.now();
    const active = f.session('still-active', 2, { status: 'active' });
    const accounted = f.session('accounted', 2);
    f.ledger.markSkipped(accounted.eventIds, 'nothing to record');
    const old = f.session('old', 2, { createdAtEpoch: now - RESUME_AUDIT_WINDOW_MS - DAY_MS });
    const recent = f.session('recent', 2, { createdAtEpoch: now - DAY_MS });

    const audited = await f.handler.resumeInterruptedAudits({ now });

    expect(audited).toEqual([recent.id]);
    expect(f.fake.sent.sort((a, b) => a - b)).toEqual(recent.eventIds);
    expect(statuses(f, active.id)).toEqual({ queued: 2 });
    expect(statuses(f, accounted.id)).toEqual({ skipped: 2 });
    expect(statuses(f, old.id)).toEqual({ queued: 2 });
  });

  it('waits while the observer is in a quota cooldown', async () => {
    const f = fixture();
    const now = Date.now();
    const cutOff = f.session('cooling-down', 2);
    recordObserverQuotaCooldown({ active: true, provider: 'claude', armedAt: now, until: now + 60 * 60 * 1000 }, healthFile());

    const audited = await f.handler.resumeInterruptedAudits({ now });

    expect(audited).toEqual([]);
    expect(f.fake.sent).toEqual([]);
    expect(statuses(f, cutOff.id)).toEqual({ queued: 2 });
  });

  it('skips a session that is reopened while an earlier session is being audited', async () => {
    const f = fixture();
    const first = f.session('first', 2);
    const reopened = f.session('reopened', 2);
    let release!: () => void;
    f.fake.gate = new Promise<void>(resolve => { release = resolve; });

    const sweep = f.handler.resumeInterruptedAudits();
    await Bun.sleep(20);
    // The user resumes the second session before the sweep reaches it; its own end will audit it.
    f.store.reopenCompletedSession(reopened.id);
    release();

    expect(await sweep).toEqual([first.id]);
    expect(statuses(f, first.id)).toEqual({ backfilled: 2 });
    expect(statuses(f, reopened.id)).toEqual({ queued: 2 });
  });

  it('does nothing when no completed session has open events', async () => {
    const f = fixture();
    f.session('active-only', 3, { status: 'active' });
    expect(await f.handler.resumeInterruptedAudits()).toEqual([]);
    expect(f.fake.sent).toEqual([]);
  });
});

describe('ObserverEventLedger.listCompletedSessionsWithUnaccounted', () => {
  it('lists only completed sessions with queued or dropped events since the given time', () => {
    const f = fixture();
    const now = Date.now();
    const completed = f.session('completed', 1);
    const dropped = f.session('dropped', 1, { dropped: 1 });
    f.session('active', 1, { status: 'active' });
    f.session('old', 1, { createdAtEpoch: now - 10 * DAY_MS });
    const accounted = f.session('accounted', 1);
    f.ledger.markSkipped(accounted.eventIds, 'nothing to record');

    expect(f.ledger.listCompletedSessionsWithUnaccounted(now - DAY_MS)).toEqual([completed.id, dropped.id]);
  });
});

describe('audits of one session', () => {
  it('run one after the other, so an event finalized twice is backfilled once', async () => {
    const f = fixture();
    const session = f.session('finalized-twice', 3, { status: 'active' });
    let release!: () => void;
    f.fake.gate = new Promise<void>(resolve => { release = resolve; });

    // First end: the audit starts and waits on the model.
    await f.handler.finalizeSession(session.id);
    // The session is resumed and ends again before that audit finishes.
    f.store.reopenCompletedSession(session.id);
    await f.handler.finalizeSession(session.id);
    await Bun.sleep(20);
    release();
    await f.handler.auditsSettled();

    expect(f.fake.sent.sort((a, b) => a - b)).toEqual(session.eventIds);
    expect(statuses(f, session.id)).toEqual({ backfilled: 3 });
    const observationIds = f.store.db.prepare('SELECT observation_ids FROM observer_event_ledger WHERE session_db_id = ?')
      .all(session.id) as Array<{ observation_ids: string }>;
    for (const row of observationIds) expect(JSON.parse(row.observation_ids)).toHaveLength(1);
  });
});
