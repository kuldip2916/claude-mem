#!/usr/bin/env bun
/**
 * Audit the observer event ledger for a finished session.
 *
 *   bun scripts/audit-observer-events.ts --session <sessionDbId|contentSessionId>
 *   bun scripts/audit-observer-events.ts --session 583 --backfill
 *   bun scripts/audit-observer-events.ts --recent 10
 *
 * Without --backfill it only lists the event ids (tool_uses row ids) that were
 * queued for the observer but never received an observation or a skip. With --backfill it sends them, in batches,
 * to a fresh observer agent and stores whatever passes validation.
 * The worker runs the same audit automatically when a session is finalized.
 */

import { SessionStore } from '../src/services/sqlite/SessionStore.js';
import { ModeManager } from '../src/services/domain/ModeManager.js';
import { SettingsDefaultsManager } from '../src/shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../src/shared/paths.js';
import { auditObserverSession } from '../src/services/worker/observer-audit.js';
import { ClaudeProvider } from '../src/services/worker/ClaudeProvider.js';

function arg(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index === -1 ? undefined : process.argv[index + 1];
}

const backfill = process.argv.includes('--backfill');
const sessionArg = arg('--session');
const recent = arg('--recent');
const batchSize = arg('--batch-size') ? Number(arg('--batch-size')) : undefined;

if (!sessionArg && !recent) {
  console.error('usage: bun scripts/audit-observer-events.ts (--session <id> | --recent <n>) [--backfill] [--batch-size <n>]');
  process.exit(2);
}

const store = new SessionStore();
// The provider is only used for its one-shot observer prompt; it starts no session.
const provider = backfill ? new ClaudeProvider({ getSessionStore: () => store } as never, null as never) : null;
const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
const mode = ModeManager.getInstance().loadMode(settings.CLAUDE_MEM_MODE || 'code');

function resolveSessionIds(): number[] {
  if (recent) {
    const rows = store.db.prepare(`
      SELECT DISTINCT session_db_id AS id FROM observer_event_ledger ORDER BY session_db_id DESC LIMIT ?
    `).all(Number(recent)) as Array<{ id: number }>;
    return rows.map(r => r.id);
  }
  if (/^\d+$/.test(sessionArg!)) return [Number(sessionArg)];
  const row = store.db.prepare('SELECT id FROM sdk_sessions WHERE content_session_id = ?').get(sessionArg!) as { id: number } | null;
  if (!row) {
    console.error(`No session found for ${sessionArg}`);
    process.exit(1);
  }
  return [row.id];
}

let exitCode = 0;
for (const sessionDbId of resolveSessionIds()) {
  const row = store.getSessionById(sessionDbId);
  const context = { sessionDbId, contentSessionId: row?.content_session_id ?? '', project: row?.project ?? '' };
  const report = await auditObserverSession(store, mode, sessionDbId, {
    complete: provider ? (prompt) => provider.runObserverBackfillPrompt(prompt, context) : undefined,
    batchSize,
    modelId: settings.CLAUDE_MEM_MODEL,
  });
  const ids = (list: number[]) => (list.length ? list.map(id => `evt_${id}`).join(' ') : '(none)');
  console.log(`session ${sessionDbId}: ${report.totalEvents} events, ${report.unobservedBefore.length} unobserved`);
  console.log(`  unobserved:    ${ids(report.unobservedBefore)}`);
  if (backfill) {
    console.log(`  backfilled:    ${ids(report.backfilled)}`);
    console.log(`  skipped:       ${ids(report.skipped)}`);
    console.log(`  still missing: ${ids(report.stillMissing)}  (batches=${report.batches}, retries=${report.retries})`);
  }
  if (report.stillMissing.length > 0) exitCode = 1;
}
process.exit(exitCode);
