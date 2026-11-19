/**
 * Post-session audit of the observer event ledger.
 *
 * Lists every queued tool event that never received an observation or a skip
 * (including events the in-RAM queue expired or the live observer dropped),
 * then optionally sends them in batches to a FRESH observer agent to backfill.
 * Backfilled output passes the same strict validator, with one retry that
 * includes the violations. Events still open afterwards are reported.
 */

import type { SessionStore } from '../sqlite/SessionStore.js';
import type { ModeConfig } from '../domain/types.js';
import { ObserverEventLedger, type LedgerEvent } from '../sqlite/ObserverEventLedger.js';
import {
  accountForBatch,
  buildRepairPrompt,
  formatEventId,
  validateObserverResponse,
  type ValidatedObservation,
  type ValidationContext,
  type Violation,
} from '../../sdk/observation-validator.js';
import { buildInitPrompt, buildObservationPrompt } from '../../sdk/prompts.js';
import { logger } from '../../utils/logger.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../shared/paths.js';

/** Events per backfill request; matches the observer's own batch ceiling. */
export const AUDIT_BATCH_SIZE = 25;

export interface AuditOptions {
  /** Fresh-agent completion. Omit for a report-only audit. */
  complete?: (prompt: string) => Promise<string | null>;
  batchSize?: number;
  modelId?: string;
  onObservationStored?: (params: { observationId: number; observation: ValidatedObservation; createdAtEpoch: number }) => void;
}

export interface AuditReport {
  sessionDbId: number;
  totalEvents: number;
  unobservedBefore: number[];
  backfilled: number[];
  skipped: number[];
  stillMissing: number[];
  batches: number;
  retries: number;
}

const BACKFILL_HEADER = `BACKFILL REQUEST
================
The tool events below happened in the primary session but were never recorded. Account for every one of them now.`;

function chunk<T>(items: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

function eventPrompt(event: LedgerEvent): string {
  return buildObservationPrompt({
    id: event.event_id,
    tool_name: event.tool_name,
    tool_input: event.tool_input ?? '{}',
    tool_output: event.tool_response ?? '{}',
    created_at_epoch: event.tool_created_at_epoch,
    cwd: event.cwd ?? undefined,
  });
}

export async function auditObserverSession(
  sessionStore: SessionStore,
  mode: ModeConfig,
  sessionDbId: number,
  options: AuditOptions = {}
): Promise<AuditReport> {
  const ledger = new ObserverEventLedger(sessionStore.db);
  const open = ledger.listUnaccounted(sessionDbId);
  const report: AuditReport = {
    sessionDbId,
    totalEvents: ledger.summarize(sessionDbId).total,
    unobservedBefore: open.map(r => r.event_id),
    backfilled: [],
    skipped: [],
    stillMissing: [],
    batches: 0,
    retries: 0,
  };

  logger.info('LEDGER', `AUDIT | sessionDbId=${sessionDbId} | events=${report.totalEvents} | unobserved=${open.length} | ids=[${open.map(r => formatEventId(r.event_id)).join(',')}]`, {
    sessionId: sessionDbId,
  });
  if (open.length === 0) return report;

  const session = sessionStore.getSessionById(sessionDbId);
  if (!options.complete || !session?.memory_session_id) {
    if (options.complete) {
      logger.warn('LEDGER', 'AUDIT cannot backfill: the session has no memory_session_id to store observations under', { sessionId: sessionDbId });
    }
    report.stillMissing = report.unobservedBefore;
    return report;
  }

  const instructions = `${buildInitPrompt(session.project, session.content_session_id, session.user_prompt || '', mode)}\n\n${BACKFILL_HEADER}`;
  const validTypes = mode.observation_types.map(t => t.id);
  const validConcepts = (mode.observation_concepts ?? []).map(c => c.id);
  const strictFields = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH).CLAUDE_MEM_OBSERVER_STRICT_FIELDS === 'true';

  for (const batch of chunk(open, options.batchSize ?? AUDIT_BATCH_SIZE)) {
    report.batches++;
    const ids = batch.map(e => e.event_id);
    const ctx: ValidationContext = { validTypes, validConcepts, knownEventIds: new Set(ids), requireEventIds: true, strictFields };
    const accepted = await backfillBatch(batch, instructions, ctx, options, report);
    storeBackfill(sessionStore, ledger, session.memory_session_id, session.project, session.platform_source, session.content_session_id, batch, accepted, options, report);
    const done = new Set([...report.backfilled, ...report.skipped]);
    for (const id of ids) if (!done.has(id)) report.stillMissing.push(id);
  }

  logger.info('LEDGER', `AUDIT_DONE | sessionDbId=${sessionDbId} | backfilled=${report.backfilled.length} | skipped=${report.skipped.length} | stillMissing=[${report.stillMissing.map(formatEventId).join(',')}] | batches=${report.batches} | retries=${report.retries}`, {
    sessionId: sessionDbId,
  });
  return report;
}

interface BackfillOutput {
  observations: ValidatedObservation[];
  skipped: { ids: number[]; reason: string | null };
}

async function backfillBatch(
  batch: LedgerEvent[],
  instructions: string,
  ctx: ValidationContext,
  options: AuditOptions,
  report: AuditReport
): Promise<BackfillOutput> {
  const out: BackfillOutput = { observations: [], skipped: { ids: [], reason: null } };
  const absorb = (text: string, allowed: number[]) => {
    const result = validateObserverResponse(text, { ...ctx, knownEventIds: new Set(allowed) });
    out.observations.push(...result.observations);
    const { skipped } = accountForBatch(result, allowed.filter(id => !out.observations.some(o => o.eventIds.includes(id))));
    out.skipped.ids.push(...skipped);
    if (skipped.length > 0) out.skipped.reason = result.skipAll?.reason ?? out.skipped.reason;
    return result.violations;
  };
  const openIds = () => batch.map(e => e.event_id)
    .filter(id => !out.observations.some(o => o.eventIds.includes(id)) && !out.skipped.ids.includes(id));

  let text: string | null = null;
  try {
    text = await options.complete!(`${instructions}\n\n${batch.map(eventPrompt).join('\n\n')}`);
  } catch (error) {
    logger.error('LEDGER', 'AUDIT backfill call failed', {}, error instanceof Error ? error : new Error(String(error)));
    return out;
  }
  const violations: Violation[] = absorb(text ?? '', batch.map(e => e.event_id));
  const remaining = openIds();
  if (remaining.length === 0) return out;

  for (const id of remaining) violations.push({ code: 'MISSING_COVERAGE', message: `no <observation> covered ${formatEventId(id)}, and there was no <skip_summary/>` });
  report.retries++;
  try {
    const retryText = await options.complete!(buildRepairPrompt({
      instructions,
      eventPrompts: batch.filter(e => remaining.includes(e.event_id)).map(eventPrompt),
      rejectedResponse: text ?? '',
      violations,
    }));
    absorb(retryText ?? '', remaining);
  } catch (error) {
    logger.error('LEDGER', 'AUDIT backfill retry failed', {}, error instanceof Error ? error : new Error(String(error)));
  }
  return out;
}

function storeBackfill(
  sessionStore: SessionStore,
  ledger: ObserverEventLedger,
  memorySessionId: string,
  project: string,
  platformSource: string,
  contentSessionId: string,
  batch: LedgerEvent[],
  accepted: BackfillOutput,
  options: AuditOptions,
  report: AuditReport
): void {
  const byId = new Map(batch.map(e => [e.event_id, e]));

  for (const obs of accepted.observations) {
    const events = obs.eventIds.map(id => byId.get(id)).filter((e): e is LedgerEvent => !!e);
    if (events.length === 0) continue;
    const first = events[0];
    const { eventIds: _eventIds, ...fields } = obs;
    const result = sessionStore.storeObservations(
      memorySessionId,
      project,
      [fields],
      null,
      first.prompt_number ?? undefined,
      0,
      first.tool_created_at_epoch,
      options.modelId,
      { committedToolUses: { platformSource, contentSessionId, toolUseIds: events.map(e => e.tool_use_id) } }
    );
    const observationId = result.observationIds[0];
    if (observationId === undefined) continue;
    ledger.markObserved(obs.eventIds, observationId, 'backfilled');
    for (const id of obs.eventIds) if (!report.backfilled.includes(id)) report.backfilled.push(id);
    options.onObservationStored?.({ observationId, observation: obs, createdAtEpoch: result.createdAtEpoch });
  }

  const skipped = accepted.skipped.ids.filter(id => !report.backfilled.includes(id));
  if (skipped.length > 0) {
    ledger.markSkipped(skipped, accepted.skipped.reason ?? 'skipped by the backfill observer');
    report.skipped.push(...skipped);
  }
}
