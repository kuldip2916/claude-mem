/**
 * The observer output contract, enforced on every reply to a batch of tool
 * events, plus the per-event ledger outcome of that reply.
 *
 * The parser (src/sdk/parser.ts) is deliberately lenient: it salvages prose,
 * falls back on a missing type, and stores whatever parses. This runs after a
 * valid parse and before storage. It keeps only observations that pass the
 * strict validator (well-formed XML, required fields, mode vocabulary, no
 * "Human:" turn, no advice), checks that every event of the batch is either
 * observed or skipped, and retries once through a fresh agent with the
 * violations included. Events still uncovered are marked 'dropped' in the
 * ledger, where the post-session audit picks them up.
 */

import { logger } from '../../../utils/logger.js';
import { ModeManager } from '../../domain/ModeManager.js';
import { buildInitPrompt, buildObservationPrompt } from '../../../sdk/prompts.js';
import type { ParsedObservation } from '../../../sdk/parser.js';
import {
  accountForBatch,
  buildRepairPrompt,
  formatEventId,
  validateObserverResponse,
  type ValidatedObservation,
  type ValidationContext,
  type Violation,
} from '../../../sdk/observation-validator.js';
import { ObserverEventLedger } from '../../sqlite/ObserverEventLedger.js';
import { observationFieldMaxChars } from '../context-window.js';
import { SettingsDefaultsManager } from '../../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../../shared/paths.js';
import type { ActiveSession, PendingMessageWithId } from '../../worker-types.js';
import type { SessionManager } from '../SessionManager.js';
import type { DatabaseManager } from '../DatabaseManager.js';

export interface ContractOutcome {
  /** Observations to store, in the shape storage expects, in the same order as eventIds. */
  observations: ParsedObservation[];
  /** Event ids each stored observation cites (parallel to observations). */
  eventIds: number[][];
  skipped: number[];
  skipReason: string | null;
  /** Batch events no accepted output covers, even after the retry. */
  uncovered: number[];
}

function citedEventIds(text: string): number[] {
  const ids = new Set<number>();
  for (const match of text.matchAll(/\bevt_(\d+)\b/g)) ids.add(Number(match[1]));
  return [...ids];
}

function toParsed(obs: ValidatedObservation): ParsedObservation {
  const { eventIds: _eventIds, ...fields } = obs;
  return fields;
}

function batchEventPrompt(message: PendingMessageWithId, fieldMaxChars: number): string {
  return buildObservationPrompt({
    id: message.eventId ?? 0,
    tool_name: message.tool_name ?? 'unknown',
    tool_input: JSON.stringify(message.tool_input),
    tool_output: JSON.stringify(message.tool_response),
    created_at_epoch: message._originalTimestamp,
    cwd: message.cwd,
  }, fieldMaxChars);
}

/**
 * Validate a reply to the claimed observation batch and repair it once.
 * Returns null when the reply does not answer an observation batch (init,
 * summarize), so the caller keeps its existing path unchanged.
 */
export async function enforceObservationContract(
  text: string,
  session: ActiveSession,
  sessionManager: SessionManager,
  agentName: string,
): Promise<ContractOutcome | null> {
  if (session.lastGeneratorSource !== 'ingest') return null;
  const claimed = sessionManager.getClaimedMessages(session.sessionDbId).filter(m => m.type === 'observation');
  if (claimed.length === 0) return null;

  const batchIds = claimed.map(m => m.eventId).filter((id): id is number => typeof id === 'number');
  const requireEventIds = batchIds.length > 0 && batchIds.length === claimed.length;
  const mode = ModeManager.getInstance().getActiveMode();
  const ctx: ValidationContext = {
    validTypes: mode.observation_types.map(t => t.id),
    validConcepts: (mode.observation_concepts ?? []).map(c => c.id),
    knownEventIds: new Set(batchIds),
    requireEventIds,
    strictFields: SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH).CLAUDE_MEM_OBSERVER_STRICT_FIELDS === 'true',
  };

  const first = validateObserverResponse(text, ctx);
  let observations = [...first.observations];
  let skipAll = first.skipAll;
  const uncoveredNow = (): number[] => {
    if (!requireEventIds) return observations.length === 0 && !skipAll ? [...batchIds] : [];
    const { observed, skipped } = accountForBatch({ observations, skipAll }, batchIds);
    return batchIds.filter(id => !observed.has(id) && !skipped.includes(id));
  };

  const violations: Violation[] = [...first.violations];
  const missing = uncoveredNow();
  for (const id of missing) {
    violations.push({ code: 'MISSING_COVERAGE', message: `no <observation> covered ${formatEventId(id)}, and there was no <skip_summary/>` });
  }
  const nothingAccepted = observations.length === 0 && !skipAll;

  if (violations.length > 0) {
    logger.warn('PARSER', `${agentName} reply REJECTED by the output contract | sessionDbId=${session.sessionDbId} | codes=[${[...new Set(violations.map(v => v.code))].join(',')}] | accepted=${observations.length} | uncovered=[${missing.map(formatEventId).join(',')}]`, {
      sessionId: session.sessionDbId,
      violations: violations.slice(0, 8).map(v => `${v.code}: ${v.message}`).join(' | '),
    });
  }

  // Retry when an event is left without a valid observation or skip. A reply
  // whose events are all covered keeps its valid blocks; its rejected extras
  // are dropped and logged above rather than re-asked.
  const needsRepair = missing.length > 0 || (!requireEventIds && nothingAccepted && violations.length > 0);
  if (needsRepair && session.observerRepair) {
    const targets = requireEventIds ? missing : [];
    const targetMessages = requireEventIds ? claimed.filter(m => targets.includes(m.eventId!)) : claimed;
    const fieldMaxChars = observationFieldMaxChars(session.observerContextWindowTokens);
    const prompt = buildRepairPrompt({
      instructions: buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode),
      eventPrompts: targetMessages.map(m => batchEventPrompt(m, fieldMaxChars)),
      rejectedResponse: text,
      violations,
    });

    let retryText: string | null = null;
    try {
      retryText = await session.observerRepair(prompt);
    } catch (error) {
      logger.error('PARSER', `${agentName} contract repair call failed; uncovered events go to the audit`, { sessionId: session.sessionDbId },
        error instanceof Error ? error : new Error(String(error)));
    }

    if (retryText !== null) {
      const retry = validateObserverResponse(retryText, requireEventIds ? { ...ctx, knownEventIds: new Set(targets) } : ctx);
      if (requireEventIds) {
        observations.push(...retry.observations.filter(o => o.eventIds.some(id => targets.includes(id))));
        if (retry.skipAll && !skipAll) skipAll = retry.skipAll;
      } else if (retry.observations.length > 0 || retry.skipAll) {
        observations = retry.observations;
        skipAll = retry.skipAll;
      }
      logger.info('PARSER', `${agentName} contract repair ${retry.ok ? 'ACCEPTED' : 'PARTIAL'} | sessionDbId=${session.sessionDbId} | obs=${retry.observations.length} | stillUncovered=[${uncoveredNow().map(formatEventId).join(',')}] | codes=[${[...new Set(retry.violations.map(v => v.code))].join(',')}]`, {
        sessionId: session.sessionDbId,
      });
    }
  }

  const { observed, skipped } = accountForBatch({ observations, skipAll }, batchIds);
  return {
    observations: observations.map(toParsed),
    eventIds: observations.map(o => o.eventIds),
    skipped,
    skipReason: skipAll?.reason ?? null,
    // Coverage is only known when every event of the batch carried an id.
    uncovered: requireEventIds ? batchIds.filter(id => !observed.has(id) && !skipped.includes(id)) : [],
  };
}

function ledgerFor(dbManager: DatabaseManager): ObserverEventLedger | null {
  try {
    return new ObserverEventLedger(dbManager.getSessionStore().db);
  } catch {
    return null;
  }
}

/** Write a stored reply's per-event outcome: observation ids, skips, and drops. */
export function recordContractOutcome(
  dbManager: DatabaseManager,
  session: ActiveSession,
  outcome: ContractOutcome,
  observationIds: readonly number[],
  agentName: string,
): void {
  const ledger = ledgerFor(dbManager);
  if (!ledger) return;
  try {
    const observedLog = new Map<number, number[]>();
    outcome.eventIds.forEach((eventIds, index) => {
      const observationId = observationIds[index];
      if (observationId === undefined || eventIds.length === 0) return;
      ledger.markObserved(eventIds, observationId);
      for (const id of eventIds) observedLog.set(id, [...(observedLog.get(id) ?? []), observationId]);
    });
    if (outcome.skipped.length > 0) ledger.markSkipped(outcome.skipped, outcome.skipReason ?? 'skipped by the observer');
    if (outcome.uncovered.length > 0) ledger.markDropped(outcome.uncovered, 'no valid observation after one retry');
    ledger.logOutcome(session.sessionDbId, observedLog, outcome.skipped, agentName);
  } catch (error) {
    logger.warn('LEDGER', 'Observer ledger outcome write failed', { sessionId: session.sessionDbId },
      error instanceof Error ? error : new Error(String(error)));
  }
}

/** Mark the claimed batch's events in the ledger when a path settles it without storage. */
export function recordClaimedBatch(
  dbManager: DatabaseManager,
  sessionManager: SessionManager,
  session: ActiveSession,
  status: 'skipped' | 'dropped',
  reason: string,
): void {
  const eventIds = sessionManager.getClaimedMessages(session.sessionDbId)
    .map(m => m.eventId)
    .filter((id): id is number => typeof id === 'number');
  if (eventIds.length === 0) return;
  const ledger = ledgerFor(dbManager);
  if (!ledger) return;
  try {
    if (status === 'skipped') ledger.markSkipped(eventIds, reason);
    else ledger.markDropped(eventIds, reason);
  } catch (error) {
    logger.warn('LEDGER', `Observer ledger ${status} write failed`, { sessionId: session.sessionDbId },
      error instanceof Error ? error : new Error(String(error)));
  }
}
