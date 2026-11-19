/**
 * Replays a long observer session through the real pipeline: ingest
 * (tool_uses row, event id, ledger), the in-RAM queue, ClaudeProvider with its
 * pacing and conversation recycling, ResponseProcessor with the output
 * contract, the real generator-exit handling, and the post-session audit run
 * by SessionCompletionHandler. A fake Agent SDK stands in for the model.
 *
 * The fake answers each event with the response the real observer gave in the
 * recording, so recorded defects (hallucinated "Human:" turns, prose, empty
 * replies, unescaped XML, advice) reach the validator as they did in
 * production. Its context is a 200K-token window that replies "Prompt is too
 * long" when exceeded. It only corrects itself when a repair prompt carries the
 * validation errors.
 *
 * Recorded fixture: tests/fixtures/private/observer-session-*.jsonl, or
 * CLAUDE_MEM_REPLAY_TRANSCRIPT. Recordings hold private session content, so the
 * directory is gitignored; without one that test is skipped and the synthetic
 * replay still runs.
 *
 * Live smoke (opt-in, spends model quota): CLAUDE_MEM_LIVE_SMOKE=1 swaps the
 * fake for the REAL Agent SDK and model (CLAUDE_MEM_LIVE_MODEL, default
 * claude-sonnet-4-6) and replays the first CLAUDE_MEM_LIVE_EVENTS (default 20)
 * recorded events. Every other test in this file is skipped in that mode.
 */
import { afterAll, beforeAll, describe, expect, it, mock, spyOn } from 'bun:test';
import { Database } from 'bun:sqlite';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import codeMode from '../../plugin/modes/code.json';

// Snapshot the real modules before installing process-global mocks; restored in afterAll.
const actualAgentSdk = { ...(await import('@anthropic-ai/claude-agent-sdk')) };
const actualFindClaude = { ...(await import('../../src/shared/find-claude-executable.js')) };
const actualEnvManager = { ...(await import('../../src/shared/EnvManager.js')) };
const actualProcessRegistry = { ...(await import('../../src/supervisor/process-registry.js')) };
const actualModeManager = { ...(await import('../../src/services/domain/ModeManager.js')) };
const actualContextGenerator = { ...(await import('../../src/services/context-generator.js')) };

const LIVE = process.env.CLAUDE_MEM_LIVE_SMOKE === '1';
const CONTEXT_LIMIT_TOKENS = 200_000;
// Conservative: JSON and code run ~3-4 chars per token, so 3 overstates tokens.
const CHARS_PER_TOKEN = 3;
// At most this share of events may still be open when the live observer stops.
const LIVE_OPEN_BUDGET = 0.15;

// ---------------------------------------------------------------------------
// Recorded session loading
// ---------------------------------------------------------------------------

interface ReplayEvent { toolName: string; toolInput: unknown; toolResponse: unknown }
interface ReplaySession { name: string; events: ReplayEvent[]; responses: string[] }

function decodeNested(text: string): unknown {
  let value: unknown = text;
  for (let i = 0; i < 3 && typeof value === 'string'; i++) {
    try { value = JSON.parse(value); } catch { break; }
  }
  return value;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.filter((c: any) => c?.type === 'text').map((c: any) => c.text ?? '').join('\n');
}

function loadRecordedSession(path: string): ReplaySession {
  const events: ReplayEvent[] = [];
  const responses: string[] = [];
  let current = -1;
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    let row: any;
    try { row = JSON.parse(line); } catch { continue; }
    if (row.type === 'user') {
      const text = textOf(row.message?.content);
      const tool = /<what_happened>([\s\S]*?)<\/what_happened>/.exec(text);
      if (!tool) { current = -1; continue; }
      const params = /<parameters>([\s\S]*?)<\/parameters>\s*<outcome>/.exec(text);
      const outcome = /<outcome>([\s\S]*)<\/outcome>\s*<\/observed_from_primary_session>/.exec(text);
      events.push({ toolName: tool[1].trim(), toolInput: decodeNested(params?.[1] ?? '{}'), toolResponse: decodeNested(outcome?.[1] ?? '') });
      responses.push('');
      current = events.length - 1;
    } else if (row.type === 'assistant' && current >= 0) {
      const text = textOf(row.message?.content);
      if (text) responses[current] = responses[current] ? `${responses[current]}\n${text}` : text;
    }
  }
  return { name: path.split('/').pop()!, events, responses };
}

function findRecording(): string | null {
  const fromEnv = process.env.CLAUDE_MEM_REPLAY_TRANSCRIPT;
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  const dir = join(import.meta.dir, '../fixtures/private');
  if (!existsSync(dir)) return null;
  const file = readdirSync(dir).find(f => /^observer-session-.*\.jsonl$/.test(f));
  return file ? join(dir, file) : null;
}

/** A deterministic stand-in with the same failure mix, for runs without a recording. */
function syntheticSession(count: number): ReplaySession {
  const events: ReplayEvent[] = [];
  const responses: string[] = [];
  for (let i = 0; i < count; i++) {
    const big = i % 7 === 3;
    events.push({
      toolName: i % 2 ? 'Read' : 'Bash',
      toolInput: i % 2 ? { file_path: `/repo/src/module-${i}.ts` } : { command: `grep -rn "handler${i}" src && echo done` },
      toolResponse: { stdout: 'line of output with <tags> & ampersands\n'.repeat(big ? 1500 : 20), exit: 0 },
    });
    const good = `<observation><type>discovery</type><title>Module ${i} wires handler${i} into the router</title><subtitle>handler${i} is registered at startup</subtitle><facts><fact>src/module-${i}.ts exports handler${i}</fact></facts><narrative>The router imports handler${i} from module-${i} and registers it during boot.</narrative><concepts><concept>how-it-works</concept></concepts><files_read><file>src/module-${i}.ts</file></files_read></observation>`;
    const kind = i % 10;
    responses.push(
      kind === 0 && i % 20 === 0 ? good.replace('during boot.', 'during boot, keeping tokens out of the agent context window.')
        : kind < 5 ? good
        : kind === 5 ? good.replace('during boot.', 'during boot && on reload.')
        : kind === 6 ? `${good}\n\nHuman: <observed_from_primary_session>invented</observed_from_primary_session>`
        : kind === 7 ? 'Skipping: nothing durable here.'
        : kind === 8 ? ''
        : good.replace('during boot.', 'during boot. Next steps: add tests for the router.')
    );
  }
  return { name: `synthetic-${count}`, events, responses };
}

// ---------------------------------------------------------------------------
// Fake Agent SDK
// ---------------------------------------------------------------------------

const escapeXml = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function eventIdsInPrompt(prompt: string): number[] {
  // Only ids inside event blocks; a rejected reply quoted in a repair prompt
  // may cite ids too, and those must not count as "sent".
  return [...prompt.matchAll(/<observed_from_primary_session>\s*<event_id>evt_(\d+)<\/event_id>/g)].map(m => Number(m[1]));
}

function cleanObservation(id: number): string {
  return `<observation><type>discovery</type><title>${escapeXml(`Replayed tool event evt_${id}`)}</title><subtitle>Observation re-recorded for a replayed tool event</subtitle><facts><fact>Event evt_${id} was recorded by the replay observer</fact></facts><narrative>The replay observer recorded tool event evt_${id}.</narrative><concepts><concept>how-it-works</concept></concepts><event_ids><event_id>evt_${id}</event_id></event_ids></observation>`;
}

interface FakeStats {
  generations: number;
  turns: number;
  maxContextTokens: number;
  overflowReplies: number;
  eventsPerGeneration: number[];
  repairCalls: number;
  repairsWithoutErrors: number;
  stubbornRepairs: number;
  backfillCalls: number;
  otherOneShots: number;
  correctedRetries: number;
}

class FakeObserverModel {
  stats: FakeStats = {
    generations: 0, turns: 0, maxContextTokens: 0, overflowReplies: 0, eventsPerGeneration: [],
    repairCalls: 0, repairsWithoutErrors: 0, stubbornRepairs: 0, backfillCalls: 0, otherOneShots: 0, correctedRetries: 0,
  };
  idle = false;
  readonly responseByEventId = new Map<number, string>();

  constructor(private readonly contextLimitTokens: number) {}

  query = (params: { prompt: string | AsyncIterable<any>; options?: any }): any =>
    typeof params.prompt === 'string' ? this.oneShot(params.prompt) : this.streaming(params.prompt, params.options?.abortController?.signal);

  private respond(message: string): string {
    // A re-asked batch that carries the validation errors gets a corrected answer.
    if (/VALIDATION ERRORS\n- \[[A-Z_]+\] /.test(message)) {
      this.stats.correctedRetries++;
      return eventIdsInPrompt(message).map(cleanObservation).join('\n');
    }
    return eventIdsInPrompt(message).map(id => {
      const recorded = this.responseByEventId.get(id) ?? '';
      if (!recorded.includes('<observation>')) return recorded;
      // A model following the new prompt cites the event; everything else stays as recorded.
      return recorded.replace(/<\/observation>/g, `<event_ids><event_id>evt_${id}</event_id></event_ids></observation>`);
    }).filter(Boolean).join('\n\n');
  }

  private async *streaming(input: AsyncIterable<any>, signal?: AbortSignal) {
    const generation = ++this.stats.generations;
    const sessionId = `fake-sdk-session-${generation}`;
    const buffer: string[] = [];
    let inputDone = false;
    let wake: (() => void) | null = null;
    const pump = async () => {
      for await (const m of input) {
        buffer.push(typeof m.message.content === 'string' ? m.message.content : JSON.stringify(m.message.content));
        wake?.();
      }
      inputDone = true;
      wake?.();
    };
    void pump().catch(() => { inputDone = true; wake?.(); });
    signal?.addEventListener('abort', () => { inputDone = true; wake?.(); }, { once: true });

    yield { type: 'system', subtype: 'init', session_id: sessionId };
    let context = 0;
    let broken = false;
    let events = 0;
    try {
      for (;;) {
        if (buffer.length === 0) {
          if (inputDone) break;
          this.idle = true;
          await new Promise<void>(resolve => { wake = resolve; });
          wake = null;
          this.idle = false;
          continue;
        }
        const message = buffer.shift()!;
        this.stats.turns++;
        events += eventIdsInPrompt(message).length;
        context += message.length;
        let text: string;
        if (broken || context / CHARS_PER_TOKEN > this.contextLimitTokens) {
          broken = true;
          this.stats.overflowReplies++;
          text = 'Prompt is too long';
        } else {
          text = this.respond(message);
          context += text.length;
        }
        const contextTokens = Math.round(context / CHARS_PER_TOKEN);
        this.stats.maxContextTokens = Math.max(this.stats.maxContextTokens, contextTokens);
        const overflow = text === 'Prompt is too long';
        yield { type: 'assistant', session_id: sessionId, message: { content: [{ type: 'thinking', thinking: '...' }], usage: { input_tokens: 1, output_tokens: 1 } } };
        yield {
          type: 'assistant', session_id: sessionId, ...(overflow ? { error: 'invalid_request' } : {}),
          message: { model: overflow ? '<synthetic>' : 'fake-model', content: [{ type: 'text', text }], usage: { input_tokens: contextTokens, output_tokens: 1 } },
        };
        yield { type: 'result', subtype: 'success', is_error: false, session_id: sessionId, usage: { input_tokens: contextTokens, output_tokens: Math.ceil(text.length / CHARS_PER_TOKEN) } };
      }
    } finally {
      this.stats.eventsPerGeneration.push(events);
      this.idle = false;
    }
  }

  private async *oneShot(prompt: string) {
    const isBackfill = prompt.includes('BACKFILL REQUEST');
    const isRepair = prompt.includes('VALIDATION ERRORS');
    const ids = eventIdsInPrompt(prompt);
    let text: string;
    if (isRepair && !/^- \[[A-Z_]+\] /m.test(prompt)) {
      // No concrete errors to act on: repeat the rejected answer.
      this.stats.repairsWithoutErrors++;
      text = /<rejected_response>\n([\s\S]*?)\n<\/rejected_response>/.exec(prompt)?.[1] ?? '';
    } else if (isRepair && !isBackfill) {
      this.stats.repairCalls++;
      if (this.stats.repairCalls % 5 === 0) {
        // Every fifth live repair is still wrong (no narrative): the pipeline
        // must re-validate it and leave those events to the audit.
        this.stats.stubbornRepairs++;
        text = ids.map(id => cleanObservation(id).replace(/<narrative>.*?<\/narrative>/, '')).join('\n');
      } else {
        text = ids.map(id => id % 4 === 0 && ids.length === 1 ? '<skip_summary reason="routine tool call with no durable finding"/>' : cleanObservation(id)).join('\n');
      }
    } else if (isBackfill && !isRepair) {
      this.stats.backfillCalls++;
      // The first backfill pass misses the last event, so the audit must retry.
      text = (ids.length > 1 ? ids.slice(0, -1) : ids).map(cleanObservation).join('\n');
    } else if (isBackfill) {
      this.stats.backfillCalls++;
      text = ids.map(cleanObservation).join('\n');
    } else {
      // Field condensing and other standalone prompts.
      this.stats.otherOneShots++;
      text = 'Condensed: the tool printed a long listing; the head and tail were kept.';
    }
    yield { type: 'system', subtype: 'init', session_id: `fake-one-shot-${Math.random()}` };
    yield { type: 'assistant', message: { content: [{ type: 'text', text }] } };
    yield { type: 'result', subtype: 'success', is_error: false, result: text, usage: { input_tokens: 1, output_tokens: 1 } };
  }
}

// ---------------------------------------------------------------------------
// Module mocks and lazily imported pipeline
// ---------------------------------------------------------------------------

let activeFake: FakeObserverModel | null = null;
let activeStore: any = null;
const logLines: string[] = [];
let tmpCwd = '';

/** Live mode: what the real SDK stream is doing, for idle detection and the report. */
const live = { lastType: '', lastAt: Date.now(), activeOneShots: 0, generations: 0, results: 0, oneShots: 0, overflowReplies: 0 };

function liveQuery(params: { prompt: string | AsyncIterable<any>; options?: any }): any {
  const streaming = typeof params.prompt !== 'string';
  if (streaming) live.generations++;
  else { live.oneShots++; live.activeOneShots++; }
  const inner = actualAgentSdk.query(params as any);
  return (async function* () {
    try {
      for await (const m of inner as AsyncIterable<any>) {
        if (streaming) {
          live.lastType = m.type;
          live.lastAt = Date.now();
          if (m.type === 'result') live.results++;
          if (m.type === 'assistant' && textOf(m.message?.content).trim() === 'Prompt is too long') live.overflowReplies++;
        }
        yield m;
      }
    } finally {
      if (!streaming) live.activeOneShots--;
    }
  })();
}

mock.module('@anthropic-ai/claude-agent-sdk', () => ({ ...actualAgentSdk,
  query: (params: any) => (LIVE ? liveQuery(params) : activeFake!.query(params)),
}));
mock.module('../../src/shared/EnvManager.js', () => ({ ...actualEnvManager,
  // Live: the CLI's own login, as the user's shell has it.
  buildIsolatedEnvWithFreshOAuth: async () => (LIVE ? { ...process.env } : { PATH: process.env.PATH ?? '' }),
  getAuthMethodDescription: () => 'test-auth',
}));
if (!LIVE) {
  mock.module('../../src/shared/find-claude-executable.js', () => ({ ...actualFindClaude, findClaudeExecutable: () => '/mock/claude' }));
  mock.module('../../src/supervisor/process-registry.js', () => ({ ...actualProcessRegistry,
    waitForSlot: async () => ({ release: () => {} }),
    createSdkSpawnFactory: () => () => { throw new Error('spawn factory must not run in this test'); },
    getSdkProcessForSession: () => undefined,
    ensureSdkProcessExit: async () => {},
  }));
}
mock.module('../../src/services/domain/ModeManager.js', () => ({ ...actualModeManager,
  ModeManager: { getInstance: () => ({ getActiveMode: () => codeMode }) },
}));
// A recycled generation is seeded with the session's own stored observations.
mock.module('../../src/services/context-generator.js', () => ({ ...actualContextGenerator,
  generateContext: async () => {
    if (!activeStore) return '';
    const rows = activeStore.db.prepare('SELECT id, type, title FROM observations ORDER BY id DESC LIMIT 60').all() as any[];
    return rows.length ? `Recent observations:\n${rows.map(r => `- #${r.id} (${r.type}) ${r.title}`).join('\n')}` : '';
  },
}));

const { SessionStore } = await import('../../src/services/sqlite/SessionStore.js');
const { SessionManager } = await import('../../src/services/worker/SessionManager.js');
const { ClaudeProvider } = await import('../../src/services/worker/ClaudeProvider.js');
const { SessionCompletionHandler } = await import('../../src/services/worker/session/SessionCompletionHandler.js');
const { handleGeneratorExit } = await import('../../src/services/worker/session/GeneratorExitHandler.js');
const { ingestObservation, setIngestContext } = await import('../../src/services/worker/http/shared.js');
const { ObserverEventLedger } = await import('../../src/services/sqlite/ObserverEventLedger.js');
const { checkObservationFields } = await import('../../src/sdk/observation-validator.js');
const { logger } = await import('../../src/utils/logger.js');
const { LIVE_OBSERVER_MAX_SESSION_MESSAGES } = await import('../../src/shared/observer-work-policy.js');

// ---------------------------------------------------------------------------
// Replay driver
// ---------------------------------------------------------------------------

async function collectOneShot(fake: FakeObserverModel, prompt: string): Promise<string> {
  const parts: string[] = [];
  for await (const m of fake.query({ prompt })) if (m.type === 'assistant') parts.push(textOf(m.message.content));
  return parts.join('\n');
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

interface ReplayResult {
  store: any;
  ledger: InstanceType<typeof ObserverEventLedger>;
  sessionDbId: number;
  fake: FakeObserverModel;
  unaccountedAfterLive: number;
  exitReasons: string[];
  liveErrors: unknown[];
  /** Events ingest refused (live buffer full) and only the ledger kept. */
  refusedAtIngest: number;
}

async function replay(
  session: ReplaySession,
  opts: { contextLimitTokens: number; burst?: boolean; liveModel?: string },
): Promise<ReplayResult> {
  const store = new SessionStore(new Database(':memory:'));
  activeStore = store;
  const fake = new FakeObserverModel(opts.contextLimitTokens);
  activeFake = fake;
  const dbManager = {
    getSessionStore: () => store,
    getSessionById: (id: number) => store.getSessionById(id),
    getChromaSync: () => null,
    getCloudSync: () => null,
  };
  const sessionManager = new SessionManager(dbManager as never);
  const provider = new ClaudeProvider(dbManager as never, sessionManager as never);
  const completion = new SessionCompletionHandler(sessionManager, { broadcastSessionCompleted: () => {} } as never, dbManager as never,
    opts.liveModel ? (prompt, context) => provider.runObserverBackfillPrompt(prompt, context) : prompt => collectOneShot(fake, prompt));
  setIngestContext({
    sessionManager,
    dbManager,
    eventBroadcaster: { broadcastObservationQueued: () => {}, broadcastSummarizeQueued: () => {} },
    ensureGeneratorRunning: async () => {},
  } as never);

  const contentSessionId = `replay-${session.name}-${opts.contextLimitTokens}`;
  const sessionDbId = store.createSDKSession(contentSessionId, 'replay-project', `Replay of ${session.name}`, undefined, 'claude');
  const ledger = new ObserverEventLedger(store.db);
  const buffer = () => sessionManager.getMessageBuffer();

  let fed = 0;
  const feedOne = async () => {
    const i = fed++;
    const event = session.events[i];
    const toolUseId = `toolu_replay_${i}`;
    await ingestObservation({ contentSessionId, platformSource: 'claude', toolName: event.toolName, toolInput: event.toolInput, toolResponse: event.toolResponse, cwd: tmpCwd, toolUseId });
    const row = store.db.prepare('SELECT id FROM tool_uses WHERE tool_use_id = ?').get(toolUseId) as { id: number } | null;
    if (row) fake.responseByEventId.set(row.id, session.responses[i]);
  };
  const modelWaiting = () => opts.liveModel
    ? live.lastType === 'result' && live.activeOneShots === 0 && Date.now() - live.lastAt > 15_000
    : fake.idle;
  const quietAndDrained = () => fed >= session.events.length && modelWaiting()
    && buffer().getPendingCount(sessionDbId) === 0 && (sessionManager.getSession(sessionDbId)?.claimedMessageIds.length ?? 0) === 0;
  const feedLimit = opts.burst ? Number.POSITIVE_INFINITY : 2;

  // A burst arrives faster than the observer drains it: past the live buffer's
  // per-session cap, ingest refuses events and only the ledger keeps them.
  if (opts.burst) while (fed < session.events.length) await feedOne();
  const refusedAtIngest = (store.db.prepare(
    "SELECT COUNT(*) AS n FROM observer_event_ledger WHERE session_db_id = ? AND status = 'dropped' AND note = 'live buffer refused the event'"
  ).get(sessionDbId) as { n: number }).n;

  const exitReasons: string[] = [];
  const liveErrors: unknown[] = [];
  for (let run = 0; run < 400; run++) {
    while (fed < session.events.length && buffer().getPendingCount(sessionDbId) < feedLimit) await feedOne();
    const active = sessionManager.getSession(sessionDbId);
    if (!active) break;
    active.abortController = new AbortController();
    active.abortReason = null;
    if (opts.liveModel) active.modelOverride = opts.liveModel;
    let settled = false;
    const runPromise = provider.startSession(active).catch(error => { liveErrors.push(error); }).finally(() => { settled = true; });
    let stable = 0;
    while (!settled) {
      if (fed < session.events.length && buffer().getPendingCount(sessionDbId) < feedLimit) { await feedOne(); continue; }
      stable = quietAndDrained() ? stable + 1 : 0;
      if (stable >= 3) {
        // Production ends a quiet observer with its idle timeout.
        active.abortReason = 'idle';
        active.abortController.abort();
        break;
      }
      await sleep(opts.liveModel ? 250 : 5);
    }
    await runPromise;
    exitReasons.push(String(active.abortReason ?? 'none'));
    await handleGeneratorExit(active, active.abortReason ?? null, { sessionManager, completionHandler: completion, resumeGenerator: () => {} } as never);
  }
  const unaccountedBeforeAudit = ledger.listUnaccounted(sessionDbId).length;
  await completion.auditsSettled();
  return { store, ledger, sessionDbId, fake, unaccountedAfterLive: unaccountedBeforeAudit, exitReasons, liveErrors, refusedAtIngest };
}

function storedObservationViolations(store: any): { count: number; violations: string[] } {
  const rows = store.db.prepare('SELECT id, type, title, subtitle, facts, narrative, concepts FROM observations').all() as any[];
  const violations = rows.flatMap(row => checkObservationFields({
    type: row.type, title: row.title ?? '', subtitle: row.subtitle ?? '',
    facts: JSON.parse(row.facts ?? '[]'), narrative: row.narrative ?? '', concepts: JSON.parse(row.concepts ?? '[]'),
  }, {
    validTypes: codeMode.observation_types.map((t: { id: string }) => t.id),
    validConcepts: codeMode.observation_concepts.map((c: { id: string }) => c.id),
  }).map(v => `#${row.id}: ${v.code} ${v.message}`));
  return { count: rows.length, violations };
}

function expectZeroMissingAndZeroViolations(result: ReplayResult, minEvents: number) {
  const summary = result.ledger.summarize(result.sessionDbId);
  // Every queued tool event has a ledger id ...
  expect(summary.total).toBeGreaterThanOrEqual(minEvents);
  // ... and ended observed, backfilled, or explicitly skipped.
  expect(result.ledger.listUnaccounted(result.sessionDbId).map(r => `evt_${r.event_id}`)).toEqual([]);

  const { count, violations } = storedObservationViolations(result.store);
  expect(count).toBeGreaterThan(0);
  expect(violations).toEqual([]);

  // Ledger links point at real observation rows, and live outcomes were logged per event.
  const existing = new Set((result.store.db.prepare('SELECT id FROM observations').all() as Array<{ id: number }>).map(r => r.id));
  const observed = result.store.db.prepare(
    "SELECT event_id, status, observation_ids FROM observer_event_ledger WHERE session_db_id = ? AND status IN ('observed', 'backfilled')"
  ).all(result.sessionDbId) as Array<{ event_id: number; status: string; observation_ids: string }>;
  const outcomeLog = logLines.filter(l => l.includes('EVENT_OUTCOME')).join('\n');
  for (const row of observed) {
    const ids: number[] = JSON.parse(row.observation_ids);
    expect(ids.length).toBeGreaterThan(0);
    for (const id of ids) expect(existing.has(id)).toBe(true);
    if (row.status === 'observed') expect(outcomeLog).toContain(`evt_${row.event_id}->#`);
  }
}

const priorStrictFields = process.env.CLAUDE_MEM_OBSERVER_STRICT_FIELDS;

beforeAll(() => {
  // These replays run the strict contract: every observation field required.
  process.env.CLAUDE_MEM_OBSERVER_STRICT_FIELDS = 'true';
  tmpCwd = mkdtempSync(join(tmpdir(), 'claude-mem-replay-cwd-'));
  spyOn(logger as any, 'log').mockImplementation((...args: unknown[]) => {
    logLines.push(args.map(a => (typeof a === 'string' ? a : '')).join(' '));
  });
});

afterAll(() => {
  if (priorStrictFields === undefined) delete process.env.CLAUDE_MEM_OBSERVER_STRICT_FIELDS;
  else process.env.CLAUDE_MEM_OBSERVER_STRICT_FIELDS = priorStrictFields;
  (logger as any).log.mockRestore?.();
  rmSync(tmpCwd, { recursive: true, force: true });
  mock.module('@anthropic-ai/claude-agent-sdk', () => actualAgentSdk);
  mock.module('../../src/shared/find-claude-executable.js', () => actualFindClaude);
  mock.module('../../src/shared/EnvManager.js', () => actualEnvManager);
  mock.module('../../src/supervisor/process-registry.js', () => actualProcessRegistry);
  mock.module('../../src/services/domain/ModeManager.js', () => actualModeManager);
  mock.module('../../src/services/context-generator.js', () => actualContextGenerator);
});

function report(label: string, result: ReplayResult, events: number) {
  const s = result.fake.stats;
  const rejected = logLines.filter(l => l.includes('REJECTED by the output contract')).length;
  console.log(`[${label}] events=${events} ledger=${JSON.stringify(result.ledger.summarize(result.sessionDbId))} generations=${s.generations} ` +
    `eventsPerGeneration(max)=${Math.max(0, ...s.eventsPerGeneration)} maxContextTokens=${s.maxContextTokens} overflowReplies=${s.overflowReplies} ` +
    `contractRejections=${rejected} repairs=${s.repairCalls} (stubborn=${s.stubbornRepairs}) openAfterLive=${result.unaccountedAfterLive} ` +
    `correctedRetries=${s.correctedRetries} backfillCalls=${s.backfillCalls} exits=${JSON.stringify(Object.fromEntries(result.exitReasons.reduce((m, r) => m.set(r, (m.get(r) ?? 0) + 1), new Map())))}`);
}

const recording = findRecording();

describe('observer replay: recorded long session', () => {
  it.skipIf(!recording || LIVE)('replays with zero missing observations and zero schema violations', async () => {
    const session = loadRecordedSession(recording!);
    expect(session.events.length).toBeGreaterThanOrEqual(100);
    logLines.length = 0;
    const result = await replay(session, { contextLimitTokens: CONTEXT_LIMIT_TOKENS });
    report(`replay ${session.name}`, result, session.events.length);

    expect(result.liveErrors).toEqual([]);
    expectZeroMissingAndZeroViolations(result, 100);
    const stats = result.fake.stats;
    // Recycling kept every generation inside the window; nothing overflowed.
    expect(stats.overflowReplies).toBe(0);
    expect(stats.maxContextTokens).toBeLessThan(CONTEXT_LIMIT_TOKENS);
    expect(stats.generations).toBeGreaterThan(1);
    // The recorded defects were caught and repaired, not stored.
    expect(logLines.some(l => l.includes('REJECTED by the output contract') && l.includes('TURN_MARKER'))).toBe(true);
    expect(stats.repairCalls).toBeGreaterThan(0);
    expect(stats.repairsWithoutErrors).toBe(0);
    // The live pipeline does the work; the audit is the safety net.
    expect(result.unaccountedAfterLive).toBeLessThanOrEqual(Math.ceil(result.ledger.summarize(result.sessionDbId).total * LIVE_OPEN_BUDGET));
  }, 300_000);

  it.skipIf(!recording || LIVE)('control: a narrow window overflows, v16 recycles, and the audit still recovers every event', async () => {
    const session = loadRecordedSession(recording!);
    logLines.length = 0;
    // Narrower than the recycle budget, so a generation does overflow.
    const result = await replay(session, { contextLimitTokens: 20_000 });
    report(`control ${session.name}`, result, session.events.length);
    expect(result.fake.stats.overflowReplies).toBeGreaterThan(0);
    expectZeroMissingAndZeroViolations(result, 100);
  }, 300_000);
});

describe('observer replay: synthetic long session', () => {
  it.skipIf(LIVE)('replays with zero missing observations and zero schema violations', async () => {
    const session = syntheticSession(80);
    logLines.length = 0;
    const result = await replay(session, { contextLimitTokens: CONTEXT_LIMIT_TOKENS });
    report('replay synthetic-80', result, 80);
    expect(result.liveErrors).toEqual([]);
    expectZeroMissingAndZeroViolations(result, 80);
    expect(result.fake.stats.overflowReplies).toBe(0);
    expect(result.fake.stats.repairCalls).toBeGreaterThan(0);
    expect(result.unaccountedAfterLive).toBeLessThanOrEqual(Math.ceil(80 * LIVE_OPEN_BUDGET));
  }, 300_000);
});

describe('observer replay: burst past the live buffer cap', () => {
  it.skipIf(LIVE)('ledgers refused events and the audit backfills them', async () => {
    const session = syntheticSession(140);
    logLines.length = 0;
    const result = await replay(session, { contextLimitTokens: CONTEXT_LIMIT_TOKENS, burst: true });
    report('burst synthetic-140', result, 140);
    // The buffer holds LIVE_OBSERVER_MAX_SESSION_MESSAGES per session; the rest were refused at ingest.
    expect(result.refusedAtIngest).toBe(140 - LIVE_OBSERVER_MAX_SESSION_MESSAGES);
    expect(result.ledger.summarize(result.sessionDbId).total).toBe(140);
    expectZeroMissingAndZeroViolations(result, 140);
  }, 300_000);
});

describe('observer replay: LIVE model smoke (opt-in)', () => {
  it.skipIf(!LIVE || !recording)('real observer model: zero missing, zero schema violations', async () => {
    const full = loadRecordedSession(recording!);
    const count = Number(process.env.CLAUDE_MEM_LIVE_EVENTS ?? 20);
    const session = { ...full, events: full.events.slice(0, count), responses: full.responses.slice(0, count) };
    const model = process.env.CLAUDE_MEM_LIVE_MODEL ?? 'claude-sonnet-4-6';
    logLines.length = 0;
    const result = await replay(session, { contextLimitTokens: CONTEXT_LIMIT_TOKENS, liveModel: model });
    const rejected = logLines.filter(l => l.includes('REJECTED by the output contract'));
    const codes = new Map<string, number>();
    for (const line of rejected) for (const code of /codes=\[([^\]]*)\]/.exec(line)?.[1].split(',') ?? []) codes.set(code, (codes.get(code) ?? 0) + 1);
    const nonXml = logLines.filter(l => l.includes('returned non-XML')).length;
    const titles = (result.store.db.prepare('SELECT type, title FROM observations ORDER BY id LIMIT 5').all() as any[]).map(r => `(${r.type}) ${r.title}`);
    console.log(`[live ${model}] events=${count} ledger=${JSON.stringify(result.ledger.summarize(result.sessionDbId))} generations=${live.generations} turns=${live.results} ` +
      `contractRejections=${rejected.length} codes=${JSON.stringify(Object.fromEntries(codes))} nonXmlReplies=${nonXml} oneShots=${live.oneShots} ` +
      `openAfterLive=${result.unaccountedAfterLive} overflowReplies=${live.overflowReplies} liveErrors=${result.liveErrors.length} exits=${JSON.stringify(result.exitReasons)}`);
    console.log(`[live sample titles]\n  ${titles.join('\n  ')}`);
    expect(live.overflowReplies).toBe(0);
    expectZeroMissingAndZeroViolations(result, Math.min(count, 10));
  }, 20 * 60_000);
});
