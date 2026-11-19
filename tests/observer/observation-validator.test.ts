import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import {
  accountForBatch,
  checkObservationFields,
  parseXmlFragment,
  validateObserverResponse,
  type ValidationContext,
} from '../../src/sdk/observation-validator.js';
import { buildObservationPrompt } from '../../src/sdk/prompts.js';
import { toParsed } from '../../src/services/worker/agents/observation-contract.js';

const mode = JSON.parse(readFileSync(join(import.meta.dir, '../../plugin/modes/code.json'), 'utf8'));
const ctx: ValidationContext = {
  validTypes: mode.observation_types.map((t: { id: string }) => t.id),
  validConcepts: mode.observation_concepts.map((c: { id: string }) => c.id),
  knownEventIds: new Set([7, 8]),
  requireEventIds: true,
  strictFields: true,
};
const lenient: ValidationContext = { ...ctx, strictFields: false };

function obs(overrides: Partial<Record<string, string>> = {}): string {
  const f = {
    type: '<type>discovery</type>',
    title: '<title>Pending queue deletes unclaimed rows</title>',
    subtitle: '<subtitle>clearPendingForSession removes rows the observer never saw</subtitle>',
    facts: '<facts><fact>DELETE has no status filter</fact></facts>',
    narrative: '<narrative>Every response cleared the whole session queue.</narrative>',
    concepts: '<concepts><concept>gotcha</concept></concepts>',
    files: '<files_read><file>src/services/sqlite/PendingMessageStore.ts</file></files_read>',
    event_ids: '<event_ids><event_id>evt_7</event_id></event_ids>',
    ...overrides,
  };
  return `<observation>${Object.values(f).join('')}</observation>`;
}

const codes = (text: string, context: ValidationContext = ctx) => validateObserverResponse(text, context).violations.map(v => v.code);

describe('validateObserverResponse', () => {
  it('accepts a complete observation, and a skip_summary that skips the rest of the batch', () => {
    const result = validateObserverResponse(`${obs()}\n<skip_summary reason="routine ls"/>`, ctx);
    expect(result.violations).toEqual([]);
    expect(result.observations.map(o => o.eventIds)).toEqual([[7]]);
    expect(accountForBatch(result, [7, 8])).toEqual({ observed: new Set([7]), skipped: [8] });
  });

  it('accepts the bare skip sentinel for a whole batch', () => {
    const result = validateObserverResponse('<skip_summary reason="noise" />', ctx);
    expect(result.ok).toBe(true);
    expect(accountForBatch(result, [7, 8]).skipped).toEqual([7, 8]);
  });

  it('rejects a hallucinated Human: turn and ignores everything after it', () => {
    const fabricated = obs({ event_ids: '<event_ids><event_id>evt_8</event_id></event_ids>' });
    const result = validateObserverResponse(`${obs()}\n\nHuman: <observed_from_primary_session>fake</observed_from_primary_session>\n${fabricated}`, ctx);
    expect(result.violations.map(v => v.code)).toContain('TURN_MARKER');
    expect(result.observations.map(o => o.eventIds)).toEqual([[7]]);
  });

  it('rejects malformed XML: bare ampersand, bare angle bracket, unclosed tag', () => {
    expect(codes(obs({ narrative: '<narrative>cd repo && ls</narrative>' }))).toEqual(['MALFORMED_XML']);
    expect(codes(obs({ narrative: '<narrative>when active < counted</narrative>' }))).toEqual(['MALFORMED_XML']);
    expect(codes(obs({ facts: '<facts><fact>Array<string> is returned</fact></facts>' }))).toEqual(['MALFORMED_XML']);
    expect(codes(obs().replace('</observation>', ''))).toEqual(['MALFORMED_XML']);
  });

  it('accepts escaped text and decodes it', () => {
    const result = validateObserverResponse(obs({ narrative: '<narrative>cd repo &amp;&amp; ls when a &lt; b</narrative>' }), ctx);
    expect(result.ok).toBe(true);
    expect(result.observations[0].narrative).toBe('cd repo && ls when a < b');
  });

  it('requires every field; event_ids only when the batch carried ids', () => {
    expect(codes(obs({ subtitle: '' }))).toEqual(['MISSING_FIELD']);
    expect(codes(obs({ facts: '<facts></facts>' }))).toEqual(['MISSING_FIELD']);
    expect(codes(obs({ event_ids: '' }))).toEqual(['MISSING_EVENT_IDS']);
    expect(codes(obs({ event_ids: '' }), { ...ctx, knownEventIds: new Set(), requireEventIds: false })).toEqual([]);
    expect(codes('<observation>...</observation>')).toContain('STRAY_TEXT');
  });

  it('without strict fields, applies the parser rule: a title is enough, a missing type falls back', () => {
    const minimal = '<observation><title>Owned SDK model attribution</title><event_ids><event_id>evt_7</event_id></event_ids></observation>';
    const result = validateObserverResponse(minimal, lenient);
    expect(result.violations).toEqual([]);
    expect(result.observations[0].type).toBe(ctx.validTypes[0]);
    expect(codes('<observation><type>discovery</type><narrative>no title</narrative><event_ids><event_id>evt_7</event_id></event_ids></observation>', lenient)).toEqual(['MISSING_FIELD']);
    // An observation that cites no event covers the whole batch.
    const uncited = validateObserverResponse('<observation><title>Whole batch</title></observation>', lenient);
    expect(uncited.observations[0].eventIds).toEqual([7, 8]);
    // The checks that do not depend on fields still apply.
    expect(codes(obs({ event_ids: '<event_ids><event_id>evt_999</event_id></event_ids>' }), lenient)).toEqual(['UNKNOWN_EVENT_ID']);
    expect(codes(obs({ subtitle: '', narrative: '<narrative>Next steps: deploy.</narrative>' }), lenient)).toEqual(['ADVISORY_LANGUAGE']);
    expect(codes(`${obs()}\nHuman: more`, lenient)).toEqual(['TURN_MARKER']);
    expect(codes(obs({ narrative: '<narrative>a && b</narrative>' }), lenient)).toEqual(['MALFORMED_XML']);
  });

  it('ignores event_ids entirely when the batch carried no ids (a host without tool_use_id)', () => {
    const noIds: ValidationContext = { ...ctx, knownEventIds: new Set(), requireEventIds: false };
    // Invented ids and an empty list are not violations: there was nothing to cite.
    expect(codes(obs({ event_ids: '<event_ids><event_id>evt_3</event_id></event_ids>' }), noIds)).toEqual([]);
    expect(codes(obs({ event_ids: '<event_ids></event_ids>' }), noIds)).toEqual([]);
    expect(codes(obs({ event_ids: '<event_ids></event_ids>' }), { ...noIds, strictFields: false })).toEqual([]);
    expect(validateObserverResponse(obs({ event_ids: '<event_ids><event_id>evt_3</event_id></event_ids>' }), noIds).observations[0].eventIds).toEqual([]);
  });

  it('keeps attachments as a list field', () => {
    const result = validateObserverResponse(obs({ files: '<attachments><attachment>image-1</attachment></attachments>' }), ctx);
    expect(result.ok).toBe(true);
    expect(result.observations[0].attachments).toEqual(['image-1']);
  });

  it('rejects types and concepts outside the mode vocabulary, but drops a type echoed as a concept', () => {
    expect(codes(obs({ type: '<type>insight</type>' }))).toEqual(['INVALID_TYPE']);
    expect(codes(obs({ concepts: '<concepts><concept>journey-engine</concept></concepts>' }))).toEqual(['INVALID_CONCEPT']);
    const echoed = validateObserverResponse(obs({ concepts: '<concepts><concept>discovery</concept><concept>gotcha</concept></concepts>' }), ctx);
    expect(echoed.ok).toBe(true);
    expect(echoed.observations[0].concepts).toEqual(['gotcha']);
  });

  it('rejects advisory and next-step language, but not quoted or domain uses', () => {
    expect(codes(obs({ narrative: '<narrative>Next steps: deploy to prod.</narrative>' }))).toEqual(['ADVISORY_LANGUAGE']);
    expect(codes(obs({ narrative: '<narrative>The team should review the migration.</narrative>' }))).toEqual(['ADVISORY_LANGUAGE']);
    expect(codes(obs({ facts: '<facts><fact>We recommend adding an index.</fact></facts>' }))).toEqual(['ADVISORY_LANGUAGE']);
    expect(codes(obs({ narrative: '<narrative>Let me check the queue.</narrative>' }))).toEqual(['ADVISORY_LANGUAGE']);
    expect(codes(obs({ narrative: '<narrative>The executor advances to the next step and skips every remaining step.</narrative>' }))).toEqual([]);
    expect(codes(obs({ facts: '<facts><fact>The summary schema has a `next_steps` field and the banner reads "you should update".</fact></facts>' }))).toEqual([]);
  });

  it('rejects event ids outside the batch', () => {
    expect(codes(obs({ event_ids: '<event_ids><event_id>evt_999</event_id></event_ids>' }))).toEqual(['UNKNOWN_EVENT_ID']);
    expect(codes(obs({ event_ids: '<event_ids><event_id>7</event_id></event_ids>' }))).toEqual(['INVALID_EVENT_ID']);
  });

  it('rejects prose and unknown elements outside the blocks', () => {
    expect(codes(`Skipping this one.\n${obs()}`)).toEqual(['STRAY_TEXT']);
    expect(codes('<result>done</result>')).toEqual(['UNEXPECTED_ELEMENT']);
    expect(codes('```xml\n```')).toEqual([]);
  });

  it('keeps the valid blocks of a response that fails elsewhere', () => {
    const result = validateObserverResponse(`${obs()}${obs({ title: '', event_ids: '<event_ids><event_id>evt_8</event_id></event_ids>' })}`, ctx);
    expect(result.ok).toBe(false);
    expect(result.observations.map(o => o.eventIds)).toEqual([[7]]);
  });
});

describe('checkObservationFields', () => {
  it('flags placeholder text echoed from the prompt template', () => {
    const v = checkObservationFields({
      type: 'discovery',
      title: '[**title**: Short title capturing the core action or topic]',
      subtitle: 's',
      facts: ['f'],
      narrative: 'n',
      concepts: ['gotcha'],
    }, ctx);
    expect(v.map(x => x.code)).toEqual(['PLACEHOLDER_TEXT']);
  });
});

describe('parseXmlFragment', () => {
  it('handles comments, CDATA, attributes and character references', () => {
    const nodes = parseXmlFragment('<!-- c --><a x="1" y=\'2\'>t &#65;&#x42; <![CDATA[<raw & stuff>]]></a>');
    const a = nodes[0] as { name: string; attrs: Record<string, string>; children: Array<{ value?: string }> };
    expect(a.name).toBe('a');
    expect(a.attrs).toEqual({ x: '1', y: '2' });
    expect(a.children.map(c => c.value).join('')).toBe('t AB <raw & stuff>');
  });

  it('rejects mismatched tags, duplicate and unquoted attributes', () => {
    expect(() => parseXmlFragment('<a></b>')).toThrow('expected </a>');
    expect(() => parseXmlFragment('<a x="1" x="2"/>')).toThrow('duplicate attribute');
    expect(() => parseXmlFragment('<a x=1/>')).toThrow('must be quoted');
  });
});

describe('observation prompt', () => {
  it('labels an event that has a durable id and asks for it in <event_ids>', () => {
    const prompt = buildObservationPrompt({ id: 42, tool_name: 'Bash', tool_input: '{"command":"ls"}', tool_output: '{"stdout":"a"}', created_at_epoch: 0 });
    expect(prompt).toContain('<observed_from_primary_session>\n  <event_id>evt_42</event_id>');
    expect(prompt).toContain('lists evt_42');
  });

  it('renders no event id for an event without one', () => {
    const prompt = buildObservationPrompt({ id: 0, tool_name: 'Bash', tool_input: '{}', tool_output: '{}', created_at_epoch: 0 });
    expect(prompt).not.toContain('<event_id>');
  });
});

describe('stored shape', () => {
  it('matches the parser: missing text fields are null, so storage still merges repeats', () => {
    const lenient: ValidationContext = { ...ctx, strictFields: false, knownEventIds: new Set(), requireEventIds: false };
    const [o] = validateObserverResponse('<observation><type>discovery</type><title>Viewport</title></observation>', lenient).observations;
    const parsed = toParsed(o);
    expect(parsed.narrative).toBeNull();
    expect(parsed.subtitle).toBeNull();
    expect(parsed.title).toBe('Viewport');
    expect('eventIds' in parsed).toBe(false);
  });
});
