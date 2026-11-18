/**
 * Strict validator for observer responses.
 *
 * The observer must answer every tool event it is shown with either one or
 * more <observation> blocks that cite the event in <event_ids>, or the
 * <skip_summary reason="..."/> sentinel, which skips every event of the batch
 * that no observation cites. Anything else (prose, a hallucinated "Human:"
 * turn, malformed XML, missing fields, advice about what to do next) is a
 * violation. Callers retry once with the violations included, and store only
 * observations that pass.
 */

import { logger } from '../utils/logger.js';

export const EVENT_ID_PATTERN = /^evt_(\d+)$/;

export function formatEventId(id: number): string {
  return `evt_${id}`;
}

export function parseEventId(raw: string): number | null {
  const match = EVENT_ID_PATTERN.exec(raw.trim());
  return match ? Number(match[1]) : null;
}

export type ViolationCode =
  | 'TURN_MARKER'
  | 'STRAY_TEXT'
  | 'MALFORMED_XML'
  | 'UNEXPECTED_ELEMENT'
  | 'DUPLICATE_FIELD'
  | 'MISSING_FIELD'
  | 'INVALID_TYPE'
  | 'INVALID_CONCEPT'
  | 'FIELD_TOO_LONG'
  | 'PLACEHOLDER_TEXT'
  | 'ADVISORY_LANGUAGE'
  | 'MISSING_EVENT_IDS'
  | 'INVALID_EVENT_ID'
  | 'UNKNOWN_EVENT_ID'
  | 'MISSING_COVERAGE';

export interface Violation {
  code: ViolationCode;
  message: string;
  /** Index of the offending top-level block, when the violation is inside one. */
  blockIndex?: number;
}

export interface ValidatedObservation {
  type: string;
  title: string;
  subtitle: string;
  facts: string[];
  narrative: string;
  concepts: string[];
  files_read: string[];
  files_modified: string[];
  /** Request labels from <attachments>, when present (media linkage). */
  attachments?: string[];
  eventIds: number[];
}

export interface ValidationContext {
  validTypes: readonly string[];
  validConcepts: readonly string[];
  /** Event ids of the batch being answered. Citing any other id is a violation. */
  knownEventIds: ReadonlySet<number>;
  /**
   * True when the batch's prompts carried <event_id>s. Strict: observations
   * must cite them. Lenient: an observation that cites none covers the batch.
   */
  requireEventIds: boolean;
  /**
   * CLAUDE_MEM_OBSERVER_STRICT_FIELDS. Strict: every field required, mode
   * vocabulary enforced, no stray markup. Otherwise the parser's own rule
   * applies (a title is required, a missing type falls back to the first mode
   * type, unknown tags are tolerated); XML, turn, advice and event checks
   * apply either way.
   */
  strictFields: boolean;
}

export interface ValidationResult {
  ok: boolean;
  violations: Violation[];
  /** Observations that passed every check, even when other blocks failed. */
  observations: ValidatedObservation[];
  /** A top-level <skip_summary/>: skips every batch event no observation cites. */
  skipAll: { reason: string | null } | null;
}

export const OBSERVATION_LIMITS = {
  titleMaxChars: 200,
  subtitleMaxWords: 40,
  maxConcepts: 5,
} as const;

// ---------------------------------------------------------------------------
// Strict XML fragment parser
// ---------------------------------------------------------------------------

interface XmlElement {
  kind: 'element';
  name: string;
  attrs: Record<string, string>;
  children: XmlNode[];
  start: number;
  end: number;
}

interface XmlText {
  kind: 'text';
  value: string;
}

type XmlNode = XmlElement | XmlText;

export class XmlSyntaxError extends Error {
  constructor(message: string, readonly position: number) {
    super(`${message} (at offset ${position})`);
  }
}

const NAME_START = /[A-Za-z_]/;
const NAME_CHAR = /[A-Za-z0-9_.-]/;

const NAMED_ENTITIES: Record<string, string> = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
};

/**
 * Parse an XML fragment (zero or more top-level nodes) under XML 1.0
 * well-formedness rules: every tag closes in order, attribute values are
 * quoted, and a literal '&' or '<' in text must be escaped.
 */
export function parseXmlFragment(input: string): XmlNode[] {
  let pos = 0;

  const fail = (message: string, at: number = pos): never => {
    throw new XmlSyntaxError(message, at);
  };

  const readName = (): string => {
    if (pos >= input.length || !NAME_START.test(input[pos])) fail('expected a tag or attribute name');
    const start = pos;
    pos++;
    while (pos < input.length && NAME_CHAR.test(input[pos])) pos++;
    return input.slice(start, pos);
  };

  const skipWhitespace = () => {
    while (pos < input.length && /\s/.test(input[pos])) pos++;
  };

  const decodeEntities = (raw: string, offset: number): string => {
    let out = '';
    for (let i = 0; i < raw.length; i++) {
      const ch = raw[i];
      if (ch === '<') fail("unescaped '<' in text; write &lt;", offset + i);
      if (ch !== '&') {
        out += ch;
        continue;
      }
      const semi = raw.indexOf(';', i);
      const body = semi === -1 ? '' : raw.slice(i + 1, semi);
      if (semi === -1 || body.length === 0 || body.length > 10) {
        fail("unescaped '&' in text; write &amp;", offset + i);
      }
      if (body.startsWith('#x') || body.startsWith('#X')) {
        const code = Number.parseInt(body.slice(2), 16);
        if (!/^[0-9a-fA-F]+$/.test(body.slice(2)) || Number.isNaN(code)) fail(`invalid character reference &${body};`, offset + i);
        out += String.fromCodePoint(code);
      } else if (body.startsWith('#')) {
        if (!/^\d+$/.test(body.slice(1))) fail(`invalid character reference &${body};`, offset + i);
        out += String.fromCodePoint(Number(body.slice(1)));
      } else if (body in NAMED_ENTITIES) {
        out += NAMED_ENTITIES[body];
      } else {
        fail(`unknown entity &${body}; (only &amp; &lt; &gt; &quot; &apos; are allowed)`, offset + i);
      }
      i = semi;
    }
    return out;
  };

  const parseNodes = (closingName: string | null): XmlNode[] => {
    const nodes: XmlNode[] = [];
    while (pos < input.length) {
      if (input.startsWith('<!--', pos)) {
        const end = input.indexOf('-->', pos + 4);
        if (end === -1) fail('unterminated comment');
        pos = end + 3;
        continue;
      }
      if (input.startsWith('<![CDATA[', pos)) {
        const end = input.indexOf(']]>', pos + 9);
        if (end === -1) fail('unterminated CDATA section');
        nodes.push({ kind: 'text', value: input.slice(pos + 9, end) });
        pos = end + 3;
        continue;
      }
      if (input.startsWith('</', pos)) {
        const at = pos;
        pos += 2;
        const name = readName();
        skipWhitespace();
        if (input[pos] !== '>') fail(`malformed closing tag </${name}`);
        pos++;
        if (closingName === null) fail(`closing tag </${name}> has no matching opening tag`, at);
        if (name !== closingName) fail(`expected </${closingName}> but found </${name}>`, at);
        return nodes;
      }
      if (input[pos] === '<') {
        nodes.push(parseElement());
        continue;
      }
      const start = pos;
      while (pos < input.length && input[pos] !== '<') pos++;
      // A '<' that cannot start a tag is a literal and must have been escaped.
      if (pos < input.length && !/[A-Za-z_/!]/.test(input[pos + 1] ?? '')) {
        fail("unescaped '<' in text; write &lt;", pos);
      }
      nodes.push({ kind: 'text', value: decodeEntities(input.slice(start, pos), start) });
    }
    if (closingName !== null) fail(`<${closingName}> is never closed`);
    return nodes;
  };

  const parseElement = (): XmlElement => {
    const start = pos;
    pos++; // '<'
    const name = readName();
    const attrs: Record<string, string> = {};
    for (;;) {
      const hadSpace = pos < input.length && /\s/.test(input[pos]);
      skipWhitespace();
      if (input.startsWith('/>', pos)) {
        pos += 2;
        return { kind: 'element', name, attrs, children: [], start, end: pos };
      }
      if (input[pos] === '>') {
        pos++;
        const children = parseNodes(name);
        return { kind: 'element', name, attrs, children, start, end: pos };
      }
      if (pos >= input.length) fail(`<${name}> tag is not terminated`, start);
      if (!hadSpace) fail(`expected whitespace before attribute in <${name}>`);
      const attrName = readName();
      skipWhitespace();
      if (input[pos] !== '=') fail(`attribute ${attrName} in <${name}> has no value`);
      pos++;
      skipWhitespace();
      const quote = input[pos];
      if (quote !== '"' && quote !== "'") fail(`attribute ${attrName} in <${name}> must be quoted`);
      const valueEnd = input.indexOf(quote, pos + 1);
      if (valueEnd === -1) fail(`attribute ${attrName} in <${name}> is not terminated`);
      if (attrName in attrs) fail(`duplicate attribute ${attrName} in <${name}>`);
      attrs[attrName] = decodeEntities(input.slice(pos + 1, valueEnd), pos + 1);
      pos = valueEnd + 1;
    }
  };

  return parseNodes(null);
}

// ---------------------------------------------------------------------------
// Content rules
// ---------------------------------------------------------------------------

/**
 * Phrases that turn an observation from a record of what happened into advice
 * about what to do. Matched case-insensitively against prose fields after
 * inline code spans and quoted strings are removed, so quoting `next_steps`,
 * a TODO comment, or UI copy is fine.
 */
export const ADVISORY_PATTERNS: ReadonlyArray<{ label: string; pattern: RegExp }> = [
  // "next step" is also a domain term (workflow and journey steps), so only
  // advisory framings count: "next steps:", "the next step is to", "expected next step".
  { label: 'next step(s)', pattern: /\bnext steps?\s*(:|are\b|include\b|would be\b|will be\b|is to\b|should\b)|\b(expected|recommended|suggested|logical|immediate|obvious|remaining) next steps?\b|\bas (a|the) next step\b/i },
  { label: 'remaining/outstanding work', pattern: /\b(remaining|outstanding) (work|tasks?|todos?|action items?)\b/i },
  { label: 'recommendation', pattern: /\b(I|we) (would )?(recommend|suggest|advise)\b|\b(it is|it's) (recommended|advisable|advised)\b/i },
  { label: 'should + action', pattern: /\bshould (now |also |still )?(be )?(consider(ed)?|add(ed)?|update[ds]?|fix(ed)?|review(ed)?|investigat(e|ed)|deploy(ed)?|merge[ds]?|run|check(ed)?|verif(y|ied)|confirm(ed)?|monitor(ed)?|follow(ed)? up|revisit(ed)?|address(ed)?|test(ed)?)\b/i },
  { label: 'addressed to reader', pattern: /\b(you|we|the user|the team) (should|must|need to|ought to|will need to)\b/i },
  { label: 'consider + action', pattern: /\bconsider (adding|using|switching|moving|implementing|updating|refactoring|running|removing|creating)\b/i },
  { label: 'follow-up needed', pattern: /\bfollow[- ]up (is |will be )?(needed|required|recommended|work|task|search)\b/i },
  { label: 'be sure / make sure', pattern: /\b(make sure|be sure|don't forget|do not forget|remember) to\b/i },
  { label: 'observer self-reference', pattern: /\b(let me|I will|I'll|I am going to|I'm going to)\b/i },
];

const PLACEHOLDER_PATTERN = /^\[\*\*[a-z_]+\*\*:|^\[(concise|knowledge-type|path\/to)|^\.{3}$|^…$/i;
const TURN_MARKER_PATTERN = /^[ \t>]*(Human|Assistant)\s*:/m;

/** Remove inline code spans and quoted strings: quoting advice is not giving it. */
function stripQuotedText(text: string): string {
  return text
    .replace(/`[^`\n]*`/g, ' ')
    .replace(/"[^"\n]*"/g, ' ')
    .replace(/“[^”\n]*”/g, ' ');
}

export function findAdvisoryLanguage(text: string): string[] {
  const prose = stripQuotedText(text);
  return ADVISORY_PATTERNS.filter(({ pattern }) => pattern.test(prose)).map(({ label }) => label);
}

function textOf(el: XmlElement): string {
  return el.children
    .map(child => (child.kind === 'text' ? child.value : ''))
    .join('')
    .trim();
}

const LEAF_FIELDS = new Set(['type', 'title', 'subtitle', 'narrative']);
const LIST_FIELDS: Record<string, string> = {
  facts: 'fact',
  concepts: 'concept',
  files_read: 'file',
  files_modified: 'file',
  attachments: 'attachment',
  event_ids: 'event_id',
};
const REQUIRED_FIELDS = ['type', 'title', 'subtitle', 'facts', 'narrative', 'concepts'] as const;
const LENIENT_REQUIRED_FIELDS = ['title'] as const;

/**
 * Field-level rules shared by response validation and by audits of stored
 * rows. Event-id rules are separate because stored rows do not carry ids.
 */
export function checkObservationFields(
  obs: Pick<ValidatedObservation, 'type' | 'title' | 'subtitle' | 'facts' | 'narrative' | 'concepts'>,
  ctx: Pick<ValidationContext, 'validTypes' | 'validConcepts'> & { strictFields?: boolean },
  blockIndex?: number
): Violation[] {
  const v: Violation[] = [];
  const at = blockIndex === undefined ? '' : ` (observation #${blockIndex + 1})`;

  if (ctx.strictFields === false) {
    if (!obs.title) v.push({ code: 'MISSING_FIELD', message: `<title> is empty${at}`, blockIndex });
    const hits = new Set<string>();
    for (const value of [obs.title, obs.subtitle, obs.narrative, ...obs.facts]) {
      if (value) for (const label of findAdvisoryLanguage(value)) hits.add(label);
    }
    if (hits.size > 0) {
      v.push({ code: 'ADVISORY_LANGUAGE', message: `observations record what happened, not what to do; remove advisory or next-step wording (${[...hits].join('; ')})${at}`, blockIndex });
    }
    return v;
  }

  if (!obs.type) v.push({ code: 'MISSING_FIELD', message: `<type> is empty${at}`, blockIndex });
  else if (!ctx.validTypes.includes(obs.type)) {
    v.push({ code: 'INVALID_TYPE', message: `<type>${obs.type}</type> is not one of: ${ctx.validTypes.join(', ')}${at}`, blockIndex });
  }

  const prose: Array<[string, string]> = [
    ['title', obs.title],
    ['subtitle', obs.subtitle],
    ['narrative', obs.narrative],
  ];
  for (const [name, value] of prose) {
    if (!value) v.push({ code: 'MISSING_FIELD', message: `<${name}> is empty${at}`, blockIndex });
    else if (PLACEHOLDER_PATTERN.test(value)) v.push({ code: 'PLACEHOLDER_TEXT', message: `<${name}> still holds template placeholder text${at}`, blockIndex });
  }
  if (obs.title && obs.title.length > OBSERVATION_LIMITS.titleMaxChars) {
    v.push({ code: 'FIELD_TOO_LONG', message: `<title> is ${obs.title.length} chars; max ${OBSERVATION_LIMITS.titleMaxChars}${at}`, blockIndex });
  }
  const subtitleWords = obs.subtitle ? obs.subtitle.split(/\s+/).filter(Boolean).length : 0;
  if (subtitleWords > OBSERVATION_LIMITS.subtitleMaxWords) {
    v.push({ code: 'FIELD_TOO_LONG', message: `<subtitle> is ${subtitleWords} words; max ${OBSERVATION_LIMITS.subtitleMaxWords}${at}`, blockIndex });
  }

  if (obs.facts.length === 0) v.push({ code: 'MISSING_FIELD', message: `<facts> needs at least one <fact>${at}`, blockIndex });
  for (const fact of obs.facts) {
    if (PLACEHOLDER_PATTERN.test(fact)) v.push({ code: 'PLACEHOLDER_TEXT', message: `a <fact> still holds template placeholder text${at}`, blockIndex });
  }

  if (obs.concepts.length === 0) v.push({ code: 'MISSING_FIELD', message: `<concepts> needs at least one <concept>${at}`, blockIndex });
  if (obs.concepts.length > OBSERVATION_LIMITS.maxConcepts) {
    v.push({ code: 'FIELD_TOO_LONG', message: `<concepts> has ${obs.concepts.length} entries; max ${OBSERVATION_LIMITS.maxConcepts}${at}`, blockIndex });
  }
  for (const concept of obs.concepts) {
    if (ctx.validConcepts.length > 0 && !ctx.validConcepts.includes(concept)) {
      v.push({ code: 'INVALID_CONCEPT', message: `<concept>${concept}</concept> is not one of: ${ctx.validConcepts.join(', ')}${at}`, blockIndex });
    }
  }

  const advisoryHits = new Set<string>();
  for (const value of [obs.title, obs.subtitle, obs.narrative, ...obs.facts]) {
    if (value) for (const label of findAdvisoryLanguage(value)) advisoryHits.add(label);
  }
  if (advisoryHits.size > 0) {
    v.push({
      code: 'ADVISORY_LANGUAGE',
      message: `observations record what happened, not what to do; remove advisory or next-step wording (${[...advisoryHits].join('; ')})${at}`,
      blockIndex,
    });
  }

  return v;
}

function validateObservationElement(
  el: XmlElement,
  index: number,
  ctx: ValidationContext,
  violations: Violation[]
): ValidatedObservation | null {
  const before = violations.length;
  const fields = new Map<string, XmlElement>();

  for (const child of el.children) {
    if (child.kind === 'text') {
      if (child.value.trim()) {
        if (ctx.strictFields) violations.push({ code: 'STRAY_TEXT', message: `text directly inside <observation> #${index + 1}; put it in a field`, blockIndex: index });
      }
      continue;
    }
    if (!LEAF_FIELDS.has(child.name) && !(child.name in LIST_FIELDS)) {
      if (ctx.strictFields) violations.push({ code: 'UNEXPECTED_ELEMENT', message: `<${child.name}> is not an observation field (observation #${index + 1})`, blockIndex: index });
      continue;
    }
    if (fields.has(child.name)) {
      violations.push({ code: 'DUPLICATE_FIELD', message: `<${child.name}> appears more than once (observation #${index + 1})`, blockIndex: index });
      continue;
    }
    fields.set(child.name, child);
  }

  const absent: string[] = (ctx.strictFields ? REQUIRED_FIELDS : LENIENT_REQUIRED_FIELDS).filter(name => !fields.has(name));
  for (const name of absent) {
    violations.push({ code: 'MISSING_FIELD', message: `<${name}> is missing (observation #${index + 1})`, blockIndex: index });
  }
  if (ctx.requireEventIds && ctx.strictFields && !fields.has('event_ids')) {
    violations.push({ code: 'MISSING_EVENT_IDS', message: `<event_ids> is missing (observation #${index + 1})`, blockIndex: index });
  }

  const leaf = (name: string): string => {
    const field = fields.get(name);
    if (!field) return '';
    if (field.children.some(c => c.kind === 'element')) {
      if (ctx.strictFields) violations.push({ code: 'UNEXPECTED_ELEMENT', message: `<${name}> must contain text only (observation #${index + 1})`, blockIndex: index });
    }
    return textOf(field);
  };

  const list = (name: string): string[] => {
    const field = fields.get(name);
    if (!field) return [];
    const itemName = LIST_FIELDS[name];
    const items: string[] = [];
    for (const child of field.children) {
      if (child.kind === 'text') {
        if (child.value.trim()) {
          if (ctx.strictFields) violations.push({ code: 'STRAY_TEXT', message: `text directly inside <${name}>; wrap it in <${itemName}> (observation #${index + 1})`, blockIndex: index });
        }
        continue;
      }
      if (child.name !== itemName || child.children.some(c => c.kind === 'element')) {
        if (ctx.strictFields) violations.push({ code: 'UNEXPECTED_ELEMENT', message: `<${name}> may only contain text <${itemName}> items (observation #${index + 1})`, blockIndex: index });
        continue;
      }
      const value = textOf(child);
      if (value) items.push(value);
    }
    return items;
  };

  const rawType = leaf('type');
  // Lenient: the parser's fallback, the first mode type when none is given.
  const type = rawType || (ctx.strictFields ? '' : ctx.validTypes[0] ?? '');
  const obs = {
    type,
    title: leaf('title'),
    subtitle: leaf('subtitle'),
    narrative: leaf('narrative'),
    facts: list('facts'),
    // Echoing an observation type as a concept is a known, lossless quirk
    // (types and concepts are separate dimensions); drop it rather than retry.
    concepts: list('concepts').filter(c => !ctx.validTypes.includes(c)),
    files_read: list('files_read'),
    files_modified: list('files_modified'),
    ...(fields.has('attachments') ? { attachments: list('attachments') } : {}),
  };
  // An absent field was already reported above; don't report it again as empty.
  violations.push(...checkObservationFields(obs, ctx, index).filter(v =>
    !(v.code === 'MISSING_FIELD' && absent.some(name => v.message.startsWith(`<${name}>`)))));

  const eventIds: number[] = [];
  if (fields.has('event_ids')) {
    const rawIds = list('event_ids');
    if (rawIds.length === 0) {
      violations.push({ code: 'MISSING_EVENT_IDS', message: `<event_ids> is empty (observation #${index + 1})`, blockIndex: index });
    }
    for (const raw of rawIds) {
      const id = parseEventId(raw);
      if (id === null) {
        violations.push({ code: 'INVALID_EVENT_ID', message: `"${raw}" is not an event id like evt_123 (observation #${index + 1})`, blockIndex: index });
      } else if (!ctx.knownEventIds.has(id)) {
        violations.push({ code: 'UNKNOWN_EVENT_ID', message: `${raw} was never sent to you; cite only event ids from <event_id> tags you received (observation #${index + 1})`, blockIndex: index });
      } else if (!eventIds.includes(id)) {
        eventIds.push(id);
      }
    }
  }

  if (violations.length > before) return null;
  // Lenient: an observation that cites nothing covers its whole batch (the
  // batch-pointer meaning tool_uses.observation_id already has).
  if (!fields.has('event_ids') && !ctx.strictFields) return { ...obs, eventIds: [...ctx.knownEventIds] };
  return { ...obs, eventIds };
}

function stripWholeResponseFence(text: string): string {
  const match = text.match(/^\s*```(?:xml)?[ \t]*\n([\s\S]*?)\n?```\s*$/i);
  return match ? match[1] : text;
}

/**
 * Validate one observer response. `observations` holds the blocks that
 * passed, so a caller that has exhausted its retry can still keep them.
 */
export function validateObserverResponse(raw: string, ctx: ValidationContext): ValidationResult {
  const violations: Violation[] = [];
  const observations: ValidatedObservation[] = [];
  let skipAll: ValidationResult['skipAll'] = null;

  const text = stripWholeResponseFence(typeof raw === 'string' ? raw : '');

  // A hallucinated "Human:" turn means everything after it was invented by the
  // model. Validate only what precedes it, and fail the response.
  const marker = TURN_MARKER_PATTERN.exec(text);
  const body = marker ? text.slice(0, marker.index) : text;
  if (marker) {
    violations.push({
      code: 'TURN_MARKER',
      message: `response contains a "${marker[0].trim()}" line, which starts an invented conversation turn; output only XML for the events you were sent`,
    });
  }

  let nodes: XmlNode[];
  try {
    nodes = parseXmlFragment(body);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    violations.push({ code: 'MALFORMED_XML', message: `response is not well-formed XML: ${message}` });
    return { ok: false, violations, observations, skipAll };
  }

  let blockIndex = 0;
  for (const node of nodes) {
    if (node.kind === 'text') {
      if (node.value.trim()) {
        const preview = node.value.trim().slice(0, 80);
        violations.push({ code: 'STRAY_TEXT', message: `text outside any XML block: "${preview}"` });
      }
      continue;
    }
    const index = blockIndex++;
    switch (node.name) {
      case 'observation': {
        const obs = validateObservationElement(node, index, ctx, violations);
        if (obs) observations.push(obs);
        break;
      }
      case 'skip_summary': {
        const reason = (node.attrs.reason ?? '').trim();
        skipAll = { reason: reason || null };
        break;
      }
      case 'summary':
        // A summary answers a summarize prompt; it is parsed and stored elsewhere.
        break;
      default:
        violations.push({ code: 'UNEXPECTED_ELEMENT', message: `<${node.name}> is not allowed; use <observation> or <skip_summary/>`, blockIndex: index });
    }
  }

  if (violations.length > 0) {
    logger.debug('PARSER', 'Observer response failed validation', {
      codes: [...new Set(violations.map(v => v.code))].join(','),
      acceptedObservations: observations.length,
      skipAll: skipAll !== null,
    });
  }
  return { ok: violations.length === 0, violations, observations, skipAll };
}

/** Batch events an accepted response observed, and those its <skip_summary/> skipped. */
export function accountForBatch(
  result: Pick<ValidationResult, 'observations' | 'skipAll'>,
  batchEventIds: readonly number[]
): { observed: Set<number>; skipped: number[] } {
  const observed = new Set<number>();
  for (const obs of result.observations) for (const id of obs.eventIds) observed.add(id);
  const skipped = result.skipAll ? batchEventIds.filter(id => !observed.has(id)) : [];
  return { observed, skipped };
}

export const OUTPUT_CONTRACT = `OUTPUT CONTRACT (enforced by a validator; violations are rejected)
- When a tool event carries an <event_id> such as evt_123, every <observation> lists the events it
  covers in <event_ids>, one <event_id> element per id. To skip every event that no observation
  covers, add <skip_summary reason="..."/>.
- Required in each <observation>: type, title, subtitle, facts (at least one fact), narrative, concepts.
- Output must be well-formed XML: write & as &amp; and < as &lt; inside text.
- Output only <observation> blocks and the <skip_summary/> sentinel. No prose, no code fences.
- Never write a line starting with "Human:" or "Assistant:". Never invent events you were not sent.
- Record what happened. Do not give advice, recommendations, or next steps.`;

/**
 * Appended to a batch that is asked again after a non-XML reply (the
 * output_retry path), so the retry says what was wrong with the first answer.
 */
export function buildRejectedReplyNote(outputClass: 'prose' | 'idle' | string, preview: string): string {
  const error = outputClass === 'idle'
    ? '- [MISSING_COVERAGE] the reply was empty, so no event was observed or skipped'
    : `- [STRAY_TEXT] the reply was prose, not XML: "${preview.replace(/\s+/g, ' ').slice(0, 120)}"`;
  return `Your previous reply to this tool event was rejected by the output validator.
VALIDATION ERRORS
${error}
Answer again with <observation> blocks that cite the <event_id> above, or <skip_summary reason="..."/> if it is not worth recording. Output only the XML.`;
}

export function buildRepairPrompt(params: {
  instructions: string;
  eventPrompts: string[];
  rejectedResponse: string;
  violations: Violation[];
}): string {
  const errors = params.violations.map(v => `- [${v.code}] ${v.message}`).join('\n');
  return `${params.instructions}

The tool events below were sent to the observer, and its response was REJECTED by the output validator.

${params.eventPrompts.join('\n\n')}

<rejected_response>
${params.rejectedResponse.replace(/<\/rejected_response>/g, '&lt;/rejected_response>')}
</rejected_response>

VALIDATION ERRORS
${errors}

Respond again with corrected output for the events above: <observation> blocks citing their <event_ids>, and <skip_summary reason="..."/> if any of them are not worth recording. Output only the XML.`;
}
