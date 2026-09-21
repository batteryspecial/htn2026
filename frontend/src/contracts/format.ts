/** JSON colouring, shared by anything that puts a spec on screen. */

export type JsonTokenKind = 'key' | 'str' | 'num' | 'bool' | 'null' | 'plain';

export interface JsonToken {
  text: string;
  kind: JsonTokenKind;
}

const TOKEN_RE =
  /("(?:\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(?:\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g;

/**
 * Tokenise pretty-printed JSON for colouring.
 *
 * Returns tokens rather than an HTML string: the old version built markup and
 * handed it to innerHTML, which React has no business doing.
 */
export function tokenizeJson(value: unknown): JsonToken[] {
  const text = JSON.stringify(value, null, 2) ?? 'null';
  const tokens: JsonToken[] = [];
  let last = 0;

  for (const match of text.matchAll(TOKEN_RE)) {
    const raw = match[0];
    const at = match.index;
    if (at > last) tokens.push({ text: text.slice(last, at), kind: 'plain' });

    let kind: JsonTokenKind = 'num';
    if (raw.startsWith('"')) kind = raw.trimEnd().endsWith(':') ? 'key' : 'str';
    else if (raw === 'true' || raw === 'false') kind = 'bool';
    else if (raw === 'null') kind = 'null';

    tokens.push({ text: raw, kind });
    last = at + raw.length;
  }

  if (last < text.length) tokens.push({ text: text.slice(last), kind: 'plain' });
  return tokens;
}
