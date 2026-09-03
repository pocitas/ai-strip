/**
 * Flight strip autofill — Cloudflare Worker
 *
 * Split of responsibilities:
 *  - The LLM only extracts RAW field values from messy Czech/English shorthand
 *    (which token is the registration, which is dep/dest, etc). It does no
 *    normalization — no hyphen insertion, no ICAO prefixing.
 *  - aircraft_type is NOT extracted by the LLM at all. It is derived purely
 *    from the LAA registration-format rule (5-char OK-YTINN suffix -> type
 *    letter). Any explicit type word in the input (e.g. "ul", "kl") is not
 *    a recognized field anymore and just falls through to "notes" as
 *    leftover text, same as any other uncategorized word.
 *  - All normalization is deterministic and lives below in plain JS, where
 *    it's testable and can never "forget" a rule the way a small model
 *    occasionally does under schema-constrained decoding.
 *
 * No API key is needed for Workers AI itself — the binding is authenticated
 * by your Cloudflare account when deployed.
 */

// ---------------------------------------------------------------------------
// MODEL
// ---------------------------------------------------------------------------

// const MODEL = "@cf/meta/llama-3.1-8b-instruct-fast";
// Confirmed json_schema-compatible, cheapest option — current default.
//
// If accuracy is still disappointing after the prompt simplification below,
// the only real step up on the confirmed-compatible list is:
const MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
// ~6x the per-token cost of the 8B model — at 300 req/day this runs past the
// 10,000 free daily neurons by roughly 3,000, which costs about $0.03–0.05/day
// on Workers Paid. Worth it only if 8B keeps failing after the JS refactor;
// not a "nearly free" default swap.

// ---------------------------------------------------------------------------
// JSON schema the model must return — RAW fields only, no normalization,
// no aircraft_type (that's derived from the registration in JS, below).
// ---------------------------------------------------------------------------

const FLIGHT_STRIP_SCHEMA = {
  type: "object",
  properties: {
    registration: { type: ["string", "null"], description: "Registration token exactly as typed in the input, e.g. ais, GUU02, d-ereu" },
    dep: { type: ["string", "null"], description: "Departure location exactly as typed — code, direction word, or place name, unmodified" },
    dest: { type: ["string", "null"], description: "Destination location exactly as typed — code, direction word, or place name, unmodified" },
    pic: { type: ["string", "null"], description: "Pilot in command's name, single word" },
    pob: { type: ["integer", "null"], description: "Persons on board" },
    language: { type: ["string", "null"], description: "Radio/communication language, only if explicitly stated" },
    notes: { type: ["string", "null"], description: "Any leftover or explicitly flagged operational notes" },
  },
  required: ["registration", "dep", "dest", "pic", "pob", "language", "notes"],
  additionalProperties: false,
};

// ---------------------------------------------------------------------------
// Prompt — only the parts that genuinely need language understanding.
// Everything algorithmic (hyphens, ICAO prefixes, type-from-registration)
// has been moved to the normalizeXxx() functions below.
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are a flight-strip data extraction assistant for a small Czech airfield.
Extract RAW field values from a short, informal flight-ops note. Notes may be in
Czech, English, or a mix, heavily abbreviated, in any word order, with fields
possibly missing.

IMPORTANT: return every value EXACTLY as it appears in the input. Do not add
prefixes, insert hyphens, expand airport codes, or infer anything not directly
stated. Normalization happens outside this step.

Aircraft type is NOT a field you extract — it is derived elsewhere from the
registration. Do not try to identify or output an aircraft type; if the input
contains a word that looks like a type abbreviation (e.g. "ul", "kl", "gl",
"vrt"), treat it like any other word that doesn't fit a recognized field (see
NOTES below).

REGISTRATION
- Always the first word/token of the input, minimum 3 characters.
- May be preceded by a marker like "r", "reg", "i", "ima" — if so, use the word
  that follows the marker, not the marker itself.
- Return exactly as typed (keep original case).

ROUTING (dep / dest)
- dep = where the flight is coming FROM. Cues: "from", "orig", "z", "ze", "od",
  "arr from X" (X is dep).
- dest = where the flight is going TO. Cues: "to", "dest", "do", "heading", "na"
  — EXCEPT "na" followed directly by a 2-digit number (e.g. "na 09") is a
  runway note, not a destination; put "na 09" in notes instead.
- If two locations are given with no dep cue on either, the first mentioned is
  dep and the second is dest.
- Return the location token exactly as typed.

PERSONS ON BOARD (pob)
- Cues: "pob", "p", "o", "os", "osob" attached to or near a number.
  "solo" / "sám" means pob = 1.
- Must end up as an integer.

PILOT IN COMMAND (pic)
- Always a single word (a name). Cues: "pic", "pilot", "velitel", "vel", "v",
  "p" immediately followed by a name-looking word.

LANGUAGE
- Only set if a radio/communication language is explicitly named:
  "eng"/"en" -> "english". Never guess.

NOTES
- Cues: "poz", "pozn", "note", "desc" followed by the note text.
- Also: any leftover text that doesn't fit another field (including apparent
  aircraft-type words — see above), and any runway/heading numbers mentioned
  after "na" (see ROUTING exception above).

GENERAL RULES
- Never invent data. Use null for anything not stated or clearly indicated by
  the cues above.
- pob is an integer or null, never a string.
- Output must strictly match the JSON schema — no extra commentary.`;

const FEW_SHOT = [
  {
    role: "user",
    content: "GUU02 ul arr from SN",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "GUU02", dep: "SN", dest: null,
      pic: null, pob: null, language: null, notes: "ul",
    }),
  },
  {
    role: "user",
    content: "UTC do LKTB 2os bez návratu",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "UTC", dep: null, dest: "LKTB",
      pic: null, pob: 2, language: null, notes: "bez návratu",
    }),
  },
  {
    role: "user",
    content: "IUU20 dep sever pic Karel solo",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "IUU20", dep: null, dest: "sever",
      pic: "Karel", pob: 1, language: null, notes: null,
    }),
  },
  {
    role: "user",
    content: "IFR eng z KV solo pozn student",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "IFR", dep: "KV", dest: null,
      pic: null, pob: 1, language: "english", notes: "student",
    }),
  },
  {
    role: "user",
    content: "1082 eng kl ze ZB solo vel Rammert na 09",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "1082", dep: "ZB", dest: null,
      pic: "Rammert", pob: 1, language: "english", notes: "kl na 09",
    }),
  },
  {
    role: "user",
    content: "dereu low fuel kt do mt eng pic Omg, os 2",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "dereu", dep: "kt", dest: "mt",
      pic: "Omg", pob: 2, language: "english", notes: "low fuel",
    }),
  },
  {
    role: "user",
    content: "wia z lt do be chce na 09",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "wia", dep: "lt", dest: "be",
      pic: null, pob: null, language: null, notes: "chce na 09",
    }),
  },
  {
    role: "user",
    content: "xwa11 odlet na z os2",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "xwa11", dep: null, dest: "z",
      pic: null, pob: 2, language: null, notes: "odlet",
    }),
  },
  {
    // The failure case that prompted the earlier refactor.
    role: "user",
    content: "ais do kv pic Šmerda",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "ais", dep: null, dest: "kv",
      pic: "Šmerda", pob: null, language: null, notes: null,
    }),
  },
];

function buildMessages(userText) {
  return [
    { role: "system", content: SYSTEM_PROMPT },
    ...FEW_SHOT,
    { role: "user", content: userText },
  ];
}

// ---------------------------------------------------------------------------
// Deterministic normalization — everything that used to be prose instructions
// is now plain JS, ported 1:1 from the rules you specified.
// ---------------------------------------------------------------------------

// Pass A prefixes for missing-hyphen insertion, longest match first so e.g.
// "CSSPA" matches "CS" (2 chars) rather than falling through to "C" (1 char).
const HYPHEN_PREFIXES = [
  "C2", "C3", "C5", "C6", "C9", "CC", "CN", "CP", "CS", "CU", "CX",
  "D2", "D4", "D6", "DQ", "GL", "MT",
  "P2", "P4", "PH", "PJ", "PK", "PP", "PR", "PT", "PZ",
  "Z3", "ZA", "ZJ", "ZK", "ZL", "ZP", "ZS", "ZT", "ZU",
  "B", "C", "D", "F", "G", "I", "M", "P", "Z", "2",
].sort((a, b) => b.length - a.length);

function insertMissingHyphen(upper) {
  for (const prefix of HYPHEN_PREFIXES) {
    if (upper.startsWith(prefix)) {
      return upper.slice(0, prefix.length) + "-" + upper.slice(prefix.length);
    }
  }
  // Pass B fallback: hyphen after the first two characters.
  return upper.slice(0, 2) + "-" + upper.slice(2);
}

function normalizeRegistration(raw) {
  if (!raw) return null;
  const upper = String(raw).trim().toUpperCase();
  if (!upper) return null;

  if (upper.includes("-")) return upper; // already a full/foreign registration

  if (/^[A-Z]{3}$/.test(upper)) return "OK-" + upper;
  if (/^[0-9]+$/.test(upper)) return "OK-" + upper; // digits-only, any length
  if (/^[A-Z]{3}[0-9]{2}$/.test(upper)) return "OK-" + upper;

  return insertMissingHyphen(upper);
}

const CZECH_AIRPORT_CODES = new Set([
  "be", "bo", "tb", "br", "ba", "bu", "ce", "cs", "dk", "er", "fr", "hb", "hd", "hc", "hv", "hs", "hk", "hn",
  "cb", "ch", "ct", "cr", "ja", "jc", "ji", "jh", "kv", "kl", "kt", "ko", "kr", "km", "ka", "ku", "ky", "pl",
  "lt", "lb", "lu", "mr", "cm", "mi", "mb", "mh", "mk", "mo", "nm", "ol", "mt", "pc", "pd", "ps", "ln", "pn",
  "pa", "pr", "vo", "pj", "po", "pm", "pi", "rk", "ra", "ry", "ro", "sz", "sk", "sn", "so", "sa", "sb", "st",
  "sr", "su", "ta", "td", "tc", "to", "ul", "uo", "vp", "vl", "vr", "vm", "vy", "za", "zb", "zl", "zn", "zm",
  "zd",
]);

function normalizeLocation(raw) {
  if (!raw) return null;
  const trimmed = String(raw).trim();
  const lower = trimmed.toLowerCase();

  if (/^[a-z]{2}$/.test(lower) && CZECH_AIRPORT_CODES.has(lower)) {
    return "LK" + lower.toUpperCase();
  }
  if (/^[a-z]{4}$/i.test(trimmed)) {
    return trimmed.toUpperCase(); // looks like a bare ICAO code already
  }
  return trimmed; // direction word, place name, single-letter direction, etc.
}

const LAA_TYPE_MAP = {
  U: "ultralight",
  K: "ultralight glider",
  W: "autogyro",
  H: "ultralight helicopter",
  Z: "motorized hang glider",
  M: "motorized paraglider",
  R: "hang glider",
  B: "ultralight balloon",
};

// The ONLY source of aircraft_type. No LLM input involved.
function inferAircraftTypeFromRegistration(registration) {
  if (!registration) return null;
  const m = /^OK-([A-Z0-9]{5})$/.exec(registration);
  if (!m) return null;
  return LAA_TYPE_MAP[m[1][1]] || null; // 2nd char of the 5-char suffix = type letter
}

function normalizeResult(raw) {
  const registration = normalizeRegistration(raw.registration);
  return {
    registration,
    aircraft_type: inferAircraftTypeFromRegistration(registration),
    dep: normalizeLocation(raw.dep),
    dest: normalizeLocation(raw.dest),
    pic: raw.pic ?? null,
    pob: typeof raw.pob === "number" ? raw.pob : null,
    language: raw.language ?? null,
    notes: raw.notes ?? null,
  };
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

function json(data, status, extraHeaders) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...extraHeaders },
  });
}

export default {
  async fetch(request, env) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    const url = new URL(request.url);
    if (request.method !== "POST" || url.pathname !== "/api/autofill") {
      return json({ error: "Not found. POST to /api/autofill" }, 404, corsHeaders);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Invalid JSON body" }, 400, corsHeaders);
    }

    const text = (body && body.text ? String(body.text) : "").trim();
    if (!text) return json({ error: "Missing 'text' field" }, 400, corsHeaders);
    if (text.length > 500) return json({ error: "Text too long (max 500 characters)" }, 400, corsHeaders);

    try {
      const result = await env.AI.run(MODEL, {
        messages: buildMessages(text),
        response_format: { type: "json_schema", json_schema: FLIGHT_STRIP_SCHEMA },
        max_tokens: 300,
        temperature: 0,
      });

      const rawParsed = result && typeof result === "object" ? result.response : null;
      if (!rawParsed) {
        return json({ error: "Model did not return schema-conformant JSON", raw: result }, 502, corsHeaders);
      }

      const parsed = normalizeResult(rawParsed);
      return json({ parsed, raw: result, model: MODEL }, 200, corsHeaders);
    } catch (err) {
      return json({ error: "AI request failed", detail: String(err && err.message ? err.message : err) }, 502, corsHeaders);
    }
  },
};
