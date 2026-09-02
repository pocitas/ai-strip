// const MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
// Cheaper/faster alternative (still supports JSON schema mode):
const MODEL = "@cf/meta/llama-3.1-8b-instruct-fast";

// ---------------------------------------------------------------------------
// JSON schema the model must return (OpenAI-compatible json_schema mode)
// ---------------------------------------------------------------------------

const FLIGHT_STRIP_SCHEMA = {
  type: "object",
  properties: {
    registration: { type: ["string", "null"], description: "Full aircraft registration, e.g. OK-AIM" },
    aircraft_type: { type: ["string", "null"], description: "Aircraft type / category, e.g. ultralight, C172" },
    dep: { type: ["string", "null"], description: "Departure aerodrome (ICAO if resolvable) or cardinal direction or raw text" },
    dest: { type: ["string", "null"], description: "Destination aerodrome (ICAO if resolvable) or cardinal direction or raw text" },
    pic: { type: ["string", "null"], description: "Pilot in command's name, single word" },
    pob: { type: ["integer", "null"], description: "Persons on board" },
    language: { type: ["string", "null"], description: "Radio/communication language, if explicitly stated" },
		notes: { type: ["string", "null"], description: "Additional operational notes or comments" },
  },
  required: ["registration", "aircraft_type", "dep", "dest", "pic", "pob", "language", "notes"],
  additionalProperties: false,
};

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are a flight-strip data extraction assistant for a small Czech airfield.
Convert short, terse operational notes into a structured JSON. Notes may be in Czech,
English, or a mix of both, are heavily abbreviated, fields may appear in any order,
and any field may be missing. Split the note into individual words and pairs.
Words are separated by spaces or commas only. Example: "word1 wo-rd-2,word3".
A same word or pair should not be used for multiple fields.

REGISTRATION
Return in uppercase. Usually the first or second word of the input. Always has minimum of 3 characters.
If the word is 3 lettters only, add "OK-" as a prefix. Example: "abc" -> "OK-ABC"
else
If the word is 4 digits only, add "OK-" as a prefix. Example: "1234" -> "OK-1234"
else
If the word is exactly 3 letters followed by 2 numbers only, add "OK-" as a prefix. Example: "abc12" -> "OK-ABC12"

After previous rules are applied, if the output still has no hyphen, add a missing hyphen following these rules:

Pass 1: If the word starts with any of the following character sequences,
    insert a hyphen after the sequence and skip Pass 2.
'C2', 'C3', 'C5', 'C6', 'C9', 'CC', 'CN', 'CP', 'CS', 'CU', 'CX',
'D2', 'D4', 'D6', 'DQ', 'GL', 'MT',
'P2', 'P4', 'PH', 'PJ', 'PK', 'PP', 'PR', 'PT', 'PZ',
'Z3', 'ZA', 'ZJ', 'ZK', 'ZL', 'ZP', 'ZS', 'ZT', 'ZU',
'B', 'C', 'D', 'F', 'G', 'I', 'M', 'P', 'Z', '2'

Examples of Pass 1: "CSSPA" -> "CS-SPA", "DAIZR" -> "D-AIZR", "GEDBA" -> "G-EDBA"

Pass 2: Insert hyphen after the second character if no hyphen has been added in Pass 1.
Examples of Pass 2: "OMMMI" -> "OM-MMI", "EIKLM" -> "EI-KLM"

AIRCRAFT TYPE
Expand known abbreviations
"ul" -> "ultralight",
"hel" -> "helicopter"
"gl" -> "glider"
"ces" -> "Cessna",
"pip" -> "Piper",
"tec" -> "Tecnam",
"twin" -> "twin-engine",
"vrt" -> "vrtulník",
"kl" -> "kluzák"

If the registration is exactly in the format "OK-XXXXX", where "X" is any character,
use the fifth character to infer the type using the following mapping:
    "U" = ultralight,
		"K" = ultralight glider,
		"W" = autogyro,
		"H" = ultralight helicopter,
		"Z" = motorized hang glider,
		"M" = motorized paraglider,
		"R" = hang glider,
		"B" = ultralight baloon

Example: "OK-PUA79" -> "ultralight", "OK-KWA33" -> "autogyro"

ROUTING
- Departure cues: "from", "orig", "z", "ze", "od". Also "from X" means the flight is
  arriving FROM X, so X is the departure aerodrome (dep), not the destination.
- Do not mistake with "dep south", "dep north", "dep east", "dep west", "dep straight", and similar phrases..
  This indicates the aircraft is departing towards the south, west, etc., not that the departure aerodrome
	is named "south". So any "dep" keyword followed by other than 2 or 4 characters should be considered as "dest" field.
	Any other strings, which follow the departure cues but do not look like an ICAO or IATA airport code
	just extract the word as is. (e.g. "from Zlonice" -> "Zlonice", "z J" -> "J")
- Destination cues: "to", "dest", "do", "heading". Also "to X" (departing to X) means the flight is departing towards X.
  So X is the destination aerodrome. 
  Examples: "heading south", "na jih", "to south", "direction south" all means the flight is heading
  towards the south. Cardinal directions can be represented by their first letter
	(e.g. "S" for "south", "N" for "north", "E" for "east", "W" for "west", "J" for "jih", "Z" for "západ", "V" for "východ").
	If it is represented by a single letter only, keep it as is and treat is as the destination field.
- In case the airport code of departure or destination is given with exactly two letters,
  prepend the missing czech prefix "LK" to form the full ICAO code. For example, "SN" becomes "LKSN", "LT" becomes "LKLT".
- If a routing endpoint is only a vague word (e.g. "east", "sever", "Zlonice") and no aerodrome/code is
  given, put that word in the field as plain text.
- Transition cues: "trans", "transition", "průlet", "pru". It means the aircraft just transits the airspace.
  It doesn't indicate any special flag for the strip, but expect two points, either entry / exit points
	(e.g. "W to E" -> dep = "W", dest = "E") or departure and destination aerodromes.
	example: "LT to TA" means dep = "LKLT", dest = "LKTA" (see adding the "LK" prefix rule mentioned above).
- The input can contain both dep and dest at the same time. If there is no dep cue, assume the first location mentioned
  is the departure point. (e.g. "TA do BE" -> dep = "LKTA", dest = "LKBE") or (e.g. "VL to EDMC" -> dep = "LKVL", dest = "EDMC")
- If both departure and destination points / airports are mentioned, these could be connected by a hyphen only
  (e.g. "LT-BE" -> dep = "LKLT", dest = "LKBE").
- Distinguish "z XX" (two letters) as indicating departure from a location (e.g. "z PR" -> dep = "LKPR")
  compared to "na Z" indicates the cardinal direction západ (e.g. "na Z" -> dest = "Z")


PERSONS ON BOARD (pob)
Cues: "pob", "p" (e.g. "pob 2"), a number directly followed or preceded by "p", "o", "os" or "osob" (Czech for
"persons") (e.g. "2p", "2 p", "2 o","2os", "pob 2"," os 2"). "solo" / "sólo" / "sám" means pob = 1.
Distinguish "p number" / "number p" (Persons on board = pob) compared to "p Name" (Pilot in command = pic).
(e.g. "p 2" -> pob = 2, "p John" -> pic = "John")

PILOT IN COMMAND (pic)
Cues: "pic", "pilot", "velitel", "vel", "v", "p" followed by a name. The pilot name is always a single word.
Distinguish "v Name" (velitel / pilot in command) compared to "to v" (indicating direction východ / east).
(e.g. "pic Poc", "pilot Smith", "velitel Burda"). Treat any other words as a different field or "notes".
(e.g. "pic Liu solo" -> "pic" = "Liu", "pob" = 1)
(e.g. "vel Noh za 2 hodiny" -> "pic" = "Noh", "notes" = "za 2 hodiny")


LANGUAGE
Only set this if the note explicitly names a language, e.g.
"eng" / "en" / "an" -> "english", otherwise null. This must be a separate word, not part of another word.
(e.g. "end" means registration "OK-END" without prefix, not the language "english" just because it contains "en").

NOTES
Can include any additional operational notes or comments relevant to the flight.
Can be recognized with "poz", "pozn", "note", "desc", followed by the actual note text.
Any uncategorized (unrecognized as a specific field) text at the end of the input can be treated as notes.

GENERAL RULES
- Never invent data. If a field is not stated and cannot be reasonably inferred from the
  cues above, output null for it.
- pob must be an integer or null, never a string.
- Output must strictly match the provided JSON schema — no extra commentary.`;

// Few-shot examples (kept consistent with the rules above; see worker.js header
// comment if you want to add more as you refine real club usage).
const FEW_SHOT = [
  {
    role: "user",
    content: "GUU02 ul arr from SN",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "OK-GUU02",
      aircraft_type: "ultralight",
      dep: "LKSN",
      dest: null,
      pic: null,
      pob: null,
      language: null,
      notes: null,
    }),
  },
  {
    role: "user",
    content: "UTC do LKTB 2os bez návratu",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "OK-UTC",
      aircraft_type: null,
      dep: null,
      dest: "LKTB",
      pic: null,
      pob: 2,
      language: null,
      notes: "bez návratu",
    }),
  },
	
  {
    role: "user",
    content: "IUU20 dep sever pic Karel solo",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "OK-IUU20",
      aircraft_type: "ultralight",
      dep: null,
      dest: "sever",
      pic: "Karel",
      pob: 1,
      language: null,
      notes: null,
    }),
  },
  {
    role: "user",
    content: "IFR eng z KV solo pozn student",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "OK-IFR",
      aircraft_type: null,
      dep: "LKKV",
      dest: null,
      pic: null,
      pob: 1,
      language: "english",
      notes: "student",
    }),
  },
  {
    role: "user",
    content: "kl 1082 eng ze ZB solo vel Rammert na 09",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "OK-1082",
      aircraft_type: "glider",
      dep: "LKZB",
      dest: null,
      pic: "Rammert",
      pob: 1,
      language: "english",
      notes: "na 09",
    }),
  },
	{
    role: "user",
    content: "dereu low fuel kt do mt eng pic Omg, os 2",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "D-EREU",
      aircraft_type: null,
      dep: "LKKT",
      dest: "LKMT",
      pic: "Omg",
      pob: 2,
      language: "english",
      notes: "low fuel",
    }),
  },
	{
    role: "user",
    content: "wia z lt do be chce na 09",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "OK-WIA",
      aircraft_type: null,
      dep: "LKLT",
      dest: "LKBE",
      pic: null,
      pob: null,
      language: null,
      notes: "chce na 09",
    }),
  },
	
	{
    role: "user",
    content: "xwa11 odlet na z os2",
  },
  {
    role: "assistant",
    content: JSON.stringify({
      registration: "OK-XWA11",
      aircraft_type: "autogyro",
      dep: null,
      dest: "z",
      pic: null,
      pob: 2,
      language: null,
      notes: "odlet",
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
      // Demo default — restrict to your Pages origin before sharing this publicly.
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
        max_tokens: 400,
        temperature: 0,
      });

      // On success, Workers AI JSON-schema mode returns { response: {...} }.
      const parsed = result && typeof result === "object" ? result.response : null;

      if (!parsed) {
        return json(
          { error: "Model did not return schema-conformant JSON", raw: result },
          502,
          corsHeaders
        );
      }

      return json({ parsed, raw: result, model: MODEL }, 200, corsHeaders);
    } catch (err) {
      return json({ error: "AI request failed", detail: String(err && err.message ? err.message : err) }, 502, corsHeaders);
    }
  },
};
