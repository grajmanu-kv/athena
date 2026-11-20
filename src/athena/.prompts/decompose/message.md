# Prompt decomposition request

You are decomposing the following prompt for the domain: **$domain**

---

## Prompt to decompose

The raw text below was sent to an LLM. Decompose it into its constituent
epistemic components following the instructions in your system prompt.

$fence
$prompt
$fence

---

## Domain description

Use this background to correctly interpret technical terms and domain-specific
language in the prompt above.

$domain_description

---

## Declared constraints

These are positive obligations the LLM response was explicitly required to
satisfy. Pass them through **unchanged** into `explicit_constraints`.
Do not modify, paraphrase, or omit any of them.

$declared_constraints

---

## Declared boundaries

These are prohibitions the LLM response was explicitly forbidden to violate.
Pass them through **unchanged** into `explicit_boundaries`.
Do not modify, paraphrase, or omit any of them.

$declared_boundaries

---

## Declared term definitions

These definitions are **authoritative** — they override general language and
the LLM's training distribution semantics. Pass them through into
`term_definitions`. If you extract additional definitions from the prompt,
merge them in but do not override these declared ones on collision.

$declared_term_definitions

---

## Your decomposition task

Working from the prompt text above:

1. **implicit_constraints** — extract constraints embedded in the prompt that
   are NOT already in the declared constraints above. Look for role framing
   ("As a cautious analyst..."), output format directives, scope restrictions,
   and epistemic authority constraints. If none found, return empty list.

2. **implicit_boundaries** — extract prohibitions embedded in the prompt that
   are NOT already in the declared boundaries above. Look for explicit
   prohibitions ("Do not...", "Never..."), role framing that implies
   prohibitions, and scope restrictions that function as prohibitions.
   If none found, return empty list.

3. **term_definitions** — merge the declared definitions above with any
   additional definitions found in the prompt (explicit: "X means...",
   or operational: "Flag readings above 3σ as anomalies").
   Declared definitions always win on collision.

4. **undeclared_terms** — technical, domain-specific, or specialised terms
   used in the prompt that have no definition in the merged term_definitions
   and are not common everyday language. These are potential semantic drift
   vectors. Do NOT flag common words.

5. **injected_facts** — factual data embedded directly in the prompt as
   context for the LLM: sensor readings, retrieved documents, prior
   conversation turns, timestamps, entity states, measurement values.
   These are distinct from standing domain knowledge. Extract each as a
   single plain-language statement preserving exact values.

6. **extraction_notes** — populate if you encountered ambiguous text,
   judgment calls, or anything that may affect downstream validation.
   Set to null if nothing notable.

Remember: extract only what is genuinely present in the prompt text.
Do not invent constraints or boundaries that are not there.