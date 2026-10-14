# Pramana: architecture, grounding method and learner model

Pramana (Sanskrit *pramāṇa*, "means of valid knowledge") is a study companion built on one rule:
**nothing reaches the student unless a mechanical check has passed.** Citations are checked by
string matching, answer keys by executed code and a second model, and mastery estimates by a
simulation with known ground truth.

## 1. System overview

```
 sources                      knowledge base (SQLite)                 student-facing
 ───────                      ───────────────────────                 ──────────────
 YouTube lecture ─┐  parse    units ── every unit keeps its           Tutor chat ── quote-lock
 slide deck PDF  ─┼────────►  origin: page / slide / seconds          Practice  ── verification gate
 textbook PDF    ─┘           │                                        Dashboard ── learner model
                              ├─ embeddings (local, 384-d) + BM25
                  enrich      ├─ figure units (vision model reads each diagram)
                 ────────►    ├─ links: slide ↔ video timestamp
                              ├─ concepts, topics, prerequisites
                              └─ unit ↔ concept tags
```

| Layer | Code | Notes |
|---|---|---|
| Ingestion | `backend/pramana/ingest/` | `parse.py` has no LLM in it; `enrich.py` holds the model-assisted stages |
| Retrieval | `backend/pramana/embed.py` | fastembed `bge-small-en-v1.5` + BM25, fused by reciprocal rank |
| Tutor | `backend/pramana/grounding.py` | coverage gate, generation, quote-lock, chat-driven mastery updates |
| Assessment | `backend/pramana/assess/` | `generate.py`, `verify.py` (gate and grading), `sandbox.py`, `report.py` |
| Learner model | `backend/pramana/learner.py` | pure Python, shared by the product and the simulation |
| Models | `backend/pramana/llm.py` | three roles (generator, verifier, bulk), each a `provider:model` string |
| API and UI | `backend/pramana/api.py`, `frontend/` | FastAPI; React (Vite) |
| Evaluation | `eval/` | retrieval, RAGAS, simulated students |

One SQLite file holds everything, so a demo machine needs no services. The index is rebuilt in
memory from that file at start-up; the default course is about a thousand units.

## 2. Multimodal knowledge base

**Units.** A unit is the smallest citable piece: one slide, part of a textbook page, a roughly
55-second transcript window, or one figure. Every unit stores `loc_type` (`slide`, `page`, `time`)
and `loc_start`/`loc_end`, which is what lets a citation open the exact place.

**No manual preprocessing.**
- *Videos.* Captions with timestamps are fetched for a YouTube URL (two fetch routes, retried and
  cached, because YouTube rate-limits bursts). A local video file goes through `faster-whisper`
  if that optional package is installed.
- *Slides.* Decks are often distributed as handouts with several slides tiled on a page. The parser
  detects the empty gutters through the page centre, applies the layout document-wide, and emits one
  unit and one image per slide.
- *Textbooks.* Running headers are stripped, hyphenation and ligatures repaired, and long pages
  split, with each part keeping its page number.

**Figures.** Each slide image, and each textbook page that carries a figure, is sent to a vision
model which describes every diagram: what is drawn, labels, numbers, and the idea shown. Each
description becomes a `figure` unit at the same location. Figure units are marked *derived*; the
viewer shows the image above the text and says the text is an AI reading, since it is the only
unit type whose text is not the author's own words.

**Triangulated sources.** Slides are shown in order, so for each lecture we choose the monotone
assignment of slides to transcript windows that maximises total embedding similarity (dynamic
programming, `align_slides_to_video`). Opening a slide then also offers "the same moment" in the
video. This uses no LLM.

**Concept graph.** One generator call reads a course outline (slide text, textbook section
headings, definitions and theorems) and returns topics → subtopics → concepts, each with a
definition, direct prerequisites and common misconceptions. Two mechanical steps follow:
- *Tagging.* Every unit is tagged with its nearest concepts by embedding similarity; each unit
  gets at least one tag.
- *Prerequisite validation.* An edge "B requires A" is kept only if A is taught no later than B
  in the material, and any edge that would close a cycle is dropped. The counts of proposed, kept
  and dropped edges are stored and shown.

## 3. Grounding method: quote-lock

1. **Retrieve** the top 8 units with hybrid search. For follow-ups and non-English messages a
   small model first rewrites the message into a self-contained English query.
2. **Coverage gate (first refusal layer).** If the best cosine similarity is below a threshold
   *and* the query has no distinctive term that the corpus uses, decline without generating. The
   threshold is fitted by `eval/run_retrieval_eval.py`. The gate is tuned never to block an
   answerable question; it catches off-topic queries.
3. **Generate in lanes.** The generator returns segments. A `source` segment must carry, for each
   claim, the id of a retrieved unit and a quote copied verbatim from it. Anything not supported by
   the excerpts must go in an `outside` segment. It also returns `coverage` (`full`, `partial`,
   `none`); `none` is the **second refusal layer**, for in-domain questions the material does not
   cover.
4. **Verify in code.** `locate_quote` checks every quote against the text of the unit it cites,
   ignoring case, punctuation and line breaks, with a 0.9 similarity allowance for caption breaks.
   A citation that fails is discarded. A `source` segment with no surviving citation is relabelled
   **unverified** and shown with a warning. A citation that survives is guaranteed to point at
   text that exists at that page, slide or timestamp.
5. **Show the lanes.** Source-backed text, outside knowledge and unverified text are visually
   distinct. A declined question shows the nearest covered material and an explicit "answer from
   general knowledge anyway" button, whose answer is rendered entirely in the outside lane.

What quote-lock guarantees is that *the quote exists where the citation says*. It does not
guarantee the sentence drawn from the quote is a fair reading of it; that is what RAGAS
faithfulness measures.

## 4. Assessment: the verification gate

A question is generated for a concept from that concept's best-tagged units, at a difficulty
chosen by the learner model. It enters the bank only if **every** applicable check passes:

| Check | How |
|---|---|
| structure | four distinct options and exactly one key; a finite numeric answer; and so on |
| quote_lock | its source quote is found in the cited unit |
| code_exact | generator-written `exact()` is executed and reproduces the key |
| code_sim | generator-written `simulate(rng, n)` Monte Carlo estimate agrees with the key |
| blind_solve | the verifier model, never shown the key, reaches the same answer |
| answerable | the verifier judges it answerable from the excerpt and unambiguous |
| novel | embedding similarity to every banked question on that concept is below 0.92 |

Rejected questions are stored with the failing check, so the rejection rate per check is a
reportable number (`/api/bank`). Checking code is screened by an AST allow-list and run in a
separate interpreter with a timeout.

**Misconception-tagged distractors.** Each wrong option is written as the answer a student with a
specific wrong belief would choose, and carries that belief as a tag. Grading a wrong option
therefore yields a named misconception. The report calls it *likely* only if it recurs.

**No repeats.** An assessment draws verified questions the student has never been served, nearest
difficulty first, and generates new ones only for gaps.

**Modes.** *Quiz*: cited feedback after each answer. *Mock exam*: timed, feedback at the end.
*Diagnostic*: each answer updates the model and the next question is chosen from the new state.

**Report.** Score by concept and topic, mastery before and after, misconceptions, root causes
(weak concepts explained by a weaker prerequisite) and, for each, the best place in the material
to study next.

## 5. Learner model

`learner.py` keeps, per student and concept, `P(mastered)`, a memory stability in days, and the
time of the last evidence.

1. **Bayesian knowledge tracing.** An answer updates `P(mastered)` by Bayes' rule. Guess depends on
   question type (0.25 multiple choice, 0.03 numerical, 0.08 short answer) and slip on difficulty,
   so a correct numerical answer is stronger evidence than a correct multiple-choice one.
2. **Prerequisite structure.**
   - *Attribution.* A question on C needs C and its direct prerequisites (noisy-AND). A wrong
     answer lowers C and any prerequisite that is already doubtful.
   - *Consistency.* Knowing a concept implies knowing what it is built on. Evidence that raises C
     lifts every ancestor to at least 0.9 × P(C), so one correct answer deep in the graph vouches
     for the chain beneath it. This is what lets an 8 to 10 question diagnostic say something about
     30 or more concepts.
   - *Structural prior.* An untested concept is discounted while its prerequisites look unknown.
3. **Forgetting.** Mastery decays as `exp(-Δt / stability)`. Stability grows 1.8× on a successful
   review and shrinks on a lapse; a concept answered correctly on first sight starts with 3× the
   base stability, as prior knowledge is already consolidated.

**Evidence sources.** Quiz and mock answers (weight 1). Diagnostic answers (weight 1, no learning
transition). Chat: when the tutor's check question is answered, the graded result counts at
weight 0.5; a misconception voiced in chat counts as soft negative evidence at weight 0.3.

**Decisions it drives.**
- *Next question.* Highest need = low mastery × readiness of prerequisites, plus a bonus for
  learned concepts that are fading; then descend to the weakest doubtful prerequisite (the root
  cause). Difficulty is the level whose predicted success is closest to 70%.
- *Diagnostic.* The concept whose answer is most informative: entropy of its estimate, weighted by
  how many ancestors and descendants it speaks for.
- *Tutor.* The prompt includes the student's mastery of the concepts in play, weak prerequisites
  and observed misconceptions, and asks for a bridge to a weak prerequisite first.
- *Schedule.* Simulates forward to the exam date and each day picks the concepts with the lowest
  predicted recall on exam day, prerequisites first.

**Cold start.** Intake asks three things (self-rated level, language, exam date). The level sets
the prior (0.15, 0.40 or 0.65). The dashboard states that these are self-reported and offers the
adaptive diagnostic, which replaces them with measured estimates.

## 6. Optional features included

- Course flow map of concepts and prerequisites, coloured by mastery.
- Flashcards for weak concepts, each with a quote-locked citation.
- Study schedule from weak concepts, forgetting and time to exam.
- Hindi and Hinglish explanations over English sources. Quotes stay in the source language.
- Voice: speech input and read-aloud through the browser's speech APIs.

## 7. Known limits

- Quote-lock verifies that quotes exist, not that every inference from them is sound.
- Figure text is model-written, the one exception to verbatim units; it is labelled.
- The answer-checking sandbox guards against accidents, not against a determined attacker.
- Unit-to-concept tagging is by embedding similarity and has not been separately evaluated.
- See `docs/EVALUATION.md` for what has and has not been measured.
