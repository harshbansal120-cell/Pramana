# Pramana

*Pramāṇa* (Sanskrit): a means of valid knowledge.

A study companion that unifies lecture videos, slide decks and a textbook into one source-cited
knowledge base, tutors from it, and adapts to the learner. Its rule is that nothing reaches the
student unless a mechanical check has passed:

- **Citations are quote-locked.** Every source-backed statement carries a verbatim quote, and code
  checks the quote is really in the cited page, slide or transcript window. Outside knowledge is
  shown in a separate lane. Questions the material does not cover are declined.
- **Questions pass a verification gate.** A second model solves each question without seeing the
  key, and numerical answers are recomputed by executed code (exact arithmetic plus Monte Carlo).
- **Mastery is modelled, not guessed.** Bayesian knowledge tracing with prerequisite structure and
  forgetting, updated after every quiz answer and every tutor check question.

Default course: MIT OCW 6.041SC *Probabilistic Systems Analysis* lectures 1-8 (videos and slides)
plus Grinstead and Snell, *Introduction to Probability*, chapters 1-6. All of it is fetched over
HTTP at ingest time; nothing is bundled.

## Quick start

Requires Python 3.11+ and Node 18+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
copy .env.example .env            # then add your API key(s) to .env

cd frontend && npm install && npm run build && cd ..

cd backend
python -m pramana.cli ingest                 # fetch and index the default course
python -m pramana.cli bank --per-concept 2   # optional: pre-generate verified questions
python -m pramana.cli serve                  # http://127.0.0.1:8000
```

`ingest` is safe to re-run: sources already in the database are skipped, captions and PDFs are
cached, and model calls made during ingestion are cached by content hash.

### Models

Three roles, each a `provider:model` string in `.env`:

| Role | Does | Default |
|---|---|---|
| `PRAMANA_GENERATOR` | answers, questions, concept graph | `anthropic:claude-opus-5-5` |
| `PRAMANA_VERIFIER` | solves and judges what the generator wrote; RAGAS judge | `anthropic:claude-sonnet-5-5` |
| `PRAMANA_BULK` | figure descriptions, query rewriting | same as generator |

For cross-model validation the verifier should differ from the generator. With a Gemini key you can
use a different model family: `PRAMANA_VERIFIER=google:<model id>`.

Without any key, parsing, embeddings, search and slide-to-video alignment still run, and the
Library and Evaluation pages work. Figure reading, the concept map, chat and question generation
need a key and say so in the UI.

## Using it

| Page | What it shows |
|---|---|
| **Library** | Sources, ingestion with a live log, upload of a PDF or YouTube link, unit browser |
| **Tutor** | Grounded chat. Click a citation to open the exact slide, page or timestamp with the quote highlighted |
| **Practice** | Quiz, timed mock exam or adaptive diagnostic on a chosen scope, with cited feedback and a report |
| **Dashboard** | Course flow map coloured by mastery, weak concepts and their root causes, study schedule, flashcards |
| **Evaluation** | The measured results in `eval/results`, with their caveats |

A new student answers three intake questions and is offered a diagnostic.

## Evaluation

```bash
python eval/run_retrieval_eval.py     # retrieval and refusal gate; no model calls
python eval/simulate.py               # simulated students; no model calls
python eval/run_ragas.py              # RAGAS and grounding checks; needs keys
python -m pytest tests -q
```

Results and how to read them: [docs/EVALUATION.md](docs/EVALUATION.md).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): architecture, grounding method, learner model
- [docs/EVALUATION.md](docs/EVALUATION.md): test set, metrics, results, limits
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md): shot list for the demonstration video

## Repository layout

```
backend/pramana/     service code
  ingest/            parse.py (no LLM), enrich.py (figures, alignment, concepts), pipeline.py
  assess/            generate.py, verify.py, sandbox.py, report.py
  grounding.py       tutor: coverage gate, generation, quote-lock
  learner.py         learner model (pure Python)
  llm.py embed.py db.py service.py extras.py api.py cli.py
frontend/            React app (Vite)
eval/                testset.json, run_retrieval_eval.py, run_ragas.py, simulate.py, results/
tests/               unit tests and scripted end-to-end tests
course.yaml          the default course manifest
```

## Licences of the default material

MIT OpenCourseWare content is CC BY-NC-SA. Grinstead and Snell is distributed under the GNU Free
Documentation License. Pramana downloads them for personal study and does not redistribute them.
