# Evaluation

Three evaluations, each a script in `eval/` that writes JSON to `eval/results/` (shown on the app's
Evaluation page). The numbers below were produced on this repository as it stands.

| Evaluation | Script | Status |
|---|---|---|
| Retrieval and refusal gate | `eval/run_retrieval_eval.py` | **Run.** No model calls. |
| Personalisation with simulated students | `eval/simulate.py` | **Run**, on a stand-in concept graph (see below). |
| Generation quality (RAGAS) and grounding | `eval/run_ragas.py` | **Not run yet.** Needs API keys. No numbers are claimed. |

Three pipeline stages also need a model and so are **untested against a real model**: figure
reading, concept-graph extraction, and question generation with its verifier. Their surrounding
code is exercised by scripted tests (`tests/test_pipeline.py`), which say nothing about model
quality.

## 1. Test set

`eval/testset.json`, written by hand for the default course:

- **58 answerable questions** with reference answers and gold source locations: 37 anchored on
  slides, 16 textbook-only, 2 video-anchored, 3 about figures.
- **10 off-topic questions** (transformers, photosynthesis, TCP vs UDP, ...).
- **10 in-domain questions the material does not cover** (Chebyshev's inequality, moment
  generating functions, Poisson *process*, ...). Each was confirmed absent by keyword search. These
  are the hard negatives: they look like course questions.

Gold locations were found by reading the slides and keyword-searching the textbook and captions,
not with the retriever. After the first run, the four questions with no gold hit were judged by
hand; the retrieved passages did answer them, so those locations were added to the gold lists
(pooled relevance judging). Before that correction hybrid hit@8 was 0.931.

## 2. Retrieval and refusal gate

996 units indexed, top 8 retrieved, 58 answerable questions.

| Retriever | hit@1 | hit@3 | hit@8 | MRR | gold-source recall |
|---|---|---|---|---|---|
| **hybrid** (used) | 0.621 | 0.845 | 1.000 | 0.757 | 0.879 |
| dense only | 0.621 | 0.793 | 0.983 | 0.741 | 0.879 |
| BM25 only | 0.517 | 0.828 | 0.897 | 0.662 | 0.799 |

Hybrid's advantage over dense alone is small and comes from proper nouns ("Galton board" has a top
embedding similarity of only 0.54 but is found lexically).

By where the answer lives (hybrid): slides hit@1 0.568, textbook 0.750, video 0.500 (n = 2),
figures 0.667 (n = 3). Slide questions rank lower at 1 because a lecture transcript often
outranks the slide it explains; all are found by rank 8.

**Refusal gate** (first refusal layer only: similarity threshold plus rare-term rescue). The
threshold was fitted on half the questions and scored on the other half.

| | answerable passed | off-topic declined | uncovered in-domain declined |
|---|---|---|---|
| Held-out half, threshold 0.735 | 100% | 100% | 80% |
| All 78 questions | 100% | 90% | 60% |
| All, without the rare-term rescue | 89.7% | 100% | 90% |

Results are flat for thresholds from 0.70 to 0.75; the product default is 0.72. The gate is
deliberately tuned not to block answerable questions. The in-domain questions that pass it are
meant to be caught by the generator's own coverage judgement, which only `run_ragas.py` measures.

Limits: the test set is small; gold lists hold verified places only, so location precision (0.276)
is a lower bound and is not a meaningful quality number; the rare-term rule was designed while
looking at this test set, so the held-out half is not a fully independent test of it; figure units
did not exist for this run, so figure questions were matched through slide text only.

## 3. Personalisation with simulated students

`eval/simulate.py` runs the production learner model (`learner.py`, unmodified) against 30
simulated students: 5 profiles (novice, average, gaps in prerequisites, forgetful, strong
reviser) × 6 seeds. Each has a hidden true knowledge state, slip rate, learning rate, forgetting
rate and two seeded misconceptions. 8 sessions of 10 questions, 2 days apart. Every policy meets
the identical students.

**This run used a stand-in concept graph** (`eval/fixtures/probability_graph.json`, 34 concepts,
written by hand to mirror the syllabus), because the real graph needs a generator key. Re-run
`python eval/simulate.py` after ingestion with a key to use the extracted graph.

### Mastery gain by question-selection policy

True mastery is the fraction of concepts the simulated student really knows.

| Policy | Mastery gain | Pramana minus this (95% CI) | Repetition rate | In-zone rate |
|---|---|---|---|---|
| **Pramana** | **+0.118** | | 0.0% | 0.401 |
| Random questions | +0.010 | +0.108 [+0.069, +0.148] | 8.3% | 0.112 |
| Syllabus order | +0.036 | +0.081 [+0.036, +0.125] | 0.0% | 0.164 |
| Pramana, no prerequisite structure | +0.099 | +0.019 [−0.016, +0.053] | 0.0% | 0.354 |
| Pramana, no forgetting | +0.121 | −0.003 [−0.044, +0.034] | 0.0% | 0.386 |

"In-zone" is the share of questions on a concept the student does not yet know but is ready to
learn. Pramana spends session 0 on a no-feedback diagnostic, so it has one fewer teaching session
than the baselines and still gains more.

By profile, Pramana gained +0.255 (novice), +0.201 (average), +0.064 (gaps in prerequisites),
+0.142 (strong reviser) and **lost** 0.074 for the forgetful profile (random lost 0.118).

### Does the mastery estimate improve with more data?

Error of the model's estimates against the hidden truth, by session (Pramana policy):

| Session | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| RMSE | 0.449 | 0.418 | 0.416 | 0.416 | 0.391 | 0.388 | 0.393 | 0.399 | 0.394 |
| AUC | 0.564 | 0.590 | 0.677 | 0.681 | 0.757 | 0.771 | 0.761 | 0.755 | 0.768 |

Estimates improve through session 4 and then plateau.

### Learner-model ablation, on identical answer logs

| Estimator | Final RMSE (2-day gaps) | Final RMSE (weekly sessions) |
|---|---|---|
| Full model | 0.394 | 0.340 |
| No prerequisite structure | 0.425 | 0.382 |
| No forgetting | 0.366 | 0.341 |
| Plain BKT | 0.383 | 0.364 |

### What did not work

- **The forgetting component did not improve estimate accuracy.** Removing it lowered RMSE at
  2-day spacing (0.366 vs 0.394) and made no difference at weekly spacing. As a selection policy
  it also showed no measurable benefit (−0.003, interval spanning zero). It is kept because the
  study schedule depends on it, but these results do not justify it on accuracy grounds.
- **The benefit of prerequisite structure for learning gains is not established.** It improves
  estimate accuracy (0.425 → 0.394) and in-zone targeting (0.354 → 0.401), but its effect on
  mastery gain (+0.019) has an interval that includes zero at n = 30.
- **Misconception detection from distractor choice was poor.** Recall 0.27 and precision 0.16 for
  Pramana. Only 28% of seeded misconceptions were even observable (their concept drew two or more
  wrong answers), since the selector tends to move to prerequisites instead. The product therefore
  labels a misconception "likely" only when it recurs and "possible" otherwise.
- **With weekly sessions every policy lost mastery**, because ten questions a week did not
  outpace forgetting in this simulation. Pramana lost least (−0.043 vs −0.118 random, −0.104
  syllabus).
- **Estimate accuracy is modest in absolute terms** (final RMSE 0.39, AUC 0.77) after 80 answers
  spread over 34 concepts.

### How much weight these numbers bear

Students are simulated, so this shows the system behaves as designed under stated assumptions; it
is not evidence of learning gains in real students. The simulated world assumes a question needs
the concept and its prerequisites, and that a concept is learned ten times more slowly while a
prerequisite is missing. The learner model makes the same structural assumption, which favours it
over the baselines. The zero repetition rate is a property of the selector (it never serves a seen
question while an unseen one exists), not a finding. During development the learner model was
revised three times after looking at these results (memory time constant, prerequisite
consistency, crediting correct answers), so the reported run is not a pre-registered test.

## 4. Generation quality: to run

`python eval/run_ragas.py` runs the production chat pipeline on the test set and reports:

- **RAGAS** on answerable questions: faithfulness, answer relevancy, context precision, context
  recall. The judge is the verifier role, called through the project's own model layer; set
  `PRAMANA_VERIFIER` to a different model from the generator so the judge is not grading itself.
- **Grounding**: quote-lock pass rate, share of segments that are source-backed, outside or
  unverified, share of verified citations landing on a gold location, false refusals on answerable
  questions, and refusal precision and recall over the off-topic and uncovered sets (both layers
  together).

Per-question rows are written to `eval/results/ragas_rows.json` for hand-checking the judge.

## 5. Tests

`python -m pytest tests -q`: 24 tests. `test_learner.py` pins the learner-model properties,
`test_checks.py` the quote-lock, sandbox, grading and parsing helpers, and `test_pipeline.py` runs
chat, the verification gate, a quiz and its report end to end with scripted model responses.
