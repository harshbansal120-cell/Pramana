# Demo video shot list (target 7 minutes)

Before recording: add keys to `.env`, run `ingest`, run `bank --per-concept 2`, run all three
evaluation scripts, and create no students yet. Use a 1440 px wide window so the source viewer has
room.

| Time | Screen | Do and say |
|---|---|---|
| 0:00 | Title | "Chatbots answer from anywhere. Pramana answers from your course, and proves it." One sentence on the three checks: quote-lock, verification gate, learner model. |
| 0:30 | Library | Show the 17 sources and the counts. Expand *Lecture 2 slides*: one PDF page held four slides and the parser split them. Click slide 7; in the viewer click "the same moment" to jump to the video timestamp. |
| 1:15 | Library | Upload a new YouTube lecture or PDF. Let the ingestion log scroll: units, figures read, alignment, tagging. |
| 1:45 | Tutor | New student: answer the three intake questions (pick Hinglish to show it later, or English first). |
| 2:00 | Tutor | Ask "What is the total probability theorem?" Click a citation: the slide opens with the quote highlighted. Click a second citation that lands on a textbook page or a timestamp. |
| 2:40 | Tutor | Ask about a diagram ("What does the tree for three coin tosses show?"). Open the figure citation and point at the "AI reading of the figure" label. |
| 3:00 | Tutor | Ask "How does a transformer neural network work?" It declines and shows the score against the threshold. Click "Answer from general knowledge anyway": the answer appears in the outside lane. Then ask "State Chebyshev's inequality" for an in-domain decline. |
| 3:30 | Tutor | Answer the check question. Point at the mastery chip moving. |
| 3:45 | Practice | Start the diagnostic. Answer three or four. Say that each answer picks the next question, and that a correct answer deep in the graph vouches for its prerequisites. |
| 4:20 | Practice | Start a quiz on *Conditioning* with all three question types. Answer one wrongly by choosing a distractor: the feedback names the misconception, cites the source, and lists the checks the question passed. |
| 5:00 | Practice | Finish. In the report show mastery before and after, the root-cause line, and "Study" opening the right slide. |
| 5:30 | Dashboard | Course flow map coloured by mastery; click a node. Show the schedule and flip a flashcard; press Listen. Switch the student to Hindi and ask one question to show quotes staying in English. |
| 6:10 | Evaluation | Retrieval table, refusal table, RAGAS numbers, simulation charts. Say one thing that did not work (see `docs/EVALUATION.md`, "What did not work"). |
| 6:40 | Diagram | Architecture figure from `docs/ARCHITECTURE.md`. Close on the rule: nothing reaches the student unchecked. |
