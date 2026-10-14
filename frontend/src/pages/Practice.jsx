import React, { useEffect, useMemo, useState } from 'react'
import { api, masteryColor, pct } from '../api.js'

const MODES = {
  quiz: ['Quiz', 'Feedback after every answer'],
  mock: ['Mock exam', 'Timed, feedback at the end'],
  diagnostic: ['Diagnostic', 'Adaptive: each answer picks the next question'],
}
const TYPES = { mcq: 'Multiple choice', numeric: 'Numerical', short: 'Short answer' }
const CHECKS = {
  structure: 'well-formed', quote_lock: 'source quote found', code_exact: 'answer recomputed by code',
  code_sim: 'Monte Carlo agrees', blind_solve: 'second model agrees', answerable: 'answerable and unambiguous', novel: 'not a duplicate',
}

function Builder({ student, intent, onStart }) {
  const [graph, setGraph] = useState(null)
  const [mode, setMode] = useState(intent?.mode || 'quiz')
  const [picked, setPicked] = useState(new Set(intent?.concept_ids || []))
  const [n, setN] = useState(intent?.mode === 'diagnostic' ? 8 : 6)
  const [types, setTypes] = useState(new Set(['mcq', 'numeric', 'short']))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => { api.get(`/api/graph?student_id=${student.id}`).then(setGraph) }, [student.id])
  const topics = useMemo(() => {
    const out = {}
    for (const c of graph?.nodes || []) (out[c.topic] ||= []).push(c)
    return out
  }, [graph])

  const toggle = (ids, on) => setPicked((p) => {
    const next = new Set(p)
    ids.forEach((id) => (on ? next.add(id) : next.delete(id)))
    return next
  })

  const start = async () => {
    setBusy(true)
    setError('')
    try {
      const a = await api.post('/api/assessments', {
        student_id: student.id, mode, concept_ids: mode === 'diagnostic' ? [] : [...picked], n, types: [...types],
      })
      onStart(a)
    } catch (e) { setError(e.message); setBusy(false) }
  }

  if (!graph) return <p className="muted">Loading…</p>
  if (graph.nodes.length === 0) {
    return <p className="note warn">There is no concept map yet. Ingest a course in the Library with a generator key configured.</p>
  }
  return (
    <>
      <div className="grid three">
        {Object.entries(MODES).map(([id, [label, hint]]) => (
          <button key={id} className={`choice ${mode === id ? 'active' : ''}`} onClick={() => setMode(id)}>
            <b>{label}</b><small>{hint}</small>
          </button>
        ))}
      </div>

      {mode !== 'diagnostic' && (
        <div className="card">
          <h3>Scope <small className="muted">{picked.size ? `${picked.size} concepts` : 'whole course'}</small></h3>
          {Object.entries(topics).map(([topic, cs]) => {
            const all = cs.every((c) => picked.has(c.id))
            return (
              <div key={topic} className="topic">
                <label className="check-row">
                  <input type="checkbox" checked={all} onChange={(e) => toggle(cs.map((c) => c.id), e.target.checked)} />
                  <b>{topic}</b>
                </label>
                <div className="concept-chips">
                  {cs.map((c) => (
                    <button key={c.id} className={`concept-chip ${picked.has(c.id) ? 'on' : ''}`}
                      onClick={() => toggle([c.id], !picked.has(c.id))}>
                      <i style={{ background: masteryColor(c.mastery) }} />{c.name}
                    </button>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      )}

      <div className="card row wrap">
        <label>Questions
          <input type="number" min={1} max={30} value={n} onChange={(e) => setN(+e.target.value)} style={{ width: 80 }} />
        </label>
        {mode !== 'diagnostic' && Object.entries(TYPES).map(([id, label]) => (
          <label key={id} className="check-row">
            <input type="checkbox" checked={types.has(id)} onChange={(e) => setTypes((t) => {
              const next = new Set(t); e.target.checked ? next.add(id) : next.delete(id); return next
            })} />{label}
          </label>
        ))}
        <div className="grow" />
        <button onClick={start} disabled={busy || (mode !== 'diagnostic' && types.size === 0)}>
          {busy ? 'Writing and verifying questions…' : `Start ${MODES[mode][0].toLowerCase()}`}
        </button>
      </div>
      {busy && <p className="muted">New questions are generated, solved independently by a second model and, for
        numerical ones, recomputed by code before you see them. This can take a minute the first time.</p>}
      {error && <p className="note bad">{error}</p>}
    </>
  )
}

function Feedback({ fb, openSource }) {
  const good = fb.score >= 1
  return (
    <div className={`feedback ${good ? 'good' : fb.score > 0 ? 'part' : 'bad'}`}>
      <b>{good ? 'Correct' : fb.score > 0 ? 'Partly correct' : 'Not quite'}</b>
      {!good && <p>Answer: {fb.correct_answer}</p>}
      {fb.note && <p>{fb.note}</p>}
      {fb.misconception && <p className="misc">Likely misconception: {fb.misconception}</p>}
      <p>{fb.solution}</p>
      {fb.citation?.unit && (
        <button className="cite" onClick={() => openSource(fb.citation.unit, fb.citation.quote)}>
          Source · {fb.citation.unit.label}
        </button>
      )}
      <div className="bubble-foot">
        {fb.mastery_updates?.map((u) => (
          <span key={u.concept_id} className={`chip ${u.after >= u.before ? 'up' : 'down'}`}>
            {u.name} {Math.round(u.before * 100)} → {Math.round(u.after * 100)}{u.direct ? '' : ' (prerequisite)'}
          </span>
        ))}
      </div>
      {fb.verification && (
        <small className="muted">Verified before serving: {Object.entries(fb.verification)
          .filter(([, v]) => v).map(([k]) => CHECKS[k] || k).join(' · ')}</small>
      )}
    </div>
  )
}

function Runner({ assessment, openSource, onFinish }) {
  const [questions, setQuestions] = useState(assessment.questions)
  const [i, setI] = useState(0)
  const [response, setResponse] = useState('')
  const [fb, setFb] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [left, setLeft] = useState(assessment.mode === 'mock' ? assessment.questions.length * 120 : null)
  const q = questions[i]
  const silent = assessment.mode !== 'quiz'
  const total = assessment.mode === 'diagnostic' ? assessment.total : questions.length

  useEffect(() => {
    if (left == null) return
    if (left <= 0) return onFinish()
    const t = setTimeout(() => setLeft(left - 1), 1000)
    return () => clearTimeout(t)
  }, [left, onFinish])

  const advance = (qs = questions) => {
    setFb(null)
    setResponse('')
    if (i + 1 < qs.length) setI(i + 1)
    else onFinish()
  }

  const submit = async (value = response) => {
    setBusy(true)
    setError('')
    try {
      const res = await api.post(`/api/assessments/${assessment.id}/answer`, { idx: q.idx, response: String(value) })
      const qs = res.next ? [...questions, res.next] : questions
      setQuestions(qs)
      if (silent) advance(qs)
      else setFb(res)
    } catch (e) { setError(e.message) }
    setBusy(false)
  }

  return (
    <div className="card question">
      <div className="row">
        <small>Question {i + 1} of {total}</small>
        <span className="tag">{q.concept.topic}</span>
        <span className="tag">{q.concept.name}</span>
        <span className="tag">difficulty {q.difficulty}/3</span>
        <div className="grow" />
        {left != null && <b>{Math.floor(left / 60)}:{String(left % 60).padStart(2, '0')}</b>}
      </div>
      <p className="stem">{q.stem}</p>
      {q.type === 'mcq' && q.options.map((o, k) => (
        <label key={k} className={`option ${String(response) === String(k) ? 'on' : ''}`}>
          <input type="radio" name="opt" disabled={!!fb} checked={String(response) === String(k)} onChange={() => setResponse(k)} />
          {o}
        </label>
      ))}
      {q.type === 'numeric' && (
        <input className="answer" disabled={!!fb} value={response} placeholder="A number, e.g. 0.375 or 3/8"
          onChange={(e) => setResponse(e.target.value)} />
      )}
      {q.type === 'short' && (
        <textarea className="answer" rows={3} disabled={!!fb} value={response} placeholder="One to three sentences"
          onChange={(e) => setResponse(e.target.value)} />
      )}
      {error && <p className="note bad">{error}</p>}
      {fb && <Feedback fb={fb} openSource={openSource} />}
      <div className="row end">
        {!fb && <button className="ghost" onClick={() => submit('')} disabled={busy}>Skip</button>}
        {!fb && <button onClick={() => submit()} disabled={busy || response === ''}>{busy ? 'Checking…' : 'Submit'}</button>}
        {fb && <button onClick={() => advance()}>{i + 1 < questions.length ? 'Next question' : 'See report'}</button>}
      </div>
    </div>
  )
}

function Report({ id, openSource, startPractice, onDone }) {
  const [r, setR] = useState(null)
  useEffect(() => { api.get(`/api/assessments/${id}/report`).then(setR) }, [id])
  if (!r) return <p className="muted">Building report…</p>
  return (
    <>
      <div className="grid four">
        <div className="stat"><b>{r.percent}%</b><small>{r.score} of {r.answered} answered</small></div>
        <div className="stat"><b>{r.weak_topics.length}</b><small>weak concepts</small></div>
        <div className="stat"><b>{r.misconceptions.length}</b><small>likely misconceptions</small></div>
        <div className="stat"><b>{r.root_causes.length}</b><small>traced to a prerequisite</small></div>
      </div>

      <div className="card">
        <h3>Mastery by concept</h3>
        <table>
          <thead><tr><th>Concept</th><th>Topic</th><th>Score</th><th>Mastery before → after</th></tr></thead>
          <tbody>
            {r.concepts.map((c) => (
              <tr key={c.concept_id}>
                <td>{c.name}</td><td className="muted">{c.topic}</td><td>{pct(c.accuracy)} of {c.questions}</td>
                <td>
                  <div className="bar"><i style={{ width: pct(c.mastery_after), background: masteryColor(c.mastery_after) }} />
                    {c.mastery_before != null && <u style={{ left: pct(c.mastery_before) }} />}</div>
                  <small>{c.mastery_before != null ? pct(c.mastery_before) : '–'} → {pct(c.mastery_after)}</small>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {(r.misconceptions.length > 0 || r.root_causes.length > 0) && (
        <div className="grid two">
          <div className="card">
            <h3>Likely misconceptions</h3>
            {r.misconceptions.length === 0 && <p className="muted">None detected.</p>}
            {r.misconceptions.map((m) => (
              <p key={m.misconception}><span className="tag">{m.strength}</span> <b>{m.count}×</b> {m.misconception}{' '}
                <small className="muted">({m.concepts.join(', ')})</small></p>
            ))}
          </div>
          <div className="card">
            <h3>Root causes</h3>
            {r.root_causes.length === 0 && <p className="muted">No weak prerequisite explains these results.</p>}
            {r.root_causes.map((c) => (
              <p key={c.concept}>Trouble with <b>{c.concept}</b> most likely comes from <b>{c.root_cause}</b> (mastery {pct(c.p_root)}).</p>
            ))}
          </div>
        </div>
      )}

      {r.next_steps.length > 0 && (
        <div className="card">
          <h3>What to do next</h3>
          {r.next_steps.map((s) => (
            <div key={s.concept_id} className="row wrap step">
              <div className="grow"><b>{s.concept}</b> <small className="muted">{s.why}</small></div>
              {s.study && <button className="cite" onClick={() => openSource(s.study, null)}>Study · {s.study.label}</button>}
              <button className="ghost small" onClick={() => { onDone(); startPractice({ mode: 'quiz', concept_ids: [s.concept_id] }) }}>Practise this</button>
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <h3>Review</h3>
        {r.items.map((it) => (
          <details key={it.question.idx} className={it.score >= 1 ? 'good' : 'bad'}>
            <summary>{it.score >= 1 ? '✓' : it.score > 0 ? '~' : '✗'} {it.question.stem}</summary>
            <p><small>Your answer:</small> {it.question.type === 'mcq' ? it.question.options[+it.response] ?? '(skipped)' : it.response || '(skipped)'}</p>
            <p><small>Answer:</small> {it.question.answer}</p>
            {it.misconception && <p className="misc">Likely misconception: {it.misconception}</p>}
            <p>{it.question.solution}</p>
            {it.question.source && (
              <button className="cite" onClick={() => openSource(it.question.source, it.question.source_quote)}>
                Source · {it.question.source.label}
              </button>
            )}
          </details>
        ))}
      </div>
      <button onClick={onDone}>New assessment</button>
    </>
  )
}

export default function Practice({ student, openSource, startPractice, intent, clearIntent }) {
  const [assessment, setAssessment] = useState(null)
  const [done, setDone] = useState(false)
  useEffect(() => { setAssessment(null); setDone(false) }, [student.id, intent])
  const reset = () => { setAssessment(null); setDone(false); clearIntent() }
  return (
    <div className="page">
      <h1>Practice</h1>
      <p className="lead">Questions are written from your sources, aimed at what you are weakest on and ready for, and
        never repeated. Each one passes a verification gate before you see it.</p>
      {!assessment && <Builder key={JSON.stringify(intent)} student={student} intent={intent} onStart={setAssessment} />}
      {assessment && !done && <Runner assessment={assessment} openSource={openSource} onFinish={() => setDone(true)} />}
      {assessment && done && <Report id={assessment.id} openSource={openSource} startPractice={startPractice} onDone={reset} />}
    </div>
  )
}
