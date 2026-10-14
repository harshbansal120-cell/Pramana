import React, { useEffect, useMemo, useState } from 'react'
import { api, masteryColor, pct, speak } from '../api.js'

const NODE_W = 168, NODE_H = 44, COL_GAP = 64, ROW_GAP = 14

// Course flow map: columns are prerequisite depth, colour is estimated mastery.
function ConceptMap({ graph, selected, onSelect }) {
  const layout = useMemo(() => {
    const cols = {}
    graph.nodes.forEach((n) => (cols[n.depth] ||= []).push(n))
    const pos = {}
    let maxRows = 0
    Object.entries(cols).forEach(([d, ns]) => {
      maxRows = Math.max(maxRows, ns.length)
      ns.forEach((n, i) => { pos[n.id] = { x: 16 + d * (NODE_W + COL_GAP), y: 16 + i * (NODE_H + ROW_GAP) } })
    })
    return { pos, width: 32 + Object.keys(cols).length * (NODE_W + COL_GAP) - COL_GAP, height: 32 + maxRows * (NODE_H + ROW_GAP) }
  }, [graph])
  const related = useMemo(() => {
    if (!selected) return null
    const s = new Set([selected])
    graph.edges.forEach((e) => { if (e.from === selected) s.add(e.to); if (e.to === selected) s.add(e.from) })
    return s
  }, [graph, selected])

  return (
    <div className="map-scroll">
      <svg width={layout.width} height={layout.height} role="img" aria-label="Concept map with prerequisites">
        {graph.edges.map((e) => {
          const a = layout.pos[e.from], b = layout.pos[e.to]
          if (!a || !b) return null
          const x1 = a.x + NODE_W, y1 = a.y + NODE_H / 2, x2 = b.x, y2 = b.y + NODE_H / 2, mx = (x1 + x2) / 2
          const hot = selected && (e.from === selected || e.to === selected)
          return <path key={`${e.from}-${e.to}`} d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
            className={`edge ${hot ? 'hot' : ''} ${selected && !hot ? 'dim' : ''}`} />
        })}
        {graph.nodes.map((n) => {
          const p = layout.pos[n.id]
          const words = n.name.split(' ')
          const mid = n.name.length > 22 ? Math.ceil(words.length / 2) : words.length
          return (
            <g key={n.id} transform={`translate(${p.x},${p.y})`} onClick={() => onSelect(n.id)}
              className={`node ${selected === n.id ? 'sel' : ''} ${related && !related.has(n.id) ? 'dim' : ''}`}>
              <title>{`${n.name}: mastery ${n.mastery != null ? pct(n.mastery) : 'unknown'}`}</title>
              <rect width={NODE_W} height={NODE_H} rx={8} fill={masteryColor(n.attempts ? n.mastery : null)} />
              <text x={NODE_W / 2} y={mid < words.length ? 18 : 27} textAnchor="middle">
                {words.slice(0, mid).join(' ')}
                {mid < words.length && <tspan x={NODE_W / 2} dy={15}>{words.slice(mid).join(' ').slice(0, 26)}</tspan>}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function Trend({ points }) {
  if (points.length < 2) return <p className="muted">Mastery over time appears after your first few answers.</p>
  const w = 520, h = 90
  const xs = (i) => 6 + (i / (points.length - 1)) * (w - 12)
  const ys = (v) => h - 8 - v * (h - 16)
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="trend" role="img" aria-label="Mean mastery after each answer">
      <polyline fill="none" points={points.map((p, i) => `${xs(i)},${ys(p.mean)}`).join(' ')} />
      <circle cx={xs(points.length - 1)} cy={ys(points.at(-1).mean)} r={3.5} />
      <text x={6} y={12}>{pct(points[0].mean)}</text>
      <text x={w - 6} y={12} textAnchor="end">{pct(points.at(-1).mean)}</text>
    </svg>
  )
}

function Flashcards({ student, openSource }) {
  const [cards, setCards] = useState(null)
  const [flipped, setFlipped] = useState({})
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const load = async () => {
    setBusy(true); setError('')
    try { setCards(await api.get(`/api/students/${student.id}/flashcards`)) } catch (e) { setError(e.message) }
    setBusy(false)
  }
  useEffect(() => { setCards(null) }, [student.id])
  return (
    <div className="card">
      <h3>Flashcards for your weak concepts</h3>
      {!cards && <button className="ghost" onClick={load} disabled={busy}>{busy ? 'Writing cards…' : 'Generate from my sources'}</button>}
      {error && <p className="note bad">{error}</p>}
      {cards?.length === 0 && <p className="muted">No weak concepts with usable source text.</p>}
      <div className="cards">
        {cards?.map((c, i) => (
          <div key={i} className={`flash ${flipped[i] ? 'back' : ''}`} onClick={() => setFlipped({ ...flipped, [i]: !flipped[i] })}>
            <small>{c.concept} · {flipped[i] ? 'answer' : 'question'}</small>
            <p>{flipped[i] ? c.back : c.front}</p>
            {flipped[i] && (
              <div className="row">
                <button className="cite" onClick={(e) => { e.stopPropagation(); openSource(c.citation.unit, c.citation.quote) }}>
                  {c.citation.unit.label}</button>
                <button className="ghost small" onClick={(e) => { e.stopPropagation(); speak(`${c.front}. ${c.back}`, student.language) }}>Listen</button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Dashboard({ student, openSource, startPractice }) {
  const [dash, setDash] = useState(null)
  const [graph, setGraph] = useState(null)
  const [plan, setPlan] = useState(null)
  const [sel, setSel] = useState(null)
  const [units, setUnits] = useState([])

  useEffect(() => {
    setSel(null)
    api.get(`/api/students/${student.id}/dashboard`).then(setDash)
    api.get(`/api/graph?student_id=${student.id}`).then(setGraph)
    api.get(`/api/students/${student.id}/schedule`).then(setPlan)
  }, [student.id])
  useEffect(() => { if (sel) api.get(`/api/concepts/${sel}/units`).then(setUnits) }, [sel])

  if (!dash || !graph) return <p className="muted page">Loading…</p>
  if (graph.nodes.length === 0) {
    return <div className="page"><h1>Dashboard</h1><p className="note warn">There is no concept map yet. Ingest a
      course in the Library with a generator key configured.</p></div>
  }
  const node = graph.nodes.find((n) => n.id === sel)
  return (
    <div className="page">
      <h1>{student.name}'s dashboard</h1>
      {!student.diagnosed && dash.events === 0 && (
        <div className="note info row">
          <span className="grow">These estimates are only your self-reported starting level. A short adaptive
            diagnostic replaces them with measured mastery.</span>
          <button onClick={() => startPractice({ mode: 'diagnostic' })}>Take the diagnostic</button>
        </div>
      )}
      <div className="grid four">
        <div className="stat"><b>{pct(dash.mean_mastery)}</b><small>mean mastery over {dash.concepts} concepts</small></div>
        <div className="stat"><b>{dash.mastered}</b><small>concepts mastered (85%+)</small></div>
        <div className="stat"><b>{dash.weak.length}</b><small>weak concepts</small></div>
        <div className="stat"><b>{dash.events}</b><small>pieces of evidence from quizzes and chat</small></div>
      </div>

      <div className="card">
        <h3>Course flow map <small className="muted">prerequisites run left to right · grey = not yet tested</small></h3>
        <div className="legend">
          {[['< 40%', 0.2], ['40–60%', 0.5], ['60–85%', 0.7], ['85%+', 0.9]].map(([l, p]) => (
            <span key={l}><i style={{ background: masteryColor(p) }} />{l}</span>
          ))}
        </div>
        <ConceptMap graph={graph} selected={sel} onSelect={setSel} />
        {node && (
          <div className="node-panel">
            <div className="row wrap">
              <div className="grow">
                <b>{node.name}</b> <small className="muted">{node.topic} · {node.subtopic}</small>
                <p>{node.definition}</p>
              </div>
              <button onClick={() => startPractice({ mode: 'quiz', concept_ids: [node.id] })}>Quiz me on this</button>
            </div>
            <div className="row wrap">
              <span className="tag">mastery {pct(node.mastery)}</span>
              <span className="tag">{node.attempts} answers</span>
              <span className="tag">memory half-fades in ~{Math.round(node.stability_days * 0.69)} days</span>
              <span className="tag">prerequisites ready {pct(node.readiness)}</span>
            </div>
            <small>Where it is taught:</small>
            <div className="row wrap">
              {units.map((u) => <button key={u.id} className="cite" onClick={() => openSource(u, null)}>{u.label}</button>)}
            </div>
          </div>
        )}
      </div>

      <div className="grid two">
        <div className="card">
          <h3>Focus next</h3>
          {dash.root_causes.map((c) => (
            <p key={c.concept} className="note info">Trouble with <b>{c.concept}</b> most likely comes from{' '}
              <b>{c.root_cause}</b> ({pct(c.p_root)}). Start there.</p>
          ))}
          {dash.weak.map((w) => (
            <div key={w.concept_id} className="row weak-row">
              <span className="grow">{w.name}</span>
              <div className="bar small"><i style={{ width: pct(w.mastery), background: masteryColor(w.mastery) }} /></div>
              <small>{pct(w.mastery)}</small>
              <button className="ghost small" onClick={() => startPractice({ mode: 'quiz', concept_ids: [w.concept_id] })}>Practise</button>
            </div>
          ))}
          {dash.weak.length === 0 && <p className="muted">Nothing below 60%.</p>}
        </div>
        <div className="card">
          <h3>Mean mastery after each answer</h3>
          <Trend points={dash.trend} />
          <h3>Misconceptions seen</h3>
          {dash.misconceptions.length === 0 && <p className="muted">None yet.</p>}
          {dash.misconceptions.map((m) => <p key={m.misconception}><b>{m.n}×</b> {m.misconception}</p>)}
        </div>
      </div>

      {plan && (
        <div className="card">
          <h3>Study schedule <small className="muted">
            {plan.exam_date ? `${plan.days_to_exam} days to your exam` : 'no exam date set, planning two weeks'} ·
            {' '}{plan.minutes_per_day} min a day</small></h3>
          <p className="muted">Each day takes the concepts you are predicted to remember least on exam day, prerequisites first.
            Learned concepts return for review just before they would fade.</p>
          <div className="schedule">
            {plan.plan.slice(0, 14).map((d) => (
              <div key={d.day} className="day">
                <small>{new Date(d.date).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })}</small>
                {d.blocks.map((b, i) => (
                  <span key={i} className={`block ${b.kind}`} title={`${b.kind}, predicted recall ${pct(b.recall_now)}`}>{b.name}</span>
                ))}
                {d.blocks.length === 0 && <span className="muted">rest</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      <Flashcards student={student} openSource={openSource} />
    </div>
  )
}
