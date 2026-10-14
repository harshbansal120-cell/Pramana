import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

const fmt = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v == null ? '–' : String(v))

function Table({ rows, columns }) {
  if (!rows?.length) return null
  const cols = columns || Object.keys(rows[0])
  return (
    <table>
      <thead><tr>{cols.map((c) => <th key={c}>{c.replace(/_/g, ' ')}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c}>{fmt(r[c])}</td>)}</tr>)}</tbody>
    </table>
  )
}

// Line chart of one metric by session for several policies.
function Lines({ series, label }) {
  const names = Object.keys(series)
  if (!names.length) return null
  const all = names.flatMap((n) => series[n])
  const lo = Math.min(...all), hi = Math.max(...all), w = 520, h = 170, n = series[names[0]].length
  const x = (i) => 40 + (i / Math.max(1, n - 1)) * (w - 56)
  const y = (v) => h - 24 - ((v - lo) / (hi - lo || 1)) * (h - 44)
  const colors = ['#3f3a9c', '#c2410c', '#0f766e', '#a16207']
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
        <line x1={40} y1={h - 24} x2={w - 16} y2={h - 24} className="axis" />
        <text x={36} y={y(hi) + 4} textAnchor="end">{hi.toFixed(2)}</text>
        <text x={36} y={y(lo) + 4} textAnchor="end">{lo.toFixed(2)}</text>
        {Array.from({ length: n }, (_, i) => <text key={i} x={x(i)} y={h - 8} textAnchor="middle">{i}</text>)}
        {names.map((name, k) => (
          <g key={name}>
            <polyline fill="none" stroke={colors[k % 4]} strokeWidth={2} points={series[name].map((v, i) => `${x(i)},${y(v)}`).join(' ')} />
            <text x={x(n - 1) - 4} y={y(series[name].at(-1)) - 6} textAnchor="end" fill={colors[k % 4]}>{name}</text>
          </g>
        ))}
      </svg>
      <figcaption>{label} (x axis: session)</figcaption>
    </figure>
  )
}

export default function Evaluation() {
  const [res, setRes] = useState(null)
  useEffect(() => { api.get('/api/eval').then(setRes) }, [])
  if (!res) return <p className="muted page">Loading…</p>
  const { retrieval, ragas, simulation } = res
  return (
    <div className="page">
      <h1>Evaluation</h1>
      <p className="lead">Results are read from <code>eval/results</code>. Each section says what was run and on what
        data. Sections that have not been run say so.</p>

      <div className="card">
        <h3>Retrieval and refusal <small className="muted">no LLM involved</small></h3>
        {!retrieval ? <p className="muted">Not run yet: <code>python eval/run_retrieval_eval.py</code></p> : (
          <>
            <p className="muted">{retrieval.description}</p>
            <Table rows={retrieval.ablation} />
            <h4>By where the answer lives</h4>
            <Table rows={retrieval.by_gold_modality} />
            <h4>Refusal gate, fitted threshold {retrieval.refusal.threshold}, scored on the held-out half</h4>
            <Table rows={[retrieval.refusal]} columns={['in_scope_answered', 'off_topic_declined', 'uncovered_in_domain_declined', 'precision', 'recall']} />
            <h4>Refusal gate across thresholds, all questions</h4>
            <Table rows={retrieval.refusal_threshold_sweep} />
            {retrieval.caveats?.map((c) => <p key={c} className="note warn">{c}</p>)}
          </>
        )}
      </div>

      <div className="card">
        <h3>Generation quality <small className="muted">RAGAS</small></h3>
        {!ragas ? <p className="muted">Not run yet. Needs model keys: <code>python eval/run_ragas.py</code></p> : (
          <>
            <p className="muted">{ragas.description}</p>
            <Table rows={[ragas.metrics]} />
            {ragas.grounding && <Table rows={[ragas.grounding]} />}
            {ragas.caveats?.map((c) => <p key={c} className="note warn">{c}</p>)}
          </>
        )}
      </div>

      <div className="card">
        <h3>Personalisation <small className="muted">simulated students</small></h3>
        {!simulation ? <p className="muted">Not run yet: <code>python eval/simulate.py</code></p> : (
          <>
            <p className="muted">{simulation.description}</p>
            <div className="grid two">
              <Lines series={simulation.true_mastery_by_session} label="True mastery (fraction of concepts known)" />
              <Lines series={simulation.estimator_rmse_by_session} label="Mastery estimate error (RMSE vs hidden truth)" />
            </div>
            <h4>By question-selection policy</h4>
            <Table rows={simulation.policies} />
            <h4>Mastery gain by student profile</h4>
            <Table rows={simulation.mastery_gain_by_profile} />
            <h4>Learner-model ablation, scored on identical answer logs</h4>
            <Table rows={simulation.estimator_ablation} />
            {simulation.weekly_sessions && (
              <>
                <h4>Same study with weekly sessions</h4>
                <p className="muted">{simulation.weekly_sessions.description}</p>
                <Table rows={simulation.weekly_sessions.policies}
                  columns={['policy', 'mastery_start', 'mastery_end', 'mastery_gain', 'pramana_minus_this_gain_95ci', 'final_estimate_rmse']} />
              </>
            )}
            {simulation.findings?.map((c) => <p key={c} className="note info">{c}</p>)}
            {simulation.caveats?.map((c) => <p key={c} className="note warn">{c}</p>)}
          </>
        )}
      </div>
    </div>
  )
}
