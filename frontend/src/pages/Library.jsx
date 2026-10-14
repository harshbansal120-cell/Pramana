import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'

const KIND_LABEL = { video: 'Lecture video', slides: 'Slide deck', textbook: 'Textbook' }

function ModelStatus({ status }) {
  if (!status) return <p className="note warn">The API is not reachable.</p>
  const missing = Object.entries(status.models).filter(([, m]) => !m.ready)
  return (
    <div className="card">
      <h3>Models</h3>
      <div className="grid three">
        {Object.entries(status.models).map(([role, m]) => (
          <div key={role} className="kv">
            <small>{role}</small>
            <b>{m.spec.split(':')[1]}</b>
            <span className={`dot ${m.ready ? 'ok' : 'bad'}`}>{m.ready ? 'key found' : 'no key'}</span>
          </div>
        ))}
      </div>
      {missing.length > 0 && (
        <p className="note warn">Add the missing key to <code>.env</code> and restart. Until then parsing, search and
          slide-to-video alignment work, but figure reading, the concept map, chat and question generation do not.</p>
      )}
    </div>
  )
}

export default function Library({ status, openSource, refreshStatus }) {
  const [sources, setSources] = useState([])
  const [open, setOpen] = useState(null)
  const [units, setUnits] = useState([])
  const [job, setJob] = useState(null)
  const [form, setForm] = useState({ kind: 'slides', title: '', youtube_url: '' })
  const [error, setError] = useState('')
  const fileRef = useRef()

  const load = () => api.get('/api/sources').then(setSources)
  useEffect(() => { load() }, [])

  useEffect(() => {
    if (!job || job.status !== 'running') return
    const t = setInterval(async () => {
      const j = await api.get(`/api/jobs/${job.id}`)
      setJob(j)
      if (j.status !== 'running') { load(); refreshStatus() }
    }, 1500)
    return () => clearInterval(t)
  }, [job, refreshStatus])

  const start = async (fn) => {
    setError('')
    try {
      const { job_id } = await fn()
      setJob({ id: job_id, status: 'running', log: [] })
    } catch (e) { setError(e.message) }
  }

  const upload = () => start(() => {
    const fd = new FormData()
    fd.append('kind', form.kind)
    fd.append('title', form.title)
    fd.append('youtube_url', form.youtube_url)
    if (fileRef.current?.files[0]) fd.append('file', fileRef.current.files[0])
    return api.upload('/api/ingest/upload', fd)
  })

  const toggle = async (s) => {
    if (open === s.id) return setOpen(null)
    setOpen(s.id)
    setUnits(await api.get(`/api/sources/${s.id}/units`))
  }

  const kb = status?.kb
  return (
    <div className="page">
      <h1>Library</h1>
      <p className="lead">Everything Pramana knows comes from the sources below. Each one is split into units that
        keep their page, slide number or video timestamp.</p>

      <ModelStatus status={status} />

      {kb && (
        <div className="grid four">
          <div className="stat"><b>{kb.sources}</b><small>sources</small></div>
          <div className="stat"><b>{Object.values(kb.units).reduce((a, b) => a + b, 0)}</b>
            <small>citable units ({kb.units.figure || 0} from figures)</small></div>
          <div className="stat"><b>{kb.concepts}</b><small>concepts, {kb.prerequisite_edges} prerequisite links</small></div>
          <div className="stat"><b>{kb.slide_video_links}</b><small>slides linked to a video timestamp</small></div>
        </div>
      )}

      <div className="card">
        <h3>Add material</h3>
        <div className="row wrap">
          <select value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value })}>
            <option value="slides">Slide deck (PDF)</option>
            <option value="textbook">Textbook or notes (PDF)</option>
            <option value="video">Lecture video</option>
          </select>
          <input placeholder="Title (optional)" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
          {form.kind === 'video' && (
            <input className="grow" placeholder="YouTube URL" value={form.youtube_url}
              onChange={(e) => setForm({ ...form, youtube_url: e.target.value })} />
          )}
          <input type="file" ref={fileRef} accept={form.kind === 'video' ? 'video/*,audio/*' : 'application/pdf'} />
          <button onClick={upload} disabled={job?.status === 'running'}>Ingest</button>
        </div>
        <div className="row wrap">
          <button className="ghost" onClick={() => start(() => api.post('/api/ingest/course'))} disabled={job?.status === 'running'}>
            Load the default course (MIT 6.041SC + Grinstead and Snell)
          </button>
          <button className="ghost" onClick={() => start(() => api.post('/api/ingest/rebuild-graph'))} disabled={job?.status === 'running'}>
            Rebuild concept map
          </button>
        </div>
        {error && <p className="note bad">{error}</p>}
        {job && (
          <div className={`joblog ${job.status}`}>
            <b>Ingestion {job.status}</b>
            <pre>{(job.log || []).slice(-12).join('\n') || 'starting...'}</pre>
          </div>
        )}
      </div>

      <div className="card">
        <h3>Sources</h3>
        {sources.length === 0 && <p className="muted">Nothing ingested yet.</p>}
        {sources.map((s) => (
          <div key={s.id} className="source">
            <button className="source-head" onClick={() => toggle(s)}>
              <span className={`tag kind-${s.kind}`}>{KIND_LABEL[s.kind]}</span>
              <b>{s.title}</b>
              <small>{s.units} units{s.figures ? `, ${s.figures} figures` : ''}{s.grp ? ` · paired as ${s.grp}` : ''}</small>
            </button>
            {open === s.id && (
              <div className="unit-list">
                {units.map((u) => (
                  <button key={u.id} onClick={() => api.get(`/api/units/${u.id}`).then((full) => openSource(full, null))}>
                    <small>{u.label.split(' of ')[0].split(' in ')[0]}</small>
                    <span>{u.text.slice(0, 150)}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
