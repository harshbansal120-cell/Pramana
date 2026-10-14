import React, { useCallback, useEffect, useState } from 'react'
import { api } from './api.js'
import SourceViewer from './SourceViewer.jsx'
import Library from './pages/Library.jsx'
import Tutor from './pages/Tutor.jsx'
import Practice from './pages/Practice.jsx'
import Dashboard from './pages/Dashboard.jsx'
import Evaluation from './pages/Evaluation.jsx'

const PAGES = [
  ['library', 'Library', 'Sources and ingestion'],
  ['tutor', 'Tutor', 'Source-grounded chat'],
  ['practice', 'Practice', 'Quizzes and mock exams'],
  ['dashboard', 'Dashboard', 'Mastery and study plan'],
  ['evaluation', 'Evaluation', 'Measured results'],
]

function Onboarding({ onDone, onCancel, existing }) {
  const [form, setForm] = useState(existing || { name: '', level: 'beginner', language: 'en', exam_date: '' })
  const [busy, setBusy] = useState(false)
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })
  const save = async () => {
    setBusy(true)
    const body = { ...form, exam_date: form.exam_date || null }
    const s = existing ? await api.patch(`/api/students/${existing.id}`, body) : await api.post('/api/students', body)
    onDone(s, !existing)
  }
  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2>{existing ? 'Edit profile' : 'Welcome to Pramana'}</h2>
        {!existing && <p className="muted">Three questions set your starting point. A short diagnostic quiz then
          replaces this guess with measured mastery.</p>}
        <label>Your name<input value={form.name} onChange={set('name')} placeholder="e.g. Asha" autoFocus /></label>
        <label>How well do you know this subject?
          <select value={form.level} onChange={set('level')}>
            <option value="beginner">New to it</option>
            <option value="intermediate">Seen it before, rusty</option>
            <option value="advanced">Comfortable, revising for an exam</option>
          </select>
        </label>
        <label>Explain to me in
          <select value={form.language} onChange={set('language')}>
            <option value="en">English</option>
            <option value="hinglish">Hinglish</option>
            <option value="hi">Hindi</option>
          </select>
        </label>
        <label>Exam date (optional)<input type="date" value={form.exam_date || ''} onChange={set('exam_date')} /></label>
        <div className="row end">
          {onCancel && <button className="ghost" onClick={onCancel}>Cancel</button>}
          <button disabled={!form.name.trim() || busy} onClick={save}>{existing ? 'Save' : 'Start'}</button>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState('library')
  const [students, setStudents] = useState([])
  const [student, setStudent] = useState(null)
  const [status, setStatus] = useState(null)
  const [viewer, setViewer] = useState(null)        // {unit, quote}
  const [modal, setModal] = useState(null)          // 'new' | 'edit'
  const [practiceIntent, setPracticeIntent] = useState(null)

  const refreshStatus = useCallback(() => api.get('/api/status').then(setStatus).catch(() => setStatus(null)), [])

  useEffect(() => {
    refreshStatus()
    api.get('/api/students').then((list) => {
      setStudents(list)
      const saved = localStorage.getItem('pramana.student')
      setStudent(list.find((s) => s.id === saved) || list[0] || null)
    })
  }, [refreshStatus])

  useEffect(() => { if (student) localStorage.setItem('pramana.student', student.id) }, [student])

  const openSource = useCallback((unit, quote) => setViewer({ unit, quote }), [])
  const startPractice = (intent) => { setPracticeIntent(intent); setPage('practice') }

  const onStudentSaved = (s, isNew) => {
    setStudents((list) => (isNew ? [...list, s] : list.map((x) => (x.id === s.id ? s : x))))
    setStudent(s)
    setModal(null)
    if (isNew && status?.kb.concepts) startPractice({ mode: 'diagnostic' })
  }

  const ctx = { student, status, openSource, startPractice, refreshStatus, setPage }
  const needsStudent = page !== 'library' && page !== 'evaluation' && !student

  return (
    <div className={`shell ${viewer ? 'with-viewer' : ''}`}>
      <nav className="sidebar">
        <div className="brand">
          <div className="brand-mark">प्र</div>
          <div><b>Pramana</b><small>proves every claim</small></div>
        </div>
        {PAGES.map(([id, label, hint]) => (
          <button key={id} className={`nav ${page === id ? 'active' : ''}`} onClick={() => setPage(id)}>
            <span>{label}</span><small>{hint}</small>
          </button>
        ))}
        <div className="spacer" />
        <div className="student-box">
          <small>Student</small>
          {students.length > 0 && (
            <select value={student?.id || ''} onChange={(e) => setStudent(students.find((s) => s.id === e.target.value))}>
              {students.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          )}
          <div className="row">
            <button className="ghost small" onClick={() => setModal('new')}>New student</button>
            {student && <button className="ghost small" onClick={() => setModal('edit')}>Edit</button>}
          </div>
        </div>
      </nav>

      <main className="main">
        {needsStudent ? (
          <div className="empty">
            <h2>Create a student profile first</h2>
            <p className="muted">Pramana tracks mastery per student.</p>
            <button onClick={() => setModal('new')}>New student</button>
          </div>
        ) : (
          <>
            {page === 'library' && <Library {...ctx} />}
            {page === 'tutor' && <Tutor {...ctx} />}
            {page === 'practice' && <Practice {...ctx} intent={practiceIntent} clearIntent={() => setPracticeIntent(null)} />}
            {page === 'dashboard' && <Dashboard {...ctx} />}
            {page === 'evaluation' && <Evaluation />}
          </>
        )}
      </main>

      {viewer && <SourceViewer {...viewer} openSource={openSource} onClose={() => setViewer(null)} />}
      {modal === 'new' && <Onboarding onDone={onStudentSaved} onCancel={students.length ? () => setModal(null) : null} />}
      {modal === 'edit' && <Onboarding existing={student} onDone={onStudentSaved} onCancel={() => setModal(null)} />}
    </div>
  )
}
