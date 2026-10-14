import React, { useEffect, useRef, useState } from 'react'
import { api, speak } from '../api.js'

const SUGGESTIONS = [
  'What is the total probability theorem?',
  'Explain Bayes\' rule with the radar example from the lecture',
  'What does the tree diagram for the two die rolls show?',
  'How does a transformer neural network work?',
]

function Citation({ n, c, openSource }) {
  return (
    <button className="cite" title={`"${c.quote}"`} onClick={() => openSource(c.unit, c.quote)}>
      {n} · {c.unit.label.split(' of ')[0].split(' in ')[0]}
    </button>
  )
}

function Assistant({ m, openSource, onOutside, lang }) {
  let n = 0
  const plain = (m.segments || []).map((s) => s.text).join(' ')
  if (m.status === 'declined') {
    return (
      <div className="bubble assistant declined">
        <b>Not in your course material</b>
        <p>{m.reason} Closest match scored {m.retrieval.top_score} against a threshold of {m.retrieval.threshold}.</p>
        {m.nearest?.length > 0 && (
          <>
            <small>Nearest things your material does cover:</small>
            {m.nearest.map((u) => (
              <button key={u.id} className="linked" onClick={() => openSource(u, null)}>{u.label}</button>
            ))}
          </>
        )}
        <button className="ghost small" onClick={onOutside}>Answer from general knowledge anyway</button>
      </div>
    )
  }
  return (
    <div className="bubble assistant">
      {m.coverage === 'partial' && <p className="note warn">Your material only partly covers this.</p>}
      {m.segments.map((s, i) => (
        <div key={i} className={`segment lane-${s.lane}`}>
          {s.lane === 'outside' && <span className="lane-label">Outside your material · not source-backed</span>}
          {s.lane === 'unverified' && <span className="lane-label">Unverified · no quote matched the cited source</span>}
          <p>{s.text}</p>
          {s.citations.length > 0 && (
            <div className="cites">
              {s.citations.map((c, j) => <Citation key={j} n={++n} c={c} openSource={openSource} />)}
            </div>
          )}
        </div>
      ))}
      {m.check_question && (
        <div className="check"><small>Check yourself</small><p>{m.check_question.question}</p></div>
      )}
      <div className="bubble-foot">
        {m.citation_check && m.citation_check.claimed > 0 && (
          <small>{m.citation_check.verified}/{m.citation_check.claimed} quotes verified against the source</small>
        )}
        {m.mastery_updates?.map((u) => (
          <span key={u.concept_id} className={`chip ${u.after >= u.before ? 'up' : 'down'}`}>
            {u.name} {Math.round(u.before * 100)} → {Math.round(u.after * 100)}
          </span>
        ))}
        <button className="ghost small" onClick={() => speak(plain, lang)}>Listen</button>
      </div>
    </div>
  )
}

export default function Tutor({ student, openSource }) {
  const [messages, setMessages] = useState([])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [listening, setListening] = useState(false)
  const endRef = useRef()

  useEffect(() => {
    setMessages([])
    api.get(`/api/students/${student.id}/chat`).then(setMessages)
  }, [student.id])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])

  const send = async (message, allowOutside = false) => {
    if (!message.trim() || busy) return
    setError('')
    setBusy(true)
    setText('')
    setMessages((ms) => [...ms, { role: 'user', text: message }])
    try {
      const res = await api.post('/api/chat', { student_id: student.id, message, allow_outside: allowOutside })
      setMessages((ms) => [...ms, { role: 'assistant', ...res }])
    } catch (e) { setError(e.message) }
    setBusy(false)
  }

  // Voice input through the browser's speech recognition, where available.
  const dictate = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) return setError('This browser has no speech recognition. Try Chrome.')
    const rec = new SR()
    rec.lang = student.language === 'en' ? 'en-IN' : 'hi-IN'
    rec.onresult = (e) => setText(e.results[0][0].transcript)
    rec.onend = () => setListening(false)
    setListening(true)
    rec.start()
  }

  const lastUser = [...messages].reverse().find((m) => m.role === 'user')
  return (
    <div className="page chat">
      <h1>Tutor</h1>
      <p className="lead">Answers come from your sources. Every source-backed statement carries a quote that was
        checked against the original; click it to open the exact page, slide or timestamp.</p>
      <div className="thread">
        {messages.length === 0 && (
          <div className="suggestions">
            {SUGGESTIONS.map((s) => <button key={s} className="ghost" onClick={() => send(s)}>{s}</button>)}
          </div>
        )}
        {messages.map((m, i) => m.role === 'user'
          ? <div key={i} className="bubble user">{m.text}</div>
          : <Assistant key={i} m={m} openSource={openSource} lang={student.language}
              onOutside={() => send(lastUser?.text || '', true)} />)}
        {busy && <div className="bubble assistant pending">Reading your sources…</div>}
        {error && <p className="note bad">{error}</p>}
        <div ref={endRef} />
      </div>
      <div className="composer">
        <textarea rows={2} value={text} placeholder="Ask about your course…" onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(text) } }} />
        <button className="ghost" onClick={dictate} disabled={listening}>{listening ? 'Listening…' : 'Speak'}</button>
        <button onClick={() => send(text)} disabled={busy || !text.trim()}>Send</button>
      </div>
    </div>
  )
}
