import React from 'react'

// Highlights the cited quote inside the unit's text. Matching ignores case and whitespace
// runs, mirroring the server-side quote check closely enough for display.
function Highlighted({ text, quote }) {
  if (!quote) return <>{text}</>
  const words = quote.trim().split(/\s+/).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  let match = null
  try { match = new RegExp(words.join('[\\s\\S]{0,3}?'), 'i').exec(text) } catch { /* quote too odd for a regex */ }
  if (!match) return <>{text}</>
  return (
    <>
      {text.slice(0, match.index)}
      <mark>{match[0]}</mark>
      {text.slice(match.index + match[0].length)}
    </>
  )
}

const KIND = { transcript: 'Lecture video', slide: 'Slide', text: 'Textbook', figure: 'Figure' }

export default function SourceViewer({ unit, quote, openSource, onClose }) {
  const o = unit.open
  return (
    <aside className="viewer">
      <header>
        <div>
          <span className={`tag kind-${unit.kind}`}>{KIND[unit.kind] || unit.kind}</span>
          <h3>{unit.label}</h3>
        </div>
        <button className="ghost small" onClick={onClose} aria-label="Close source viewer">Close</button>
      </header>

      <div className="viewer-stage">
        {o.type === 'youtube' && (
          <iframe key={`${o.video}-${o.start}`} title="Lecture video" allowFullScreen
            allow="accelerometer; autoplay; encrypted-media; picture-in-picture"
            src={`https://www.youtube.com/embed/${o.video}?start=${o.start}&autoplay=1`} />
        )}
        {o.type === 'image' && <img src={o.url} alt={unit.label} />}
        {o.type === 'pdf' && <iframe key={`${o.url}-${o.page}`} title="Textbook page" src={`${o.url}#page=${o.page}&view=FitH`} />}
      </div>

      {unit.derived && (
        <p className="note warn">This text is an AI reading of the figure above, not the author's words. Check it
          against the image.</p>
      )}
      <h4>{quote ? 'Cited passage' : 'Text of this unit'}</h4>
      <div className="unit-text"><Highlighted text={unit.text} quote={quote} /></div>

      {unit.linked?.length > 0 && (
        <>
          <h4>Same moment in other sources</h4>
          {unit.linked.map((l) => (
            <button key={l.id} className="linked" onClick={() => openSource({ ...l, linked: [] }, null)}>
              <span className={`tag kind-${l.kind}`}>{KIND[l.kind]}</span> {l.label}
            </button>
          ))}
        </>
      )}
    </aside>
  )
}
