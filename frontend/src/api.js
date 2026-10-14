// Thin fetch wrapper. Errors carry the server's message so the UI can show it verbatim.
async function request(method, path, body, isForm) {
  const opts = { method, headers: {} }
  if (body && isForm) opts.body = body
  else if (body) {
    opts.headers['Content-Type'] = 'application/json'
    opts.body = JSON.stringify(body)
  }
  const res = await fetch(path, opts)
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail || detail } catch { /* not JSON */ }
    const err = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    err.status = res.status
    throw err
  }
  return res.json()
}

export const api = {
  get: (path) => request('GET', path),
  post: (path, body) => request('POST', path, body),
  patch: (path, body) => request('PATCH', path, body),
  upload: (path, form) => request('POST', path, form, true),
}

export const pct = (x) => `${Math.round((x ?? 0) * 100)}%`

// Mastery colour scale shared by the map, bars and chips.
export function masteryColor(p) {
  if (p == null) return '#d6d3d1'
  if (p < 0.4) return '#e5796b'
  if (p < 0.6) return '#eeb155'
  if (p < 0.85) return '#b5cf6b'
  return '#5fb37a'
}

export function speak(text, lang) {
  if (!('speechSynthesis' in window)) return false
  window.speechSynthesis.cancel()
  const u = new SpeechSynthesisUtterance(text)
  u.lang = lang === 'en' ? 'en-IN' : 'hi-IN'
  window.speechSynthesis.speak(u)
  return true
}
