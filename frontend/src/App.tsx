import { FormEvent, useEffect, useRef, useState } from 'react'

type SessionStatus = 'running' | 'paused' | 'stopped' | 'error'
type CaptureStatus = 'ok' | 'no_face' | 'camera_error' | null

type Snapshot = {
  session_id: string
  subject: string
  goal: string
  duration_seconds: number
  elapsed_seconds: number
  status: SessionStatus
  capture_status: CaptureStatus
  tentative_class: string | null
  error: string | null
  offer: Offer | null
  enabled_offers: string[]
}

type Offer = {
  offer_id: string
  cue_class: string
  guide_mode: 'explain' | 'steps' | 'engage' | 'break'
  message: string
  sample_count: number
  window_seconds: number
}

type Health = {
  service: string
  model_present: boolean
  key_configured: boolean
  tutor_ready: boolean
  session_status: string
  demo_mode: boolean
}

type GuideMode = 'explain' | 'steps' | 'engage'
type Preview = { prompt: string; digest: string }
type Guide = { request_id: string; mode: GuideMode; answer: string }

type SessionEvent = {
  session_id: string | null
  sequence: number
  timestamp: string
  type: 'snapshot'
  payload: Snapshot | null
}

const formatTime = (seconds: number) => {
  const whole = Math.floor(seconds)
  return `${String(Math.floor(whole / 60)).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}`
}

function App() {
  const [token, setToken] = useState('')
  const [health, setHealth] = useState<Health | null>(null)
  const [session, setSession] = useState<Snapshot | null>(null)
  const [connected, setConnected] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [subject, setSubject] = useState('')
  const [goal, setGoal] = useState('')
  const [duration, setDuration] = useState(25)
  const [monitoring, setMonitoring] = useState(true)
  const [guideMode, setGuideMode] = useState<GuideMode>('explain')
  const [topic, setTopic] = useState('')
  const [excerpt, setExcerpt] = useState('')
  const [preview, setPreview] = useState<Preview | null>(null)
  const [guide, setGuide] = useState<Guide | null>(null)
  const [helpBusy, setHelpBusy] = useState(false)
  const [helpError, setHelpError] = useState('')
  const [feedback, setFeedback] = useState('')
  const [acceptedOfferId, setAcceptedOfferId] = useState<string | null>(null)
  const sequence = useRef(-1)

  useEffect(() => {
    let cancelled = false
    async function restore() {
      try {
        const boot = await fetch('/bootstrap').then((response) => response.json())
        const status = await fetch('/health').then((response) => response.json())
        const current = await fetch('/sessions/current', {
          headers: { 'X-StudySync-Token': boot.token },
        }).then((response) => response.json())
        if (!cancelled) {
          setToken(boot.token)
          setHealth(status)
          setSession(current)
          if (current) {
            setSubject(current.subject)
            setGoal(current.goal)
          }
        }
      } catch {
        if (!cancelled) setError('The local service is unavailable. Start run_web.py and reload.')
      }
    }
    void restore()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!token || !session?.session_id) return
    let closed = false
    let retry: number | undefined
    let socket: WebSocket | undefined
    const sessionId = session.session_id

    function connect() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${window.location.host}/sessions/${sessionId}/events`)
      socket.onopen = () => socket?.send(JSON.stringify({ token }))
      socket.onmessage = (message) => {
        const event: SessionEvent = JSON.parse(message.data)
        if (event.type === 'snapshot' && event.payload && event.sequence >= sequence.current) {
          sequence.current = event.sequence
          setSession(event.payload)
          setConnected(true)
          setError('')
        }
      }
      socket.onclose = () => {
        setConnected(false)
        if (!closed) retry = window.setTimeout(connect, 1500)
      }
      socket.onerror = () => socket?.close()
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(retry)
      socket?.close()
      setConnected(false)
    }
  }, [token, session?.session_id])

  async function request<T>(path: string, body?: object): Promise<T> {
    const response = await fetch(path, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-StudySync-Token': token,
      },
      body: body ? JSON.stringify(body) : undefined,
    })
    if (!response.ok) {
      const failure = await response.json().catch(() => null)
      const detail = failure?.detail
      throw new Error(typeof detail === 'string' ? detail : detail?.message || `Request failed (${response.status})`)
    }
    return response.json()
  }

  async function start(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      sequence.current = -1
      setSession(await request<Snapshot>('/sessions', {
        subject,
        goal,
        duration_minutes: duration,
        monitoring,
      }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not start the session')
    } finally {
      setBusy(false)
    }
  }

  async function control(action: 'pause' | 'resume' | 'stop') {
    if (!session) return
    setBusy(true)
    setError('')
    try {
      setSession(await request<Snapshot>(`/sessions/${session.session_id}/${action}`))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : `Could not ${action} the session`)
    } finally {
      setBusy(false)
    }
  }

  function changeHelpText(field: 'mode' | 'topic' | 'excerpt', value: string) {
    setPreview(null)
    setHelpError('')
    if (field === 'mode') setGuideMode(value as GuideMode)
    if (field === 'topic') setTopic(value)
    if (field === 'excerpt') setExcerpt(value)
  }

  async function previewHelp(event: FormEvent) {
    event.preventDefault()
    if (!session) return
    setHelpBusy(true)
    setHelpError('')
    try {
      setPreview(await request<Preview>(`/sessions/${session.session_id}/help/preview`, {
        mode: guideMode, topic, excerpt,
      }))
    } catch (cause) {
      setHelpError(cause instanceof Error ? cause.message : 'Could not prepare the preview')
    } finally {
      setHelpBusy(false)
    }
  }

  async function sendHelp() {
    if (!session || !preview) return
    setHelpBusy(true)
    setHelpError('')
    setGuide(null)
    setFeedback('')
    try {
      const result = await request<Guide>(`/sessions/${session.session_id}/help`, {
        mode: guideMode, topic, excerpt,
        approved_digest: preview.digest,
        request_id: crypto.randomUUID().replaceAll('-', ''),
        offer_id: acceptedOfferId,
      })
      setGuide(result)
      setPreview(null)
      setAcceptedOfferId(null)
    } catch (cause) {
      setHelpError(cause instanceof Error ? cause.message : 'Could not get a guide')
    } finally {
      setHelpBusy(false)
    }
  }

  async function actOnOffer(action: 'accept' | 'dismiss' | 'snooze' | 'wrong_cue') {
    if (!session?.offer) return
    setError('')
    try {
      const result = await request<{ offer: Offer }>(
        `/sessions/${session.session_id}/offers/${session.offer.offer_id}/action`, { action },
      )
      setSession((current) => current ? { ...current, offer: null } : current)
      if (action === 'accept') {
        if (result.offer.guide_mode === 'break') {
          if (session.status === 'running') await control('pause')
        } else {
          setGuideMode(result.offer.guide_mode)
          setTopic((current) => current || session.goal)
          setPreview(null)
          setAcceptedOfferId(result.offer.offer_id)
          document.getElementById('guide-panel')?.scrollIntoView({ behavior: 'smooth' })
        }
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not update the offer')
    }
  }

  async function toggleOfferClass(cueClass: string) {
    if (!session) return
    const selected = new Set(session.enabled_offers)
    if (selected.has(cueClass)) selected.delete(cueClass)
    else selected.add(cueClass)
    try {
      setSession(await request<Snapshot>(`/sessions/${session.session_id}/offer-classes`, {
        enabled: [...selected],
      }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not update offer settings')
    }
  }

  async function sendFeedback(choice: string) {
    if (!session) return
    try {
      await request(`/sessions/${session.session_id}/feedback`, { choice })
      setFeedback('Thanks. Your feedback was counted on this computer.')
    } catch (cause) {
      setHelpError(cause instanceof Error ? cause.message : 'Could not record feedback')
    }
  }

  const active = session?.status === 'running' || session?.status === 'paused'
  const progress = session ? Math.min(100, session.elapsed_seconds / session.duration_seconds * 100) : 0

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">S</span><span>StudySync</span></div>
        <div className="top-status"><span className={connected ? 'dot live' : 'dot'} />
          {connected ? 'Local session connected' : 'Local study space'}</div>
      </header>

      <main>
        <section className="hero">
          <div className="eyebrow">MODEL-LED STUDY SUPPORT</div>
          <h1>Stay with the work.<br /><em>Get help when you choose.</em></h1>
          <p>The original six-class model looks for tentative cues during a session you start.
            Camera frames stay on this computer. You can pause or stop monitoring at any time.</p>
        </section>

        {error && <div className="alert" role="alert">{error}</div>}
        {health?.demo_mode && <div className="demo-alert" role="status">
          Simulation mode: cues come from synthetic data. No camera or saved model is being used.
        </div>}

        <div className="dashboard">
          <section className="panel setup-panel">
            <div className="panel-title"><span className="step">01</span><h2>Your study session</h2></div>
            <form onSubmit={start}>
              <label htmlFor="subject">Subject</label>
              <input id="subject" value={subject} onChange={(event) => setSubject(event.target.value)}
                placeholder="e.g. Calculus" maxLength={120} required disabled={active} />
              <label htmlFor="goal">What are you working on?</label>
              <textarea id="goal" value={goal} onChange={(event) => setGoal(event.target.value)}
                placeholder="e.g. Practice derivatives and the chain rule"
                maxLength={500} rows={3} required disabled={active} />
              <label htmlFor="duration">Session length</label>
              <div className="duration-field"><input id="duration" type="number" min="1" max="240"
                value={duration} onChange={(event) => setDuration(Number(event.target.value))}
                disabled={active} /><span>minutes</span></div>
              <label className="monitor-choice"><input type="checkbox" checked={monitoring}
                onChange={(event) => setMonitoring(event.target.checked)} disabled={active} />
                Use camera cues during this session</label>
              {!active && <button className="primary" type="submit" disabled={!token || busy || (monitoring && !health?.model_present && !health?.demo_mode)}>
                {busy ? 'Starting…' : monitoring ? 'Start monitoring' : 'Start without camera'}</button>}
            </form>
          </section>

          <section className="panel live-panel">
            <div className="panel-title"><span className="step">02</span><h2>Session status</h2></div>
            <div className="clock">{formatTime(session?.elapsed_seconds || 0)}
              <span> / {formatTime(session?.duration_seconds || duration * 60)}</span></div>
            <div className="progress" role="progressbar" aria-valuenow={Math.round(progress)}
              aria-valuemin={0} aria-valuemax={100}><div style={{ width: `${progress}%` }} /></div>
            <div className="status-grid">
              <div><span>Monitoring</span><strong>{session?.status || 'Not started'}</strong></div>
              <div><span>Camera</span><strong>{health?.demo_mode ? 'Simulation only' : session?.capture_status === 'ok' ? 'Face visible' :
                session?.capture_status === 'no_face' ? 'Face not visible' :
                session?.status === 'running' ? 'Starting…' : 'Off'}</strong></div>
              <div><span>{health?.demo_mode ? 'Synthetic cue' : 'Model cue'}</span><strong>{session?.tentative_class || 'No cue'}</strong></div>
              <div><span>Model file</span><strong>{health?.model_present ? 'Ready' : 'Missing'}</strong></div>
            </div>
            {session?.error && <p className="inline-error" role="alert">{session.error}</p>}
            <div className="controls">
              {session?.status === 'running' && <button type="button" onClick={() => control('pause')} disabled={busy}>Pause</button>}
              {session?.status === 'paused' && <button type="button" onClick={() => control('resume')} disabled={busy}>Resume</button>}
              {active && <button className="stop" type="button" onClick={() => control('stop')} disabled={busy}>Stop session</button>}
            </div>
            <p className="fine-print">The model output is a tentative cue, not a statement about how you feel.
              Switching tabs or apps does not pause a session you started. Closing this page pauses it after a short grace period.</p>
            {active && <fieldset className="offer-settings"><legend>Optional model offers</legend>
              {(['Confused', 'Frustrated', 'Bored', 'Drowsy'] as const).map((cueClass) =>
                <label key={cueClass}><input type="checkbox"
                  checked={session?.enabled_offers?.includes(cueClass) || false}
                  onChange={() => toggleOfferClass(cueClass)} />{cueClass}</label>)}
            </fieldset>}
          </section>
        </div>

        {session?.offer && active && <section className="offer-card" aria-live="polite">
          <span className="offer-eyebrow">OPTIONAL MODEL CUE</span>
          <h2>{session.offer.message}</h2>
          <p>The saved model has shown a sustained, tentative {session.offer.cue_class.toLowerCase()} cue.
            You can correct or ignore it. No study text is sent unless you preview and approve a guide.</p>
          <div className="offer-actions">
            <button type="button" className="primary" onClick={() => actOnOffer('accept')}>
              {session.offer.guide_mode === 'break' ? 'Pause for a break' : 'Prepare a guide'}</button>
            <button type="button" onClick={() => actOnOffer('dismiss')}>Dismiss</button>
            <button type="button" onClick={() => actOnOffer('snooze')}>Snooze 5 min</button>
            <button type="button" onClick={() => actOnOffer('wrong_cue')}>This cue is wrong</button>
          </div>
        </section>}

        <section className="panel guide-panel" id="guide-panel">
          <div className="panel-title"><span className="step">03</span><h2>Ask for a study guide</h2></div>
          <p>Choose the kind of help you want. Review the exact text before it goes to Gemini.
            Camera frames, model scores, and your API key are never included.</p>
          {acceptedOfferId && <p className="guide-hint">A model offer selected this guide type. You can change it before approving.</p>}
          <form onSubmit={previewHelp} className="guide-form">
            <label htmlFor="guide-mode">Guide type</label>
            <select id="guide-mode" value={guideMode}
              onChange={(event) => changeHelpText('mode', event.target.value)}>
              <option value="explain">Plain explanation and example</option>
              <option value="steps">Small steps and a next action</option>
              <option value="engage">Analogy, application, and challenge</option>
            </select>
            <label htmlFor="topic">Your question or topic</label>
            <textarea id="topic" value={topic} maxLength={500} rows={3} required
              onChange={(event) => changeHelpText('topic', event.target.value)}
              placeholder="What would you like help with?" />
            <label htmlFor="excerpt">Optional excerpt from your notes</label>
            <textarea id="excerpt" value={excerpt} maxLength={1600} rows={4}
              onChange={(event) => changeHelpText('excerpt', event.target.value)}
              placeholder="Paste only text you are comfortable sending." />
            <small>{excerpt.length}/1600 characters</small>
            <button type="submit" disabled={!active || helpBusy}>Preview text to send</button>
          </form>
          {!active && <p className="guide-hint">Start a session first. You can leave camera cues off.</p>}
          {preview && <div className="preview-box"><h3>Text to Gemini</h3>
            <pre>{preview.prompt}</pre>
            <button type="button" className="primary" disabled={helpBusy}
              onClick={sendHelp}>{helpBusy ? 'Writing…' : 'Send this text and get guide'}</button>
          </div>}
          {!health?.tutor_ready && <p className="guide-hint">Gemini is currently off. Set a new local
            GEMINI_API_KEY and confirm this project’s Free Tier before enabling requests.</p>}
          {helpError && <p className="inline-error" role="alert">{helpError}</p>}
          {guide && <div className="answer-box" aria-live="polite"><h3>Your guide</h3>
            <p>{guide.answer}</p><div className="feedback-row">
              <span>Was this helpful?</span>
              {(['useful', 'irrelevant', 'too_easy', 'too_hard', 'incorrect', 'wrong_cue'] as const).map((choice) =>
                <button key={choice} type="button" onClick={() => sendFeedback(choice)}>{choice.replace('_', ' ')}</button>)}</div>
            {feedback && <small role="status">{feedback}</small>}
          </div>}
        </section>
      </main>
      <footer>Local by design · The camera is released when monitoring pauses or stops.</footer>
    </div>
  )
}

export default App
