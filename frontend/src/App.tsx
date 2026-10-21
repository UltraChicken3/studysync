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
}

type Health = {
  service: string
  model_present: boolean
  key_configured: boolean
  session_status: string
  demo_mode: boolean
}

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
  const [showHelp, setShowHelp] = useState(false)
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

  async function request(path: string, body?: object): Promise<Snapshot> {
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
      throw new Error(failure?.detail || `Request failed (${response.status})`)
    }
    return response.json()
  }

  async function start(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      sequence.current = -1
      setSession(await request('/sessions', {
        subject,
        goal,
        duration_minutes: duration,
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
      setSession(await request(`/sessions/${session.session_id}/${action}`))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : `Could not ${action} the session`)
    } finally {
      setBusy(false)
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
              {!active && <button className="primary" type="submit" disabled={!token || busy || (!health?.model_present && !health?.demo_mode)}>
                {busy ? 'Starting…' : 'Start monitoring'}</button>}
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
          </section>
        </div>

        <section className="help-strip">
          <div><span className="step">03</span><h2>Need a hand?</h2>
            <p>Manual study guides and model-led offers are the next implementation step.</p></div>
          <button type="button" onClick={() => setShowHelp(!showHelp)}>
            {showHelp ? 'Hide details' : 'How help will work'}</button>
          {showHelp && <p className="help-details">You will choose whether to send your typed topic and selected notes
            to Gemini. No camera frame or model score will be sent. Until this is connected, the desktop app still
            supports manual questions.</p>}
        </section>
      </main>
      <footer>Local by design · The camera is released when monitoring pauses or stops.</footer>
    </div>
  )
}

export default App
