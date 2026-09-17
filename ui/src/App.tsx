import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { DUR, EASE } from './motion'
import { api, connect } from './api'
import type { Ev, FieldSeries, FileRow, Session } from './types'
import Nav from './components/Nav'
import Idle from './components/Idle'
import Stage from './components/Stage'
import { toSteps, finalAnswer } from './components/Ledger'
import Phases from './components/Phases'
import Answer from './components/Answer'
import SlideOver from './components/SlideOver'
import RunState, { type Outcome } from './components/RunState'

export default function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [events, setEvents] = useState<Ev[]>([])
  const [prompt, setPrompt] = useState('')
  const [started, setStarted] = useState(false)
  const [outcome, setOutcome] = useState<Outcome>('running')
  const [failure, setFailure] = useState<{ message: string; traceback?: string } | null>(null)
  const [startedAt, setStartedAt] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const busy = started && outcome === 'running'
  const [status, setStatus] = useState('')
  const [menu, setMenu] = useState(false)
  const [field, setField] = useState<FieldSeries | null>(null)
  const [files, setFiles] = useState<FileRow[]>([])
  const [cwd, setCwd] = useState('')
  const [models, setModels] = useState<{ id: string; label: string }[]>([])
  const [modes, setModes] = useState<string[]>([])
  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    api.models().then(setModels)
    api.modes().then(setModes)
    /* ?session=<id> reopens a finished run. Useful on its own, and it is what
       lets a recorder replay a real run instead of staging one. */
    const wanted = new URLSearchParams(location.search).get('session')
    const load = wanted ? api.session(wanted) : api.newSession()
    load.then((s) => {
      if (s.events?.length) {
        setEvents(s.events)
        let seen: Outcome = 'running'
        for (const e of s.events) {
          if (e.type === 'error') seen = (e as { outcome?: Outcome }).outcome || 'failed'
          else if (e.type === 'done') {
            const stated = (e as { outcome?: Outcome }).outcome
            if (stated) seen = stated
            else if (seen === 'running') seen = 'completed'
          }
        }
        setOutcome(seen)
        const err = [...s.events].reverse().find((e) => e.type === 'error')
        if (err) setFailure({ message: err.message || 'the run failed',
                              traceback: (err as { traceback?: string }).traceback })
        const first = s.events.find((e) => e.type === 'user_msg')
        if (first?.text) { setPrompt(first.text); setStarted(true) }
        void findField()
      }
      setSession(s)
      ws.current = connect(s.id, (raw) => {
        const e = raw as Ev
        if (e.type === 'status') {
          if (e.session) setSession((p) => ({ ...(p as Session), ...e.session }))
          setStatus(e.message === 'connected' ? '' : e.message || '')
          return
        }
        if (e.type === 'tool_result' || e.type === 'done') {
          void findField()   // only ever this run's own output
        }
        if (e.type === 'error') {
          setFailure({ message: e.message || 'the run failed',
                       traceback: (e as { traceback?: string }).traceback })
          setOutcome((e as { outcome?: Outcome }).outcome || 'failed')
        }
        if (e.type === 'done') {
          // The server decides the terminal state. The interface used to say
          // "finished" for every run that stopped, including a crash.
          setOutcome((prev) => {
            const o = (e as { outcome?: Outcome }).outcome
            return o && o !== 'running' ? o : (prev === 'running' ? 'completed' : prev)
          })
          setStatus('')
        }
        if (e.type === 'token_count') {
          setSession((p) => p && {
            ...p,
            tokens_in: p.tokens_in + (e.input || 0),
            tokens_out: p.tokens_out + (e.output || 0),
          })
        }
        setEvents((p) => [...p, e])
      })
    })
    return () => ws.current?.close()
  }, [])

  useEffect(() => {
    if (!busy || !startedAt) return
    const id = setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 1000)
    return () => clearInterval(id)
  }, [busy, startedAt])

  const refreshFiles = (rel = cwd) =>
    api.files(rel).then((r) => { setFiles(r.entries); setCwd(r.rel || '') })

  /* A run may only show what that run produced.

     This used to walk the whole sandbox and open the first field file it found.
     Any prompt at all then displayed whatever result happened to be lying
     there, including for a run that computed nothing. That is fabrication: the
     interface would assert a result the run never made. The search is now
     confined to this session's own working directory. */
  async function findField() {
    if (!session?.id) return false
    const root = `webui_${session.id}`
    const walk = async (rel: string, depth: number): Promise<boolean> => {
      if (depth > 2) return false
      const r = await api.files(rel).catch(() => null)
      if (!r) return false
      for (const f of r.entries) {
        if (f.is_dir) { if (await walk(f.rel_path, depth + 1)) return true; continue }
        if (!f.name.endsWith('.json')) continue
        const v = await api.viz(f.rel_path).catch(() => null)
        if (v?.kind === 'field_series') { setField(v as FieldSeries); return true }
      }
      return false
    }
    return walk(root, 0)
  }

  useEffect(() => { if (menu) refreshFiles() }, [menu])

  function send() {
    if (!prompt.trim() || !ws.current) return
    ws.current.send(JSON.stringify({ type: 'prompt', text: prompt }))
    setStarted(true)
    setOutcome('running'); setFailure(null)
    setStartedAt(Date.now()); setElapsed(0)
  }
  const stopRun = () => ws.current?.send(JSON.stringify({ type: 'stop' }))

  /* Leaving mid-run abandons work that is still going. Saying so first is the
     rule, and it costs one sentence. */
  const leave = (to: string) => {
    if (busy && !confirm('This run is still going. Leave it and start again?')) return
    location.href = to
  }
  const decide = (call_id: string, ok: boolean) =>
    ws.current?.send(JSON.stringify(
      ok ? { type: 'approve', call_id } : { type: 'reject', call_id, reason: 'not now' }))

  async function openFile(f: FileRow) {
    if (f.is_dir) { refreshFiles(f.rel_path); return }
    const v = await api.viz(f.rel_path)
    if (v.kind !== 'field_series') return
    // Same rule as findField: this run may only display what this run wrote.
    // Without this check two clicks put any field in the sandbox onto the
    // stage, rendered identically to one the run produced.
    if (!session?.id || !f.rel_path.startsWith(`webui_${session.id}/`)) {
      setFailure({ message:
        `That file was not produced by this run (${f.rel_path}). It is not shown as a result.` })
      return
    }
    setField(v as FieldSeries); setMenu(false)
  }

  const [cut, setCut] = useState<number | null>(null)

  /* The other half of the capture hook: show the run as it stood after N
     steps. The events are the real ones; this only decides how many of them
     have arrived yet. */
  useEffect(() => {
    ;(window as unknown as Record<string, unknown>).__film = {
      start: (text: string) => { setPrompt(text); setStarted(true) },
      steps: (n: number | null) => setCut(n),
    }
  }, [])

  const allSteps = toSteps(events)
  const steps = cut === null ? allSteps : allSteps.slice(0, cut)
  const answer = (outcome === 'completed' && (cut === null || cut >= allSteps.length))
    ? finalAnswer(events) : ''

  /* There is no verdict panel. It was a headline sentence with a hardcoded
     empty figures array, so it could never show the numbers that would justify
     it, and the tool that decides it has never been called in any recorded
     session. It comes back when it can be fed the quantity, the refinement
     sequence, the value per level and the tolerance. */

  return (
    <>
      <Nav
        inRun={started}
        onMenu={() => setMenu(true)}
        onHome={() => leave('/')}
        onOpenRun={(id) => leave(`/?session=${id}`)}
      />

      {!started ? (
        <Idle value={prompt} onChange={setPrompt} onSend={send} busy={busy} />
      ) : (
        <>
          <section className="w-[1224px] mx-auto pt-12">
            <div className="eyebrow flex items-center gap-3">
              <span>{session?.model ?? ''}</span>
              {steps.length > 0 && <>
                <span className="text-graphit">&middot;</span>
                <span>{steps.length} {steps.length === 1 ? 'step' : 'steps'}</span>
              </>}
              {status && busy && <>
                <span className="text-graphit">&middot;</span>
                <span>{status}</span>
              </>}
            </div>
            <h1 title={prompt}
                className="mt-4 text-[28px] font-medium tracking-[-0.02em] leading-[1.25]
                           text-ink max-w-[1100px] overflow-hidden text-ellipsis
                           whitespace-nowrap">
              {prompt}
            </h1>
          </section>

          <div className="w-[1224px] mx-auto mt-8">
            <RunState outcome={outcome} message={failure?.message}
                      traceback={failure?.traceback} elapsed={elapsed}
                      onStop={busy ? stopRun : undefined} />
          </div>

          <AnimatePresence>
            {field && (
              <motion.div key="stage"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                transition={{ duration: 0.6, ease: 'linear' }}>
                <Stage series={field} />
              </motion.div>
            )}
          </AnimatePresence>

          <div className="w-[1224px] mx-auto mt-12 grid grid-cols-[1fr_476px] gap-16 items-start">
            <Phases steps={steps} running={busy}
                    onApprove={(id) => decide(id, true)}
                    onReject={(id) => decide(id, false)} />
            <div className="space-y-12">
            <Answer text={answer} />
            </div>
          </div>
          <div className="h-16" />
        </>
      )}

      <SlideOver
        open={menu} onClose={() => setMenu(false)} session={session}
        models={models} modes={modes} files={files} cwd={cwd}
        onModel={(m) => ws.current?.send(JSON.stringify({ type: 'set_model', model: m }))}
        onMode={(m) => ws.current?.send(JSON.stringify({ type: 'set_mode', mode: m }))}
        onOpenFile={openFile}
        onUp={() => refreshFiles(cwd.split('/').slice(0, -1).join('/'))}
      />
    </>
  )
}
