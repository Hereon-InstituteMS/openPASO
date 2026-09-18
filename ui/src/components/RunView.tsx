import { useEffect, useMemo, useRef, useState } from 'react'
import { api, clock, fileUrl, money } from '../api'
import { navigate } from '../route'
import type { AppConfig, FieldSeries, ModelGroup } from '../types'
import { useRun } from '../useRun'
import Composer, { ModePicker, findModel } from './Composer'
import FilesDrawer from './FilesDrawer'
import Glyph, { STATE } from './Glyph'
import { runsChanged } from './Shell'
import Stage from './Stage'
import Transcript from './Transcript'

function pref(key: string, fallback: string) {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}

/** The run's own field animation, if it wrote one. Only this run's folder is
    searched: showing a result from anywhere else is how a crashed run once
    displayed somebody else's vortex street as its answer. */
function useField(runId: string, settle: number) {
  const [field, setField] = useState<FieldSeries | null>(null)
  const [pictures, setPictures] = useState<{ rel: string; name: string; mtime: number }[]>([])
  const seen = useRef<Map<string, number>>(new Map())
  useEffect(() => { setField(null); setPictures([]); seen.current = new Map() }, [runId])
  useEffect(() => {
    let dead = false
    const pics: { rel: string; name: string; mtime: number }[] = []
    const walk = async (sub: string, depth: number): Promise<FieldSeries | null> => {
      if (depth > 3) return null
      const d = await api.files(runId, sub).catch(() => null)
      if (!d) return null
      for (const f of d.entries) {
        if (f.is_dir) {
          if (/^(\.|__pycache__|node_modules|uploads$)/.test(f.name)) continue
          const hit = await walk(f.sub, depth + 1); if (hit) return hit
          continue
        }
        if (/\.(png|jpe?g|svg|gif|webp)$/i.test(f.name)) pics.push({ rel: f.rel_path, name: f.sub || f.name, mtime: f.mtime })
        if (!f.name.endsWith('.json') || (f.size ?? 0) < 200) continue
        if (seen.current.get(f.rel_path) === f.mtime) continue
        seen.current.set(f.rel_path, f.mtime)
        const v = await api.viz(f.rel_path).catch(() => null)
        if (v?.kind === 'field_series') return v as unknown as FieldSeries
      }
      return null
    }
    walk('', 0).then((hit) => {
      if (dead) return
      if (hit) setField(hit)
      setPictures(pics.sort((a, b) => b.mtime - a.mtime).slice(0, 6))
    })
    return () => { dead = true }
  }, [runId, settle])
  return { field, pictures }
}

export default function RunView({ id, config, groups }: {
  id: string; config: AppConfig | null; groups: ModelGroup[] | null
}) {
  const { session, events, link, notice, setNotice, send } = useRun(id)
  const [files, setFiles] = useState(false)
  const [reasoning, setReasoning] = useState(pref('openpaso.reasoning', 'shown') === 'shown')
  const [confirmStop, setConfirmStop] = useState(false)
  const [now, setNow] = useState(Date.now())
  const [busy, setBusy] = useState(false)
  const scroller = useRef<HTMLDivElement>(null)
  const pinned = useRef(true)

  const running = !!session?.running
  const outcome = running ? 'running' : (session?.outcome ?? 'running')
  const settle = events.filter((e) => e.type === 'tool_result' || e.type === 'done').length
  const { field, pictures } = useField(id, settle)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const model = findModel(groups, session?.model ?? null)

  useEffect(() => { if (!running) return; const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t) }, [running])
  useEffect(() => { setConfirmStop(false) }, [running, id])
  useEffect(() => { runsChanged() }, [running, outcome])

  // stay at the bottom while new work arrives, unless you scrolled up to read
  useEffect(() => {
    const el = scroller.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  }, [events.length, field])

  const turnStart = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) if (events[i].type === 'turn_start' || events[i].type === 'user_msg') return events[i].t
    return undefined
  }, [events])
  const lastDone = useMemo(() => [...events].reverse().find((e) => e.type === 'done')?.t, [events])
  const seconds = turnStart ? ((running ? now : (lastDone ?? now)) - turnStart) / 1000 : 0
  const turnsSoFar = events.filter((e) => e.type === 'turn_start').length
  const steps = events.filter((e) => e.type === 'tool_call_pending').length
  const prompt = events.find((e) => e.type === 'user_msg')?.text
  // "Waiting for you" only while a Run this step button is on the screen: the
  // same events decide both, so the header cannot say it with nothing to press
  const waiting = running && (() => {
    const pend = new Set<string>()
    for (const e of events) {
      if (e.type === 'tool_call_pending') pend.add(e.call_id!)
      if (['tool_call_executing', 'tool_call_rejected', 'tool_result', 'tool_error'].includes(e.type)) pend.delete(e.call_id!)
      if (e.type === 'done') pend.clear()
    }
    return pend.size > 0
  })()
  const modelName = session?.model_detail || session?.claude_model
  // the running total arrives with each token count; the snapshot is from when the page opened
  const liveCost = useMemo(() => { for (let i = events.length - 1; i >= 0; i--) if (events[i].cost_usd_total != null) return events[i].cost_usd_total; return null }, [events])
  const cost = money(liveCost ?? session?.cost_usd)
  const firstStart = events.find((e) => e.type === 'turn_start' || e.type === 'user_msg')?.t
  const allSeconds = firstStart ? ((running ? now : (lastDone ?? now)) - firstStart) / 1000 : 0
  const kindWords = model?.kind === 'openrouter' ? 'hosted on OpenRouter' : model?.kind === 'claude-code' ? 'your Claude login'
    : model?.kind === 'local' ? 'on this machine' : ''

  if (link === 'missing') {
    return (
      <div className="h-full grid place-items-center">
        <div className="text-center">
          <p className="text-[20px] text-ink">This run does not exist.</p>
          <p className="mt-2 text-[15px] text-muted">It may have been deleted.</p>
          <button onClick={() => navigate({ view: 'home' })} className="mt-5 h-10 px-5 rounded-[8px] border line text-[15px] text-ink hover:bg-card">Start a new run</button>
        </div>
      </div>
    )
  }

  async function followUp(text: string, attach: File[]) {
    setBusy(true)
    try {
      let names: string[] = []
      if (attach.length) names = (await api.upload(id, attach)).saved.map((x) => x.name)
      if (running) {
        const note = names.length ? `${text}\n\n(I uploaded: ${names.map((n) => `uploads/${n}`).join(', ')})` : text
        return send({ type: 'steer', text: note })
      }
      return send({ type: 'prompt', text, attachments: names })
    } catch (e) {
      setNotice(String((e as Error).message || e)); return false
    } finally { setBusy(false) }
  }

  async function remove() {
    setConfirmDelete(false)
    try { await api.deleteRun(id); runsChanged(); navigate({ view: 'home' }) }
    catch (e) { setNotice(String((e as Error).message || e)) }
  }

  return (
    <div className="h-full flex flex-col relative">
      {/* ── what this run is, and its state ─────────────────────────────── */}
      <div className="border-b line px-10 pt-6 pb-4">
        <div className="max-w-[1040px] mx-auto">
          <div className="flex items-start gap-4">
            <h1 className="flex-1 min-w-0 text-[22px] font-semibold leading-[1.35] tracking-[-0.01em] text-ink line-clamp-2" title={prompt}>
              {prompt || (session ? 'New run' : 'Loading…')}
            </h1>
            <div className="flex items-center gap-2 shrink-0">
              <button onClick={() => setFiles(true)} className="h-9 px-3.5 rounded-[8px] border line text-[14px] text-body hover:text-ink hover:border-strong">Files</button>
              <button onClick={() => setConfirmDelete(true)} disabled={running} title={running ? 'Stop the run before deleting it' : 'Delete this run from this machine'}
                      className="h-9 px-3.5 rounded-[8px] border line text-[14px] text-body hover:text-bad hover:border-bad/50 disabled:opacity-40 disabled:hover:text-body">Delete</button>
            </div>
          </div>

          <div className="mt-3 flex items-center gap-x-5 gap-y-2 flex-wrap text-[14px]" role="status">
            <span className={`flex items-center gap-2 text-[15px] font-medium ${STATE[outcome]?.tone}`} title={waiting ? undefined : STATE[outcome]?.means}>
              <Glyph outcome={outcome} /> {waiting ? 'Waiting for you' : STATE[outcome]?.word}
            </span>
            {turnStart && <span className="num text-muted" title="Time taken by the latest prompt or follow-up">{turnsSoFar > 1 ? `last turn ${clock(seconds)} · ${clock(allSeconds)} since the first prompt` : clock(seconds)}</span>}
            <span className="text-muted">{steps} step{steps === 1 ? '' : 's'}</span>
            {session && (
              <span className="text-body">
                {model?.label ?? session.model}{modelName ? <span className="num text-muted"> ({modelName})</span> : null}
                {kindWords && <span className="text-muted"> · {kindWords}</span>}
              </span>
            )}
            {cost
              ? <span className="num text-muted" title={model?.kind === 'claude-code' ? 'As reported by Claude Code' : "Estimated from token counts and OpenRouter's list price"}>{cost}</span>
              : session && model?.kind === 'openrouter' && steps > 0
                ? <span className="text-muted" title="No token counts or no price were reported for this run">cost unknown</span> : null}
            {running && (
              confirmStop ? (
                <span className="ml-auto flex items-center gap-2">
                  <span className="text-[14px] text-ink2">Stop this run and end everything it started?</span>
                  <button onClick={() => { send({ type: 'stop' }); setConfirmStop(false) }}
                          className="h-9 px-4 rounded-[8px] bg-bad text-white text-[14px] font-semibold">Stop</button>
                  <button onClick={() => setConfirmStop(false)} className="h-9 px-3 rounded-[8px] text-[14px] text-body hover:bg-card">Keep running</button>
                </span>
              ) : (
                <button onClick={() => setConfirmStop(true)}
                        className="ml-auto h-9 px-4 rounded-[8px] border line text-[14px] text-ink hover:border-bad/60 hover:text-bad">Stop</button>
              )
            )}
          </div>

          {confirmDelete && !running && (
            <div className="mt-3 flex items-center gap-2 flex-wrap" role="alertdialog" aria-label="Delete this run">
              <span className="text-[14px] text-ink2">Delete this run from this machine? Its record and every file in its folder are removed. This cannot be undone.</span>
              <button onClick={remove} className="h-9 px-4 rounded-[8px] bg-bad text-white text-[14px] font-semibold">Delete</button>
              <button onClick={() => setConfirmDelete(false)} className="h-9 px-3 rounded-[8px] text-[14px] text-body hover:bg-card">Keep it</button>
            </div>
          )}
          {link === 'lost' && (
            <p className="mt-3 text-[14px] text-bad">Connection to the server lost. Reconnecting… The run keeps working on the server.</p>
          )}
          {outcome === 'unfinished' && (
            <p className="mt-3 text-[14px] text-body">
              This run stopped without recording an end, usually because the server was restarted while it worked.
              Nothing is running for it now. You can send a follow-up to continue.
            </p>
          )}
          {notice && (
            <p className="mt-3 text-[14px] text-ink2 flex gap-3" role="alert">
              {notice}<button onClick={() => setNotice(null)} className="text-muted hover:text-ink underline underline-offset-4">Dismiss</button>
            </p>
          )}
        </div>
      </div>

      {/* ── the work ────────────────────────────────────────────────────── */}
      <div ref={scroller} className="flex-1 overflow-y-auto scroll"
           onScroll={(e) => { const el = e.currentTarget; pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80 }}>
        <div className="max-w-[1040px] mx-auto px-10 py-6">
          {field && (
            <section className="mb-8" aria-label="Result">
              <div className="text-[13px] font-medium text-muted mb-2">Field this run wrote</div>
              <Stage series={field} />
            </section>
          )}

          {pictures.length > 0 && (
            <section className="mb-8" aria-label="Pictures this run made">
              <div className="text-[13px] font-medium text-muted mb-2">Pictures this run made</div>
              <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
                {pictures.map((p) => (
                  <figure key={p.rel} className="bg-soft border line rounded-[8px] p-3">
                    <img src={fileUrl(p.rel)} alt={p.name} loading="lazy" className="w-full rounded-[6px] bg-white" />
                    <figcaption className="num mt-2 text-[13px] text-muted break-all">{p.name}</figcaption>
                  </figure>
                ))}
              </div>
            </section>
          )}

          <div className="flex items-center gap-3 mb-3">
            <span className="text-[13px] font-medium text-muted">What happened</span>
            <span className="ml-auto text-[13px] text-muted">Reasoning</span>
            <div role="radiogroup" aria-label="Reasoning" className="flex rounded-[8px] border line p-0.5">
              {(['shown', 'hidden'] as const).map((v) => (
                <button key={v} role="radio" aria-checked={(v === 'shown') === reasoning}
                        onClick={() => { setReasoning(v === 'shown'); try { localStorage.setItem('openpaso.reasoning', v) } catch { /* */ } }}
                        className={`h-7 px-3 rounded-[6px] text-[13px] capitalize ${(v === 'shown') === reasoning ? 'bg-card text-ink' : 'text-muted hover:text-ink'}`}>
                  {v}
                </button>
              ))}
            </div>
          </div>

          <Transcript events={events} live={running} showReasoning={reasoning} now={now} modelKind={model ? model.kind : session?.model === 'mock' ? 'test' : undefined}
                      onEndStep={model?.kind === 'claude-code' ? undefined : (cid) => send({ type: 'end_step', call_id: cid })}
                      onDecide={(cid, ok) => send(ok ? { type: 'approve', call_id: cid } : { type: 'reject', call_id: cid, reason: 'skipped by the user' })} />
        </div>
      </div>

      {/* ── say something to it ─────────────────────────────────────────── */}
      <div className="border-t line px-10 py-4">
        <div className="max-w-[1040px] mx-auto">
          <Composer
            placeholder={running
              ? 'Send a correction. openPASO reads it as soon as the current step finishes.'
              : 'Ask a follow-up about this run, or tell openPASO what to change and run again'}
            submitLabel={running ? 'Send correction' : 'Send'}
            onSubmit={followUp} busy={busy} draftKey={`run.${id}`}
            left={session && (
              <ModePicker modes={config?.modes} value={session.mode} planAllowed={session.model !== 'claude-code'}
                          onChange={(m) => send({ type: 'set_mode', mode: m })} />
            )} />
        </div>
      </div>

      <FilesDrawer runId={id} open={files} onClose={() => setFiles(false)} refreshKey={settle} />
    </div>
  )
}
