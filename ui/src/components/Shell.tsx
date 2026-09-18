import { useEffect, useState, type ReactNode } from 'react'
import { ago, api } from '../api'
import { navigate } from '../route'
import type { AppConfig, RunRow } from '../types'
import Glyph, { STATE } from './Glyph'
import Logo from './Logo'

/* The frame every page shares: where you are, your runs, and how to start one.

   Runs used to sit behind a dropdown that showed a 12-character id, and opening
   one reloaded the page, which killed whatever you were watching. The panel
   lists what you asked, how it ended and with which model; switching costs
   nothing, because every run keeps working on the server. It can be hidden,
   and runs can be deleted from this machine, one or several at a time. */
export function useRuns() {
  const [rows, setRows] = useState<RunRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let dead = false, t = 0
    const load = () => api.runs()
      .then((d) => { if (!dead) { setRows(d.sessions); setError(null) } })
      .catch((e) => { if (!dead) setError(String(e.message || e)) })
      .finally(() => { if (!dead) t = window.setTimeout(load, 3000) })
    load()
    const bump = () => { clearTimeout(t); load() }
    window.addEventListener('runs-changed', bump)
    return () => { dead = true; clearTimeout(t); window.removeEventListener('runs-changed', bump) }
  }, [])
  return { rows, error }
}

export const runsChanged = () => window.dispatchEvent(new Event('runs-changed'))

function stored(key: string, fallback: string) {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}

function word(r: RunRow) {
  return r.waiting ? 'Waiting for you' : STATE[r.outcome]?.word
}

export default function Shell({ current, view, config, children }: {
  current: string | null; view: string; config: AppConfig | null; children: ReactNode
}) {
  const { rows, error } = useRuns()
  const [open, setOpen] = useState(stored('openpaso.panel', 'open') === 'open')
  const [selecting, setSelecting] = useState(false)
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [problem, setProblem] = useState<string | null>(null)
  const running = rows?.filter((r) => r.running).length ?? 0
  const waiting = rows?.filter((r) => r.waiting).length ?? 0
  // A tab left open across an update keeps running the code it loaded with, so
  // a control fixed since then still misbehaves in front of you. The page can
  // see that: it knows which bundle it is, and the server says which it serves.
  // only a built page can be out of date: under the dev server this module is
  // its own file (Shell.tsx) and would always disagree with the served bundle
  const mine = import.meta.url.split('/').pop()
  const built = /^index-.*\.js$/.test(mine || '')
  const stale = built && !!config?.build && config.build !== mine

  useEffect(() => {
    const cur = rows?.find((r) => r.id === current)
    const base = cur?.prompt ? `${cur.prompt.slice(0, 48)} · openPASO`
      : view === 'solvers' ? 'Solvers · openPASO' : 'openPASO'
    document.title = waiting ? `(${waiting} waiting for you) ${base}` : running ? `(${running} working) ${base}` : base
  }, [rows, current, view, running, waiting])

  function toggle() {
    const next = !open
    setOpen(next)
    try { localStorage.setItem('openpaso.panel', next ? 'open' : 'closed') } catch { /* */ }
  }

  const [asking, setAsking] = useState<string[] | null>(null)

  async function remove(ids: string[]) {
    const list = rows?.filter((r) => ids.includes(r.id)) ?? []
    const busy = list.filter((r) => r.running)
    if (busy.length) { setProblem(`Stop ${busy.length === 1 ? 'the run that is' : `the ${busy.length} runs that are`} still working before deleting.`); return }
    setProblem(null)
    setAsking(ids)
  }

  async function reallyRemove(ids: string[]) {
    setAsking(null)
    const failed: string[] = []
    for (const id of ids) {
      try { await api.deleteRun(id) } catch (e) { failed.push(String((e as Error).message || e)) }
    }
    if (failed.length) setProblem(`${failed.length} could not be deleted: ${failed[0]}`)
    setPicked(new Set()); setSelecting(false)
    if (current && ids.includes(current)) navigate({ view: 'home' })
    runsChanged()
  }

  return (
    <div className="h-dvh grid grid-rows-[60px_1fr]" style={{ gridTemplateColumns: open ? '296px 1fr' : '0px 1fr' }}>
      <header className="col-span-2 flex items-center px-3 border-b line gap-1">
        <button onClick={toggle} aria-expanded={open} aria-controls="runs-panel"
                title={open ? 'Hide the runs panel' : 'Show the runs panel'}
                className="h-9 w-9 rounded-[8px] grid place-items-center text-body hover:text-ink hover:bg-card">
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden>
            <rect x="1.75" y="2.75" width="14.5" height="12.5" rx="2.25" stroke="currentColor" strokeWidth="1.5" />
            <path d="M6.5 3v12" stroke="currentColor" strokeWidth="1.5" />
            {open && <rect x="2.5" y="3.5" width="3.5" height="11" rx="1" fill="currentColor" opacity=".35" />}
          </svg>
          <span className="sr-only">{open ? 'Hide runs' : 'Show runs'}</span>
        </button>
        <a href="/" onClick={(e) => { e.preventDefault(); navigate({ view: 'home' }) }}
           className="ml-1 flex items-center gap-3 rounded-[8px] pr-2">
          <Logo size={28} />
          <span className="text-[17px] font-semibold text-ink tracking-[-0.01em]">openPASO</span>
        </a>
        {!open && (
          <button onClick={() => navigate({ view: 'home' })}
                  className="ml-4 h-9 px-3 rounded-[8px] border line text-[14px] text-ink hover:bg-card">
            <span aria-hidden className="text-coral mr-1.5">+</span>New run
          </button>
        )}
        {!open && (running > 0 || waiting > 0) && (
          <button onClick={toggle} className={`ml-2 h-9 px-3 rounded-[8px] text-[13px] ${waiting ? 'text-coral' : 'text-muted'} hover:bg-card`}>
            {waiting ? `${waiting} waiting for you` : `${running} working`}
          </button>
        )}
        {stale && (
          <button onClick={() => location.reload()}
                  className="ml-4 h-9 px-3.5 rounded-[8px] border border-coral/50 bg-coral/[0.08] text-[14px] text-ink
                             hover:bg-coral/[0.14] transition-colors">
            openPASO was updated · reload this page
          </button>
        )}
        <nav className="ml-auto flex items-center gap-1 text-[15px]">
          <a href="/?view=solvers" onClick={(e) => { e.preventDefault(); navigate({ view: 'solvers' }) }}
             aria-current={view === 'solvers' ? 'page' : undefined}
             className={`h-9 px-3 rounded-[8px] inline-flex items-center transition-colors
                         ${view === 'solvers' ? 'text-ink bg-card' : 'text-body hover:text-ink hover:bg-card'}`}>
            Solvers
          </a>
          {config && (
            <a href={config.docs_url} target="_blank" rel="noreferrer"
               className="h-9 px-3 rounded-[8px] inline-flex items-center gap-1.5 text-body hover:text-ink hover:bg-card transition-colors">
              Docs<span className="sr-only"> (opens in a new tab)</span>
              <span aria-hidden className="text-muted">↗</span>
            </a>
          )}
        </nav>
      </header>

      <aside id="runs-panel" aria-label="Your runs" inert={!open} aria-hidden={!open}
             className={`line flex flex-col min-h-0 overflow-hidden ${open ? 'border-r' : 'invisible'}`}>
        <div className="p-3">
          <button onClick={() => navigate({ view: 'home' })}
                  className={`w-full h-10 rounded-[8px] text-[15px] font-medium flex items-center gap-2 px-3 transition-colors
                              ${view === 'home' ? 'bg-card text-ink border border-strong' : 'border line text-ink hover:bg-card'}`}>
            <span aria-hidden className="text-coral text-[18px] leading-none">+</span> New run
          </button>
        </div>

        <div className="px-5 pt-2 pb-2 flex items-center gap-2 min-h-[36px]">
          {selecting ? (
            <>
              <span className="text-[13px] text-ink2">{picked.size} selected</span>
              <button disabled={!picked.size} onClick={() => void remove([...picked])}
                      className="ml-auto h-7 px-2.5 rounded-[6px] text-[13px] text-bad hover:bg-bad/10 disabled:opacity-40">Delete</button>
              <button onClick={() => { setSelecting(false); setPicked(new Set()) }}
                      className="h-7 px-2.5 rounded-[6px] text-[13px] text-body hover:bg-card">Done</button>
            </>
          ) : (
            <>
              <span className="text-[13px] font-medium text-muted">Runs</span>
              {waiting > 0 ? <span className="num text-[12px] text-coral">{waiting} waiting for you</span>
                : running > 0 ? <span className="num text-[12px] text-coral">{running} working</span> : null}
              {!!rows?.length && (
                <button onClick={() => setSelecting(true)}
                        className="ml-auto h-7 px-2.5 rounded-[6px] text-[13px] text-muted hover:text-ink hover:bg-card">Select</button>
              )}
            </>
          )}
        </div>
        {problem && <p className="px-5 pb-2 text-[13px] text-bad" role="alert">{problem}</p>}
        {asking && (
          <div className="mx-3 mb-2 p-3 rounded-[8px] border border-bad/40 bg-bad/[0.06]" role="alertdialog" aria-label="Delete runs">
            <p className="text-[14px] text-ink2">
              Delete {asking.length === 1
                ? <>“{(rows?.find((r) => r.id === asking[0])?.prompt || 'this run').slice(0, 80)}”</>
                : `${asking.length} runs`} from this machine? The record and every file in {asking.length === 1 ? 'its folder' : 'their folders'} are removed. This cannot be undone.
            </p>
            <div className="mt-2 flex gap-2">
              <button onClick={() => void reallyRemove(asking)} className="h-8 px-3 rounded-[8px] bg-bad text-white text-[13px] font-semibold">Delete</button>
              <button onClick={() => setAsking(null)} className="h-8 px-3 rounded-[8px] text-[13px] text-body hover:bg-card">Keep</button>
            </div>
          </div>
        )}

        <ol className="flex-1 overflow-y-auto scroll px-2 pb-3 min-h-0">
          {error && <li className="px-3 py-2 text-[13px] text-bad">Could not load runs: {error}</li>}
          {rows && rows.length === 0 && <li className="px-3 py-2 text-[14px] text-muted">No runs yet.</li>}
          {rows?.map((r) => {
            const on = r.id === current
            const checked = picked.has(r.id)
            const tone = r.waiting || r.running ? 'text-coral' : r.outcome === 'failed' || r.outcome === 'unfinished' ? 'text-bad' : ''
            return (
              <li key={r.id} className="group relative">
                {selecting && (
                  <input type="checkbox" checked={checked} aria-label={`Select “${(r.prompt || '').slice(0, 40)}”`}
                         onChange={() => { const n = new Set(picked); checked ? n.delete(r.id) : n.add(r.id); setPicked(n) }}
                         className="absolute left-3 top-3.5 z-10 w-4 h-4 accent-[#FF6B4A]" />
                )}
                <a href={`/?run=${r.id}`}
                   onClick={(e) => {
                     e.preventDefault()
                     if (selecting) { const n = new Set(picked); checked ? n.delete(r.id) : n.add(r.id); setPicked(n); return }
                     navigate({ run: r.id })
                   }}
                   aria-current={on ? 'page' : undefined}
                   className={`block rounded-[8px] py-2.5 pr-9 transition-colors ${selecting ? 'pl-9' : 'pl-3'}
                               ${on ? 'bg-card' : checked ? 'bg-soft' : 'hover:bg-soft'}`}>
                  <div className="flex items-start gap-2.5">
                    {!selecting && <Glyph outcome={r.outcome} className="text-[13px] mt-[3px] w-3 shrink-0 text-center" />}
                    <span className={`text-[14px] leading-[1.35] line-clamp-2 ${on ? 'text-ink' : 'text-ink2'}`}>{r.prompt}</span>
                  </div>
                  <div className={`${selecting ? '' : 'pl-[22px]'} mt-1 text-[13px] leading-[1.45] text-muted`}>
                    <div className={tone} title={r.waiting ? 'openPASO is waiting for you to run or skip a step.' : STATE[r.outcome]?.means}>{word(r)} · {ago(r.updated_at)}</div>
                    <div className="truncate">{r.model_label}</div>
                  </div>
                </a>
                {!selecting && (
                  <button onClick={() => void remove([r.id])} disabled={r.running}
                          title={r.running ? 'Stop this run before deleting it' : 'Delete this run from this machine'}
                          aria-label={`Delete “${(r.prompt || '').slice(0, 40)}”`}
                          className="absolute right-2 top-2.5 w-7 h-7 rounded-[6px] grid place-items-center text-muted
                                     opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-bad hover:bg-card
                                     disabled:hidden transition-opacity">
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
                      <path d="M2.5 3.5h9M5.5 3.5V2.25h3V3.5M3.5 3.5l.6 8.25h5.8l.6-8.25" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                )}
              </li>
            )
          })}
        </ol>
      </aside>

      <main className="min-h-0 overflow-hidden relative">{children}</main>
    </div>
  )
}
