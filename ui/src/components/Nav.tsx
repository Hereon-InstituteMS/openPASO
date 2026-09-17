import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

type Run = { id: string; model?: string; n_events?: number; created_at?: number }

function when(ts?: number) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  const mins = (Date.now() - d.getTime()) / 60000
  if (mins < 1) return 'just now'
  if (mins < 60) return `${Math.floor(mins)} min ago`
  if (mins < 60 * 24) return `${Math.floor(mins / 60)} h ago`
  return d.toLocaleDateString()
}

/* Navigation, because there was none.

   The wordmark goes home, and past runs are reachable. Sessions were saved to
   disk from the start and the endpoint that lists them was never called, so
   reopening yesterday's work meant copying a hex id out of a panel by hand. */
export default function Nav({
  onHome, onOpenRun, onMenu, inRun,
}: {
  onHome: () => void
  onOpenRun: (id: string) => void
  onMenu: () => void
  inRun: boolean
}) {
  const [open, setOpen] = useState(false)
  const [runs, setRuns] = useState<Run[]>([])
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    api.sessions().then((r) => setRuns((r as Run[]).filter((x) => (x.n_events ?? 0) > 0)))
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false)
    }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', esc)
    return () => {
      document.removeEventListener('mousedown', away)
      document.removeEventListener('keydown', esc)
    }
  }, [open])

  return (
    <nav className="h-16 flex items-center px-12" data-testid="nav">
      <button onClick={onHome}
              className="text-[17px] font-medium text-ink tracking-[-0.014em]
                         transition-opacity duration-150 hover:opacity-80">
        openPASO
      </button>

      {inRun && (
        <button onClick={onHome} data-testid="new-run"
                className="ml-8 h-8 px-3.5 rounded-[6px] border line text-[14px] text-muted
                           transition-colors duration-150 hover:text-ink2">
          New run
        </button>
      )}

      <div className="ml-auto flex items-center gap-7 text-[14px] font-medium text-muted">
        <div className="relative" ref={box}>
          <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
                  data-testid="runs"
                  className="transition-colors duration-150 hover:text-ink2">
            Runs
          </button>
          {open && (
            <div className="absolute right-0 top-9 w-[380px] max-h-[420px] overflow-y-auto
                            scroll bg-s4 border line rounded-[10px] edge-top p-2 z-20">
              {runs.length === 0 && (
                <div className="px-3 py-4 text-[14px] text-muted">No earlier runs yet.</div>
              )}
              {runs.map((r) => (
                <button key={r.id}
                        onClick={() => { setOpen(false); onOpenRun(r.id) }}
                        className="w-full text-left px-3 py-2.5 rounded-[6px]
                                   transition-colors duration-150 hover:bg-s3">
                  <div className="flex items-center gap-3">
                    <span className="num text-[13px] text-ink2">{r.id.slice(0, 8)}</span>
                    <span className="text-[13px] text-muted ml-auto">{when(r.created_at)}</span>
                  </div>
                  <div className="num text-[13px] text-muted mt-1">
                    {r.model} &middot; {r.n_events} events
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        <a href="https://github.com/Hereon-InstituteMS/openPASO"
           className="transition-colors duration-150 hover:text-ink2">Docs</a>

        <button onClick={onMenu} aria-label="Session, model and files"
                title="Session, model and files"
                className="w-8 h-8 rounded-[6px] border line grid place-items-center
                           transition-colors duration-150 hover:text-ink2">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"
               stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path d="M1.5 3.5h11M1.5 7h11M1.5 10.5h11" />
          </svg>
        </button>
      </div>
    </nav>
  )
}
