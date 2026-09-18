import { useEffect, useRef, useState } from 'react'
import type { ModelGroup, ModelInfo, ModeInfo } from '../types'
import Popover from './Popover'

const ACCEPT = '.step,.stp,.iges,.igs,.stl,.brep,.geo,.msh,.vtk,.vtu,.xdmf,.h5,.hdf5,.exo,.e,.med,.inp,.dat,.yaml,.yml,.json,.xml,.feb,.py,.csv,.txt,.md,.npy,.npz,.png,.jpg,.jpeg,.pdf'

export function findModel(groups: ModelGroup[] | null, id: string | null): ModelInfo | null {
  for (const g of groups || []) for (const m of g.models) if (m.id === id) return m
  return null
}

const usd = (x?: number | null) => (x == null ? '' : x < 0.01 ? 'under $0.01' : `$${x.toFixed(2)}`)
const perM = (x?: number | null) => (x == null ? null : x < 0.1 ? `$${x.toFixed(3)}` : `$${x.toFixed(2)}`)

/* Where each model runs, what it costs, where the data goes, and whether it
   works right now. The list used to be seven flat names: three local models
   with no server behind them looked exactly like the three that worked. */
export function ModelPicker({ groups, value, onChange, disabled, up = true }: {
  groups: ModelGroup[] | null; value: string | null; onChange: (id: string) => void; disabled?: boolean; up?: boolean
}) {
  const [open, setOpen] = useState(false)
  const cur = findModel(groups, value)
  return (
    <div className="relative">
      <button type="button" disabled={disabled} onClick={() => setOpen((o) => !o)}
              aria-haspopup="dialog" aria-expanded={open}
              className="h-9 pl-3 pr-2.5 rounded-[8px] border line text-[14px] text-body flex items-center gap-2
                         hover:text-ink hover:border-strong transition-colors disabled:opacity-60 disabled:cursor-default">
        <span className="text-muted">Model</span>
        <span className="text-ink">{cur ? cur.label : groups ? 'Choose a model' : 'Loading…'}</span>
        {!disabled && <span aria-hidden className="text-muted text-[11px]">▾</span>}
      </button>
      <Popover open={open} onClose={() => setOpen(false)} up={up} width={520}>
        {groups?.map((g) => (
          <section key={g.kind} className="p-2">
            <div className="px-2 pt-1 text-[14px] font-semibold text-ink">{g.title}</div>
            <p className="px-2 mt-1 text-[13px] leading-[1.45] text-muted">{g.note}</p>
            {g.kind === 'openrouter' && g.key_source && <p className="px-2 mt-1 text-[13px] text-muted">Your OpenRouter key was found.</p>}
            <ul className="mt-2">
              {g.models.map((m) => {
                const sel = m.id === value
                return (
                  <li key={m.id}>
                    <button type="button" disabled={!m.available}
                            onClick={() => { onChange(m.id); setOpen(false) }}
                            aria-pressed={sel}
                            className={`w-full text-left rounded-[8px] px-2 py-2 flex items-start gap-3 transition-colors
                                        ${sel ? 'bg-card' : 'hover:bg-card'} disabled:cursor-not-allowed`}>
                      <span aria-hidden className={`mt-[3px] w-3 text-[12px] ${sel ? 'text-coral' : 'text-transparent'}`}>●</span>
                      <span className="flex-1 min-w-0">
                        <span className={`block text-[15px] ${m.available ? 'text-ink' : 'text-muted'}`}>{m.label}</span>
                        <span className={`block text-[13px] ${m.available ? 'text-muted' : 'text-bad'}`}>{m.status}</span>
                        {m.past_runs ? (
                          <span className="block text-[13px] text-muted">
                            {m.past_runs === 1 ? 'Your one run with it cost ' : `Your ${m.past_runs} runs with it cost `}
                            {m.past_cost_low === m.past_cost_high ? usd(m.past_cost_low) : `between ${usd(m.past_cost_low)} and ${usd(m.past_cost_high)} each`}
                          </span>
                        ) : null}
                      </span>
                      {(m.price_in != null) && (
                        <span className="num text-[12px] text-muted text-right shrink-0 leading-[1.5]">
                          {perM(m.price_in)} to read<br />{perM(m.price_out)} to write<br />
                          <span className="text-faint">per million tokens</span>
                        </span>
                      )}
                    </button>
                  </li>
                )
              })}
            </ul>
          </section>
        ))}
      </Popover>
    </div>
  )
}

export function ModePicker({ modes, value, onChange, planAllowed, up = true }: {
  modes: ModeInfo[] | undefined; value: string; onChange: (m: string) => void; planAllowed: boolean; up?: boolean
}) {
  const [open, setOpen] = useState(false)
  const cur = modes?.find((m) => m.id === value)
  return (
    <div className="relative">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-haspopup="dialog" aria-expanded={open}
              className="h-9 pl-3 pr-2.5 rounded-[8px] border line text-[14px] text-body flex items-center gap-2
                         hover:text-ink hover:border-strong transition-colors">
        <span className="text-muted">Steps</span>
        <span className="text-ink">{cur?.label ?? value}</span>
        <span aria-hidden className="text-muted text-[11px]">▾</span>
      </button>
      <Popover open={open} onClose={() => setOpen(false)} up={up} width={440}>
        <ul>
          {modes?.map((m) => {
            const blocked = m.id === 'plan' && !planAllowed
            const sel = m.id === value
            return (
              <li key={m.id}>
                <button type="button" disabled={blocked} aria-pressed={sel}
                        onClick={() => { onChange(m.id); setOpen(false) }}
                        className={`w-full text-left rounded-[8px] px-3 py-2.5 flex gap-3 transition-colors
                                    ${sel ? 'bg-card' : 'hover:bg-card'} disabled:cursor-not-allowed`}>
                  <span aria-hidden className={`mt-[3px] w-3 text-[12px] ${sel ? 'text-coral' : 'text-transparent'}`}>●</span>
                  <span>
                    <span className={`block text-[15px] ${blocked ? 'text-muted' : 'text-ink'}`}>{m.label}</span>
                    <span className="block mt-0.5 text-[13px] leading-[1.45] text-muted">
                      {blocked ? 'Not available with Claude Code: it runs headless and cannot stop to ask.' : m.detail}
                    </span>
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      </Popover>
    </div>
  )
}

export default function Composer({
  placeholder, submitLabel, onSubmit, left, busy, autoFocus, hint, allowFiles = true, initial = '', draftKey,
}: {
  placeholder: string
  submitLabel: string
  onSubmit: (text: string, files: File[]) => Promise<boolean> | boolean
  left?: React.ReactNode
  busy?: boolean
  autoFocus?: boolean
  hint?: React.ReactNode
  allowFiles?: boolean
  initial?: string
  /** keeps what you typed if you leave the page and come back */
  draftKey?: string
}) {
  const stored = () => { try { return draftKey ? sessionStorage.getItem(`openpaso.draft.${draftKey}`) : null } catch { return null } }
  const [text, setText] = useState(() => stored() ?? initial)
  const [files, setFiles] = useState<File[]>([])
  const [shake, setShake] = useState(false)
  const area = useRef<HTMLTextAreaElement>(null)
  const pick = useRef<HTMLInputElement>(null)

  useEffect(() => { if (initial) setText(initial) }, [initial])
  useEffect(() => { setText(stored() ?? initial) }, [draftKey])  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!draftKey) return
    try {
      if (text) sessionStorage.setItem(`openpaso.draft.${draftKey}`, text)
      else sessionStorage.removeItem(`openpaso.draft.${draftKey}`)
    } catch { /* private window */ }
  }, [text, draftKey])
  useEffect(() => {
    const a = area.current; if (!a) return
    a.style.height = 'auto'
    a.style.height = Math.min(220, a.scrollHeight) + 'px'
  }, [text])

  async function submit() {
    if (!text.trim() || busy) { if (!text.trim()) { setShake(true); setTimeout(() => setShake(false), 250); area.current?.focus() } return }
    const ok = await onSubmit(text.trim(), files)
    if (ok) { setText(''); setFiles([]) }
  }

  return (
    <div className={`bg-card border line rounded-[12px] focus-within:border-strong transition-colors ${shake ? 'animate-[shake_.25s]' : ''}`}>
      <label className="sr-only" htmlFor="prompt">{placeholder}</label>
      <textarea id="prompt" data-testid="prompt" ref={area} rows={2} value={text} autoFocus={autoFocus}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void submit() } }}
                placeholder={placeholder}
                className="block w-full resize-none bg-transparent outline-none px-5 pt-4 pb-2 text-[17px] leading-[1.5] text-ink placeholder:text-muted" />
      {files.length > 0 && (
        <ul className="px-4 pb-1 flex flex-wrap gap-2" aria-label="Files to upload">
          {files.map((f, i) => (
            <li key={f.name + i} className="h-8 pl-3 pr-1 rounded-[8px] bg-elevated text-[13px] text-ink2 flex items-center gap-2">
              <span className="num">{f.name}</span>
              <span className="text-muted">{(f.size / 1024 / 1024).toFixed(f.size > 1e6 ? 1 : 2)} MB</span>
              <button type="button" onClick={() => setFiles(files.filter((_, k) => k !== i))}
                      aria-label={`Remove ${f.name}`} className="w-6 h-6 rounded-[6px] text-muted hover:text-ink hover:bg-card">×</button>
            </li>
          ))}
        </ul>
      )}
      <div className="flex items-center gap-2 px-3 pb-3 pt-1">
        {allowFiles && (
          <>
            <input ref={pick} type="file" multiple accept={ACCEPT} className="hidden"
                   onChange={(e) => { setFiles([...files, ...Array.from(e.target.files || [])]); e.target.value = '' }} />
            <button type="button" onClick={() => pick.current?.click()}
                    className="h-9 px-3 rounded-[8px] border line text-[14px] text-body hover:text-ink hover:border-strong transition-colors"
                    title="Give openPASO your own files: geometry (STEP, STL), meshes (MSH, VTU, XDMF), input decks (INP, YAML) or data (CSV). They go into the run's uploads folder and openPASO is told where they are.">
              Attach files
            </button>
          </>
        )}
        {left}
        <span className="ml-auto" />
        {hint && <span className="text-[13px] text-muted mr-2 hidden xl:inline">{hint}</span>}
        <button type="button" data-testid="send" onClick={() => void submit()} disabled={busy}
                className="h-10 px-6 rounded-[8px] bg-coral text-on-coral text-[15px] font-semibold
                           hover:bg-coral-h active:translate-y-[1px] transition-colors disabled:opacity-60">
          {busy ? 'Starting…' : submitLabel}
        </button>
      </div>
    </div>
  )
}
