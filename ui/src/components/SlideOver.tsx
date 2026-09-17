import type { FileRow, Session } from '../types'

/* The machinery. Reachable, and no longer competing with the result. */
export default function SlideOver({
  open, onClose, session, models, modes, files, onModel, onMode, onOpenFile, cwd, onUp,
}: {
  open: boolean
  onClose: () => void
  session: Session | null
  models: { id: string; label: string }[]
  modes: string[]
  files: FileRow[]
  cwd: string
  onModel: (m: string) => void
  onMode: (m: string) => void
  onOpenFile: (f: FileRow) => void
  onUp: () => void
}) {
  return (
    <>
      <div
        onClick={onClose} aria-hidden={!open}
        className={`fixed inset-0 bg-black/50 transition-opacity duration-200
                    ${open ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
      />
      <aside
        data-testid="slideover"
        aria-hidden={!open}
        className={`fixed top-0 right-0 h-full w-[420px] bg-s2 border-l line z-10
                    overflow-y-auto scroll p-8 transition-transform duration-[260ms]
                    ${open ? 'translate-x-0' : 'translate-x-full'}`}
        style={{ transitionTimingFunction: 'var(--ease-out)' }}
      >
        <div className="flex items-center">
          <span className="eyebrow">Session</span>
          <button onClick={onClose} aria-label="Close"
                  className="ml-auto text-muted text-sm hover:text-ink2">Close</button>
        </div>
        <div className="num text-[13px] text-muted mt-2">{session?.id ?? ''}</div>

        <div className="eyebrow mt-8">Model</div>
        <select value={session?.model ?? ''} onChange={(e) => onModel(e.target.value)}
                aria-label="Model"
                className="mt-3 w-full h-10 px-3 bg-s1 border line rounded-[6px]
                           text-sm text-ink2">
          {models.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>

        <div className="eyebrow mt-8">Mode</div>
        <div className="mt-3 grid grid-cols-3 gap-2">
          {modes.map((m) => (
            <button key={m} onClick={() => onMode(m)}
                    className={`h-9 rounded-[6px] text-[13px] capitalize transition-colors duration-150
                      ${session?.mode === m
                        ? 'bg-s3 text-coral font-medium' : 'border line text-muted hover:text-ink2'}`}>
              {m}
            </button>
          ))}
        </div>

        <div className="eyebrow mt-8 flex items-center">
          <span>Files</span>
          <button onClick={onUp} className="ml-auto normal-case tracking-normal
                    font-sans text-[13px] text-muted hover:text-ink2">Up</button>
        </div>
        <div className="num text-[13px] text-muted mt-2">/{cwd}</div>
        <div className="mt-3">
          {files.length === 0 && <div className="text-sm text-graphit italic">empty</div>}
          {files.map((f) => (
            <button key={f.rel_path} onClick={() => onOpenFile(f)}
                    className="w-full flex items-center gap-3 h-9 text-left
                               border-t line-soft first:border-t-0
                               transition-colors duration-150 hover:text-ink2">
              <span className="text-[13px] text-body">{f.is_dir ? f.name + '/' : f.name}</span>
              <span className="num text-[13px] text-muted ml-auto">
                {f.is_dir ? '' : `${(f.size / 1024).toFixed(1)} KB`}
              </span>
            </button>
          ))}
        </div>

        <div className="eyebrow mt-8">Tokens</div>
        <div className="mt-3 flex gap-6 num text-[13px] text-muted">
          <span>in {session?.tokens_in ?? 0}</span>
          <span>out {session?.tokens_out ?? 0}</span>
        </div>
      </aside>
    </>
  )
}
