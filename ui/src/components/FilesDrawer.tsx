import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { FileRow } from '../types'
import FileView from './FileView'

function size(n: number | null) {
  if (n == null) return ''
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 ** 2).toFixed(1)} MB`
}

/* This run's folder: what it wrote, and what you gave it. It used to open at the
   root of the whole sandbox, listing every run's directory side by side. */
export default function FilesDrawer({ runId, open, onClose, refreshKey }: {
  runId: string; open: boolean; onClose: () => void; refreshKey: number
}) {
  const [sub, setSub] = useState('')
  const [rows, setRows] = useState<FileRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [viewing, setViewing] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const pick = useRef<HTMLInputElement>(null)
  const panel = useRef<HTMLDivElement>(null)

  useEffect(() => { setSub('') }, [runId])
  useEffect(() => {
    if (!open) return
    setRows(null); setError(null)
    api.files(runId, sub).then((d) => setRows(d.entries)).catch((e) => setError(String(e.message || e)))
  }, [runId, sub, open, refreshKey])
  useEffect(() => {
    if (!open) return
    const returnTo = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { if (viewing) setViewing(null); else onClose(); return }
      if (e.key !== 'Tab' || viewing) return
      // a drawer you can tab out of leaves the keyboard on the page behind it
      const inside = panel.current?.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])')
      if (!inside?.length) return
      const first = inside[0], last = inside[inside.length - 1]
      const here = document.activeElement
      if (!e.shiftKey && (here === last || here === panel.current)) { e.preventDefault(); first.focus() }
      else if (e.shiftKey && (here === first || here === panel.current)) { e.preventDefault(); last.focus() }
    }
    document.addEventListener('keydown', key)
    return () => {
      document.removeEventListener('keydown', key)
      returnTo?.focus?.()                 // back to the button that opened it
    }
  }, [open, onClose, viewing])

  async function upload(files: File[]) {
    if (!files.length) return
    setUploading(true); setError(null)
    try {
      await api.upload(runId, files)
      setSub('uploads')
      const d = await api.files(runId, 'uploads'); setRows(d.entries)
    } catch (e) { setError(String((e as Error).message || e)) }
    finally { setUploading(false) }
  }

  if (!open) return null
  const parts = sub ? sub.split('/') : []
  return (
    <>
      <div className="absolute inset-0 z-20 bg-black/40" onClick={onClose} aria-hidden />
      <aside ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-label="Files of this run"
             className="absolute top-0 right-0 bottom-0 z-30 w-[480px] max-w-full bg-elevated border-l border-strong flex flex-col outline-none">
        <div className="h-16 px-5 flex items-center gap-3 border-b line">
          <h2 className="text-[17px] font-semibold text-ink">Files</h2>
          <span className="text-[13px] text-muted">this run's folder</span>
          <button onClick={onClose} className="ml-auto h-9 px-3 rounded-[8px] text-[14px] text-body hover:text-ink hover:bg-card">Close</button>
        </div>
        <div className="px-5 py-3 flex items-center gap-1 flex-wrap text-[14px] border-b line">
          <button onClick={() => setSub('')} className="text-body hover:text-ink">run folder</button>
          {parts.map((p, i) => (
            <span key={i} className="flex items-center gap-1">
              <span className="text-faint">/</span>
              <button onClick={() => setSub(parts.slice(0, i + 1).join('/'))} className="num text-body hover:text-ink">{p}</button>
            </span>
          ))}
          <input ref={pick} type="file" multiple className="hidden"
                 onChange={(e) => { void upload(Array.from(e.target.files || [])); e.target.value = '' }} />
          <button onClick={() => pick.current?.click()} disabled={uploading}
                  className="ml-auto h-8 px-3 rounded-[8px] border line text-[13px] text-body hover:text-ink hover:border-strong disabled:opacity-60">
            {uploading ? 'Uploading…' : 'Upload files'}
          </button>
        </div>
        <div className="flex-1 overflow-y-auto scroll">
          {error && <p className="px-5 py-3 text-[14px] text-bad">{error}</p>}
          {!rows && !error && <p className="px-5 py-3 text-[14px] text-muted">Loading…</p>}
          {rows && rows.length === 0 && (
            <p className="px-5 py-4 text-[14px] text-muted">
              {sub ? 'This folder is empty.' : 'Nothing here yet. Files appear as the run writes them, and uploads go into uploads/.'}
            </p>
          )}
          <ul>
            {rows?.map((f) => (
              <li key={f.rel_path}>
                <button onClick={() => (f.is_dir ? setSub(f.sub) : setViewing(f.rel_path))}
                        className="w-full text-left px-5 py-2.5 flex items-center gap-3 hover:bg-card">
                  <span aria-hidden className="w-4 text-muted text-[13px]">{f.is_dir ? '▸' : ''}</span>
                  <span className={`num text-[14px] truncate ${f.is_dir ? 'text-ink' : 'text-ink2'}`}>{f.name}{f.is_dir ? '/' : ''}</span>
                  <span className="ml-auto num text-[12px] text-muted shrink-0">{size(f.size)}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div className="px-5 py-3 border-t line">
          <a href={api.manifestUrl(runId)} download={`openpaso-run-${runId}.json`}
             className="text-[14px] text-body hover:text-ink underline underline-offset-4">
            Download the run record
          </a>
          <p className="mt-1 text-[12px] text-muted">Prompt, model, every event, and a checksum for every file.</p>
        </div>
      </aside>
      {viewing && <FileView rel={viewing} onClose={() => setViewing(null)} />}
    </>
  )
}
