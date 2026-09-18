import { useEffect, useMemo, useState } from 'react'
import { fileUrl } from '../api'

/* Whatever the run wrote, readable.

   Clicking a file used to be a silent no-op unless the payload happened to be a
   field animation, so the script the agent wrote, the mesh, the log and the
   tables were listed and unopenable. The bytes were always served; nothing
   asked for them. */

type Viz =
  | { kind: 'text'; text: string; syntax?: string }
  | { kind: 'json'; obj: unknown }
  | { kind: 'table'; rows: string[][]; header: string[] }
  | { kind: 'image'; rel: string }
  | { kind: 'error'; error: string }
  | { kind: string; [k: string]: unknown }

/** A plain line chart. No library, no chart chrome, one series per column. */
function Chart({ header, rows }: { header: string[]; rows: string[][] }) {
  const series = useMemo(() => {
    const xs: number[] = []
    const cols: number[][] = header.slice(1).map(() => [])
    for (const r of rows) {
      const x = Number(r[0])
      if (!Number.isFinite(x)) continue
      xs.push(x)
      header.slice(1).forEach((_, i) => cols[i].push(Number(r[i + 1])))
    }
    return { xs, cols }
  }, [header, rows])

  // a line only means something along an ordered axis; a table of x, y, value
  // points drawn as a line is a meaningless zig-zag
  const ordered = series.xs.every((x, i) => i === 0 || x >= series.xs[i - 1])
  if (series.xs.length < 2 || !ordered || new Set(series.xs).size < series.xs.length * 0.9) return null
  const W = 640, H = 240, P = 32
  const xmin = Math.min(...series.xs), xmax = Math.max(...series.xs)
  const all = series.cols.flat().filter(Number.isFinite)
  const ymin = Math.min(...all), ymax = Math.max(...all)
  const sx = (v: number) => P + (v - xmin) / (xmax - xmin || 1) * (W - 2 * P)
  const sy = (v: number) => H - P - (v - ymin) / (ymax - ymin || 1) * (H - 2 * P)
  const stroke = ['#FF6B4A', '#94A3B8', '#AFBCCB']

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full mt-4" role="img"
         aria-label={`${header.slice(1).join(', ')} against ${header[0]}`}>
      <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="rgb(100 116 139 / .28)" />
      <line x1={P} y1={P} x2={P} y2={H - P} stroke="rgb(100 116 139 / .28)" />
      {series.cols.map((col, i) => (
        <polyline key={i} fill="none" strokeWidth="1.5" stroke={stroke[i % stroke.length]}
          points={col.map((v, j) => `${sx(series.xs[j])},${sy(v)}`).join(' ')} />
      ))}
      <text x={P} y={H - 10} fill="#94A3B8" fontSize="11" fontFamily="monospace">
        {header[0]}
      </text>
      <text x={P} y={P - 10} fill="#94A3B8" fontSize="11" fontFamily="monospace">
        {ymax.toPrecision(4)}
      </text>
    </svg>
  )
}

export default function FileView({ rel, onClose }: { rel: string; onClose: () => void }) {
  const [viz, setViz] = useState<Viz | null>(null)

  useEffect(() => {
    // opening files quickly could show the first file's contents under the
    // second file's name, whichever answer happened to arrive last
    setViz(null)
    const stop = new AbortController()
    fetch(`/api/viz?rel=${encodeURIComponent(rel)}`, { signal: stop.signal })
      .then((r) => r.json())
      .then(setViz)
      .catch((e) => { if (!stop.signal.aborted) setViz({ kind: 'error', error: String(e) }) })
    return () => stop.abort()
  }, [rel])

  const name = rel.split('/').pop() || rel

  return (
    <div className="fixed inset-0 z-30 flex" role="dialog" aria-modal="true" aria-label={name}>
      <div className="flex-1 bg-black/55" onClick={onClose} />
      <aside className="w-[760px] h-full bg-card border-l line overflow-y-auto scroll p-8
                        overscroll-contain">
        <div className="flex items-center gap-4">
          <span className="num text-[14px] text-ink2 min-w-0 overflow-hidden
                           text-ellipsis whitespace-nowrap">{name}</span>
          <a href={fileUrl(rel)} download
             className="ml-auto h-8 px-3.5 rounded-[6px] border line text-[13px] text-muted
                        grid place-items-center transition-colors duration-150
                        hover:text-ink2">
            Download
          </a>
          <button onClick={onClose}
                  className="h-8 px-3.5 rounded-[6px] border line text-[13px] text-muted
                             transition-colors duration-150 hover:text-ink2">Close</button>
        </div>
        <div className="num text-[13px] text-muted mt-2">{rel}</div>

        <div className="mt-6">
          {!viz && <div className="text-[14px] text-muted">Opening…</div>}

          {viz?.kind === 'error' && (
            <div className="text-[14px] text-[#D85A6F]">{String(viz.error)}</div>
          )}

          {viz?.kind === 'text' && (
            <pre className="num text-[13px] leading-[1.65] text-body whitespace-pre-wrap
                            break-words">{(viz as { text: string }).text}</pre>
          )}

          {viz?.kind === 'json' && (
            <pre className="num text-[13px] leading-[1.65] text-body whitespace-pre-wrap">
              {JSON.stringify((viz as { obj: unknown }).obj, null, 2)}
            </pre>
          )}

          {viz?.kind === 'table' && (
            <>
              <Chart header={(viz as { header: string[] }).header}
                     rows={(viz as { rows: string[][] }).rows} />
              <table className="w-full mt-6 num text-[13px]">
                <thead>
                  <tr>{(viz as { header: string[] }).header.map((h) => (
                    <th key={h} className="text-left text-muted font-normal pb-2
                                           border-b line-soft">{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {(viz as { rows: string[][] }).rows.slice(0, 40).map((r, i) => (
                    <tr key={i}>{r.map((c, j) => (
                      <td key={j} className="text-body py-1 border-b line-soft">{c}</td>
                    ))}</tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {viz?.kind === 'image' && (
            <img src={fileUrl(rel)} alt={name}
                 className="max-w-full rounded-[6px] border line" />
          )}

          {viz?.kind === 'vtk' && (
            <div className="text-[15px] leading-[1.55] text-body">
              <p>A mesh or result file in {String(viz.format || 'VTK').toUpperCase()} format. openPASO does not draw it here.</p>
              <p className="mt-2 text-muted">Download it and open it in ParaView, VisIt or PyVista. A run can also write a picture of a field, which is shown on the run page.</p>
            </div>
          )}

          {viz?.kind === 'hdf' && (
            <div className="text-[15px] leading-[1.55] text-body">
              <p>A data file in HDF5 format. openPASO does not draw it here.</p>
              {Array.isArray(viz.keys) && viz.keys.length > 0 && (
                <p className="num mt-2 text-[14px] text-muted break-all">It contains: {(viz.keys as string[]).join(', ')}</p>
              )}
              <p className="mt-2 text-muted">Download it and open it in ParaView or h5py.</p>
            </div>
          )}

          {viz && !['text', 'json', 'table', 'image', 'error', 'vtk', 'hdf'].includes(viz.kind) && (
            <div className="text-[15px] text-body">
              openPASO cannot show a {viz.kind} file here. Download it and open it in a program that reads it.
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}
