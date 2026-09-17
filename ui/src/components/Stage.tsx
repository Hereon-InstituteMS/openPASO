import { useEffect, useRef, useState } from 'react'
import type { FieldSeries } from '../types'

/* The stage: a field the solver produced, one frame per stored timestep.

   Vorticity is signed, so the ramp diverges with the well showing through at
   zero: coral for one rotation, graphite for the other. The two peaks are
   matched in luminance (0.470 against 0.494) so neither sign visually outweighs
   the other. */
const NEG = [0xaf, 0xbc, 0xcb]
const ZERO = [0x08, 0x0b, 0x11]
const POS = [0xff, 0x9d, 0x82]
const HOLE = [0x1d, 0x25, 0x30]

function ramp() {
  const lut = new Uint8Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * 2 - 1
    const a = Math.pow(Math.abs(t), 1.1)
    const end = t < 0 ? NEG : POS
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(ZERO[c] + (end[c] - ZERO[c]) * a)
  }
  return lut
}

const b64 = (s: string) => {
  const bin = atob(s)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

type Data = {
  nx: number; ny: number
  mask: Uint8Array; frames: Uint8Array
  times: number[]; fps: number
}

export default function Stage({ series }: { series: FieldSeries }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const data = useRef<Data | null>(null)
  const lut = useRef(ramp())
  const raf = useRef(0)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [time, setTime] = useState(0)
  const [n, setN] = useState(0)

  // draw is deliberately not a hook dependency: it reads refs, so a redraw
  // never re-renders React.
  function draw(i: number) {
    const d = data.current
    const cv = canvas.current
    if (!d || !cv) return
    const ctx = cv.getContext('2d')
    if (!ctx) return
    const img = ctx.createImageData(d.nx, d.ny)
    const base = i * d.nx * d.ny
    for (let row = 0; row < d.ny; row++) {
      const src = (d.ny - 1 - row) * d.nx // the grid starts at the bottom, canvas at the top
      for (let col = 0; col < d.nx; col++) {
        const s = src + col
        const o = (row * d.nx + col) * 4
        if (!d.mask[s]) {
          img.data[o] = HOLE[0]; img.data[o + 1] = HOLE[1]
          img.data[o + 2] = HOLE[2]; img.data[o + 3] = 255
          continue
        }
        const v = d.frames[base + s] * 3
        img.data[o] = lut.current[v]
        img.data[o + 1] = lut.current[v + 1]
        img.data[o + 2] = lut.current[v + 2]
        img.data[o + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
    setTime(d.times[i] ?? 0)
  }

  useEffect(() => {
    let dead = false
    fetch(series.url)
      .then((r) => r.json())
      .then((m) => {
        if (dead) return
        data.current = {
          nx: m.nx, ny: m.ny,
          mask: b64(m.mask), frames: b64(m.frames),
          times: m.times, fps: m.fps || 25,
        }
        setN(m.times.length)
        const cv = canvas.current
        if (cv) { cv.width = m.nx; cv.height = m.ny }
        setFrame(0)
        draw(0)
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) setPlaying(false)
      })
    return () => { dead = true; cancelAnimationFrame(raf.current) }
  }, [series.url])

  /* Capture hook. Rendering a film means every frame must be a pure function
     of its number, so a recorder can park the field on an exact timestep
     instead of racing the animation. Read only; it changes nothing otherwise. */
  useEffect(() => {
    ;(window as unknown as Record<string, unknown>).__fieldFrame = (i: number) => {
      setPlaying(false); setFrame(i); draw(i)
    }
    ;(window as unknown as Record<string, unknown>).__fieldCount = () =>
      data.current?.times.length ?? 0
  }, [n])

  useEffect(() => {
    if (!playing || !n) return
    let last = 0
    let i = frame
    const tick = (now: number) => {
      const d = data.current
      if (d && now - last >= 1000 / d.fps) {
        i = (i + 1) % n
        setFrame(i)
        draw(i)
        last = now
      }
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, n])

  const prov = series.provenance
  const pct = prov?.saturated_fraction != null
    ? `${(prov.saturated_fraction * 100).toFixed(1)}%` : 'some'
  const span = series.dx && series.dy
    ? `${(series.dx * series.nx).toFixed(2)} x ${(series.dy * series.ny).toFixed(2)} m`
    : ''

  return (
    <section className="bg-well pt-12 pb-5 mt-12" data-testid="stage">
      {/* Aspect comes from the data. Forcing width and height independently
          stretched the field by an amount that changed with the window, so two
          people measuring vortex spacing off it got different numbers. */}
      <canvas ref={canvas} aria-label={series.field}
              className="block mx-auto w-auto max-w-full"
              style={{ height: 'min(420px, 42vw)',
                       aspectRatio: `${series.nx} / ${series.ny}` }} />
      <div className="w-[1224px] mx-auto mt-5 flex items-center gap-4">
        <button
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? 'Pause the field' : 'Play the field'}
          className="h-7 px-3 rounded-[6px] border line text-muted text-xs
                     transition-colors duration-150 hover:text-ink2"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <span className="num text-[13px] text-muted">t = {time.toFixed(3)} s</span>
        <input
          type="range" min={0} max={Math.max(0, n - 1)} value={frame}
          aria-label="Frame"
          onChange={(e) => {
            const i = Number(e.target.value)
            setPlaying(false); setFrame(i); draw(i)
          }}
          className="w-64 accent-coral"
        />
        <span className="ml-auto flex items-center gap-2.5 font-mono text-[13px] text-muted">
          <span>{series.vmin}</span>
          <span className="w-[132px] h-1.5 rounded-sm"
                style={{ background: 'linear-gradient(90deg,#AFBCCB,#64748B,#080B11,#C94A30,#FF9D82)' }} />
          <span>+{series.vmax}</span>
          <span>{series.unit}</span>
        </span>
      </div>

      {prov && (
        <p className="w-[1224px] mx-auto mt-4 font-mono text-[13px] text-muted leading-relaxed">
          {span && `${span} · `}
          {series.nx} x {series.ny} grid, values at cell centres.
          {typeof prov.true_min === 'number' && typeof prov.true_max === 'number' && (
            <> Field reaches {prov.true_min.toFixed(1)} to {prov.true_max.toFixed(1)} {series.unit};
              colours span {series.vmin} to {series.vmax}
              {prov.clip_percentile ? ` (${prov.clip_percentile}th percentile)` : ''},
              so {pct} of the domain is saturated and reads as a bound, not a value.</>
          )}
          {prov.quantisation_step && <> {prov.quantisation_step.toFixed(3)} {series.unit} per colour level.</>}
          {prov.solver && <> {prov.solver}.</>}
        </p>
      )}
    </section>
  )
}
