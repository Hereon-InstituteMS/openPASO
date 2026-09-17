import { useEffect, useRef, useState } from 'react'
import { DUR } from '../motion'

/* A number arriving at its value.

   Residuals span many orders of magnitude, so the interpolation runs in log
   space: counting linearly from 1.0 to 4.2e-9 would spend almost the whole
   animation displaying zero. */
export default function Counter({
  to, decimals = 3, log = false, className = '',
}: { to: number; decimals?: number; log?: boolean; className?: string }) {
  const [v, setV] = useState(to)
  const raf = useRef(0)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setV(to); return }
    const from = log ? Math.log10(Math.max(v, 1e-30)) : v
    const dest = log ? Math.log10(Math.max(to, 1e-30)) : to
    const t0 = performance.now()
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / (DUR.count * 1000))
      const e = 1 - Math.pow(1 - p, 4)          // out-expo feel, settles hard
      const cur = from + (dest - from) * e
      setV(log ? Math.pow(10, cur) : cur)
      if (p < 1) raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [to])

  const text = log
    ? v.toExponential(2)
    : v.toLocaleString('en-US', { minimumFractionDigits: decimals,
                                  maximumFractionDigits: decimals })
  return <span className={`num ${className}`}>{text}</span>
}
