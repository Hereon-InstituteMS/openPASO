import { useEffect, useRef } from 'react'

/* The brand mark, animated, from the logo package's own module.

   It redraws each frame at the screen's pixel density, so it stays sharp at any
   size, and it holds a still frame under prefers-reduced-motion by itself.

   Two deliberate choices. It is mounted in graphite, not coral: the accent is
   spent on the one primary action, and a coral sphere would outshout it. And it
   turns at half speed, because anything quicker reads as a loading spinner on a
   screen where there is nothing to wait for. */

declare global {
  interface Window {
    openPASOLogo?: {
      mount: (sel: string | HTMLElement, opts?: {
        farbe?: string; tempo?: number; dichte?: 'fine' | 'coarse'
      }) => { pause: () => void; play: () => void }
    }
  }
}

let loading: Promise<void> | null = null

function load(): Promise<void> {
  if (window.openPASOLogo) return Promise.resolve()
  if (!loading) {
    loading = new Promise<void>((resolve, reject) => {
      const s = document.createElement('script')
      s.src = '/static/openpaso-logo.js'
      s.onload = () => resolve()
      s.onerror = () => reject(new Error('logo script did not load'))
      document.head.appendChild(s)
    })
  }
  return loading
}

export default function Logo({ size = 280, dim = false }: { size?: number; dim?: boolean }) {
  const host = useRef<HTMLDivElement>(null)
  const handle = useRef<{ pause: () => void; play: () => void } | null>(null)

  useEffect(() => {
    let dead = false
    load().then(() => {
      if (dead || !host.current || !window.openPASOLogo) return
      host.current.innerHTML = ''
      handle.current = window.openPASOLogo.mount(host.current, {
        farbe: '#64748B', tempo: 0.5, dichte: 'fine',
      })
    }).catch(() => { /* the page is fine without it */ })
    return () => { dead = true; handle.current?.pause() }
  }, [])

  // While someone is writing physics, the screen belongs to them.
  useEffect(() => {
    if (dim) handle.current?.pause()
    else handle.current?.play()
  }, [dim])

  return (
    <div ref={host} aria-hidden="true" data-testid="logo"
         style={{ width: size, height: size,
                  opacity: dim ? 0.5 : 1,
                  transition: 'opacity 200ms cubic-bezier(.23,1,.32,1)' }} />
  )
}
