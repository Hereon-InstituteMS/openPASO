import { useCallback, useEffect, useRef, useState } from 'react'
import { socket } from './api'
import type { Ev, Session } from './types'

export type Link = 'connecting' | 'live' | 'lost' | 'missing'

export function useRun(id: string) {
  const [session, setSession] = useState<Session | null>(null)
  const [events, setEvents] = useState<Ev[]>([])
  const [link, setLink] = useState<Link>('connecting')
  const [notice, setNotice] = useState<string | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const buffer = useRef<Ev[]>([])
  const frame = useRef(0)

  useEffect(() => {
    let dead = false
    let retry = 0
    let timer = 0
    setSession(null); setEvents([]); setLink('connecting'); setNotice(null)

    // token chunks arrive many times a second; publishing them once per frame
    // keeps a long run from re-rendering the whole transcript on every word
    const flush = () => {
      frame.current = 0
      const add = buffer.current.splice(0)
      if (add.length) setEvents((p) => p.concat(add))
    }

    const open = () => {
      const s = socket(id)
      ws.current = s
      s.onopen = () => { retry = 0 }
      s.onmessage = (m) => {
        // a socket that has been replaced can still deliver what it had
        // queued; without this its events land in the run now on screen
        if (dead || ws.current !== s) return
        let e: Ev & { session?: Session; events?: Ev[] }
        try { e = JSON.parse(m.data) } catch { return }
        if (e.type === 'hello') {
          // the replay is the whole truth; anything buffered from the socket
          // that just closed is in it already and would be shown twice
          buffer.current = []
          if (frame.current) { cancelAnimationFrame(frame.current); frame.current = 0 }
          setSession(e.session!); setEvents(e.events || []); setLink('live')
          return
        }
        if (e.type === 'session') { setSession(e.session!); return }
        if (e.type === 'notice') { setNotice(e.message || null); return }
        if (e.type === 'error' && !e.seq && /no such run/.test(e.message || '')) {
          setLink('missing'); dead = true; return
        }
        if (e.type === 'status') return
        buffer.current.push(e)
        if (!frame.current) frame.current = requestAnimationFrame(flush)
      }
      s.onclose = () => {
        if (dead) return
        // the run keeps working on the server; this is only our view of it
        setLink('lost')
        timer = window.setTimeout(open, Math.min(8000, 800 * 2 ** retry++))
      }
    }
    open()
    return () => {
      dead = true
      clearTimeout(timer)
      if (frame.current) cancelAnimationFrame(frame.current)
      buffer.current = []
      ws.current?.close()
    }
  }, [id])

  const send = useCallback((msg: Record<string, unknown>) => {
    const s = ws.current
    if (s && s.readyState === WebSocket.OPEN) { s.send(JSON.stringify(msg)); return true }
    setNotice('Not connected to the server right now. Reconnecting…')
    return false
  }, [])

  return { session, events, link, notice, setNotice, send }
}
