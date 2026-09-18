import { useEffect, useRef, type ReactNode } from 'react'

/* A panel anchored to a control. Closes on Escape and on a click outside, and
   hands focus back to the control that opened it. */
export default function Popover({ open, onClose, children, align = 'left', up = false, width = 420 }: {
  open: boolean; onClose: () => void; children: ReactNode
  align?: 'left' | 'right'; up?: boolean; width?: number
}) {
  const box = useRef<HTMLDivElement>(null)
  // The pickers pass a new onClose on every render of their parent, and a run
  // view re-renders once a second while a run works. Depending on it here meant
  // this effect tore down and set up every second, handing focus back to the
  // opener mid-click. Only opening and closing may run it.
  const close = useRef(onClose)
  close.current = onClose
  useEffect(() => {
    if (!open) return
    const opener = document.activeElement as HTMLElement | null
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); close.current() } }
    const down = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)
          && !(opener && opener.contains(e.target as Node))) close.current()
    }
    document.addEventListener('keydown', key)
    document.addEventListener('mousedown', down)
    return () => {
      document.removeEventListener('keydown', key)
      document.removeEventListener('mousedown', down)
      opener?.focus?.()
    }
  }, [open])
  if (!open) return null
  return (
    <div ref={box} role="dialog"
         style={{ width }}
         className={`absolute z-40 ${up ? 'bottom-full mb-2' : 'top-full mt-2'} ${align === 'right' ? 'right-0' : 'left-0'}
                     max-h-[70vh] overflow-y-auto scroll bg-elevated border border-strong rounded-[12px] p-2
                     shadow-[0_16px_48px_rgba(0,0,0,0.5)]`}>
      {children}
    </div>
  )
}
