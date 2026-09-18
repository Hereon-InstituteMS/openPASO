import { useEffect, useRef, type ReactNode } from 'react'

/* A panel anchored to a control. Closes on Escape and on a click outside, and
   hands focus back to the control that opened it. */
export default function Popover({ open, onClose, children, align = 'left', up = false, width = 420 }: {
  open: boolean; onClose: () => void; children: ReactNode
  align?: 'left' | 'right'; up?: boolean; width?: number
}) {
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const opener = document.activeElement as HTMLElement | null
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }
    const down = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)
          && !(opener && opener.contains(e.target as Node))) onClose()
    }
    document.addEventListener('keydown', key)
    document.addEventListener('mousedown', down)
    return () => {
      document.removeEventListener('keydown', key)
      document.removeEventListener('mousedown', down)
      opener?.focus?.()
    }
  }, [open, onClose])
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
