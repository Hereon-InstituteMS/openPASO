import type { ReactNode } from 'react'

/* Models write markdown whether or not you asked for it, so bold and code are
   rendered rather than printed. Anything else is left as plain words: a run
   summary is not a document. */
function render(text: string): ReactNode[] {
  const out: ReactNode[] = []
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g
  let last = 0
  let m: RegExpExecArray | null
  let k = 0
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index))
    const tok = m[0]
    if (tok.startsWith('**')) {
      out.push(<strong key={k++} className="font-medium text-ink">{tok.slice(2, -2)}</strong>)
    } else {
      out.push(<code key={k++} className="num text-[15px] text-muted">{tok.slice(1, -1)}</code>)
    }
    last = m.index + tok.length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

/** What the model actually said, once, at the end. Not a log line. */
export default function Answer({ text }: { text: string }) {
  if (!text) return null
  return (
    <section className="mt-12" data-testid="answer">
      <div className="eyebrow">Answer</div>
      <p className="mt-4 text-[19px] leading-[1.6] tracking-[-0.012em] text-body max-w-[65ch]">
        {render(text)}
      </p>
    </section>
  )
}
