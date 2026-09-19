import 'katex/dist/katex.min.css'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { tidy } from '../api'

/* Models answer in markdown with LaTeX maths: lists, tables, code, $…$ and $$…$$.
   Formulas used to print as source, and a "\n" inside "\nabla" was turned into a
   line break on the way. */
export default function Markdown({ text, className = '' }: { text: string; className?: string }) {
  // models also write \( … \) and \[ … \]; remark-math reads dollars only
  const src = tidy(text)
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => `\n$$\n${m}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => `$${m}$`)
  return (
    <div className={`md ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]}
                     rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: 'ignore' }]]}
                     components={{ a: (p) => <a {...p} target="_blank" rel="noreferrer" /> }}>
        {src}
      </ReactMarkdown>
    </div>
  )
}
