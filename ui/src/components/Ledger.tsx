import { AnimatePresence, motion } from 'motion/react'
import type { Ev } from '../types'
import { rowIn } from '../motion'

/* What openPASO did, in short lines.

   The old interface printed every raw event, including JSON.stringify of the
   connect handshake. A person watching a simulation needs the sequence, not the
   wire format, so each tool call collapses to one row that changes state. */

export type Step = {
  id: string
  tool: string
  what: string
  detail: string
  state: 'running' | 'done' | 'waiting' | 'failed' | 'rejected'
}

const SHORT: Record<string, string> = {
  discover: 'Looked at the installed solvers',
  prepare_simulation: 'Read what this solver needs',
  run_simulation: 'Ran the solver',
  run_with_generator: 'Wrote the input and ran it',
  verify_mesh_independence: 'Refined the mesh and checked',
  verify_pde_consistency: 'Checked the field against its equation',
  audit_results: 'Audited its own output',
  knowledge: 'Looked up a known trap',
  generate_mesh: 'Built the mesh',
  visualize: 'Plotted the result',
  spawn_subagent: 'Asked a second model to review',
  examples: 'Opened a real example file',
  web_search: 'Searched the web',
  run_bash: 'Ran a command',
  write_file: 'Wrote the solver script',
  read_file: 'Read a file back',
  couple: 'Coupled two solvers',
  setup_backend: 'Checked an install',
}

const say = (tool?: string) => (tool && SHORT[tool]) || tool || 'Worked'

/** What a tool actually returned, in a few readable words.

   The server hands back MCP content blocks, whose repr looks like
   [{'type': 'text', 'text': '- **4C Multiphysics** ...'}]. Printing that is
   worse than printing nothing, so pull the text out and take one clean line. */
function detailOf(raw: string): string {
  let t = raw
  const m = t.match(/'text':\s*(['"])([\s\S]*?)\1\s*[,}]/)
  if (m) t = m[2]
  t = t.replace(/\\n/g, '\n')
  // A bracketed line is usually a repr fragment, but the harness reports a
  // timeout as "[timeout after 900s; ...]". Filtering that as noise made a
  // quarter hour of nothing render as a completed step.
  const timeout = t.match(/\[timeout after [^\]]+\]/)
  if (timeout) return timeout[0].replace(/^\[|\]$/g, '')
  const noise = (l: string) =>
    !l ||
    /^[[{]/.test(l) ||                    // a JSON or repr fragment
    /\\[a-zA-Z]+\{|\^\{|_\{|\\to|\\ast/.test(l) ||   // stray LaTeX
    /^#+\s*$/.test(l) ||
    /^(Traceback|File ")/.test(l)
  const line = t.split('\n').map((l) => l.trim()).find((l) => !noise(l)) || ''
  return line
    .replace(/\*\*/g, '')
    .replace(/^[-#*]+\s*/, '')
    .replace(/\\'/g, "'")                       // escaped quotes from a repr
    .replace(/\/home\/[^/\s:'"]+\//g, '~/')     // never show whose machine this is
    .slice(0, 64)
}

/** Fold the event stream into steps.

   Only things openPASO did become rows. The model's running commentary is not
   a step, and its final answer is an answer, so both are handled elsewhere. */
export function toSteps(events: Ev[]): Step[] {
  const out: Step[] = []
  const at = new Map<string, number>()
  for (const e of events) {
    const id = e.call_id || ''
    switch (e.type) {
      case 'tool_call_pending':
        at.set(id, out.length)
        out.push({ id, tool: e.tool || '', what: say(e.tool),
                   detail: 'waiting for you', state: 'waiting' })
        break
      case 'tool_call_executing': {
        const i = at.get(id)
        if (i !== undefined) { out[i].state = 'running'; out[i].detail = 'running' }
        break
      }
      case 'tool_result': {
        const i = at.get(id)
        if (i !== undefined) {
          const raw = e.result || ''
          const timedOut = /\[timeout after /.test(raw)
          out[i].state = timedOut ? 'failed' : 'done'
          out[i].detail = detailOf(raw)
        }
        break
      }
      case 'tool_error': {
        const i = at.get(id)
        if (i !== undefined) {
          out[i].state = 'failed'
          // The reason is emitted by the server and used to be thrown away.
          out[i].detail = detailOf(e.error || e.message || '') || 'failed'
        }
        break
      }
      case 'tool_call_rejected': {
        const i = at.get(id)
        if (i !== undefined) {
          out[i].state = 'rejected'
          out[i].detail = e.reason ? `you rejected this: ${e.reason}` : 'you rejected this'
        }
        break
      }
    }
  }
  // The same call repeated back to back reads as a stutter, not as two steps.
  return out.filter((s, i) => !(i && out[i - 1].what === s.what && out[i - 1].detail === s.detail))
}

/** The model's closing answer, if it gave one. */
export function finalAnswer(events: Ev[]): string {
  const msgs = events.filter((e) => e.type === 'agent_msg' && e.text?.trim())
  return msgs.length ? (msgs[msgs.length - 1].text as string).trim() : ''
}

const MARK: Record<Step['state'], string> = {
  running:  'bg-coral',
  waiting:  'bg-coral',
  done:     'bg-transparent border border-muted',
  failed:   'bg-[#D85A6F]',
  rejected: 'bg-transparent border border-[#D85A6F]',
}

export default function Ledger({
  steps, onApprove, onReject,
}: {
  steps: Step[]
  onApprove: (id: string) => void
  onReject: (id: string) => void
}) {
  return (
    <section data-testid="ledger">
      <div className="eyebrow">What openPASO did</div>
      <div className="mt-5" role="log" aria-live="polite" aria-label="Run steps">
        <AnimatePresence initial={false}>
        {steps.map((s, i) => (
          <motion.div key={s.id} {...rowIn(i)} layout
               className={`grid grid-cols-[14px_300px_1fr] items-center gap-x-4
                           min-h-[52px] py-2 ${i ? 'border-t line-soft' : ''}`}>
            <span className={`w-2 h-2 rounded-full ${MARK[s.state]}
                              ${s.state === 'running' ? 'animate-pulse' : ''}`} />
            <span className="text-[16px] text-ink2 leading-snug">{s.what}</span>
            <span className={`num text-[14px] leading-snug overflow-hidden min-w-0
                             text-ellipsis whitespace-nowrap
                             ${s.state === 'failed' || s.state === 'rejected'
                               ? 'text-[#D85A6F]' : 'text-muted'}`}>{s.detail}</span>
            {s.state === 'waiting' && (
              <span className="col-start-3 flex gap-2">
                <button onClick={() => onApprove(s.id)}
                        className="h-7 px-3.5 rounded-[6px] bg-coral text-canvas text-xs font-medium
                                   transition-colors duration-150 hover:bg-coral-h">Approve</button>
                <button onClick={() => onReject(s.id)}
                        className="h-7 px-3.5 rounded-[6px] border line text-muted text-xs font-medium
                                   transition-colors duration-150 hover:text-ink2">Reject</button>
              </span>
            )}
          </motion.div>
        ))}
        </AnimatePresence>
      </div>
    </section>
  )
}
