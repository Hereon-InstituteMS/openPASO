import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { EASE } from '../motion'
import type { Step } from './Ledger'

/* A run, grouped into the phases it actually works in.

   A flat list of twenty rows, fifteen of them saying "Ran a command", is
   unreadable and tells you nothing about shape. Folded into phases, a
   twenty-step run opens at five headers instead of eleven hundred pixels, and
   you can see where the time went. */

const PHASES = [
  { id: 'orient',   name: 'Understood the problem',
    tools: ['discover', 'prepare_simulation', 'examples', 'setup_backend'] },
  { id: 'research', name: 'Looked things up',
    tools: ['knowledge', 'web_search'] },
  { id: 'build',    name: 'Built the case',
    tools: ['write_file', 'generate_mesh', 'read_file', 'run_bash'] },
  { id: 'solve',    name: 'Ran the solver',
    tools: ['run_simulation', 'run_with_generator', 'couple', 'transfer_field'] },
  { id: 'verify',   name: 'Checked the answer',
    tools: ['verify_mesh_independence', 'verify_pde_consistency', 'audit_results',
            'spawn_subagent', 'visualize'] },
] as const

function phaseOf(step: Step): string {
  for (const p of PHASES) if (p.tools.includes(step.tool as never)) return p.id
  return 'build'
}

const MARK: Record<Step['state'], string> = {
  running:  'bg-coral',
  waiting:  'bg-coral',
  done:     'bg-transparent border border-muted',
  failed:   'bg-[#D85A6F]',
  rejected: 'bg-transparent border border-[#D85A6F]',
}

type Group = { id: string; name: string; steps: Step[] }

export default function Phases({
  steps, running, onApprove, onReject,
}: {
  steps: Step[]
  running: boolean
  onApprove: (id: string) => void
  onReject: (id: string) => void
}) {
  const groups: Group[] = PHASES
    .map((p) => ({ id: p.id, name: p.name,
                   steps: steps.filter((s) => phaseOf(s) === p.id) }))
    .filter((g) => g.steps.length > 0)

  // The phase that is working, or the last one, is the one worth reading.
  const live = groups.findLast((g) =>
    g.steps.some((s) => s.state === 'running' || s.state === 'waiting'))
  const trouble = groups.filter((g) =>
    g.steps.some((s) => s.state === 'failed' || s.state === 'rejected'))
  const openByDefault = new Set(
    [live?.id, ...(trouble.map((g) => g.id)),
     ...(running ? [] : [groups[groups.length - 1]?.id])].filter(Boolean) as string[])

  const [open, setOpen] = useState<Set<string> | null>(null)
  const shown = open ?? openByDefault
  const toggle = (id: string) => {
    const next = new Set(shown)
    next.has(id) ? next.delete(id) : next.add(id)
    setOpen(next)
  }

  return (
    <section data-testid="ledger">
      <div className="eyebrow">What openPASO did</div>

      <div className="mt-5" role="log" aria-live="polite" aria-label="Run steps">
        {groups.map((g) => {
          const isOpen = shown.has(g.id)
          const failed = g.steps.some((s) => s.state === 'failed')
          const busy = g.steps.some((s) => s.state === 'running' || s.state === 'waiting')
          return (
            <div key={g.id} className="border-t line-soft first:border-t-0">
              <button onClick={() => toggle(g.id)} aria-expanded={isOpen}
                      className="w-full flex items-center gap-4 h-14 text-left
                                 transition-colors duration-150 group">
                <span className={`w-2 h-2 rounded-full shrink-0 ${
                  failed ? 'bg-[#D85A6F]'
                    : busy ? 'bg-coral animate-pulse'
                    : 'bg-transparent border border-muted'}`} />
                <span className="text-[16px] text-ink2">{g.name}</span>
                <span className="num text-[13px] text-muted ml-auto">
                  {g.steps.length} {g.steps.length === 1 ? 'step' : 'steps'}
                </span>
                <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"
                     className={`text-muted transition-transform duration-200 ${
                       isOpen ? 'rotate-90' : ''}`}>
                  <path d="M4 2l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" />
                </svg>
              </button>

              <AnimatePresence initial={false}>
                {isOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.22, ease: EASE }}
                    className="overflow-hidden">
                    <div className="pb-3 pl-6">
                      {g.steps.map((s, i) => (
                        <div key={s.id}
                             className={`grid grid-cols-[10px_280px_1fr] items-center gap-x-4
                                         min-h-[36px] py-1.5
                                         ${i ? 'border-t line-soft' : ''}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${MARK[s.state]}
                            ${s.state === 'running' ? 'animate-pulse' : ''}`} />
                          <span className="text-[14px] text-body leading-snug">{s.what}</span>
                          <span className={`num text-[13px] leading-snug min-w-0 overflow-hidden
                            text-ellipsis whitespace-nowrap
                            ${s.state === 'failed' || s.state === 'rejected'
                              ? 'text-[#D85A6F]' : 'text-muted'}`}
                                title={s.detail}>{s.detail}</span>
                          {s.state === 'waiting' && (
                            <span className="col-start-3 flex gap-2 mt-1">
                              <button onClick={() => onApprove(s.id)}
                                className="h-7 px-3.5 rounded-[6px] bg-coral text-canvas
                                           text-[13px] font-medium">Approve</button>
                              <button onClick={() => onReject(s.id)}
                                className="h-7 px-3.5 rounded-[6px] border line text-muted
                                           text-[13px] font-medium">Reject</button>
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )
        })}
      </div>
    </section>
  )
}
