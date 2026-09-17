import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { heroIn } from '../motion'
import Logo from './Logo'

type Solver = { name: string; status: string; version: string | null; physics: number }

/* What is installed, asked rather than asserted.

   This row used to be nine hardcoded names above the words "Nine solvers
   online". On a machine with none installed it still said nine. */
function useSolvers() {
  const [state, setState] = useState<{ ok: boolean; solvers: Solver[]; error?: string }>(
    { ok: true, solvers: [] })
  useEffect(() => {
    fetch('/api/solvers').then((r) => r.json()).then(setState)
      .catch((e) => setState({ ok: false, solvers: [], error: String(e) }))
  }, [])
  return state
}

const EXAMPLES = [
  'Flow past a cylinder at Reynolds 100',
  'Heat through a wall with a steel stud',
  'A bracket under load, and check the mesh',
]

export default function Idle({
  value, onChange, onSend, busy,
}: {
  value: string
  onChange: (v: string) => void
  onSend: () => void
  busy: boolean
}) {
  const { ok, solvers, error } = useSolvers()
  const checked = solvers.length > 0 || !ok
  const ready = solvers.filter((s) => s.status === 'available')
  const [shake, setShake] = useState(false)

  function submit() {
    if (!value.trim()) {
      // A dead control is worse than a live one that says no.
      setShake(true); setTimeout(() => setShake(false), 200)
      return
    }
    onSend()
  }

  return (
    <section className="pt-[120px] pb-24" data-testid="idle">
      <div className="w-[1160px] mx-auto">

        <div className="grid grid-cols-[1fr_320px] gap-16 items-center min-h-[380px]">
          <div>
            <motion.div className="eyebrow" {...heroIn(0)}>
              {!checked ? 'Checking what is installed'
                : !ok ? 'Could not check what is installed'
                : `${ready.length} of ${solvers.length} solvers ready`}
            </motion.div>

            <motion.h1 {...heroIn(1)}
              className="mt-6 text-[72px] font-medium tracking-[-0.028em]
                         leading-[1.02] text-ink">
              Describe the physics.
            </motion.h1>

            <motion.p {...heroIn(2)}
              className="mt-6 max-w-[46ch] text-[19px] leading-[1.55]
                         tracking-[-0.014em] text-body">
              We choose the solver, write its input, and run it for real. Every
              step it takes is shown to you.
            </motion.p>
          </div>

          <motion.div {...heroIn(3)} className="justify-self-center">
            <Logo size={280} dim={value.length > 0} />
          </motion.div>
        </div>

        <motion.div
          initial={heroIn(4).initial}
          animate={{ ...heroIn(4).animate, x: shake ? [0, -5, 5, -3, 0] : 0 }}
          transition={shake ? { duration: 0.2 } : heroIn(4).transition}
          className="mt-10 w-full h-16 bg-s2 border line rounded-[10px] edge-top
                     flex items-center pl-5 pr-2 focus-within:border-coral
                     transition-colors duration-150">
          <label htmlFor="prompt" className="sr-only">Describe a physics problem</label>
          <input
            id="prompt" data-testid="prompt" value={value} autoComplete="off"
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) submit() }}
            placeholder="A steel bracket under load, and tell me if I can trust the answer…"
            className="flex-1 bg-transparent outline-none text-[17px] tracking-[-0.011em]
                       text-ink2 placeholder:text-[#5A6673]"
          />
          <button onClick={submit} disabled={busy} data-testid="send"
            className="h-10 px-5 rounded-[6px] bg-coral text-canvas text-[14px] font-medium
                       transition-colors duration-150 hover:bg-coral-h">
            Run
          </button>
        </motion.div>

        <motion.div {...heroIn(5)} className="mt-5 flex flex-wrap gap-3">
          {EXAMPLES.map((e) => (
            <button key={e} onClick={() => onChange(e)}
              className="h-9 px-4 rounded-[6px] border line text-[14px] text-body
                         transition-colors duration-150 hover:text-ink2 hover:border-strong">
              {e}
            </button>
          ))}
        </motion.div>

        <motion.div {...heroIn(6)} className="mt-20 pt-8 border-t line-soft">
          <div className="flex flex-wrap gap-x-8 gap-y-4 font-mono text-[14px]">
            {!checked && <span className="text-muted">checking…</span>}
            {checked && !ok && (
              <span className="text-[#D85A6F]">{error || 'the check failed'}</span>
            )}
            {solvers.map((s) => (
              <span key={s.name} className="flex items-center gap-2.5"
                    title={s.version || s.status}>
                <span className={`w-1.5 h-1.5 rounded-full ${
                  s.status === 'available'
                    ? 'bg-muted' : 'bg-transparent border border-muted'}`} />
                <span className={s.status === 'available' ? 'text-body' : 'text-muted'}>
                  {s.name}
                </span>
              </span>
            ))}
          </div>
        </motion.div>

      </div>
    </section>
  )
}
