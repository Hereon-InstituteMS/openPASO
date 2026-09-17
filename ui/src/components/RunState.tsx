import { useState } from 'react'

/* What happened to this run, stated.

   A run ends in exactly one of three states and the person is told which. The
   interface used to render nothing at all for a failure: the server emitted a
   correct, actionable message and no code existed to receive it, so a crashed
   run and a clean one were the same screen. */

export type Outcome = 'running' | 'completed' | 'failed' | 'interrupted'

/** Turn an exception into a sentence a person can act on. */
function advise(message: string): string | null {
  const m = message.toLowerCase()
  if (m.includes('no openrouter key'))
    return 'Copy .env.example to .env and paste your key after OPENROUTER_API_KEY=, then run again.'
  if (m.includes('claude code is not on path'))
    return 'Install Claude Code, or choose a model that uses an API key.'
  if (m.includes('claude code exited'))
    return 'Check that you are signed in to Claude Code, then run again.'
  if (m.includes('connection') || m.includes('connect') || m.includes('refused'))
    return 'That model is served locally and nothing is listening. Start it, or pick a different model.'
  if (m.includes('recursionlimit') || m.includes('recursion limit'))
    return 'The run used all its steps without finishing. Ask for a smaller piece of the problem.'
  if (m.includes('timeout'))
    return 'The step ran out of time. A coarser mesh or a shorter end time will finish.'
  return null
}

export default function RunState({
  outcome, message, traceback, elapsed, onStop,
}: {
  outcome: Outcome
  message?: string
  traceback?: string
  elapsed: number
  onStop?: () => void
}) {
  const [open, setOpen] = useState(false)
  const mins = Math.floor(elapsed / 60)
  const secs = Math.floor(elapsed % 60)
  const clock = mins ? `${mins}m ${String(secs).padStart(2, '0')}s` : `${secs}s`

  if (outcome === 'running') {
    return (
      <div className="flex items-center gap-4" data-testid="runstate">
        <span className="w-2 h-2 rounded-full bg-coral" />
        <span className="text-[15px] text-ink2">Working</span>
        {elapsed > 0 && <span className="num text-[14px] text-muted">{clock}</span>}
        {onStop && (
          <button onClick={onStop}
                  className="ml-2 h-8 px-3.5 rounded-[6px] border line text-muted text-[13px]
                             transition-colors duration-150 hover:text-ink2">
            Stop
          </button>
        )}
      </div>
    )
  }

  if (outcome === 'completed') {
    return (
      <div className="flex items-center gap-4" data-testid="runstate">
        <span className="w-2 h-2 rounded-full border border-muted" />
        <span className="text-[15px] text-ink2">Finished</span>
        {elapsed > 0 && <span className="num text-[14px] text-muted">{clock}</span>}
      </div>
    )
  }

  const stopped = outcome === 'interrupted'
  const fix = message ? advise(message) : null

  return (
    <div data-testid="runstate"
         className={`border rounded-[10px] p-6 ${stopped
           ? 'border-line bg-s2' : 'border-[#D85A6F]/40 bg-[#D85A6F]/[0.06]'}`}>
      <div className="flex items-center gap-3">
        <span className={`w-2 h-2 rounded-full ${stopped ? 'bg-muted' : 'bg-[#D85A6F]'}`} />
        <span className="text-[17px] text-ink2">
          {stopped ? 'You stopped this run.' : 'This run did not finish.'}
        </span>
        {elapsed > 0 && <span className="num text-[14px] text-muted ml-auto">{clock}</span>}
      </div>

      {!stopped && (
        <>
          <p className="num text-[14px] text-body mt-4 break-words">{message}</p>
          {fix && <p className="text-[15px] text-ink2 mt-3">{fix}</p>}
          <p className="text-[14px] text-muted mt-4">
            Nothing below this point is a result.
          </p>
          {traceback && (
            <>
              <button onClick={() => setOpen((o) => !o)}
                      aria-expanded={open}
                      className="mt-4 text-[13px] text-muted underline underline-offset-4
                                 transition-colors duration-150 hover:text-ink2">
                {open ? 'Hide the detail' : 'Show the detail'}
              </button>
              {open && (
                <pre className="num text-[13px] text-muted mt-3 p-4 bg-well rounded-[6px]
                                overflow-x-auto whitespace-pre-wrap">{traceback}</pre>
              )}
            </>
          )}
        </>
      )}
    </div>
  )
}
