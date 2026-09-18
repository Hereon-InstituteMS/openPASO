import type { Outcome } from '../types'

/* A run's state, the same everywhere: a shape and a word, so it survives a
   screenshot, greyscale and colour blindness. Success carries no colour; only
   trouble does. */
export const STATE: Record<Outcome, { glyph: string; word: string; tone: string; means?: string }> = {
  running:     { glyph: '◐', word: 'Working',                tone: 'text-coral' },
  completed:   { glyph: '✓', word: 'Finished',               tone: 'text-ink2', means: 'An openPASO solver ran and openPASO verified its result.' },
  unverified:  { glyph: '△', word: 'Ran, not verified',      tone: 'text-coral', means: 'A solver ran, but openPASO did not verify the result. Check it before relying on it.' },
  no_result:   { glyph: '○', word: 'Ended',                  tone: 'text-ink2', means: 'The model replied, but no solver result was produced. Open the run to see why.' },
  failed:      { glyph: '✕', word: 'Failed',                 tone: 'text-bad', means: 'The run hit an error. Open it for the message.' },
  interrupted: { glyph: '■', word: 'Stopped',                tone: 'text-muted', means: 'You stopped this run.' },
  unfinished:  { glyph: '◌', word: 'Stopped unexpectedly',   tone: 'text-bad', means: 'The run has no recorded end, usually because the server restarted while it worked.' },
}

export default function Glyph({ outcome, className = '' }: { outcome: Outcome; className?: string }) {
  const s = STATE[outcome] ?? STATE.no_result
  return <span aria-hidden className={`num ${s.tone} ${outcome === 'running' ? 'spin-slow inline-block' : ''} ${className}`}>{s.glyph}</span>
}
