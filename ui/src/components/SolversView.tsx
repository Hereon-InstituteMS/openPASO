import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Solver } from '../types'

// From the descriptions openPASO's own server publishes for each backend.
const WHAT: Record<string, string> = {
  '4C Multiphysics': 'Fluid-structure interaction, thermo-structure, contact, beams, particles, cardiovascular',
  'FEniCSx (dolfinx)': 'Weak forms in UFL; Navier-Stokes, hyperelasticity; Gmsh meshes',
  'deal.II': 'Adaptive hp-refinement, matrix-free, parallel on MPI and GPU',
  'FEBio': 'Biomechanics: biphasic and multiphasic tissue, active contraction',
  'NGSolve': 'High-order methods; Maxwell, Helmholtz, DG/HDG, eigenvalue problems',
  'scikit-fem': 'Pure Python with assembly-level control; Stokes, biharmonic',
  'Kratos Multiphysics': 'Structural, fluid, FSI, DEM, MPM, co-simulation',
  'DUNE-fem': 'DG methods, virtual elements, h/p-adaptivity',
  'SPARTA (DSMC)': 'Rarefied gas flow by direct simulation Monte Carlo (a particle code, not finite elements)',
}

/* What openPASO can use on this machine, checked by the same Python it runs in. */
export default function SolversView() {
  const [data, setData] = useState<{ ok: boolean; error?: string; checked_at: number; solvers: Solver[] } | null>(null)
  const [busy, setBusy] = useState(false)
  const load = (refresh = false) => {
    setBusy(true)
    api.solvers(refresh).then(setData).catch((e) => setData({ ok: false, error: String(e), checked_at: 0, solvers: [] }))
      .finally(() => setBusy(false))
  }
  useEffect(() => load(), [])
  const ready = data?.solvers.filter((s) => s.status === 'available').length ?? 0

  return (
    <div className="h-full overflow-y-auto scroll">
      <div className="max-w-[1040px] mx-auto px-10 py-12">
        <h1 className="text-[40px] font-semibold tracking-[-0.02em] text-ink">Solvers on this machine</h1>
        <p className="mt-3 text-[17px] text-body max-w-[70ch]">
          openPASO chooses among these for each run. You can also name one in your prompt.
        </p>
        <div className="mt-8 flex items-center gap-4">
          <span className="text-[15px] text-ink2">
            {!data ? 'Checking…' : data.ok ? `${ready} of ${data.solvers.length} found on this machine` : 'The check failed'}
          </span>
          {data?.checked_at ? <span className="text-[14px] text-muted">checked at {new Date(data.checked_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span> : null}
          <button onClick={() => load(true)} disabled={busy}
                  className="ml-auto h-9 px-4 rounded-[8px] border line text-[14px] text-body hover:text-ink hover:border-strong disabled:opacity-60">
            {busy ? 'Checking…' : 'Check again'}
          </button>
        </div>
        {data && !data.ok && <p className="mt-4 text-[15px] text-bad">{data.error}</p>}
        {data?.ok && (
          <table className="mt-6 w-full text-left">
            <thead>
              <tr className="border-b line text-[13px] text-muted">
                <th className="py-3 font-medium">Solver</th>
                <th className="py-3 font-medium">Good for</th>
                <th className="py-3 font-medium">Status</th>
                <th className="py-3 pl-6 font-medium text-right">Version</th>
                <th className="py-3 pl-6 font-medium text-right whitespace-nowrap" title="Kinds of problem openPASO has a ready-made input for, per solver">Ready-made set-ups</th>
              </tr>
            </thead>
            <tbody>
              {data.solvers.map((s) => (
                <tr key={s.name} className="border-b line align-top">
                  <td className="py-4 pr-4 text-[16px] text-ink whitespace-nowrap">{s.name}</td>
                  <td className="py-4 pr-4 text-[15px] text-body">{WHAT[s.name] ?? ''}</td>
                  <td className="py-4 pr-4 text-[15px] whitespace-nowrap">
                    {s.status === 'available'
                      ? <span className="text-ink2"><span aria-hidden className="text-ok mr-1.5">✓</span>Found</span>
                      : <span className="text-bad"><span aria-hidden className="mr-1.5">✕</span>{s.status.replace('_', ' ')}</span>}
                  </td>
                  <td className="py-4 pl-6 num text-[14px] text-right text-muted whitespace-nowrap">{s.version ?? 'not reported'}</td>
                  <td className="py-4 pl-6 num text-[14px] text-right text-body">{s.physics ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
