/* Motion tokens.

   Motivated only: something arrived, something changed state, focus moved.
   Never `ease-in` on interface motion, because it delays the exact moment the
   eye is watching. Only transform and opacity. */
export const EASE = [0.23, 1, 0.32, 1] as const

export const DUR = {
  press: 0.14, hover: 0.16, row: 0.2, panel: 0.26, hero: 0.52, count: 0.7,
} as const

/** A thing arriving: rises 6px and fades in. */
export const arrive = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: DUR.row, ease: EASE },
}

/** The hero, staggered 80ms. Rare and first impression, so slower. */
export const heroIn = (i: number) => ({
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: DUR.hero, ease: EASE, delay: 0.08 * i },
})

/** Rows stagger 40ms, capped: five rows of delay is the ceiling before a list
    feels laggy, and a solver emits them in bursts. */
export const rowIn = (i: number) => ({
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: DUR.row, ease: EASE, delay: Math.min(i, 4) * 0.04 },
})
