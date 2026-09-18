import { useEffect, useState } from 'react'

/* The address bar is the state. Going back used to reload the page and kill the
   run you were watching; now it switches views, and every run and page has an
   address you can bookmark or open in a second tab. */
export type Route = { run: string | null; view: 'home' | 'run' | 'solvers' }

function read(): Route {
  const q = new URLSearchParams(location.search)
  const run = q.get('run') || q.get('session')
  if (run) return { run, view: 'run' }
  if (q.get('view') === 'solvers') return { run: null, view: 'solvers' }
  return { run: null, view: 'home' }
}

export function navigate(to: { run?: string; view?: 'home' | 'solvers' }, replace = false) {
  const url = to.run ? `/?run=${to.run}` : to.view === 'solvers' ? '/?view=solvers' : '/'
  if (url === location.pathname + location.search) return
  history[replace ? 'replaceState' : 'pushState']({}, '', url)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

export function useRoute(): Route {
  const [r, setR] = useState(read)
  useEffect(() => {
    const on = () => setR(read())
    window.addEventListener('popstate', on)
    return () => window.removeEventListener('popstate', on)
  }, [])
  return r
}
