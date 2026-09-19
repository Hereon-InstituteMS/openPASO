import { useEffect, useState } from 'react'
import { api } from './api'
import Home from './components/Home'
import RunView from './components/RunView'
import Shell from './components/Shell'
import SolversView from './components/SolversView'
import { useRoute } from './route'
import type { AppConfig, ModelGroup } from './types'

export default function App() {
  const route = useRoute()
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [groups, setGroups] = useState<ModelGroup[] | null>(null)

  useEffect(() => {
    api.config().then(setConfig).catch(() => {})
    // whether a model works can change (a local server started, a key added)
    const load = () => api.models().then((d) => setGroups(d.groups)).catch(() => {})
    load()
    const t = setInterval(load, 60000)
    return () => clearInterval(t)
  }, [])

  return (
    <Shell current={route.run} view={route.view} config={config}>
      {route.view === 'run' && route.run
        ? <RunView key={route.run} id={route.run} config={config} groups={groups} />
        : route.view === 'solvers'
          ? <SolversView />
          : <Home config={config} groups={groups} />}
    </Shell>
  )
}
