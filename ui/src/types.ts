export type Ev = {
  type: string
  text?: string
  tool?: string
  args?: Record<string, unknown>
  result?: string
  call_id?: string
  role?: string
  task?: string
  message?: string
  input?: number
  output?: number
  reason?: string
  session?: Session
}

export type Session = {
  id: string
  model: string
  mode: string
  mcp_servers: string[]
  tokens_in: number
  tokens_out: number
  events?: Ev[]
}

export type FileRow = {
  name: string
  rel_path: string
  is_dir: boolean
  size: number
  kind: string
}

export type FieldSeries = {
  kind: 'field_series'
  url: string
  field: string
  unit: string
  nx: number
  ny: number
  vmin: number
  vmax: number
  n_frames: number
  x0?: number
  y0?: number
  dx?: number
  dy?: number
  provenance?: {
    true_min?: number
    true_max?: number
    clip_percentile?: number | null
    saturated_fraction?: number
    quantisation_step?: number
    interpolation?: string
    solver?: string | null
    source?: string | null
    sha256?: string
  }
}
