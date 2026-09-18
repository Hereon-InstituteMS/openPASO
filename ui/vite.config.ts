import { rmSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwind from '@tailwindcss/vite'

// The built bundle is committed, so a fresh clone runs the interface with no
// Node at all: FastAPI serves webui/static and returns its index.html at /.
// webui/static also holds files Vite does not make (openpaso-logo.js, the mark),
// so the folder is not emptied; only the hashed bundles in assets/ are, all of
// which Vite writes. Without this every build left the previous bundles behind,
// committed and served next to the current ones.
const cleanAssets = (): Plugin => ({
  name: 'openpaso-clean-assets',
  apply: 'build',
  buildStart() { rmSync(resolve(__dirname, '../webui/static/assets'), { recursive: true, force: true }) },
})

export default defineConfig({
  plugins: [react(), tailwind(), cleanAssets()],
  base: '/static/',
  build: { outDir: '../webui/static', emptyOutDir: false, assetsDir: 'assets' },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8080',
      '/sandbox-file': 'http://127.0.0.1:8080',
      '/ws': { target: 'ws://127.0.0.1:8080', ws: true },
    },
  },
})
