import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwind from '@tailwindcss/vite'

// The built bundle is committed, so a fresh clone runs the interface with no
// Node at all: FastAPI serves webui/static and returns its index.html at /.
export default defineConfig({
  plugins: [react(), tailwind()],
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
