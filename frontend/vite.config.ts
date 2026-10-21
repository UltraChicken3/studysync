import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backend = 'http://127.0.0.1:8765'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: Object.fromEntries(
      ['/health', '/bootstrap', '/sessions'].map((path) => [
        path,
        { target: backend, changeOrigin: true, ws: path === '/sessions' },
      ]),
    ),
  },
})
