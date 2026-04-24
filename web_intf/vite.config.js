import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({

  // IMPORTANT for SageMaker proxy path
  base: './',

  plugins: [react()],

  server: {

    // MUST for SageMaker
    host: '0.0.0.0',

    port: 5173,

    strictPort: true,

    // IMPORTANT
    hmr: false,

    // VERY IMPORTANT
    allowedHosts: 'all',

    proxy: {

      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },

      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})