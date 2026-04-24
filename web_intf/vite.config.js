import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/jupyterlab/default/proxy/5173/',
  server: {
    // Allow access from SageMaker JupyterLab proxy
    allowedHosts: [
      'cyk8dhyvk3zadsd.studio.sagemaker.ap-southeast-2.app.aws',
      'localhost'
    ],
    // Listen on all network interfaces
    host: true,
    port: 5173,
    // Configure HMR for proxy environment
    hmr: {
      protocol: 'wss',
      host: 'cyk8dhyvk3zadsd.studio.sagemaker.ap-southeast-2.app.aws',
      port: 443,
      clientPort: 443
    }
  }
})
