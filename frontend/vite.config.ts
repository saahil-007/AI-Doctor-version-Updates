import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "::",
    port: 8080,
    proxy: {
      '/api/chat': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      },
      '/api/translate': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      },
      '/api/preprocess': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      },
      '/api/health': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      },
      '/api/text-to-speech': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      },
      '/api/stop-speech': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  build: {
    chunkSizeWarningLimit: 1000, // Increase limit to 1000kb to avoid warnings
    rollupOptions: {
      output: {
        manualChunks: {
          // Split vendor chunks to reduce bundle size
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-tooltip'],
          utils: ['@tanstack/react-query', 'framer-motion', 'lucide-react']
        }
      }
    }
  },
  plugins: [react()]
}
))