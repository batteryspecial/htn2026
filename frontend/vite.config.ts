import { fileURLToPath, URL } from 'node:url';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * The dev server is the only "backend" this frontend has, and it exists for one
 * reason: to be the seam where cross-origin calls can be made same-origin.
 *
 * By default the app talks to perception and the orchestrator directly, using
 * the URLs in Settings. That works because perception ships CORS middleware.
 * If a service is reachable but CORS is not (an orchestrator someone is still
 * writing, a tunnel, a phone IP-cam), set PROXY_* in .env.local and point the
 * matching Settings field at the local path instead:
 *
 *   PROXY_PERCEPTION=http://localhost:8001   ->  pipeline base "/perception"
 *   PROXY_ORCHESTRATOR=http://localhost:8000 ->  API base      "/orchestrator"
 *
 * Nothing else changes; both paths speak the same HTTP and the same sockets.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const proxy: Record<string, object> = {};

  const mount = (path: string, target?: string) => {
    if (!target) return;
    proxy[path] = {
      target,
      changeOrigin: true,
      ws: true,
      rewrite: (p: string) => p.replace(new RegExp(`^${path}`), ''),
    };
  };

  mount('/perception', env.PROXY_PERCEPTION);
  mount('/orchestrator', env.PROXY_ORCHESTRATOR);

  return {
    plugins: [react()],
    resolve: {
      alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
    },
    server: {
      port: 3000,
      // Fail loudly rather than silently moving to 3001 — a demo where the URL
      // on the projector is not the URL the app is on wastes everyone's time.
      strictPort: true,
      proxy,
    },
    preview: { port: 3000, strictPort: true },
  };
});
