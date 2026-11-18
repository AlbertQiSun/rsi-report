import { defineConfig } from 'astro/config';

// GitHub Pages: set SITE and BASE when the repository name is known,
// e.g. SITE=https://<user>.github.io BASE=/rsi-report npm run build
export default defineConfig({
  site: process.env.SITE || 'http://localhost:4321',
  base: process.env.BASE || '/',
  output: 'static',
});
