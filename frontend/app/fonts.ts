import localFont from 'next/font/local';

/**
 * Self-hosted, latin subset only. No CDN request is made at build or at run time,
 * because the deployment target is an offline container.
 *
 * Saira Condensed  - industrial signage grotesque. Headings only.
 * Archivo          - high-performance grotesque. Running text.
 * IBM Plex Mono    - every number, label and identifier. Instrumentation reads mono.
 */

export const display = localFont({
  src: [
    { path: './fonts/SairaCondensed-500-latin.woff2', weight: '500', style: 'normal' },
    { path: './fonts/SairaCondensed-600-latin.woff2', weight: '600', style: 'normal' },
    { path: './fonts/SairaCondensed-700-latin.woff2', weight: '700', style: 'normal' },
  ],
  variable: '--font-display',
  display: 'swap',
  fallback: ['Arial Narrow', 'Haettenschweiler', 'sans-serif'],
  adjustFontFallback: false,
});

export const body = localFont({
  src: [{ path: './fonts/Archivo-latin.woff2', weight: '400 600', style: 'normal' }],
  variable: '--font-body',
  display: 'swap',
  fallback: ['Helvetica Neue', 'Arial', 'sans-serif'],
  adjustFontFallback: false,
});

export const mono = localFont({
  src: [
    { path: './fonts/IBMPlexMono-400-latin.woff2', weight: '400', style: 'normal' },
    { path: './fonts/IBMPlexMono-500-latin.woff2', weight: '500', style: 'normal' },
    { path: './fonts/IBMPlexMono-600-latin.woff2', weight: '600', style: 'normal' },
  ],
  variable: '--font-mono',
  display: 'swap',
  fallback: ['Consolas', 'Menlo', 'monospace'],
  adjustFontFallback: false,
});
