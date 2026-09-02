import type { Metadata, Viewport } from 'next';
import { body, display, mono } from './fonts';
import './globals.css';
import { Nav } from '@/components/Nav';
import { EvidenceBar, EvidenceProvider } from '@/components/Evidence';
import { data } from '@/lib/data';

export const metadata: Metadata = {
  title: 'LocalMind — measured results',
  description:
    'A 31M-parameter decoder-only LM inside an agentic RAG system, built on free-tier compute. Every figure is labelled measured, synthetic or not run. The model is not trained.',
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: '#0a0b0d',
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>
        <a className="skip" href="#main">
          Skip to content
        </a>
        <EvidenceProvider>
          <Nav />
          <EvidenceBar />
          <main id="main">{children}</main>
        </EvidenceProvider>
        <div className="shell">
          <footer className="foot">
            <span>Apache-2.0</span>
            <span>Cash cost to date $0.00 · 0 GPU-hours · docs/compute_log.md</span>
            <span>Figures extracted {new Date(data.generatedAt).toISOString().slice(0, 16).replace('T', ' ')}Z</span>
            <span>{data.generatedBy}</span>
          </footer>
        </div>
      </body>
    </html>
  );
}
