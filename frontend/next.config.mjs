const nextConfig = {
  reactStrictMode: true,
  // Next 16 writes CLAUDE.md / AGENTS.md into the project on first run. Not authored here.
  agentRules: false,
  // Offline-capable container: no remote images, no external fetches at build or run time.
  images: { unoptimized: true },
  env: {
    NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000",
  },
};

export default nextConfig;
