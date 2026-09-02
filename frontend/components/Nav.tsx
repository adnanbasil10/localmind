'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const LINKS = [
  { href: '/', label: 'Results' },
  { href: '/architecture', label: 'Request path' },
  { href: '/query', label: 'Query' },
];

/**
 * The mark is a block allocator: nine KV blocks, five in use, one hot. It is the
 * repo's headline result rendered at 22 pixels, which is more honest than a logo.
 */
function Glyph() {
  const cells = [1, 1, 0, 1, 2, 0, 1, 0, 0];
  return (
    <svg className="nav__glyph" viewBox="0 0 22 22" aria-hidden="true">
      {cells.map((v, i) => (
        <rect
          key={i}
          x={(i % 3) * 8}
          y={Math.floor(i / 3) * 8}
          width="6"
          height="6"
          fill={v === 2 ? 'var(--red)' : v === 1 ? 'var(--ink)' : 'var(--rule-2)'}
        />
      ))}
    </svg>
  );
}

export function Nav() {
  const path = usePathname();
  return (
    <nav className="nav" aria-label="Primary">
      <div className="nav__in">
        <Link href="/" className="nav__mark">
          <Glyph />
          <span className="nav__word">LocalMind</span>
        </Link>
        <div className="nav__links">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="nav__link"
              aria-current={path === l.href ? 'page' : undefined}
            >
              {l.label}
            </Link>
          ))}
        </div>
        <span className="nav__status">
          <span aria-hidden="true">◇</span> model not trained
        </span>
      </div>
    </nav>
  );
}
