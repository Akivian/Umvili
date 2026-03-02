'use client';

import { useEffect, useState } from 'react';
import { useReducedMotion } from '../common/useReducedMotion';

export function ParallaxDotGrid() {
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const reduced = useReducedMotion();

  useEffect(() => {
    if (reduced) return;
    const handler = (e: MouseEvent) => {
      setPos({ x: e.clientX / 50, y: e.clientY / 50 });
    };
    window.addEventListener('mousemove', handler, { passive: true });
    return () => window.removeEventListener('mousemove', handler);
  }, [reduced]);

  const transform = reduced ? undefined : `translate3d(${pos.x}px, ${pos.y}px, 0)`;

  return (
    <div
      className={`fixed inset-0 pointer-events-none z-0 ${reduced ? '' : 'transform-gpu'}`}
      aria-hidden
      style={{
        backgroundImage: 'radial-gradient(circle, rgba(255, 255, 255, 0.08) 0.5px, transparent 0.5px)',
        backgroundSize: '24px 24px',
        backgroundAttachment: 'fixed',
        transform: transform ?? 'none',
        willChange: reduced ? 'auto' : 'transform',
      }}
    />
  );
}
