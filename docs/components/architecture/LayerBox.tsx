'use client';

import { motion } from 'framer-motion';
import { cn } from '../../common/utils';

/** Container for topology blocks: dot grid (20px), 1px zinc-800 border, JetBrains Mono */
export function LayerBox({
  children,
  className,
  layout,
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  layout?: boolean;
  delay?: number;
}) {
  const Wrapper = layout ? motion.div : 'div';
  const layoutProps = layout
    ? {
        initial: { opacity: 0, y: 12 },
        whileInView: { opacity: 1, y: 0 },
        viewport: { once: true, margin: '-30px' as const },
        transition: { type: 'spring' as const, stiffness: 120, damping: 20, delay },
      }
    : {};

  return (
    <Wrapper
      className={cn(
        'relative z-10 rounded-sm min-h-[80px] overflow-hidden',
        'bg-[rgba(9,9,11,0.6)] backdrop-blur-xl',
        'border border-[rgba(39,39,42,0.8)]',
        'shadow-[inset_0_1px_1px_0_rgba(255,255,255,0.1)]',
        className
      )}
      {...layoutProps}
    >
      <div className="relative z-10 p-4">{children}</div>
    </Wrapper>
  );
}

/** Mono label for high-density sub-components. Use tag for physical badge style. */
export function TechLabel({
  children,
  className,
  tag,
}: {
  children: React.ReactNode;
  className?: string;
  tag?: boolean;
}) {
  return (
    <span
      className={cn(
        'font-mono text-xs uppercase tracking-wider',
        tag ? 'inline-flex px-2 py-0.5 bg-zinc-900 border border-zinc-800 rounded-sm text-zinc-400' : 'text-zinc-400',
        className
      )}
    >
      {children}
    </span>
  );
}
