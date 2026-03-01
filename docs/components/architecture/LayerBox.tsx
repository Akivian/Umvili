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
        'relative rounded-sm border border-zinc-800 bg-black min-h-[80px] overflow-hidden',
        className
      )}
      {...layoutProps}
    >
      <div className="relative z-10 p-4">{children}</div>
    </Wrapper>
  );
}

/** Mono label for high-density sub-components */
export function TechLabel({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        'font-mono text-xs text-zinc-500 uppercase tracking-wider',
        className
      )}
    >
      {children}
    </span>
  );
}
