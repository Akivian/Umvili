'use client';

import { motion } from 'framer-motion';
import { cn } from '../../common/utils';
import { useReducedMotion } from '../../common/useReducedMotion';

interface HairlineCardProps {
  children: React.ReactNode;
  className?: string;
  as?: 'div' | 'article' | 'section';
}

const MotionComponents = {
  div: motion.div,
  article: motion.article,
  section: motion.section,
} as const;

export function HairlineCard({
  children,
  className,
  as: Component = 'div',
}: HairlineCardProps) {
  const Motion = MotionComponents[Component];
  const reduced = useReducedMotion();

  return (
    <Motion
      className={cn(
        'relative z-10 rounded-sm p-5',
        'bg-[rgba(9,9,11,0.6)] backdrop-blur-xl',
        'border border-[rgba(39,39,42,0.8)]',
        'shadow-[inset_0_1px_1px_0_rgba(255,255,255,0.1)]',
        'transition-[border-color,box-shadow] duration-300 ease-out',
        'hover:border-[rgba(161,161,170,0.4)]',
        'hover:shadow-[inset_0_1px_1px_0_rgba(255,255,255,0.1),0_0_20px_0_rgba(255,255,255,0.05)]',
        'transform-gpu',
        className
      )}
      whileHover={
        reduced
          ? undefined
          : {
              y: -8,
              scale: 1.02,
              backgroundColor: 'rgba(24, 24, 27, 0.8)',
              boxShadow:
                'inset 0 1px 1px 0 rgba(255,255,255,0.1), 0 0 20px 0 rgba(255,255,255,0.1)',
              transition: { type: 'spring', stiffness: 300, damping: 20 },
            }
      }
      style={{ willChange: reduced ? 'auto' : 'transform' }}
    >
      {children}
    </Motion>
  );
}
