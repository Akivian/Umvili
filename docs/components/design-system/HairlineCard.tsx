'use client';

import { motion } from 'framer-motion';
import { cn } from '../../common/utils';

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
        className
      )}
      whileHover={{
        y: -8,
        scale: 1.01,
        transition: { type: 'spring', stiffness: 300, damping: 20 },
      }}
    >
      {children}
    </Motion>
  );
}
