'use client';

import { AnimatePresence, motion } from 'framer-motion';
import { useReducedMotion } from '../common/useReducedMotion';

interface AnimatedPageContentProps {
  children: React.ReactNode;
  locale: string;
}

export function AnimatedPageContent({ children, locale }: AnimatedPageContentProps) {
  const reduced = useReducedMotion();

  if (reduced) {
    return <>{children}</>;
  }

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={locale}
        initial={{ opacity: 0, filter: 'blur(4px)', y: 5 }}
        animate={{ opacity: 1, filter: 'blur(0px)', y: 0 }}
        exit={{ opacity: 0, filter: 'blur(4px)', y: -5 }}
        transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
        style={{ willChange: 'transform, opacity, filter' }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
