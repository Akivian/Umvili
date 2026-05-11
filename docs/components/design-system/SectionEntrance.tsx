'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { useReducedMotion } from '../../common/useReducedMotion';

const SPRING = { type: 'spring' as const, stiffness: 80, damping: 15, mass: 1 };
const VIEWPORT = { once: true, margin: '-10% 0px' as const };

const containerVariants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.05, delayChildren: 0 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 30, filter: 'blur(4px)' },
  visible: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: SPRING,
  },
};

interface SectionEntranceProps {
  children: React.ReactNode;
  className?: string;
  /** Optional delay before cascade starts */
  delay?: number;
}

export function SectionEntrance({ children, className, delay = 0 }: SectionEntranceProps) {
  const reduced = useReducedMotion();

  if (reduced) {
    return <div className={className}>{children}</div>;
  }

  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="visible"
      viewport={VIEWPORT}
      variants={{
        ...containerVariants,
        visible: {
          ...containerVariants.visible,
          transition: { staggerChildren: 0.05, delayChildren: delay },
        },
      }}
      style={{ willChange: 'transform, opacity, filter' }}
    >
      {React.Children.map(children, (child, i) => (
        <motion.div
          key={i}
          variants={itemVariants}
          className="transform-gpu"
          style={{ willChange: 'transform, opacity, filter' }}
        >
          {child}
        </motion.div>
      ))}
    </motion.div>
  );
}
