'use client';

import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { useReducedMotion } from '../../common/useReducedMotion';

/** SVG paths (0.5px) with pathLength draw animation + sine pulse on photons */
export function FlowPaths() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.2 });
  const reduced = useReducedMotion();

  const pathTransition = reduced
    ? { duration: 0 }
    : { duration: 1.5, ease: [0.4, 0, 0.2, 1] };

  return (
    <div
      ref={ref}
      className="absolute left-0 top-0 bottom-0 w-16 pointer-events-none hidden xl:block"
      aria-hidden
    >
      <svg viewBox="0 0 64 100" className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id="flowArchA" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(63,63,70,0.5)" />
            <stop offset="100%" stopColor="rgba(63,63,70,0.1)" />
          </linearGradient>
          <linearGradient id="flowArchB" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="rgba(63,63,70,0.5)" />
            <stop offset="100%" stopColor="rgba(63,63,70,0.1)" />
          </linearGradient>
        </defs>
        {/* Path A: Config Builder -> Config Entry (down) — pathLength draw */}
        <motion.path
          d="M 16 15 L 16 50"
          fill="none"
          stroke="url(#flowArchA)"
          strokeWidth="0.5"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: inView && !reduced ? 1 : 0 }}
          transition={pathTransition}
        />
        <motion.path
          d="M 16 15 L 16 50"
          fill="none"
          stroke="rgba(161,161,170,0.6)"
          strokeWidth="0.5"
          strokeLinecap="round"
          strokeDasharray="6 8"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: inView && !reduced ? 1 : 0 }}
          transition={pathTransition}
        />
        <path
          d="M 16 15 L 16 50"
          fill="none"
          stroke="rgba(161,161,170,0.6)"
          strokeWidth="0.5"
          strokeLinecap="round"
          strokeDasharray="6 8"
          strokeDashoffset="0"
          className="flow-photon-down"
          style={{ opacity: inView ? 1 : 0 }}
        />
        {/* Path B: Analytics Hub -> Sim Data (up) */}
        <motion.path
          d="M 48 85 L 48 50"
          fill="none"
          stroke="url(#flowArchB)"
          strokeWidth="0.5"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: inView && !reduced ? 1 : 0 }}
          transition={pathTransition}
        />
        <path
          d="M 48 85 L 48 50"
          fill="none"
          stroke="rgba(161,161,170,0.6)"
          strokeWidth="0.5"
          strokeLinecap="round"
          strokeDasharray="6 8"
          strokeDashoffset="0"
          className="flow-photon-up"
          style={{ opacity: inView ? 1 : 0 }}
        />
      </svg>
    </div>
  );
}
