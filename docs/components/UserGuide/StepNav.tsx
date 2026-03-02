'use client';

import { motion } from 'framer-motion';
import { cn } from '../../common/utils';

export interface StepItem {
  id: number;
  label: string;
  command: string;
  output: string[];
}

interface StepNavProps {
  steps: StepItem[];
  activeStep: number;
  onStepSelect: (id: number) => void;
}

export function StepNav({
  steps,
  activeStep,
  onStepSelect,
}: StepNavProps) {
  return (
    <nav
      className={cn(
        'flex gap-2',
        'md:flex-col',
        'max-md:flex-row max-md:overflow-x-auto max-md:pb-2 max-md:-mx-1'
      )}
      role="tablist"
      aria-label="Guide steps"
    >
      {steps.map(({ id, label }) => {
        const isActive = activeStep === id;
        return (
          <motion.button
            key={id}
            role="tab"
            aria-selected={isActive}
            aria-controls={`terminal-step-${id}`}
            id={`step-tab-${id}`}
            onClick={() => onStepSelect(id)}
            className={cn(
              'relative flex items-center gap-3 px-4 py-3 rounded-sm font-mono text-sm text-left',
              'bg-zinc-950/30 backdrop-blur-md',
              'border border-zinc-800',
              'shadow-[inset_0_1px_1px_0_rgba(255,255,255,0.05)]',
              'transition-colors duration-200',
              'hover:border-zinc-700',
              isActive
                ? 'text-zinc-100 border-l-[3px] border-l-emerald-500 max-md:border-l-0 max-md:border-b-[3px] max-md:border-b-emerald-500'
                : 'text-zinc-500',
              'max-md:shrink-0 max-md:min-w-[140px]'
            )}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 400, damping: 25 }}
          >
            <span className="text-zinc-500">{String(id).padStart(2, '0')}</span>
            <span>{label}</span>
          </motion.button>
        );
      })}
    </nav>
  );
}
