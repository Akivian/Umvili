'use client';

import 'katex/dist/katex.min.css';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { LucideIcon } from 'lucide-react';
import { cn } from '../../common/utils';

export type AccentColor = 'mint' | 'cream' | 'pink';

const accentGlow: Record<AccentColor, string> = {
  mint: 'hover:shadow-glow-mint hover:border-[var(--accent-mint)]',
  cream: 'hover:shadow-glow-cream hover:border-[var(--accent-cream)]',
  pink: 'hover:shadow-glow-pink hover:border-[var(--accent-pink)]',
};

interface ModuleCardProps {
  label: string;
  icon: LucideIcon;
  accent?: AccentColor;
  /** LaTeX string for deep-dive tooltip (e.g. "Q_{tot}(\\\\tau, \\\\mathbf{u})") */
  formula?: string;
  children?: React.ReactNode;
  className?: string;
}

export function ModuleCard({
  label,
  icon: Icon,
  accent = 'mint',
  formula,
  children,
  className,
}: ModuleCardProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [formulaHtml, setFormulaHtml] = useState<string | null>(null);

  const loadFormula = () => {
    if (!formula || formulaHtml !== null) return;
    import('katex').then((katex) => {
      try {
        const html = katex.default.renderToString(formula, {
          displayMode: false,
          throwOnError: false,
          output: 'html',
        });
        setFormulaHtml(html);
      } catch {
        setFormulaHtml(null);
      }
    });
  };

  return (
    <motion.div
      className={cn(
        'relative rounded-sm border border-[#27272A] bg-zinc-950/80 backdrop-blur-sm transition-all duration-200 hover:bg-zinc-900/50',
        accentGlow[accent],
        className
      )}
      onMouseEnter={() => {
        setShowTooltip(true);
        if (formula) loadFormula();
      }}
      onMouseLeave={() => setShowTooltip(false)}
      whileHover={{ scale: 1.02, backgroundColor: 'rgba(24, 24, 27, 0.8)' }}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
      className="transform-gpu"
    >
      <div className="p-4 flex items-start gap-3">
        <span className="text-[#27272A] shrink-0" aria-hidden>
          <Icon
            size={18}
            strokeWidth={1.2}
            className="text-muted"
          />
        </span>
        <div className="min-w-0 flex-1">
          <span className="font-mono text-xs text-muted uppercase tracking-wider">
            {label}
          </span>
          {children && (
            <div className="mt-1.5 text-sm text-[#A1A1AA]">{children}</div>
          )}
        </div>
      </div>
      <AnimatePresence>
        {showTooltip && formula && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15 }}
            className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-10 px-3 py-2 rounded-sm border border-[#27272A] bg-black shadow-lg min-w-[120px]"
          >
            <p className="text-[10px] font-mono text-muted uppercase tracking-wider mb-1.5">
              Deep-Dive
            </p>
            {formulaHtml ? (
              <div
                className="text-sm text-white [&>.katex]:text-sm"
                dangerouslySetInnerHTML={{ __html: formulaHtml }}
              />
            ) : (
              <span className="text-sm text-muted">Loading…</span>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
