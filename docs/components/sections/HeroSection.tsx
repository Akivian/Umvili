'use client';

import { motion } from 'framer-motion';
import { Divider, InteractiveTerminal } from '../design-system';

interface HeroSectionProps {
  badge: string;
  title: string;
  tagline: string;
  cta: string;
  command: string;
}

export function HeroSection({ badge, title, tagline, cta, command }: HeroSectionProps) {
  return (
    <section className="relative">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-24 sm:py-32">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="space-y-6"
        >
          <span className="inline-flex items-center px-2 py-1 font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400 bg-zinc-900 border border-zinc-800 rounded-sm mb-3">
            {badge}
          </span>
          <h1 className="text-4xl sm:text-5xl font-semibold text-white" style={{ letterSpacing: '-0.05em' }}>
            {title}
          </h1>
          <p className="text-lg text-zinc-400 max-w-2xl">{tagline}</p>
          <div className="pt-2">
            <p className="text-xs text-zinc-400 uppercase tracking-wider mb-2 font-mono">
              {cta}
            </p>
            <InteractiveTerminal>
              <pre className="m-0 overflow-x-auto">
                <code>
                  <span className="text-zinc-500">$</span>{' '}
                  <span className="text-zinc-200">{command}</span>
                  <InteractiveTerminal.BlinkCursor />
                </code>
              </pre>
            </InteractiveTerminal>
          </div>
        </motion.div>
      </div>
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10">
        <Divider />
      </div>
    </section>
  );
}
