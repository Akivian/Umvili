'use client';

import { motion } from 'framer-motion';

interface HeroSectionProps {
  title: string;
  tagline: string;
  cta: string;
  command: string;
}

export function HeroSection({ title, tagline, cta, command }: HeroSectionProps) {
  return (
    <section className="relative border-b border-[#27272A]">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-24 sm:py-32">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="space-y-6"
        >
          <h1 className="text-4xl sm:text-5xl font-semibold tracking-tighter text-white">
            {title}
          </h1>
          <p className="text-lg text-zinc-400 max-w-2xl">{tagline}</p>
          <div className="pt-2">
            <p className="text-xs text-muted uppercase tracking-wider mb-2 font-mono">
              {cta}
            </p>
            <div className="border border-[#27272A] rounded-sm bg-[#09090b] p-4 font-mono text-sm text-[#A1A1AA] overflow-x-auto">
              <code className="text-[#e4e4e7]">
                <span className="text-[#71717a]">$</span> {command}
              </code>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
