'use client';

import { motion } from 'framer-motion';
import { HairlineCard, TechnicalHeader } from '../design-system';

interface BackgroundSectionProps {
  originTitle: string;
  originText: string;
  problemTitle: string;
  problemText: string;
}

export function BackgroundSection({
  originTitle,
  originText,
  problemTitle,
  problemText,
}: BackgroundSectionProps) {
  return (
    <section id="background" className="border-b border-[#27272A]">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-8"
        >
          <HairlineCard as="article" accent="mint">
            <TechnicalHeader title={originTitle} />
            <div className="mt-4 h-px bg-zinc-800/50" />
            <p className="mt-4 text-sm text-zinc-400 leading-relaxed">{originText}</p>
          </HairlineCard>
          <HairlineCard as="article" accent="cream">
            <TechnicalHeader title={problemTitle} />
            <div className="mt-4 h-px bg-zinc-800/50" />
            <p className="mt-4 text-sm text-zinc-400 leading-relaxed">{problemText}</p>
          </HairlineCard>
        </motion.div>
      </div>
    </section>
  );
}
