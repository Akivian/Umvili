'use client';

import { motion } from 'framer-motion';
import { HairlineCard, TechnicalHeader } from '@/components/design-system';

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
      <div className="max-w-5xl mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-8"
        >
          <HairlineCard as="article">
            <TechnicalHeader title={originTitle} />
            <div className="mt-4 h-px bg-[#27272A]" />
            <p className="mt-4 text-sm text-[#A1A1AA] leading-relaxed">{originText}</p>
          </HairlineCard>
          <HairlineCard as="article">
            <TechnicalHeader title={problemTitle} />
            <div className="mt-4 h-px bg-[#27272A]" />
            <p className="mt-4 text-sm text-[#A1A1AA] leading-relaxed">{problemText}</p>
          </HairlineCard>
        </motion.div>
      </div>
    </section>
  );
}
