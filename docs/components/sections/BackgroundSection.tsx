'use client';

import { motion } from 'framer-motion';
import { Divider, HairlineCard, SectionEntrance, TechnicalHeader } from '../design-system';

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
    <section id="background">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-20">
        <SectionEntrance className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <HairlineCard as="article">
            <TechnicalHeader title={originTitle} />
            <Divider className="mt-4" />
            <p className="mt-4 text-sm text-zinc-400 leading-relaxed">{originText}</p>
          </HairlineCard>
          <HairlineCard as="article">
            <TechnicalHeader title={problemTitle} />
            <Divider className="mt-4" />
            <p className="mt-4 text-sm text-zinc-400 leading-relaxed">{problemText}</p>
          </HairlineCard>
        </SectionEntrance>
      </div>
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10">
        <Divider />
      </div>
    </section>
  );
}
