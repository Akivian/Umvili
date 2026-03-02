'use client';

import { motion } from 'framer-motion';
import { Divider, HairlineCard, SectionEntrance, TechnicalHeader } from '../design-system';
import { Cpu, Layers, LineChart, Settings } from 'lucide-react';

const iconClass = 'w-5 h-5 text-zinc-400';

const cells = [
  { key: 'algorithms', Icon: Cpu, title: 'IQL & QMIX', desc: 'Independent Q-Learning and QMIX with prioritized replay.' },
  { key: 'environment', Icon: Layers, title: 'Sugar / Spice / Hazard', desc: 'Multi-resource environment with dynamic zones.' },
  { key: 'visualization', Icon: LineChart, title: 'Real-time Metrics', desc: 'Loss, Q-value, TD error, exploration rate, policy entropy.' },
  { key: 'config', Icon: Settings, title: 'JSON/YAML Config', desc: 'Flexible configuration and CLI overrides.' },
];

interface BentoSectionProps {
  title: string;
  subtitle: string;
}

export function BentoSection({ title, subtitle }: BentoSectionProps) {
  return (
    <section id="functionality">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-20">
        <SectionEntrance className="space-y-8">
          <TechnicalHeader title={title} subtitle={subtitle} />
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {cells.map(({ key, Icon, title: cellTitle, desc }) => (
              <HairlineCard key={key}>
                <Icon className={iconClass} strokeWidth={1.2} />
                <h3 className="mt-3 font-medium text-white text-sm">{cellTitle}</h3>
                <p className="mt-1 text-xs text-zinc-400">{desc}</p>
              </HairlineCard>
            ))}
          </div>
        </SectionEntrance>
      </div>
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10">
        <Divider />
      </div>
    </section>
  );
}
