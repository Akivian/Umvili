'use client';

import { motion } from 'framer-motion';
import { HairlineCard, TechnicalHeader } from '@/components/design-system';
import { Cpu, Layers, LineChart, Settings } from 'lucide-react';

const iconClass = 'w-5 h-5 text-[#A1A1AA] stroke-[1.5px]';

const cells = [
  {
    key: 'algorithms',
    Icon: Cpu,
    title: 'IQL & QMIX',
    desc: 'Independent Q-Learning and QMIX with prioritized replay.',
  },
  {
    key: 'environment',
    Icon: Layers,
    title: 'Sugar / Spice / Hazard',
    desc: 'Multi-resource environment with dynamic zones.',
  },
  {
    key: 'visualization',
    Icon: LineChart,
    title: 'Real-time Metrics',
    desc: 'Loss, Q-value, TD error, exploration rate, policy entropy.',
  },
  {
    key: 'config',
    Icon: Settings,
    title: 'JSON/YAML Config',
    desc: 'Flexible configuration and CLI overrides.',
  },
];

interface BentoSectionProps {
  title: string;
  subtitle: string;
}

export function BentoSection({ title, subtitle }: BentoSectionProps) {
  return (
    <section id="functionality" className="border-b border-[#27272A]">
      <div className="max-w-5xl mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
          className="space-y-8"
        >
          <TechnicalHeader title={title} subtitle={subtitle} />
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {cells.map(({ key, Icon, title: cellTitle, desc }, i) => (
              <motion.div
                key={key}
                initial={{ opacity: 0, y: 8 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: i * 0.05 }}
              >
                <HairlineCard>
                  <Icon className={iconClass} />
                  <h3 className="mt-3 font-medium text-white text-sm">{cellTitle}</h3>
                  <p className="mt-1 text-xs text-[#A1A1AA]">{desc}</p>
                </HairlineCard>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
