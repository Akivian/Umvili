'use client';

import { motion } from 'framer-motion';
import { TechnicalHeader } from '../design-system';

interface ShowcaseSectionProps {
  title: string;
  subtitle: string;
}

export function ShowcaseSection({ title, subtitle }: ShowcaseSectionProps) {
  return (
    <section id="showcase" className="border-b border-[#27272A]">
      <div className="max-w-5xl mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <TechnicalHeader title={title} subtitle={subtitle} />
          <div className="border border-[#27272A] rounded-sm bg-[#09090b] overflow-hidden">
            <div className="flex items-center gap-2 border-b border-[#27272A] px-4 py-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#71717a]" />
              <span className="w-2.5 h-2.5 rounded-full bg-[#71717a]" />
              <span className="w-2.5 h-2.5 rounded-full bg-[#71717a]" />
            </div>
            <pre className="p-4 font-mono text-sm overflow-x-auto">
              <code>
                <span className="text-[#71717a]">$</span>{' '}
                <span className="text-[#e4e4e7]">python main.py</span>
                {'\n'}
                <span className="text-[#71717a]">$</span>{' '}
                <span className="text-[#e4e4e7]">python main.py</span>{' '}
                <span className="text-[#a1a1aa]">--config config/default.json</span>
                {'\n'}
                <span className="text-[#71717a]">$</span>{' '}
                <span className="text-[#e4e4e7]">python main.py</span>{' '}
                <span className="text-[#a1a1aa]">--simulation-type comparative</span>{' '}
                <span className="text-[#22d3ee]">--grid-size 100</span>{' '}
                <span className="text-[#22d3ee]">--agents 200</span>
              </code>
            </pre>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
