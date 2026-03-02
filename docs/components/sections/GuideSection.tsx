'use client';

import { motion } from 'framer-motion';
import { Divider, HairlineCard, TechnicalHeader } from '../design-system';

const steps = [
  { id: 1, label: 'Clone & install', code: 'git clone https://github.com/Akivian/Umvili.git && cd Umvili && pip install -r requirements.txt' },
  { id: 2, label: 'Run', code: 'python main.py' },
  { id: 3, label: 'Optional config', code: 'python main.py --config config/default.json' },
];

interface GuideSectionProps {
  title: string;
  subtitle: string;
}

export function GuideSection({ title, subtitle }: GuideSectionProps) {
  return (
    <section id="guide">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
          className="space-y-8"
        >
          <TechnicalHeader title={title} subtitle={subtitle} />
          <div className="relative grid grid-cols-1 gap-6 max-w-2xl">
            {/* Vertical progress line connecting steps */}
            <div
              className="absolute left-[38px] top-6 bottom-6 w-px bg-[linear-gradient(to_bottom,transparent_0%,#27272a_10%,#27272a_90%,transparent_100%)]"
              aria-hidden
            />
            {steps.map(({ id, label, code }, i) => (
              <motion.div
                key={id}
                initial={{ opacity: 0, y: 8 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: i * 0.08 }}
                className="relative"
              >
                <HairlineCard>
                  <div className="flex gap-4">
                    <span className="shrink-0 inline-flex w-9 h-9 items-center justify-center font-mono text-xs bg-zinc-900 border border-zinc-800 rounded-sm text-zinc-400">
                      {String(id).padStart(2, '0')}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-white">{label}</p>
                      <pre className="mt-3 p-3 rounded-sm bg-[#18181b] border border-zinc-800/80 font-mono text-xs text-zinc-400 overflow-x-auto">
                        {code}
                      </pre>
                    </div>
                  </div>
                </HairlineCard>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10">
        <Divider />
      </div>
    </section>
  );
}
