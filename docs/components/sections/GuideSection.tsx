'use client';

import { motion } from 'framer-motion';
import { TechnicalHeader } from '../design-system';

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
    <section id="guide" className="border-b border-[#27272A]">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
          className="flex flex-col md:flex-row gap-12"
        >
          <aside className="md:w-48 shrink-0">
            <nav className="sticky top-24 space-y-1 text-sm">
              <a href="#guide" className="block text-zinc-400 hover:text-white font-mono transition-colors">
                01 — Quick start
              </a>
              <a href="#guide" className="block text-zinc-400 hover:text-white font-mono transition-colors">
                02 — Configuration
              </a>
              <a href="#guide" className="block text-zinc-400 hover:text-white font-mono transition-colors">
                03 — Agent types
              </a>
            </nav>
          </aside>
          <div className="flex-1 space-y-8">
            <TechnicalHeader title={title} subtitle={subtitle} />
            <ol className="space-y-6">
              {steps.map(({ id, label, code }) => (
                <li key={id} className="border-l border-zinc-800 pl-5">
                  <span className="inline-flex px-2 py-0.5 font-mono text-xs bg-zinc-900 border border-zinc-800 rounded-sm text-zinc-400">
                    {String(id).padStart(2, '0')}
                  </span>
                  <p className="mt-2 text-sm text-white">{label}</p>
                  <pre className="mt-2 p-3 rounded-sm bg-[rgba(9,9,11,0.6)] border border-zinc-800 font-mono text-xs text-zinc-400 overflow-x-auto">
                    {code}
                  </pre>
                </li>
              ))}
            </ol>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
