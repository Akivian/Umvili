'use client';

import { motion } from 'framer-motion';
import { TechnicalHeader } from '@/components/design-system';

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
      <div className="max-w-5xl mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
          className="flex flex-col md:flex-row gap-12"
        >
          <aside className="md:w-48 shrink-0">
            <nav className="sticky top-24 space-y-1 text-sm">
              <a href="#guide" className="block text-muted hover:text-white font-mono">
                01 — Quick start
              </a>
              <a href="#guide" className="block text-muted hover:text-white font-mono">
                02 — Configuration
              </a>
              <a href="#guide" className="block text-muted hover:text-white font-mono">
                03 — Agent types
              </a>
            </nav>
          </aside>
          <div className="flex-1 space-y-8">
            <TechnicalHeader title={title} subtitle={subtitle} />
            <ol className="space-y-6">
              {steps.map(({ id, label, code }) => (
                <li key={id} className="border-l border-[#27272A] pl-5">
                  <span className="text-xs font-mono text-muted">{String(id).padStart(2, '0')}</span>
                  <p className="mt-1 text-sm text-white">{label}</p>
                  <pre className="mt-2 p-3 rounded-sm bg-[#09090b] border border-[#27272A] font-mono text-xs text-[#A1A1AA] overflow-x-auto">
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
