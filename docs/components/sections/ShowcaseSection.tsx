'use client';

import { motion } from 'framer-motion';
import { Divider, InteractiveTerminal, SectionEntrance, TechnicalHeader } from '../design-system';

interface ShowcaseSectionProps {
  title: string;
  subtitle: string;
}

/** Zinc theme: prompt zinc-500, command zinc-200, args zinc-400, flag values cyan-400 */
export function ShowcaseSection({ title, subtitle }: ShowcaseSectionProps) {
  return (
    <section id="showcase">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-20">
        <SectionEntrance className="space-y-6">
          <TechnicalHeader title={title} subtitle={subtitle} />
          <InteractiveTerminal>
            <pre className="m-0 overflow-x-auto">
              <code>
                <span className="text-zinc-500">$</span>{' '}
                <span className="text-zinc-200">python main.py</span>
                {'\n'}
                <span className="text-zinc-500">$</span>{' '}
                <span className="text-zinc-200">python main.py</span>{' '}
                <span className="text-zinc-400">--config config/default.json</span>
                {'\n'}
                <span className="text-zinc-500">$</span>{' '}
                <span className="text-zinc-200">python main.py</span>{' '}
                <span className="text-zinc-400">--simulation-type comparative</span>{' '}
                <span className="text-cyan-400">--grid-size 100</span>{' '}
                <span className="text-cyan-400">--agents 200</span>
                <InteractiveTerminal.BlinkCursor />
              </code>
            </pre>
          </InteractiveTerminal>
        </SectionEntrance>
      </div>
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10">
        <Divider />
      </div>
    </section>
  );
}
