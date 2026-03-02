'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Divider, SectionEntrance, TechnicalHeader } from '../design-system';
import { StepNav, TerminalConsole, GUIDE_STEPS } from '../UserGuide';

interface GuideSectionProps {
  title: string;
  subtitle: string;
}

export function GuideSection({ title, subtitle }: GuideSectionProps) {
  const [activeStep, setActiveStep] = useState(1);
  const activeStepData = GUIDE_STEPS.find((s) => s.id === activeStep) ?? GUIDE_STEPS[0];

  return (
    <section id="guide">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-10% 0px' }}
          transition={{ type: 'spring', stiffness: 100, damping: 20 }}
          className="space-y-8"
        >
          <TechnicalHeader title={title} subtitle={subtitle} />

          {/* Two-column: Nav (1-4) + Console (5-12). Stack on < 768px */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-8">
            {/* Step sequencer: cols 1-4 on md+ */}
            <div className="md:col-span-4">
              <StepNav
                steps={GUIDE_STEPS}
                activeStep={activeStep}
                onStepSelect={setActiveStep}
              />
            </div>

            {/* Mock terminal: cols 5-12 on md+ */}
            <div className="md:col-span-8">
              <TerminalConsole
                step={activeStepData}
                onCopy={() => {}}
              />
            </div>
          </div>
        </motion.div>
      </div>
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10">
        <Divider />
      </div>
    </section>
  );
}
