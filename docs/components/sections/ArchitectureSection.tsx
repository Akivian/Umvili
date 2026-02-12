'use client';

import { motion } from 'framer-motion';
import { TechnicalHeader, HairlineCard } from '@/components/design-system';

interface ArchitectureSectionProps {
  title: string;
  subtitle: string;
  labels: {
    config: string;
    core: string;
    marl: string;
    utils: string;
    env: string;
    agents: string;
    simulation: string;
    visualization: string;
  };
}

export function ArchitectureSection({
  title,
  subtitle,
  labels,
}: ArchitectureSectionProps) {
  return (
    <section id="architecture" className="border-b border-[#27272A]">
      <div className="max-w-5xl mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
          className="space-y-10"
        >
          <TechnicalHeader title={title} subtitle={subtitle} />

          {/* Blueprint-style flowchart: SVG lines + boxes */}
          <div className="relative">
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none"
              preserveAspectRatio="none"
            >
              <defs>
                <marker
                  id="arrow"
                  markerWidth="6"
                  markerHeight="6"
                  refX="5"
                  refY="3"
                  orient="auto"
                >
                  <path d="M0,0 L6,3 L0,6 Z" fill="#27272A" />
                </marker>
              </defs>
              {/* Vertical spine */}
              <line
                x1="50%"
                y1="0"
                x2="50%"
                y2="100%"
                stroke="#27272A"
                strokeWidth="1"
                markerEnd="url(#arrow)"
              />
              {/* Horizontal branches to boxes */}
              <line x1="50%" y1="20%" x2="20%" y2="20%" stroke="#27272A" strokeWidth="1" />
              <line x1="50%" y1="45%" x2="20%" y2="45%" stroke="#27272A" strokeWidth="1" />
              <line x1="50%" y1="70%" x2="20%" y2="70%" stroke="#27272A" strokeWidth="1" />
              <line x1="50%" y1="85%" x2="80%" y2="85%" stroke="#27272A" strokeWidth="1" />
            </svg>

            <div className="relative grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Left column: modules */}
              <div className="space-y-6">
                <HairlineCard className="relative">
                  <span className="text-xs font-mono text-muted uppercase tracking-wider">
                    {labels.config}
                  </span>
                  <p className="mt-2 text-sm text-[#A1A1AA]">
                    app_config, config_loader, defaults
                  </p>
                </HairlineCard>
                <HairlineCard>
                  <span className="text-xs font-mono text-muted uppercase tracking-wider">
                    {labels.core}
                  </span>
                  <p className="mt-2 text-sm text-[#A1A1AA]">
                    {labels.env}, {labels.agents}, {labels.simulation}, reward_calculator
                  </p>
                </HairlineCard>
                <HairlineCard>
                  <span className="text-xs font-mono text-muted uppercase tracking-wider">
                    {labels.marl}
                  </span>
                  <p className="mt-2 text-sm text-[#A1A1AA]">
                    IQL, QMIX, networks, replay_buffer
                  </p>
                </HairlineCard>
              </div>

              {/* Right column: visualization */}
              <div className="md:pt-0 pt-4">
                <HairlineCard className="md:max-w-sm">
                  <span className="text-xs font-mono text-muted uppercase tracking-wider">
                    {labels.utils} → {labels.visualization}
                  </span>
                  <p className="mt-2 text-sm text-[#A1A1AA]">
                    MultiLineChart, QValueHeatmap, NetworkStatePanel, MARLSimulationRenderer
                  </p>
                </HairlineCard>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
