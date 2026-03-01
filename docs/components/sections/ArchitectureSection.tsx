'use client';

import { motion } from 'framer-motion';
import {
  Monitor,
  FlaskConical,
  Cpu,
  Users,
  Map,
  GitBranch,
  Network,
  Settings,
  Loader2,
  Database,
} from 'lucide-react';
import { TechnicalHeader } from '../design-system';
import {
  ArchitectureLayer,
  FlowLine,
  ModuleCard,
} from '../architecture';

export interface ArchitectureLabels {
  title: string;
  subtitle: string;
  layer1: string;
  layer2: string;
  layer3: string;
  layer4: string;
  modViz: string;
  modUi: string;
  modEngine: string;
  modAgents: string;
  modEnv: string;
  modIql: string;
  modQmix: string;
  modDefaults: string;
  modLoader: string;
  modSchema: string;
  dataPipeline: string;
}

export function ArchitectureSection({ labels }: { labels: ArchitectureLabels }) {
  return (
    <section id="architecture" className="border-b border-[#27272A]">
      <div className="max-w-5xl mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
          className="space-y-0"
        >
          <TechnicalHeader
            title={labels.title}
            subtitle={labels.subtitle}
          />

          <div className="mt-12 flex flex-col">
            {/* Layer 1: Presentation */}
            <ArchitectureLayer
              index={1}
              title={labels.layer1}
              delay={0}
            >
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <ModuleCard
                  label={labels.modViz}
                  icon={Monitor}
                  accent="mint"
                />
                <ModuleCard
                  label={labels.modUi}
                  icon={FlaskConical}
                  accent="mint"
                />
              </div>
            </ArchitectureLayer>

            <Connector direction="up" />

            {/* Layer 2: Core Simulation */}
            <ArchitectureLayer
              index={2}
              title={labels.layer2}
              delay={0.1}
            >
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <ModuleCard
                  label={labels.modEngine}
                  icon={Cpu}
                  accent="cream"
                />
                <ModuleCard
                  label={labels.modAgents}
                  icon={Users}
                  accent="cream"
                />
                <ModuleCard
                  label={labels.modEnv}
                  icon={Map}
                  accent="cream"
                />
              </div>
            </ArchitectureLayer>

            <Connector direction="down" />

            {/* Layer 3: MARL */}
            <ArchitectureLayer
              index={3}
              title={labels.layer3}
              delay={0.2}
            >
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <ModuleCard
                  label={labels.modIql}
                  icon={GitBranch}
                  accent="pink"
                  formula="Q_i(\\tau^i, u^i)"
                />
                <ModuleCard
                  label={labels.modQmix}
                  icon={Network}
                  accent="pink"
                  formula="Q_{tot}(\\tau, \\mathbf{u})"
                />
              </div>
            </ArchitectureLayer>

            <Connector direction="up" />

            {/* Layer 4: Configuration */}
            <ArchitectureLayer
              index={4}
              title={labels.layer4}
              delay={0.3}
            >
              <div className="space-y-3">
                <p className="font-mono text-xs text-muted uppercase tracking-wider mb-3">
                  {labels.dataPipeline}
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <ModuleCard
                    label={labels.modDefaults}
                    icon={Settings}
                    accent="mint"
                  />
                  <ModuleCard
                    label={labels.modLoader}
                    icon={Loader2}
                    accent="cream"
                  />
                  <ModuleCard
                    label={labels.modSchema}
                    icon={Database}
                    accent="pink"
                  />
                </div>
              </div>
            </ArchitectureLayer>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function Connector({ direction }: { direction: 'up' | 'down' }) {
  return (
    <div className="relative h-8 sm:h-10 flex justify-center shrink-0">
      <FlowLine
        pathD="M 0.5 0 L 0.5 1"
        direction={direction}
        id={`conn-${direction}`}
        responsive={true}
      />
    </div>
  );
}
