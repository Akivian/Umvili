'use client';

import { motion } from 'framer-motion';
import {
  GitBranch,
  Network,
  LayoutGrid,
  Map,
  LayoutList,
  Settings,
  ArrowRight,
  Cpu,
  Boxes,
  BarChart3,
  Copy,
  Database,
  ArrowDown,
} from 'lucide-react';
import { TechnicalHeader } from '../design-system';
import { LayerBox, TechLabel } from '../architecture';

export interface ArchitectureLabels {
  title: string;
  subtitle: string;
  marlDeepDive: string;
  presentation: string;
  coreLayer: string;
  configLayer: string;
  simData: string;
  configBuilder: string;
  maps: string;
  mapsSugar: string;
  mapsSpice: string;
  mapsHazard: string;
  tabs: string;
  tabsOverview: string;
  tabsBehavior: string;
  configEntry: string;
  envEngine: string;
  marlSimplified: string;
  logicController: string;
  analyticsHub: string;
  dataSnapshot: string;
  snapshotLabels: string;
  defaults: string;
  loaderCopy: string;
  schemaAppSimUi: string;
  modIql: string;
  modQmix: string;
  iqlQNet: string;
  iqlPolicy: string;
  iqlPER: string;
  iqlTarget: string;
  qmixMixing: string;
  qmixHyper: string;
  qmixAgentNets: string;
}

const ICON_PROPS = { size: 14, strokeWidth: 1 };

export function ArchitectureSection({ labels }: { labels: ArchitectureLabels }) {
  return (
    <section id="architecture" className="border-b border-zinc-800 bg-black">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
        >
          <TechnicalHeader title={labels.title} subtitle={labels.subtitle} />

          {/* Grid: Top (40/60), Middle (full), Bottom (full) */}
          <div className="mt-10 grid grid-cols-1 lg:grid-cols-[2fr_3fr] gap-6 overflow-hidden">
            {/* ─── Top Row Left: MARL Deep-Dive (40%) ─── */}
            <div className="overflow-hidden rounded-sm min-w-0">
              <LayerBox layout delay={0} className="min-h-[220px]">
                <TechLabel className="block mb-3">{labels.marlDeepDive}</TechLabel>
                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-2 text-white font-mono text-xs">
                      <GitBranch {...ICON_PROPS} className="text-zinc-500 shrink-0" />
                      {labels.modIql}
                    </div>
                    <ul className="pl-5 font-mono text-xs text-zinc-500 space-y-0.5">
                      <li>{labels.iqlQNet}</li>
                      <li>{labels.iqlPolicy}</li>
                      <li>{labels.iqlPER}</li>
                      <li>{labels.iqlTarget}</li>
                    </ul>
                  </div>
                  <div className="border-t border-zinc-800 pt-3 space-y-1.5">
                    <div className="flex items-center gap-2 text-white font-mono text-xs">
                      <Network {...ICON_PROPS} className="text-zinc-500 shrink-0" />
                      {labels.modQmix}
                    </div>
                    <ul className="pl-5 font-mono text-xs text-zinc-500 space-y-0.5">
                      <li>{labels.qmixMixing}</li>
                      <li>{labels.qmixHyper}</li>
                      <li>{labels.qmixAgentNets}</li>
                    </ul>
                  </div>
                </div>
              </LayerBox>
            </div>

            {/* ─── Top Row Right: Presentation Layer (60%) ─── */}
            <div className="overflow-hidden rounded-sm min-w-0">
              <LayerBox layout delay={0.05} className="min-h-[220px]">
                <TechLabel className="block mb-3">{labels.presentation}</TechLabel>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <TechLabel className="flex items-center gap-1.5 mb-1">
                      <ArrowDown {...ICON_PROPS} /> {labels.simData} (Input)
                    </TechLabel>
                    <p className="font-mono text-xs text-zinc-500">State, Metrics</p>
                  </div>
                  <div>
                    <TechLabel className="flex items-center gap-1.5 mb-1">
                      <Map {...ICON_PROPS} /> {labels.maps}
                    </TechLabel>
                    <p className="font-mono text-xs text-zinc-500">
                      {labels.mapsSugar}, {labels.mapsSpice}, {labels.mapsHazard}
                    </p>
                  </div>
                  <div>
                    <TechLabel className="flex items-center gap-1.5 mb-1">
                      <LayoutList {...ICON_PROPS} /> {labels.tabs}
                    </TechLabel>
                    <p className="font-mono text-xs text-zinc-500">
                      {labels.tabsOverview}, {labels.tabsBehavior}, …
                    </p>
                  </div>
                  <div>
                    <TechLabel className="flex items-center gap-1.5 mb-1">
                      <Settings {...ICON_PROPS} /> {labels.configBuilder} (Output)
                    </TechLabel>
                    <p className="font-mono text-xs text-zinc-500">Config → Core</p>
                  </div>
                </div>
              </LayerBox>
            </div>
          </div>

          {/* ─── Middle Row: Core Pipeline (full width) ─── */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-30px' }}
            transition={{ type: 'spring', stiffness: 120, damping: 20, delay: 0.1 }}
            className="mt-6"
          >
            <LayerBox className="min-h-[100px]">
              <TechLabel className="block mb-3">{labels.coreLayer}</TechLabel>
              <div className="flex flex-wrap items-center gap-2 sm:gap-4">
                <PipelineNode icon={Settings} label={labels.configEntry} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={Cpu} label={labels.envEngine} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={Boxes} label={labels.marlSimplified} id="core-marl-node" />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={LayoutGrid} label={labels.logicController} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={BarChart3} label={labels.analyticsHub} />
              </div>
              <div className="mt-4 pt-3 border-t border-zinc-800 flex items-center gap-2">
                <ArrowDown {...ICON_PROPS} className="text-zinc-500 shrink-0" />
                <TechLabel>{labels.dataSnapshot}</TechLabel>
                <span className="font-mono text-xs text-zinc-600">[{labels.snapshotLabels}]</span>
              </div>
            </LayerBox>
          </motion.div>

          {/* ─── Bottom Row: Configuration Layer ─── */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-30px' }}
            transition={{ type: 'spring', stiffness: 120, damping: 20, delay: 0.15 }}
            className="mt-6"
          >
            <LayerBox>
              <TechLabel className="block mb-3">{labels.configLayer}</TechLabel>
              <div className="flex flex-wrap items-center gap-2 sm:gap-4">
                <PipelineNode icon={Database} label={labels.defaults} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={Copy} label={labels.loaderCopy} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={Database} label={labels.schemaAppSimUi} />
              </div>
            </LayerBox>
          </motion.div>

        </motion.div>
      </div>
    </section>
  );
}

function PipelineNode({
  icon: Icon,
  label,
  id,
}: {
  icon: React.ElementType;
  label: string;
  id?: string;
}) {
  return (
    <span
      id={id}
      className="inline-flex items-center gap-1.5 px-2 py-1 rounded-sm border border-zinc-800 bg-zinc-900/30 font-mono text-xs text-zinc-300"
    >
      <Icon {...ICON_PROPS} className="text-zinc-500 shrink-0" />
      {label}
    </span>
  );
}

