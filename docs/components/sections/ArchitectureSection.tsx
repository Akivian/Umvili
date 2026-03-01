'use client';

import { motion } from 'framer-motion';
import {
  GitBranch,
  Network,
  GitFork,
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
  marlMaster: string;
  marlSimplified: string;
  logicController: string;
  analyticsHub: string;
  dataSnapshot: string;
  snapshotLabels: string;
  algorithmImpl: string;
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

          {/* Level 1: PRESENTATION LAYER — clean horizontal row */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-30px' }}
            transition={{ type: 'spring', stiffness: 120, damping: 20, delay: 0 }}
            className="mt-10"
          >
            <LayerBox className="min-h-[88px]">
              <TechLabel className="block mb-3">{labels.presentation}</TechLabel>
              <div className="flex flex-wrap items-center gap-6 sm:gap-8">
                <PresentationModule icon={ArrowDown} label={`${labels.simData} (Input)`} sub="State, Metrics" />
                <PresentationModule icon={Map} label={labels.maps} sub={`${labels.mapsSugar}, ${labels.mapsSpice}, ${labels.mapsHazard}`} />
                <PresentationModule icon={LayoutList} label={labels.tabs} sub={`${labels.tabsOverview}, ${labels.tabsBehavior}, …`} />
                <PresentationModule icon={Settings} label={`${labels.configBuilder} (Output)`} sub="Config → Core" />
              </div>
            </LayerBox>
          </motion.div>

          {/* Level 2: CORE LAYER (SIMULATION ENGINE) — largest container, shell */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-30px' }}
            transition={{ type: 'spring', stiffness: 120, damping: 20, delay: 0.05 }}
            className="mt-6"
          >
            <div className="rounded-sm border border-zinc-800 bg-zinc-950/50 p-6 overflow-hidden">
              <TechLabel className="block mb-4">{labels.coreLayer}</TechLabel>

              {/* Pipeline: Config Entry → Env Engine → MARL Master → Logic Controller → Analytics Hub */}
              <div className="flex flex-wrap items-center gap-2 sm:gap-4">
                <PipelineNode icon={Settings} label={labels.configEntry} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={Cpu} label={labels.envEngine} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={Boxes} label={labels.marlMaster} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={LayoutGrid} label={labels.logicController} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={BarChart3} label={labels.analyticsHub} />
              </div>

              {/* Level 3 (nested in Core): ALGORITHM IMPLEMENTATIONS — MARL portal sub-grid */}
              <div className="mt-6 pt-5 border-t border-zinc-800">
                <TechLabel className="block mb-3">{labels.algorithmImpl}</TechLabel>
                <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto_1fr] gap-4 items-start">
                  <MarlCard
                    icon={GitBranch}
                    title={labels.modIql}
                    items={[labels.iqlQNet, labels.iqlPolicy, labels.iqlPER, labels.iqlTarget]}
                  />
                  <div className="hidden sm:flex items-center justify-center pt-8 text-zinc-600" aria-hidden>
                    <GitFork size={18} strokeWidth={1} />
                  </div>
                  <MarlCard
                    icon={Network}
                    title={labels.modQmix}
                    items={[labels.qmixMixing, labels.qmixHyper, labels.qmixAgentNets]}
                  />
                </div>
              </div>

              {/* Exit: Simulation Data Snapshot at bottom of Core */}
              <div className="mt-6 pt-4 border-t border-zinc-800 flex items-center gap-2">
                <ArrowDown {...ICON_PROPS} className="text-zinc-500 shrink-0" />
                <TechLabel>{labels.dataSnapshot}</TechLabel>
                <span className="font-mono text-xs text-zinc-600">[{labels.snapshotLabels}]</span>
              </div>
            </div>
          </motion.div>

          {/* Level 4: CONFIGURATION LAYER */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-30px' }}
            transition={{ type: 'spring', stiffness: 120, damping: 20, delay: 0.1 }}
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

function PresentationModule({
  icon: Icon,
  label,
  sub,
}: {
  icon: React.ElementType;
  label: string;
  sub: string;
}) {
  return (
    <div className="flex items-start gap-2 min-w-0">
      <Icon {...ICON_PROPS} className="text-zinc-500 shrink-0 mt-0.5" />
      <div>
        <span className="font-mono text-xs text-white">{label}</span>
        <p className="font-mono text-xs text-zinc-500 mt-0.5">{sub}</p>
      </div>
    </div>
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

/** MARL implementation card: path label core.marl.v1, border-zinc-900, subtle glow */
function MarlCard({
  icon: Icon,
  title,
  items,
}: {
  icon: React.ElementType;
  title: string;
  items: string[];
}) {
  return (
    <div className="rounded-sm border border-zinc-900 bg-zinc-900/30 p-4 shadow-[0_0_10px_rgba(255,255,255,0.05)]">
      <p className="font-mono text-[10px] text-zinc-600 uppercase tracking-wider mb-2">core.marl.v1</p>
      <div className="flex items-center gap-2 text-white font-mono text-xs mb-2">
        <Icon {...ICON_PROPS} className="text-zinc-500 shrink-0" />
        {title}
      </div>
      <ul className="pl-5 font-mono text-xs text-zinc-500 space-y-0.5">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
