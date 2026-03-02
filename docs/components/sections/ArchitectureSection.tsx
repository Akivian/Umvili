'use client';

import { useState } from 'react';
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
import { Divider, TechnicalHeader } from '../design-system';
import { CarrierContainer, FlowPaths, TechLabel } from '../architecture';

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
  marlKernel: string;
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

const ICON_PROPS = { size: 14, strokeWidth: 1.2 };

export function ArchitectureSection({ labels }: { labels: ArchitectureLabels }) {
  const [marlHovered, setMarlHovered] = useState(false);

  return (
    <section id="architecture" className="bg-black relative">
      <FlowPaths />
      <div className="relative w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-20">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.35 }}
        >
          <TechnicalHeader title={labels.title} subtitle={labels.subtitle} />

          {/* Level 1: PRESENTATION LAYER — L-brackets, LAYER_ID: 0x01 */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-30px' }}
            transition={{ type: 'spring', stiffness: 120, damping: 20, delay: 0 }}
            className="mt-10"
          >
            <CarrierContainer hexId="LAYER_ID: 0x01" className="min-h-[88px]">
              <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-500 block mb-3">
                {labels.presentation}
              </span>
              <div className="flex flex-wrap items-center gap-6 sm:gap-8">
                <PresentationModule
                  icon={ArrowDown}
                  label={`${labels.simData} (Input)`}
                  sub="State, Metrics"
                />
                <PresentationModule
                  icon={Map}
                  label={labels.maps}
                  sub={`${labels.mapsSugar}, ${labels.mapsSpice}, ${labels.mapsHazard}`}
                />
                <PresentationModule
                  icon={LayoutList}
                  label={labels.tabs}
                  sub={`${labels.tabsOverview}, ${labels.tabsBehavior}, …`}
                />
                <PresentationModule
                  icon={Settings}
                  label={`${labels.configBuilder} (Output)`}
                  sub="Config → Core"
                />
              </div>
            </CarrierContainer>
          </motion.div>

          {/* Level 2: CORE LAYER — largest, CORE_V2.0, pipeline + MARL nested */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-30px' }}
            transition={{ type: 'spring', stiffness: 120, damping: 20, delay: 0.05 }}
            className="mt-6"
          >
            <CarrierContainer hexId="CORE_V2.0" className="min-h-[280px]">
              <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-500 block mb-4">
                {labels.coreLayer}
              </span>

              {/* Internal pipeline: Config Entry → Env Engine → MARL KERNEL → Logic Controller → Analytics Hub */}
              <div className="flex flex-wrap items-center gap-2 sm:gap-4">
                <PipelineNode icon={Settings} label={labels.configEntry} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={Cpu} label={labels.envEngine} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <MarlKernelNode
                  label={labels.marlKernel}
                  onHoverChange={setMarlHovered}
                />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={LayoutGrid} label={labels.logicController} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={BarChart3} label={labels.analyticsHub} />
              </div>

              {/* Variable injection: t mod 4 == 0, env.grow_back() */}
              <p className="mt-2 font-mono text-[10px] text-zinc-500">
                t mod 4 == 0, env.grow_back()
              </p>

              {/* Engine Room: ALGORITHM IMPLEMENTATIONS — sunken sub-grid below MARL KERNEL */}
              <div className="mt-6 pt-5">
                <div className="h-px w-full bg-[linear-gradient(to_right,transparent_0%,#18181b_15%,#18181b_85%,transparent_100%)] mb-5" />
                <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-500 block mb-3">
                  {labels.algorithmImpl}
                </span>
                <div
                  className={`
                    grid grid-cols-1 sm:grid-cols-[1fr_auto_1fr] gap-4 items-start p-4 rounded-sm
                    bg-zinc-950/60 backdrop-blur-sm
                    border border-dashed border-zinc-800
                    shadow-[inset_0_2px_8px_rgba(0,0,0,0.3)]
                    transition-opacity duration-200
                    ${marlHovered ? 'opacity-100 ring-1 ring-zinc-600/30' : 'opacity-90'}
                  `}
                >
                  <MarlCard
                    icon={GitBranch}
                    title={labels.modIql}
                    items={[labels.iqlQNet, labels.iqlPolicy, labels.iqlPER, labels.iqlTarget]}
                    meta="γ=0.99, Soft Update (τ)"
                  />
                  <div className="hidden sm:flex items-center justify-center pt-8 text-zinc-600" aria-hidden>
                    <GitFork size={18} strokeWidth={1.2} />
                  </div>
                  <MarlCard
                    icon={Network}
                    title={labels.modQmix}
                    items={[labels.qmixMixing, labels.qmixHyper, labels.qmixAgentNets]}
                    meta="γ=0.99, Soft Update (τ)"
                  />
                </div>
              </div>

              {/* Exit: Simulation Data Snapshot */}
              <div className="mt-6 pt-4 flex items-center gap-2">
                <ArrowDown {...ICON_PROPS} className="text-zinc-500 shrink-0" />
                <TechLabel>{labels.dataSnapshot}</TechLabel>
                <span className="font-mono text-[10px] text-zinc-500">[{labels.snapshotLabels}]</span>
              </div>
            </CarrierContainer>
          </motion.div>

          {/* Level 3: CONFIGURATION LAYER */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-30px' }}
            transition={{ type: 'spring', stiffness: 120, damping: 20, delay: 0.1 }}
            className="mt-6"
          >
            <CarrierContainer hexId="LAYER_ID: 0x03">
              <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-500 block mb-3">
                {labels.configLayer}
              </span>
              <div className="flex flex-wrap items-center gap-2 sm:gap-4">
                <PipelineNode icon={Database} label={labels.defaults} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={Copy} label={labels.loaderCopy} />
                <ArrowRight {...ICON_PROPS} className="text-zinc-600 shrink-0" />
                <PipelineNode icon={Database} label={labels.schemaAppSimUi} />
              </div>
            </CarrierContainer>
          </motion.div>
        </motion.div>
      </div>
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10">
        <Divider />
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
        <span className="font-mono text-[12px] text-white">{label}</span>
        <p className="font-mono text-[10px] text-zinc-400 mt-0.5">{sub}</p>
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
      className="inline-flex items-center gap-1.5 px-2 py-1 rounded-sm border border-zinc-800 bg-zinc-900/30 font-mono text-[12px] text-zinc-300"
    >
      <Icon {...ICON_PROPS} className="text-zinc-500 shrink-0" />
      {label}
    </span>
  );
}

/** MARL KERNEL node with pulsing LED and hover to highlight nested sub-cards */
function MarlKernelNode({
  label,
  onHoverChange,
}: {
  label: string;
  onHoverChange: (hovered: boolean) => void;
}) {
  return (
    <span
      className="inline-flex items-center gap-2 px-2 py-1 rounded-sm border border-zinc-800 bg-zinc-900/30 font-mono text-[12px] text-zinc-300 cursor-default"
      onMouseEnter={() => onHoverChange(true)}
      onMouseLeave={() => onHoverChange(false)}
    >
      <span
        className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"
        aria-hidden
      />
      <span className="flex items-center gap-1.5">
        <Boxes {...ICON_PROPS} className="text-zinc-500 shrink-0" />
        {label}
      </span>
    </span>
  );
}

/** MARL implementation card: bg-zinc-950, dashed border, γ=0.99 meta */
function MarlCard({
  icon: Icon,
  title,
  items,
  meta,
}: {
  icon: React.ElementType;
  title: string;
  items: string[];
  meta?: string;
}) {
  return (
    <div className="rounded-sm border border-dashed border-zinc-800 bg-zinc-950 p-4 shadow-[inset_0_1px_2px_rgba(0,0,0,0.2)]">
      <p className="font-mono text-[10px] text-zinc-600 uppercase tracking-wider mb-1">core.marl.v1</p>
      {meta && (
        <p className="font-mono text-[10px] text-zinc-500 mb-2">{meta}</p>
      )}
      <div className="flex items-center gap-2 text-white font-mono text-[12px] mb-2">
        <Icon {...ICON_PROPS} className="text-zinc-500 shrink-0" />
        {title}
      </div>
      <ul className="pl-5 font-mono text-[12px] text-zinc-400 space-y-0.5">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
