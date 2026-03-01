'use client';

import { motion } from 'framer-motion';
import { cn } from '../../common/utils';

interface ArchitectureLayerProps {
  index: number;
  title: string;
  children: React.ReactNode;
  className?: string;
  /** For staggered entrance */
  delay?: number;
}

const layerIndexFormat = (i: number) => String(i).padStart(2, '0');

export function ArchitectureLayer({
  index,
  title,
  children,
  className,
  delay = 0,
}: ArchitectureLayerProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{
        type: 'spring',
        stiffness: 120,
        damping: 20,
        delay,
      }}
      className={cn('relative', className)}
    >
      {/* Scientific Blueprint header */}
      <div className="flex items-baseline gap-3 mb-4">
        <span className="font-mono text-xs text-[#27272A] tracking-widest">
          {layerIndexFormat(index)}
        </span>
        <h3 className="font-mono text-sm font-medium text-white uppercase tracking-wider">
          {title}
        </h3>
      </div>
      {/* Bento container with dot grid */}
      <div
        className={cn(
          'relative rounded-sm border border-[#27272A] p-5',
          'bg-[#000]',
          'min-h-[100px]',
          'before:absolute before:inset-0 before:rounded-sm before:pointer-events-none before:opacity-[0.35]',
          'before:bg-[length:20px_20px] before:bg-[radial-gradient(circle_at_50%_50%,#27272A_1px,transparent_1px)]'
        )}
      >
        <div className="relative z-10">{children}</div>
      </div>
    </motion.div>
  );
}
