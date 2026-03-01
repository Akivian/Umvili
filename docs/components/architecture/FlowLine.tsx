'use client';

import { motion } from 'framer-motion';

export type FlowDirection = 'up' | 'down';

interface FlowLineProps {
  pathD: string;
  direction: FlowDirection;
  id?: string;
  className?: string;
  responsive?: boolean;
}

export function FlowLine({
  pathD,
  direction: _direction,
  id = 'flow',
  className = '',
  responsive = true,
}: FlowLineProps) {
  // 流动动画已暂时关闭（原意：虚线沿连接线移动表示「数据流」向上/向下，避免闪烁）
  return (
    <svg
      className={`absolute inset-0 w-full h-full pointer-events-none overflow-visible ${className} ${responsive ? 'hidden sm:block' : ''}`}
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      aria-hidden
    >
      <defs>
        <linearGradient id={`${id}-gradient`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#27272A" stopOpacity="0.4" />
          <stop offset="50%" stopColor="#27272A" stopOpacity="1" />
          <stop offset="100%" stopColor="#27272A" stopOpacity="0.4" />
        </linearGradient>
      </defs>
      <motion.path
        d={pathD}
        fill="none"
        stroke="#27272A"
        strokeWidth="1"
        strokeOpacity="0.5"
        vectorEffect="non-scaling-stroke"
        initial={{ pathLength: 0 }}
        whileInView={{ pathLength: 1 }}
        viewport={{ once: true, margin: '-20px' }}
        transition={{ duration: 0.8, ease: 'easeInOut' }}
      />
    </svg>
  );
}

export function FlowLineVertical({
  fromY,
  toY,
  x = '50%',
  direction,
  id = 'flow-v',
}: {
  fromY: string;
  toY: string;
  x?: string;
  direction: FlowDirection;
  id?: string;
}) {
  const pathD = `M ${x} ${fromY} L ${x} ${toY}`;
  return (
    <FlowLine pathD={pathD} direction={direction} id={id} responsive={true} />
  );
}
