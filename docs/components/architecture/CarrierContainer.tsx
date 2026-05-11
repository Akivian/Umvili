'use client';

import { cn } from '../../common/utils';

/** L-shaped corner bracket (2px zinc-700) for blueprint schematic feel */
function LBracket({ position }: { position: 'tl' | 'tr' | 'bl' | 'br' }) {
  const base = 'absolute w-3 h-3 border-[2px] border-zinc-700 pointer-events-none';
  const positions = {
    tl: 'top-0 left-0 border-r-0 border-b-0 rounded-tl',
    tr: 'top-0 right-0 border-l-0 border-b-0 rounded-tr',
    bl: 'bottom-0 left-0 border-r-0 border-t-0 rounded-bl',
    br: 'bottom-0 right-0 border-l-0 border-t-0 rounded-br',
  };
  return <div className={cn(base, positions[position])} aria-hidden />;
}

interface CarrierContainerProps {
  children: React.ReactNode;
  className?: string;
  /** Hex metadata in top-right, e.g. "LAYER_ID: 0x01" */
  hexId?: string;
}

/** Container with L-brackets and optional hex metadata */
export function CarrierContainer({ children, className, hexId }: CarrierContainerProps) {
  return (
    <div
      className={cn(
        'relative rounded-sm overflow-hidden',
        'bg-[rgba(9,9,11,0.6)] backdrop-blur-xl',
        'border border-[rgba(39,39,42,0.8)]',
        'shadow-[inset_0_1px_1px_0_rgba(255,255,255,0.1)]',
        className
      )}
    >
      <LBracket position="tl" />
      <LBracket position="tr" />
      <LBracket position="bl" />
      <LBracket position="br" />
      {hexId && (
        <span className="absolute top-2 right-2 font-mono text-[10px] text-zinc-500 tracking-wider">
          {hexId}
        </span>
      )}
      <div className="relative z-10 p-4 sm:p-6">{children}</div>
    </div>
  );
}
