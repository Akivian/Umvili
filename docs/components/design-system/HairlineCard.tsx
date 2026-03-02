'use client';

import { cn } from '../../common/utils';

type AccentGlow = 'mint' | 'cream' | 'pink';

const accentHoverGlow: Record<AccentGlow, string> = {
  mint: 'hover:shadow-[0_0_24px_rgba(167,243,208,0.06)]',
  cream: 'hover:shadow-[0_0_24px_rgba(254,243,199,0.06)]',
  pink: 'hover:shadow-[0_0_24px_rgba(251,207,232,0.06)]',
};

interface HairlineCardProps {
  children: React.ReactNode;
  className?: string;
  as?: 'div' | 'article' | 'section';
  accent?: AccentGlow;
}

export function HairlineCard({
  children,
  className,
  as: Component = 'div',
  accent,
}: HairlineCardProps) {
  return (
    <Component
      className={cn(
        'rounded-sm p-5',
        'bg-zinc-950/50 backdrop-blur-md',
        'border border-zinc-800/50',
        'transition-all duration-300 ease-out',
        'hover:-translate-y-1 hover:border-zinc-400',
        accent && accentHoverGlow[accent],
        className
      )}
    >
      {children}
    </Component>
  );
}
