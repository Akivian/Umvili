'use client';

import { cn } from '@/lib/utils';

interface HairlineCardProps {
  children: React.ReactNode;
  className?: string;
  as?: 'div' | 'article' | 'section';
}

export function HairlineCard({ children, className, as: Component = 'div' }: HairlineCardProps) {
  return (
    <Component
      className={cn(
        'border border-[#27272A] bg-black rounded-sm p-5',
        'hover:border-[#3f3f46] transition-colors duration-200',
        className
      )}
    >
      {children}
    </Component>
  );
}
