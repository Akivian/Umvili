'use client';

import { cn } from '@/lib/utils';

interface TechnicalHeaderProps {
  title: string;
  subtitle?: string;
  className?: string;
  mono?: boolean;
}

export function TechnicalHeader({ title, subtitle, className, mono }: TechnicalHeaderProps) {
  return (
    <header className={cn('space-y-1', className)}>
      <h2
        className={cn(
          'text-xl font-semibold tracking-tight text-white',
          mono && 'font-mono text-base'
        )}
      >
        {title}
      </h2>
      {subtitle && (
        <p className="text-sm text-muted text-[#A1A1AA] max-w-xl">{subtitle}</p>
      )}
    </header>
  );
}
