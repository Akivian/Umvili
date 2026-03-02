'use client';

import { cn } from '../../common/utils';

/** 1px zinc-900 divider with subtle gradient fade at both ends */
export function Divider({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'w-full h-px',
        'bg-[linear-gradient(to_right,transparent_0%,#18181b_15%,#18181b_85%,transparent_100%)]',
        className
      )}
    />
  );
}
