'use client';

import { cn } from '../../common/utils';

/** macOS-style traffic light dots */
function TrafficLights({ className }: { className?: string }) {
  return (
    <div className={cn('flex items-center gap-1.5', className)}>
      <span className="w-2.5 h-2.5 rounded-full bg-[#FF5F57]" />
      <span className="w-2.5 h-2.5 rounded-full bg-[#FEBC2E]" />
      <span className="w-2.5 h-2.5 rounded-full bg-[#28C840]" />
    </div>
  );
}

/** Blinking underscore cursor */
function BlinkCursor({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        'inline-block w-3 h-px ml-0.5 bg-zinc-400 align-middle',
        'animate-[blink_1s_ease-in-out_infinite]',
        className
      )}
      aria-hidden
    />
  );
}

interface InteractiveTerminalProps {
  children: React.ReactNode;
  className?: string;
  /** Show traffic lights in header. Default true. */
  trafficLights?: boolean;
}

export function InteractiveTerminal({
  children,
  className,
  trafficLights = true,
}: InteractiveTerminalProps) {
  return (
    <div
      className={cn(
        'rounded-sm overflow-hidden',
        'bg-[#18181b]',
        'border border-[rgba(39,39,42,0.8)]',
        'shadow-[inset_0_1px_1px_0_rgba(255,255,255,0.05)]',
        className
      )}
    >
      <div className="flex items-center gap-2 border-b border-zinc-800/80 px-4 py-2.5">
        {trafficLights && <TrafficLights />}
      </div>
      <div className="p-4 font-mono text-sm overflow-x-auto">{children}</div>
    </div>
  );
}

InteractiveTerminal.TrafficLights = TrafficLights;
InteractiveTerminal.BlinkCursor = BlinkCursor;
