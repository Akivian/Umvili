'use client';

/** Flow direction type (kept for API compatibility). FlowLine no longer renders any SVG to avoid stray lines. */
export type FlowDirection = 'up' | 'down';

interface FlowLineProps {
  pathD?: string;
  direction?: FlowDirection;
  id?: string;
  className?: string;
  responsive?: boolean;
}

/** No-op: renders nothing. Connector lines were removed to fix stray solid/dashed lines in MARL & Presentation areas. */
export function FlowLine(_props: FlowLineProps) {
  return null;
}

export function FlowLineVertical(_props: {
  fromY: string;
  toY: string;
  x?: string;
  direction: FlowDirection;
  id?: string;
}) {
  return null;
}
