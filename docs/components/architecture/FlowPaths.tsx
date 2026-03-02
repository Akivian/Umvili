'use client';

/** SVG paths (0.5px) with stroke-dashoffset photon animation for data flow direction */
export function FlowPaths() {
  return (
    <div className="absolute left-0 top-0 bottom-0 w-16 pointer-events-none hidden xl:block" aria-hidden>
      <svg viewBox="0 0 64 100" className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id="flowArchA" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(63,63,70,0.5)" />
            <stop offset="100%" stopColor="rgba(63,63,70,0.1)" />
          </linearGradient>
          <linearGradient id="flowArchB" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="rgba(63,63,70,0.5)" />
            <stop offset="100%" stopColor="rgba(63,63,70,0.1)" />
          </linearGradient>
        </defs>
        {/* Path A: Config Builder -> Config Entry (down) */}
        <path
          d="M 16 15 L 16 50"
          fill="none"
          stroke="url(#flowArchA)"
          strokeWidth="0.5"
          strokeLinecap="round"
          opacity={0.5}
        />
        <path
          d="M 16 15 L 16 50"
          fill="none"
          stroke="rgba(161,161,170,0.6)"
          strokeWidth="0.5"
          strokeLinecap="round"
          strokeDasharray="6 8"
          strokeDashoffset="0"
          className="flow-photon-down"
        />
        {/* Path B: Analytics Hub -> Sim Data (up) */}
        <path
          d="M 48 85 L 48 50"
          fill="none"
          stroke="url(#flowArchB)"
          strokeWidth="0.5"
          strokeLinecap="round"
          opacity={0.5}
        />
        <path
          d="M 48 85 L 48 50"
          fill="none"
          stroke="rgba(161,161,170,0.6)"
          strokeWidth="0.5"
          strokeLinecap="round"
          strokeDasharray="6 8"
          strokeDashoffset="0"
          className="flow-photon-up"
          style={{ animationDelay: '0.5s' }}
        />
      </svg>
    </div>
  );
}
