import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#000000',
        foreground: '#FFFFFF',
        muted: '#A1A1AA',
        border: '#27272A',
        borderSubtle: '#18181B',
      },
      fontFamily: {
        sans: ['var(--font-geist-sans)', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['var(--font-jetbrains-mono)', 'JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        sm: '4px',
        DEFAULT: '4px',
      },
      borderWidth: {
        hairline: '1px',
      },
      boxShadow: {
        glow: '0 0 20px rgba(255,255,255,0.03)',
      },
    },
  },
  plugins: [],
};

export default config;
