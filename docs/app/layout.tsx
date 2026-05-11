import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { ParallaxDotGrid } from '../components/ParallaxDotGrid';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-geist-sans',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
});

export const metadata: Metadata = {
  title: 'Umvili — MARL Visualization Platform',
  description: 'Sandbox-style multi-agent reinforcement learning algorithm comparison with real-time visualization.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="antialiased scroll-smooth">
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans text-foreground min-h-screen relative`}>
        <ParallaxDotGrid />
        {children}
      </body>
    </html>
  );
}
