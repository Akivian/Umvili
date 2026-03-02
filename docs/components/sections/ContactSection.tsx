'use client';

import Link from 'next/link';
import { Github, FileText } from 'lucide-react';
import { Divider } from '../design-system';

interface ContactSectionProps {
  github: string;
  docs: string;
}

export function ContactSection({ github, docs }: ContactSectionProps) {
  return (
    <footer id="contact">
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10">
        <Divider />
      </div>
      <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 py-12">
        <div className="flex flex-wrap items-center justify-between gap-6">
          <p className="text-xs text-zinc-400 font-mono">Umvili · MARL Visualization Platform</p>
          <nav className="flex items-center gap-6">
            <Link
              href="https://github.com/Akivian/Umvili"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-zinc-400 hover:text-white transition-colors"
            >
              <Github className="w-4 h-4" strokeWidth={1.2} />
              {github}
            </Link>
            <Link
              href="https://github.com/Akivian/Umvili/tree/main/docs/reference"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-zinc-400 hover:text-white transition-colors"
            >
              <FileText className="w-4 h-4" strokeWidth={1.2} />
              {docs}
            </Link>
          </nav>
        </div>
      </div>
    </footer>
  );
}
