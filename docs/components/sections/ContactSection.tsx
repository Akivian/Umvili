'use client';

import Link from 'next/link';
import { Github, FileText } from 'lucide-react';

interface ContactSectionProps {
  github: string;
  docs: string;
}

export function ContactSection({ github, docs }: ContactSectionProps) {
  return (
    <footer id="contact" className="border-t border-[#27272A]">
      <div className="max-w-5xl mx-auto px-6 py-12">
        <div className="flex flex-wrap items-center justify-between gap-6">
          <p className="text-xs text-muted font-mono">Umvili · MARL Visualization Platform</p>
          <nav className="flex items-center gap-6">
            <Link
              href="https://github.com/Akivian/Umvili"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-muted hover:text-white transition-colors"
            >
              <Github className="w-4 h-4 stroke-[1.5px]" />
              {github}
            </Link>
            <Link
              href="https://github.com/Akivian/Umvili/tree/main/docs/reference"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-muted hover:text-white transition-colors"
            >
              <FileText className="w-4 h-4 stroke-[1.5px]" />
              {docs}
            </Link>
          </nav>
        </div>
      </div>
    </footer>
  );
}
