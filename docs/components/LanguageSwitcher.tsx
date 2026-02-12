'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '../lib/utils';
import { locales, type Locale } from '../lib/i18n';

const labels: Record<Locale, string> = { en: 'EN', zh: '中文' };

export function LanguageSwitcher() {
  const pathname = usePathname();
  const base = pathname?.replace(/^\/(en|zh)/, '') || '';

  return (
    <nav className="flex items-center gap-0 border border-[#27272A] rounded-sm overflow-hidden">
      {locales.map((locale) => {
        const href = `/${locale}${base || ''}`;
        const isActive = pathname?.startsWith(`/${locale}`);
        return (
          <Link
            key={locale}
            href={href}
            className={cn(
              'px-3 py-1.5 text-xs font-medium transition-colors',
              isActive
                ? 'bg-white text-black'
                : 'text-muted hover:text-white hover:bg-[#18181B]'
            )}
          >
            {labels[locale]}
          </Link>
        );
      })}
    </nav>
  );
}
