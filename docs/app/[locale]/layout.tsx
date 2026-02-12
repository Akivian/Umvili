import { notFound } from 'next/navigation';
import { locales, type Locale } from '../../common/i18n';
import { LanguageSwitcher } from '../../components/LanguageSwitcher';
import Link from 'next/link';
import { getMessages } from '../../common/i18n';

export function generateStaticParams() {
  return locales.map((locale) => ({ locale }));
}

export default function LocaleLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { locale: string };
}) {
  const locale = params.locale as Locale;
  if (!locales.includes(locale)) notFound();
  const t = getMessages(locale);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-50 border-b border-[#27272A] bg-black/95 backdrop-blur supports-[backdrop-filter]:bg-black/80">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
          <Link href={`/${locale}`} className="font-semibold text-white tracking-tight">
            Umvili
          </Link>
          <nav className="hidden sm:flex items-center gap-6 text-sm text-muted">
            <a href={`/${locale}#background`} className="hover:text-white transition-colors">
              {t['nav.background']}
            </a>
            <a href={`/${locale}#functionality`} className="hover:text-white transition-colors">
              {t['nav.functionality']}
            </a>
            <a href={`/${locale}#showcase`} className="hover:text-white transition-colors">
              {t['nav.showcase']}
            </a>
            <a href={`/${locale}#architecture`} className="hover:text-white transition-colors">
              {t['nav.architecture']}
            </a>
            <a href={`/${locale}#guide`} className="hover:text-white transition-colors">
              {t['nav.guide']}
            </a>
            <a href={`/${locale}#contact`} className="hover:text-white transition-colors">
              {t['nav.contact']}
            </a>
          </nav>
          <LanguageSwitcher />
        </div>
      </header>
      <main className="flex-1">{children}</main>
    </div>
  );
}
