import { notFound } from 'next/navigation';
import { locales, type Locale } from '../../common/i18n';
import { Divider } from '../../components/design-system';
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
    <div className="min-h-screen flex flex-col relative z-10">
      <header className="sticky top-0 z-50 bg-black/95 backdrop-blur supports-[backdrop-filter]:bg-black/80">
        <div className="w-full max-w-[1400px] mx-auto px-6 sm:px-8 lg:px-10 h-14 flex items-center">
          <Link href={`/${locale}`} className="font-semibold text-white tracking-tight shrink-0">
            Umvili
          </Link>
          <div className="flex items-center gap-6 sm:gap-8 ml-auto">
            <nav className="hidden sm:flex items-center gap-6 text-sm text-zinc-400">
              <a href="#background" className="hover:text-white transition-colors">
                {t['nav.background']}
              </a>
              <a href="#functionality" className="hover:text-white transition-colors">
                {t['nav.functionality']}
              </a>
              <a href="#showcase" className="hover:text-white transition-colors">
                {t['nav.showcase']}
              </a>
              <a href="#architecture" className="hover:text-white transition-colors">
                {t['nav.architecture']}
              </a>
              <a href="#guide" className="hover:text-white transition-colors">
                {t['nav.guide']}
              </a>
              <a href="#contact" className="hover:text-white transition-colors">
                {t['nav.contact']}
              </a>
            </nav>
            <LanguageSwitcher />
          </div>
        </div>
        <Divider />
      </header>
      <main className="flex-1">{children}</main>
    </div>
  );
}
