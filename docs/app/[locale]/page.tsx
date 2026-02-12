import { getMessages } from '../../common/i18n';
import type { Locale } from '../../common/i18n';
import { HeroSection } from '../../components/sections/HeroSection';
import { BackgroundSection } from '../../components/sections/BackgroundSection';
import { BentoSection } from '../../components/sections/BentoSection';
import { ShowcaseSection } from '../../components/sections/ShowcaseSection';
import { ArchitectureSection } from '../../components/sections/ArchitectureSection';
import { GuideSection } from '../../components/sections/GuideSection';
import { ContactSection } from '../../components/sections/ContactSection';

export default function Home({ params }: { params: { locale: string } }) {
  const locale = params.locale as Locale;
  const t = getMessages(locale);

  return (
    <>
      <HeroSection
        title={t['hero.title']}
        tagline={t['hero.tagline']}
        cta={t['hero.cta']}
        command={t['hero.command']}
      />
      <BackgroundSection
        originTitle={t['background.origin']}
        originText={t['background.origin.text']}
        problemTitle={t['background.problem']}
        problemText={t['background.problem.text']}
      />
      <BentoSection
        title={t['functionality.title']}
        subtitle={t['functionality.subtitle']}
      />
      <ShowcaseSection title={t['showcase.title']} subtitle={t['showcase.subtitle']} />
      <ArchitectureSection
        title={t['arch.title']}
        subtitle={t['arch.subtitle']}
        labels={{
          config: t['arch.config'],
          core: t['arch.core'],
          marl: t['arch.marl'],
          utils: t['arch.utils'],
          env: t['arch.env'],
          agents: t['arch.agents'],
          simulation: t['arch.simulation'],
          visualization: t['arch.visualization'],
        }}
      />
      <GuideSection title={t['guide.title']} subtitle={t['guide.subtitle']} />
      <ContactSection github={t['footer.github']} docs={t['footer.docs']} />
    </>
  );
}
