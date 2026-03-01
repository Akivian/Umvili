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
        labels={{
          title: t['arch.title'],
          subtitle: t['arch.subtitle'],
          layer1: t['arch.layer1'],
          layer2: t['arch.layer2'],
          layer3: t['arch.layer3'],
          layer4: t['arch.layer4'],
          modViz: t['arch.mod.viz'],
          modUi: t['arch.mod.ui'],
          modEngine: t['arch.mod.engine'],
          modAgents: t['arch.mod.agents'],
          modEnv: t['arch.mod.env'],
          modIql: t['arch.mod.iql'],
          modQmix: t['arch.mod.qmix'],
          modDefaults: t['arch.mod.defaults'],
          modLoader: t['arch.mod.loader'],
          modSchema: t['arch.mod.schema'],
          dataPipeline: t['arch.dataPipeline'],
        }}
      />
      <GuideSection title={t['guide.title']} subtitle={t['guide.subtitle']} />
      <ContactSection github={t['footer.github']} docs={t['footer.docs']} />
    </>
  );
}
