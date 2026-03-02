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
        badge={t['hero.badge']}
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
          presentation: t['arch.presentation'],
          coreLayer: t['arch.coreLayer'],
          configLayer: t['arch.configLayer'],
          simData: t['arch.simData'],
          configBuilder: t['arch.configBuilder'],
          maps: t['arch.maps'],
          mapsSugar: t['arch.mapsSugar'],
          mapsSpice: t['arch.mapsSpice'],
          mapsHazard: t['arch.mapsHazard'],
          tabs: t['arch.tabs'],
          tabsOverview: t['arch.tabsOverview'],
          tabsBehavior: t['arch.tabsBehavior'],
          configEntry: t['arch.configEntry'],
          envEngine: t['arch.envEngine'],
          marlMaster: t['arch.marlMaster'],
          marlKernel: t['arch.marlKernel'],
          marlSimplified: t['arch.marlSimplified'],
          logicController: t['arch.logicController'],
          analyticsHub: t['arch.analyticsHub'],
          dataSnapshot: t['arch.dataSnapshot'],
          snapshotLabels: t['arch.snapshotLabels'],
          algorithmImpl: t['arch.algorithmImpl'],
          defaults: t['arch.defaults'],
          loaderCopy: t['arch.loaderCopy'],
          schemaAppSimUi: t['arch.schemaAppSimUi'],
          modIql: t['arch.mod.iql'],
          modQmix: t['arch.mod.qmix'],
          iqlQNet: t['arch.iqlQNet'],
          iqlPolicy: t['arch.iqlPolicy'],
          iqlPER: t['arch.iqlPER'],
          iqlTarget: t['arch.iqlTarget'],
          qmixMixing: t['arch.qmixMixing'],
          qmixHyper: t['arch.qmixHyper'],
          qmixAgentNets: t['arch.qmixAgentNets'],
        }}
      />
      <GuideSection title={t['guide.title']} subtitle={t['guide.subtitle']} />
      <ContactSection github={t['footer.github']} docs={t['footer.docs']} />
    </>
  );
}
