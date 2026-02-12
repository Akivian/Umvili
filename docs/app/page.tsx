import { redirect } from 'next/navigation';
import { defaultLocale } from '../common/i18n';

export default function Home() {
  redirect(`/${defaultLocale}`);
}
