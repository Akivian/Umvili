# Umvili project site (Next.js)

Linear-style showcase for Umvili. EN / 中文 via top-right switcher.

## Run locally

```bash
cd docs
npm install
npm run dev
```

Open [http://localhost:3001](http://localhost:3001). Root redirects to `/en`.

## Build for GitHub Pages

```bash
npm run build
```

Static output is in `out/`. For **Project Pages** (e.g. `https://<user>.github.io/Umvili/`), the app is built with `basePath: '/Umvili'`. Deploy the contents of `out/` to your Pages branch or docs folder as needed (e.g. copy `out/*` into `docs/` if you publish from the `docs` folder).

## Stack

- Next.js 14 (App Router), static export
- Tailwind CSS (black #000, zinc borders, Inter + JetBrains Mono)
- Framer Motion, Lucide React
- i18n: `lib/i18n.ts` (EN / ZH)
