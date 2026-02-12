/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: process.env.NODE_ENV === 'production' ? '/Umvili' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/Umvili/' : '',
  trailingSlash: true,
  images: { unoptimized: true },
};

module.exports = nextConfig;
