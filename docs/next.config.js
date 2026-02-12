/** @type {import('next').NextConfig} */
const path = require('path');

const nextConfig = {
  output: 'export',
  basePath: process.env.NODE_ENV === 'production' ? '/Umvili' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/Umvili/' : '',
  trailingSlash: true,
  images: { unoptimized: true },
  webpack: (config) => {
    config.resolve.alias['@'] = path.resolve(__dirname);
    return config;
  },
};

module.exports = nextConfig;
