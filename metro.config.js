const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
// const config = getDefaultConfig(__dirname);

module.exports = (async () => {
  const defaultConfig = await getDefaultConfig(__dirname);
  const { resolver } = defaultConfig.resolver;
  defaultConfig.resolver = {
    ...resolver,
    assetExts: ['bin', 'txt', 'jpg', 'ttf', 'png'],
  }
  return defaultConfig
})();
