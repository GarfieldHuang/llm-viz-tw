/** @type {import('next').NextConfig} */

// GitHub Pages 的 project site 會掛在 https://<user>.github.io/<repo>/ 底下，
// 需要 basePath；部署在網域根目錄（或 user site）時留空即可。
// 由 .github/workflows/deploy.yml 在 build 時注入。
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

const nextConfig = {
    // 靜態匯出：產生純 HTML/JS/CSS 到 out/，不需要 Node server。
    output: 'export',

    // 每個路由輸出成 <route>/index.html，GitHub Pages 才不會 404。
    trailingSlash: true,

    basePath: basePath,
    assetPrefix: basePath || undefined,

    // 靜態匯出不能用 Next 的圖片最佳化伺服器。
    images: { unoptimized: true },

    reactStrictMode: false,
    productionBrowserSourceMaps: true,
    experimental: {
        appDir: true,
    },

    // 注意：output: 'export' 不支援 redirects()／API routes，
    // 原上游的 /llm-viz -> /llm 轉址已移除。
};

module.exports = nextConfig;
