/**
 * 資產路徑解析：讓專案能同時部署在網域根目錄與 GitHub Pages 子路徑下。
 *
 * GitHub Pages 的 project site 會把網站掛在 `/<repo>/` 底下，
 * 此時 next.config.js 會設定 basePath，並透過 NEXT_PUBLIC_BASE_PATH 傳進前端。
 * public/ 內的靜態檔案不會被 Next 自動加上 basePath，必須自己補。
 */
const BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH ?? '').replace(/\/$/, '');

export function assetPath(path: string): string {
    let clean = path.replace(/^\//, '');
    return `${BASE_PATH}/${clean}`;
}
