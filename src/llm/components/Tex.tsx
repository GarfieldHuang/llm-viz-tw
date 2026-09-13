import React from 'react';
import katex from 'katex';

/**
 * KaTeX 公式。
 *
 * 側邊欄在懸停時每一幀都會重新渲染，同一條式子反覆排版太浪費，
 * 所以把排好的 HTML 快取起來。
 */
const cache = new Map<string, string>();

export function texToHtml(tex: string, display: boolean): string {
    let key = (display ? 'D:' : 'I:') + tex;
    let html = cache.get(key);
    if (html === undefined) {
        html = katex.renderToString(tex, {
            displayMode: display,
            throwOnError: false,
            strict: 'ignore',
        });
        if (cache.size > 4000) {
            cache.clear();
        }
        cache.set(key, html);
    }
    return html;
}

export const Tex: React.FC<{ tex: string, block?: boolean, className?: string }> = ({ tex, block, className }) => {
    return <span className={className} dangerouslySetInnerHTML={{ __html: texToHtml(tex, !!block) }} />;
};
