import React from 'react';
import { Phase } from "./Walkthrough";
import { commentary, embed, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import s from './Walkthrough.module.scss';
import { Vec3 } from '@/src/utils/vector';

let minGptLink = 'https://github.com/karpathy/minGPT';
let pytorchLink = 'https://pytorch.org/';
let andrejLink = 'https://karpathy.ai/';
let zeroToHeroLink = 'https://karpathy.ai/zero-to-hero.html';

export function walkthrough01_Prelim(args: IWalkthroughArgs) {
    let { state, walkthrough: wt } = args;

    if (wt.phase !== Phase.Intro_Prelim) {
        return;
    }

    setInitialCamera(state, new Vec3(184.744, 0.000, -636.820), new Vec3(296.000, 16.000, 13.500));

    let c0 = commentary(wt, null, 0)`
在深入瞭解演算法的複雜性之前，我們先來做個簡單的回顧。

本指南側重於 _推理_ 而非訓練，因此只是整個機器學習過程的一小部分。在我們的例子中，模型的權重已經預先訓練好，我們使用推理過程來生成輸出。這可以直接在瀏覽器中執行。
在我們的例子中，模型的權重已經預先訓練好，我們使用推理過程來生成輸出。這可以直接在瀏覽器中執行。

這裡展示的模型是 GPT（生成式預訓練轉換器）系列的一部分，可以說是 "基於上下文的 token 預測器"。OpenAI 在 2018 年引入了這一家族，其著名成員包括 GPT-2、GPT-3 和 GPT-3.5 Turbo，後者是廣泛使用的 ChatGPT 的基礎。它還可能與 GPT-4 有關，但具體細節仍不得而知。

本指南受到 ${embedLink('minGPT', minGptLink)} GitHub 專案的啟發，該專案是 ${embedLink('Andrej Karpathy', andrejLink)} 在 ${embedLink('PyTorch', pytorchLink)} 中建立的最小 GPT 實現。他的 YouTube ${embedLink("Neural Networks: Zero to Hero", zeroToHeroLink)} 系列和 minGPT 專案是建立本指南的寶貴參考資源。這裡介紹的玩具模型就是基於 minGPT 專案中的一個模型。

好的，讓我們開始吧！
`;

}

export function embedLink(a: React.ReactNode, href: string) {
    return embedInline(<a className={s.externalLink} href={href} target="_blank" rel="noopener noreferrer">{a}</a>);
}

export function embedInline(a: React.ReactNode) {
    return { insertInline: a };
}


// Another similar model is BERT (bidirectional encoder representations from transformers), a "context-aware text encoder" commonly
// used for tasks like document classification and search.  Newer models like Facebook's LLaMA (large language model architecture), continue to use
// a similar transformer architecture, albeit with some minor differences.
