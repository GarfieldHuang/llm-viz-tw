import React from 'react';
import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { flyAccumulate } from "./BackpropAnim";
import { getBlockValueAtIdx } from "../components/DataFlow";

export function walkthrough16_BackEmbedding(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Embedding) {
        return;
    }

    setInitialCamera(state, new Vec3(15.654, 0.000, -80.905), new Vec3(287.000, 14.500, 3.199));
    wt.dimHighlightBlocks = [layout.tokEmbedObj, layout.posEmbedObj];

    commentary(wt, null, 0)`
這是最後一站。梯度從損失出發，穿過輸出層、三層 transformer、注意力和 Layer Norm，
現在回到了最開頭的兩張查表。

嵌入層的反向很特別，因為前向那一步**根本不是運算** —— 它是查表。
一個運算的反向是另一個運算，但查表的反向是什麼？`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.6);
    let t_fade = afterTime(null, 1.3);

    breakAfter();
    commentary(wt)`
答案是 **scatter-add**：把梯度加回去被查到的那一行。

前向時 token "C"（索引 2）去 ${c_blockRef('詞嵌入表', layout.tokEmbedObj)} 取了第 2 行。
反向時，這個位置的梯度就整條加回第 2 行。沒被查到的行，一個字都不會動。

把滑鼠移到嵌入表上，浮層寫的是 **scatter-add, only rows that were looked up** ——
而且用的是「加等於」而不是「等於」，因為同一行會被加很多次。`;
    breakAfter();

    let t_dTok = afterTime(null, 4.8);

    breakAfter();
    commentary(wt)`
看清楚這張表的形狀：詞彙表只有 ${c_dimRef('三個 token', DimStyle.n_vocab)}（A、B、C），
但序列有六個位置。所以**同一行會被好幾個位置同時加到**。

這就是 scatter-add 那個 add 的來源 —— 不是覆蓋，是累加。
如果序列裡出現了三次 "B"，那三個位置的責任全部疊在 "B" 那一行上。

這也解釋了為什麼常見的詞學得快、罕見的詞學得慢：**梯度的累積次數就是出現次數**。`;
    breakAfter();

    let t_dPos = afterTime(null, 4.8);

    breakAfter();
    commentary(wt)`
${c_blockRef('位置嵌入表', layout.posEmbedObj)} 的情況剛好相反。

位置 0 只會對應到第 0 行、位置 1 只會對應第 1 行 —— 一對一，不會重複。
所以它每一行最多只拿到一份梯度。

而且因為這個例子的損失只看位置 5，**位置 6 以後的行全是灰的**。
它們在這次前向裡根本沒被用到（因果遮罩擋住了），自然也沒有責任。`;
    breakAfter();

    let t_end = afterTime(null, 3.2);

    breakAfter();
    commentary(wt)`
到這裡，反向傳播結束了。

沒有下一步了 —— 嵌入表是葉節點，它的梯度就是終點。
從損失那三個數字 [1, 0, −1] 開始，一路分流、轉置相乘、穿過 softmax 和 GELU、
被 Layer Norm 扣掉整行的平均，最後散落回這兩張表上。

整趟走完，模型裡**每一個參數**都拿到了一個數字，回答同一個問題：
「把我調高一點點，損失會變多少？」

optimizer 接手的就是這些數字。反向傳播的工作，到此為止。`;
    breakAfter();

    moveCameraTo(state, t_moveCamera, new Vec3(9.5, 0, -48.4), new Vec3(287, 14.5, 1.4));

    let relevant = new Set([
        layout.idxObj,
        layout.tokEmbedObj,
        layout.posEmbedObj,
        layout.residual0,
    ]);
    focusBackwardScene(state, relevant, t_fade.t);

    if (t_dTok.t > 0) {
        // scatter-add 的重點是「累加」：讓前六個位置的梯度一個接一個飛進詞嵌入表，
        // 重複出現的 token 就會看到同一行被加了好幾次。
        let srcs = [];
        let dests = [];
        for (let t = 0; t <= 5; t++) {
            let tok = getBlockValueAtIdx(layout.idxObj, new Vec3(t, 0, 0));
            if (tok === null || tok === undefined) continue;
            srcs.push({ blk: layout.residual0, colIdx: t });
            dests.push({ blk: layout.tokEmbedObj, colIdx: Math.round(tok) });
        }
        flyAccumulate(state, t_dTok, srcs, dests);

        processBackwardChain(state, t_dTok, [layout.residual0, layout.tokEmbedObj]);
    }
    if (t_dPos.t > 0) {
        processBackwardChain(state, t_dPos, [layout.residual0, layout.posEmbedObj]);
    }
    if (t_end.t > 0) {
        processBackwardChain(state, t_end, [
            layout.residual0, layout.tokEmbedObj, layout.posEmbedObj,
        ]);
    }
}
