import React from 'react';
import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { flySliceTo } from "./BackpropAnim";

export function walkthrough10_BackLoss(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Loss) {
        return;
    }

    // 損失只問這個位置，動畫也都圍著它演
    let LOSS_POS = 5;

    setInitialCamera(state, new Vec3(-20.203, 0.000, -1642.819), new Vec3(281.600, -7.900, 2.298));
    wt.dimHighlightBlocks = [layout.logits, layout.logitsSoftmax];

    commentary(wt, null, 0)`
反向傳播要有起點。前向跑完得到一串機率，但「機率」本身不是我們要最小化的東西 ——
**損失**才是。這一章就是把損失變成第一份梯度。

先說這個範例的損失怎麼來的。這個 nano-gpt 在排序任務上已經完全收斂，
每個位置的預測機率都是 1.0，用正解去算梯度會得到一片近乎全零的畫面 ——
那雖然正確，但什麼也看不到。

所以我們問一個**反事實**的問題：在位置 ${c_dimRef('t = 5', DimStyle.T)}，
模型深信下一個字是 "A"，如果正解其實是 "C" 呢？誤差會怎麼往回傳？
這樣算出來的梯度乾淨、好看，而且好驗證。

（接下來各章的矩陣式，一律用**畫面上看到的形狀**來寫：每個區塊就是一個矩陣，
列是橫的、行是直的。例如 Layer Norm 在畫面上是 (C, t)，式子裡就當它是 (C, t)。
滑鼠移到任一格上，側邊欄會用同一套座標把算式展開。）`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.6);
    let t_fade = afterTime(null, 1.3);

    breakAfter();
    commentary(wt)`
現在看 ${c_blockRef('Logits Softmax', layout.logitsSoftmax)} 這一塊 ——
把滑鼠移上去，浮層會告訴你**這一塊沒有梯度資料**。

那不是漏掉，是刻意的。我們**不對機率微分**。

softmax 和 cross-entropy 如果分開算，中間會冒出一個 Jacobian 矩陣，數值也不穩定。
但把這兩步合起來推導，那個矩陣會整個消掉，只剩一個好得不像話的式子：

dL/dlogits = p − y

式子裡的 p 是模型吐出的機率，y 是正解的 one-hot。**預測減去答案**，就是梯度。
所以實作上 softmax 這一層在反向根本不存在，梯度是直接落在 ${c_blockRef('Logits', layout.logits)} 上的。`;
    breakAfter();

    let t_dLogits = afterTime(null, 4.0);

    breakAfter();
    commentary(wt)`
梯度落下來了。注意兩件事。

第一，**只有位置 5 那一行有東西**，其他位置全是灰的。因為損失只問了那一個位置，
其他位置對這個損失完全沒有貢獻 —— 沒有貢獻，就沒有責任。

第二，那一行的三個數字是 [1, 0, −1]：模型給 "A" 的機率是 1（要往下壓），
給 "C" 的是 0 而正解是 C（要往上推），"B" 沒被選也不是答案（不動）。
整個訓練訊號，就濃縮成這三個數字。`;
    breakAfter();

    let t_dWlm = afterTime(null, 4.0);

    breakAfter();
    commentary(wt)`
接著梯度分兩路走。

一路進 ${c_blockRef('LM Head 權重', layout.lmHeadWeight)}：dWlm = dLogits · LNfᵀ。
這是輸出層要怎麼調 —— optimizer 拿得走的第一份東西。

另一路往下，回到 ${c_blockRef('最後的 Layer Norm', layout.ln_f.lnResid)}：dLNf = Wlm ᵀ · dLogits。
這份梯度接下來要穿過三層 transformer，一路回到嵌入表。`;
    breakAfter();

    let t_dLnf = afterTime(null, 4.0);

    moveCameraTo(state, t_moveCamera, new Vec3(-24.4, 0, -1660.9), new Vec3(281.6, -7.9, 1.5));

    let relevant = new Set([
        layout.ln_f.lnResid,
        layout.lmHeadWeight,
        layout.logits,
        layout.logitsAgg1,
        layout.logitsAgg2,
        layout.logitsSoftmax,
    ]);
    focusBackwardScene(state, relevant, t_fade.t);

    if (t_dLogits.t > 0) {
        // 先演給你看：機率那一行整條飛下來，減去正解，落成 dLogits。
        // 「預測減去答案」這句話，用看的比用讀的快。
        flySliceTo(state, t_dLogits,
            { blk: layout.logitsSoftmax, colIdx: LOSS_POS },
            { blk: layout.logits, colIdx: LOSS_POS },
            { symbol: '—' });
        processBackwardChain(state, t_dLogits, [layout.logits], { animateFirst: true });
    }
    if (t_dWlm.t > 0) {
        processBackwardChain(state, t_dWlm, [layout.logits, layout.lmHeadWeight]);
    }
    if (t_dLnf.t > 0) {
        processBackwardChain(state, t_dLnf, [layout.logits, layout.ln_f.lnResid]);
    }
}
