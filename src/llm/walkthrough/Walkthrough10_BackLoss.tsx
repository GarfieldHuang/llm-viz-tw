import React from 'react';
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { argmaxAbs, demoAngle, focusCell, sceneLossSeed, scenePairDot } from "./BackpropScenes";
import { embedInline } from "./Walkthrough01_Prelim";
import { Tex } from "../components/Tex";
import { fwdAt } from "../components/GradMath";

export function walkthrough10_BackLoss(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Loss) {
        return;
    }

    let POS = state.gradData?.lossPos ?? 5;
    let TARGET = state.gradData?.lossTarget ?? 2;
    let { C } = layout.shape;
    let cell = layout.cell;
    let lnf = layout.ln_f.lnResid;

    setInitialCamera(state, new Vec3(-20.203, 0.000, -1642.819), new Vec3(281.600, -7.900, 2.298));
    wt.dimHighlightBlocks = [layout.logits, layout.logitsSoftmax, layout.lmHeadWeight, lnf];

    // 示範用的通道：最後 Layer Norm 在位置 5 上絕對值最大的那一格，乘出來的東西最看得清楚
    let cStar = argmaxAbs(C, c => fwdAt(state, lnf, new Vec3(POS, c, 0)));

    commentary(wt, null, 0)`
反向傳播要有起點。前向跑完得到一串機率，但我們要最小化的不是機率，是**損失**。
這一章把損失變成第一份梯度，再往回送一步。

先說這個範例的損失怎麼來的。畫面上的輸入是 C B A B B C，模型要把它排序。
在位置 ${c_dimRef('t = 5', DimStyle.T)}，它要生出排序後的第一個字，而它非常確定是 "A" —— 也答對了。
答對的時候梯度幾乎全是 0，畫面上什麼也看不到。

所以我們問一個**反事實**的問題：如果正解其實是 "C" 呢？誤差會怎麼往回傳？

（隨時可以點一下 3D 畫面裡的任一格，側邊欄會從前向式出發，一步一步推到那一格的梯度，
最後跟 PyTorch 算出來的數字對答案。）`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.6);
    let t_fade = afterTime(null, 1.3);

    breakAfter();
    commentary(wt)`
看位置 5 的 ${c_blockRef('機率', layout.logitsSoftmax)}：三格分別是 "A"、"B"、"C" 的機率，
"A" 那一格幾乎是 1，另外兩格幾乎是 0。

梯度的規則出奇地簡單：**每一格機率，減掉它的答案**。正解 "C" 那一格減 1，其他格減 0。
看著三格機率飛下來，各自減掉答案，落進 ${c_blockRef('Logits', layout.logits)}。`;
    breakAfter();

    let t_zoomSeed = afterTime(null, 0.8);
    let t_seed = afterTime(null, 4.5, 0.3);
    let t_seedFill = afterTime(null, 1.0);

    breakAfter();
    commentary(wt)`
落下來的三個數字是 [1, 0, −1]。
"A" 的機率是 1 卻不是答案，要往下壓（梯度為正，代表「調高會讓損失變大」）；
"C" 是答案但機率是 0，要往上推；"B" 兩邊都不是，不動。整個訓練訊號，就濃縮成這三個數字。
其他位置全是灰的：損失只問了位置 5，其他位置沒有責任。

${embedInline(<Tex block tex={String.raw`dz_{v,5} = p_{v,5} - \mathbb{1}[v = \text{C}]`} />)}

為什麼 softmax 那一塊沒有梯度？softmax 和 cross-entropy 合在一起微分時，
中間那個牽連整列的 Jacobian 會整個消掉，只剩「預測減答案」—— 梯度是直接種在 logits 上的。`;
    breakAfter();

    commentary(wt)`
接著梯度分兩路。先看 ${c_blockRef('LM Head 權重', layout.lmHeadWeight)}。

前向時，權重的同一格被**每個位置**都用了一次。所以它的梯度要把每個位置的貢獻加起來：
dLogits 的那一列（沿著位置 t）和 ${c_blockRef('Layer Norm', lnf)} 的那一列，逐項相乘再相加。

看成對飛出來的格子：只有 t = 5 那一對不是 0，因為其他位置的 dLogits 全是 0。`;
    breakAfter();

    let t_zoomW = afterTime(null, 0.8);
    let t_dWlmDemo = afterTime(null, 5.0, 0.3);
    let t_dWlmFill = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`dW^{\text{lm}}_{v,c} = \sum_{t} dz_{v,t}\;\text{LN}_{c,t}`} />)}

這是 optimizer 拿得走的第一份東西。

另一路往下，回到 ${c_blockRef('最後的 Layer Norm', lnf)}。這次加總的方向換了：
Layer Norm 的一格在前向時被**三個詞彙**的 logits 用到，所以沿詞彙 v 加總 ——
位置 5 那一行的三格 dLogits，各乘上權重的一格，再相加。`;
    breakAfter();

    let t_zoomL = afterTime(null, 0.8);
    let t_dLnfDemo = afterTime(null, 4.5, 0.3);
    let t_dLnfFill = afterTime(null, 2.0);
    let t_zoomOut = afterTime(null, 0.8);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`d\text{LN}_{c,t} = \sum_{v} dz_{v,t}\;W^{\text{lm}}_{v,c}`} />)}

同一個矩陣乘法，對兩個運算元各有一條反向式，差別只在「沿哪一軸加總」。
這份梯度接下來要穿過三層 transformer，一路回到嵌入表。`;

    // 相機依時間順序排：moveCameraTo 靠呼叫順序找「上一個」相機位置
    let overview = new Vec3(-24.4, 0, -1660.9);
    moveCameraTo(state, t_moveCamera, overview, new Vec3(281.6, -7.9, 1.5));
    moveCameraTo(state, t_zoomSeed, focusCell(state, layout.logits, new Vec3(POS, 1, 0), cell * 2, -cell * 2), demoAngle(0.5));
    moveCameraTo(state, t_zoomW, focusCell(state, layout.lmHeadWeight, new Vec3(cStar, 0, 0), -cell * 6, -cell * 5), demoAngle(0.85));
    moveCameraTo(state, t_zoomL, focusCell(state, lnf, new Vec3(POS, cStar, 0), -cell * 6, -cell * 3), demoAngle(0.85));
    moveCameraTo(state, t_zoomOut, overview, new Vec3(281.6, -7.9, 1.5));

    focusBackwardScene(state, new Set([
        lnf, layout.lmHeadWeight, layout.logits, layout.logitsAgg1, layout.logitsAgg2, layout.logitsSoftmax,
    ]), t_fade.t);

    // 先亮已經算好的區塊，再演動畫
    if (t_seedFill.t > 0) {
        processBackwardChain(state, t_seedFill, [layout.logits], { animateFirst: true });
    }
    if (t_dWlmFill.t > 0) {
        processBackwardChain(state, t_dWlmFill, [layout.logits, layout.lmHeadWeight]);
    }
    if (t_dLnfFill.t > 0) {
        processBackwardChain(state, t_dLnfFill, [layout.logits, lnf]);
    }

    sceneLossSeed(state, t_seed, layout.logitsSoftmax, layout.logits, POS, TARGET);

    scenePairDot(state, t_dWlmDemo,
        { blk: layout.logits, fixDim: Dim.Y, fixIdx: 0, kind: 'grad' },
        { blk: lnf, fixDim: Dim.Y, fixIdx: cStar, kind: 'fwd' },
        { blk: layout.lmHeadWeight, idx: new Vec3(cStar, 0, 0) },
        { maxPairs: POS + 1 });

    scenePairDot(state, t_dLnfDemo,
        { blk: layout.logits, fixDim: Dim.X, fixIdx: POS, kind: 'grad' },
        { blk: layout.lmHeadWeight, fixDim: Dim.X, fixIdx: cStar, kind: 'fwd' },
        { blk: lnf, idx: new Vec3(POS, cStar, 0) });
}
