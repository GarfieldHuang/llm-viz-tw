import React from 'react';
import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { flyPairDot } from "./BackpropAnim";

export function walkthrough11_BackMlp(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Mlp) {
        return;
    }

    let block0 = layout.blocks[0];

    setInitialCamera(state, new Vec3(-154.755, 0.000, -460.042), new Vec3(289.100, -8.900, 2.298));
    wt.dimHighlightBlocks = [block0.mlpFc, block0.mlpAct, block0.mlpResult];

    commentary(wt, null, 0)`
MLP 的反向是整個 transformer 裡最單純的一段，值得先看，因為它可以當成對照組 ——
等一下看 attention 時，你才會知道那裡為什麼特別。

前向是三步：先升維到 ${c_dimRef('4C', DimStyle.C)}、過 GELU、再降回 ${c_dimRef('C', DimStyle.C)}。
反向就是倒著走這三步，而且**每一格各算各的**，彼此不干涉。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.6);
    let t_fade = afterTime(null, 1.3);

    breakAfter();
    commentary(wt)`
梯度從 ${c_blockRef('MLP 殘差', block0.mlpResidual)} 分一路進來，落到
${c_blockRef('MLP Result', block0.mlpResult)}。

這是加法的分支，所以梯度**原封不動**地複製過來 —— 一個字都沒改。
（那條分流我們在「殘差分流」那一章專門講。）`;
    breakAfter();

    let t_dMlpOut = afterTime(null, 3.2);

    breakAfter();
    commentary(wt)`
接著往回穿過第二個線性層，進到 ${c_blockRef('GELU 的輸出', block0.mlpAct)}，
再穿過 GELU 本身，到 ${c_blockRef('GELU 的輸入', block0.mlpFc)}。

GELU 的反向只有一件事：**乘上導數**。

dFc = dGelu ⊙ gelu′(Fc)

滑鼠移到 ${c_blockRef('MLP Activation', block0.mlpAct)} 上，浮層畫的不再是 GELU 本身，
而是 **gelu′ 的曲線**，並且在你這一格的 x 位置標一個點。

看那條曲線的左半邊：x 越負，導數越接近 0。也就是說，**被 GELU 壓掉的神經元，
梯度也一起被壓掉了** —— 它對這次的錯誤不負責，也就學不到東西。
這正是激活函數同時決定「誰能出聲」和「誰要負責」的地方。`;
    breakAfter();

    let t_dGelu = afterTime(null, 5.6);

    breakAfter();
    commentary(wt)`
再往回一步就回到 ${c_blockRef('Layer Norm 2', block0.ln2.lnResid)}，這一段的中間量就結束了。

注意整段的關鍵對照：這裡每一格的梯度只跟**它自己的前向值**有關 ——
dFc 的第 i 格只看 Fc 的第 i 格，不看同一列的其他人。

等一下你會看到 attention 完全不是這樣：那裡整列是綁在一起的。`;
    breakAfter();

    let t_dLn2 = afterTime(null, 4.0);

    breakAfter();
    commentary(wt)`
最後收權重。MLP 佔了整個模型大約三分之二的參數，所以這四塊是這一層裡
optimizer 拿走最多東西的地方：

dWfc = dFc ᵀ · LN2　　dWproj = dMlp ᵀ · Gelu

偏置的梯度更簡單 —— 它對每個位置都加同一個數，所以反向就是**把整批位置加起來**。
浮層在偏置上會多一個 Σ，就是這個意思。`;
    breakAfter();

    let t_dW = afterTime(null, 4.8);

    moveCameraTo(state, t_moveCamera, new Vec3(-160.2, 0, -455.6), new Vec3(289.1, -8.9, 1.7));

    let relevant = new Set([
        block0.ln2.lnResid,
        block0.mlpFcWeight,
        block0.mlpFcBias,
        block0.mlpFc,
        block0.mlpAct,
        block0.mlpProjWeight,
        block0.mlpProjBias,
        block0.mlpResult,
        block0.mlpResidual,
    ]);
    focusBackwardScene(state, relevant, t_fade.t);

    if (t_dMlpOut.t > 0) {
        processBackwardChain(state, t_dMlpOut, [block0.mlpResidual, block0.mlpResult]);
    }
    if (t_dGelu.t > 0) {
        // 穿過第二個線性層到 GELU 輸出，再穿過 GELU 到它的輸入
        processBackwardChain(state, t_dGelu, [block0.mlpResult, block0.mlpAct, block0.mlpFc]);
    }
    if (t_dLn2.t > 0) {
        processBackwardChain(state, t_dLn2, [block0.mlpFc, block0.ln2.lnResid]);
    }
    if (t_dW.t > 0) {
        // 權重梯度是「沿著整批位置的點積」—— 讓格子成對飛出來相乘再相加，
        // 比印一個 dot( , ) 清楚得多。只演前幾對，不然畫面塞不下。
        flyPairDot(state, t_dW,
            { blk: block0.mlpFc, alongX: false, fixed: 30 },
            { blk: block0.ln2.lnResid, alongX: true, fixed: 12 },
            { blk: block0.mlpFcWeight, idx: new Vec3(30, 12, 0) },
            { maxPairs: 6 });

        processBackwardChain(state, t_dW, [
            block0.mlpFcWeight, block0.mlpFcBias,
            block0.mlpProjWeight, block0.mlpProjBias,
        ], { animateFirst: true });
    }
}
