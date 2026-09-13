import React from 'react';
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { argmaxAbs, demoAngle, focusCell, sceneCollapse, sceneElemMul, sceneMoveSlice, scenePairDot, shiftToBlock } from "./BackpropScenes";
import { embedInline } from "./Walkthrough01_Prelim";
import { Tex } from "../components/Tex";
import { fwdAt, gradAt } from "../components/GradMath";
import { drawDataFlow } from "../components/DataFlow";

export function walkthrough11_BackMlp(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Mlp) {
        return;
    }

    // 反向是倒著走的：梯度離開最後的 Layer Norm，第一個抵達的是最後一層的 MLP
    let li = layout.blocks.length - 1;
    let blk = layout.blocks[li];
    let POS = state.gradData?.lossPos ?? 5;
    let { C } = layout.shape;
    let cell = layout.cell;
    let cam = (x: number, z: number) => shiftToBlock(layout, li, new Vec3(x, 0, z));

    setInitialCamera(state, cam(-154.755, -460.042), new Vec3(289.100, -8.900, 2.298));
    wt.dimHighlightBlocks = [blk.mlpFc, blk.mlpAct, blk.mlpResult];

    // 示範用的神經元：一個「有在出力」的（x 為正）、一個「被 GELU 壓掉」的（x 很負）
    let fcX = (k: number) => fwdAt(state, blk.mlpFc, new Vec3(k, POS, 0));
    let dAct = (k: number) => gradAt(blk.mlpAct, new Vec3(k, POS, 0));
    let kAlive = argmaxAbs(4 * C, k => { let x = fcX(k); return x !== null && x > 0.5 ? dAct(k) : 0; });
    let kDead = argmaxAbs(4 * C, k => { let x = fcX(k); return x !== null && x < -2 ? dAct(k) : 0; });
    let cStar = argmaxAbs(C, c => gradAt(blk.ln2.lnResid, new Vec3(POS, c, 0)));

    commentary(wt, null, 0)`
MLP 的反向是整個 transformer 裡最單純的一段，值得先看 —— 它可以當對照組，
等一下看 attention 時，你才知道那裡為什麼特別。

我們從**最後一層**看起：反向傳播是倒著走的，梯度離開最後的 Layer Norm 之後，第一個抵達的就是這裡。

前向是三步：升維到 ${c_dimRef('4C', DimStyle.C)}、過 GELU、再降回 ${c_dimRef('C', DimStyle.C)}。反向就是倒著走這三步。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.6);
    let t_fade = afterTime(null, 1.3);

    breakAfter();
    commentary(wt)`
梯度從 ${c_blockRef('MLP 殘差', blk.mlpResidual)} 分一份進來，落到 ${c_blockRef('MLP 的輸出', blk.mlpResult)}。
這是加法的分支，所以那一行梯度**原封不動**地搬過去。（殘差那一章會細講這件事。）`;
    breakAfter();

    let t_copy = afterTime(null, 3.0, 0.3);
    let t_copyFill = afterTime(null, 0.8);

    breakAfter();
    commentary(wt)`
接著往回穿過第二個線性層。

前向時，${c_blockRef('MLP 輸出', blk.mlpResult)} 的每一格，都是 ${c_blockRef('GELU 輸出', blk.mlpAct)} 那一列
和權重的一列做點積。所以 GELU 輸出的一格（第 k 個神經元），影響過輸出的**每一個通道 c**。

它的梯度就要把這些影響全部收回來：dMlpOut 那一行（48 格）逐格乘上 ${c_blockRef('權重', blk.mlpProjWeight)} 的第 k 行，再加起來。`;
    breakAfter();

    let t_dActDemo = afterTime(null, 5.5, 0.3);
    let t_dActFill = afterTime(null, 2.5);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`d\text{Gelu}_{t,k} = \sum_{c} d\text{MlpOut}_{c,t}\;W^{\text{mlp}}_{c,k}`} />)}

再穿過 GELU。這一步只做一件事：**乘上 GELU 在那一點的斜率** gelu′(x)。

先看一個有在出力的神經元：它的 x 是正的，斜率接近 1，梯度幾乎原封不動地通過。
浮層上那條曲線就是 gelu′，點標的是這一格的 x。`;
    breakAfter();

    let t_zoomAlive = afterTime(null, 0.8);
    let t_aliveDemo = afterTime(null, 3.5, 0.5);

    breakAfter();
    commentary(wt)`
再看一個被 GELU 壓掉的神經元：它的 x 很負，斜率幾乎是 0。
乘上去之後，梯度也跟著消失了 —— **這個神經元這次沒出聲，所以也不負責，學不到東西**。

這正是激活函數同時決定「誰能出聲」和「誰要負責」的地方。`;
    breakAfter();

    let t_zoomDead = afterTime(null, 0.8);
    let t_deadDemo = afterTime(null, 3.5, 0.5);
    let t_zoomBack = afterTime(null, 0.8);
    let t_geluFill = afterTime(null, 2.5);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`d\text{Fc}_{t,k} = d\text{Gelu}_{t,k}\cdot \text{gelu}'\!\left(\text{Fc}_{t,k}\right)`} />)}

注意這一步和剛才的線性層不同：這裡每一格**只看自己**，同一列的其他格完全不參與。

最後穿過第一個線性層，回到 ${c_blockRef('Layer Norm 2', blk.ln2.lnResid)}。
道理一樣，只是這次沿 192 個神經元 k 加總。`;
    breakAfter();

    let t_dLnDemo = afterTime(null, 5.0, 0.3);
    let t_dLnFill = afterTime(null, 2.5);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`d\text{LN}_{c,t} = \sum_{k} d\text{Fc}_{t,k}\;W^{\text{fc}}_{c,k}`} />)}

最後收權重。MLP 佔了整個模型大約三分之二的參數，所以這裡是 optimizer 拿走最多東西的地方。

權重的一格 W[c, k] 在前向時被**每個位置**都用過，所以它的梯度沿位置 t 加總 ——
加總的方向翻面了：剛才算 dLN 是沿 k 加，現在算 dW 是沿 t 加。
偏置更簡單：它對每個位置加同一個數，反向就是把每個位置的梯度加起來。`;
    breakAfter();

    let t_dWDemo = afterTime(null, 5.0, 0.3);
    let t_biasDemo = afterTime(null, 3.0, 0.3);
    let t_dWFill = afterTime(null, 3.0);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`dW^{\text{fc}}_{c,k} = \sum_{t} d\text{Fc}_{t,k}\;\text{LN}_{c,t}\qquad db^{\text{fc}}_{k} = \sum_{t} d\text{Fc}_{t,k}`} />)}

整段的關鍵對照：MLP 裡每一格的梯度只跟**它自己的前向值**有關。等一下你會看到 attention 完全不是這樣。`;

    // 相機依時間順序排：moveCameraTo 靠呼叫順序找「上一個」相機位置
    let overview = cam(-160.2, -455.6);
    moveCameraTo(state, t_moveCamera, overview, new Vec3(289.1, -8.9, 1.7));
    moveCameraTo(state, t_zoomAlive, focusCell(state, blk.mlpFc, new Vec3(kAlive, POS, 0), -cell * 2, -cell * 3), demoAngle(0.6));
    moveCameraTo(state, t_zoomDead, focusCell(state, blk.mlpFc, new Vec3(kDead, POS, 0), -cell * 2, -cell * 3), demoAngle(0.6));
    moveCameraTo(state, t_zoomBack, overview, new Vec3(289.1, -8.9, 1.7));

    focusBackwardScene(state, new Set([
        blk.ln2.lnResid, blk.mlpFcWeight, blk.mlpFcBias, blk.mlpFc, blk.mlpAct,
        blk.mlpProjWeight, blk.mlpProjBias, blk.mlpResult, blk.mlpResidual,
    ]), t_fade.t);

    // 殘差的梯度是上一章送下來的，一開始就亮著
    if (t_fade.t > 0) {
        processBackwardChain(state, t_fade, [blk.mlpResidual]);
    }
    if (t_copyFill.t > 0) {
        processBackwardChain(state, t_copyFill, [blk.mlpResidual, blk.mlpResult]);
    }
    if (t_dActFill.t > 0) {
        processBackwardChain(state, t_dActFill, [blk.mlpResult, blk.mlpAct]);
    }
    if (t_geluFill.t > 0) {
        processBackwardChain(state, t_geluFill, [blk.mlpAct, blk.mlpFc]);
    }
    if (t_dLnFill.t > 0) {
        processBackwardChain(state, t_dLnFill, [blk.mlpFc, blk.ln2.lnResid]);
    }
    if (t_dWFill.t > 0) {
        processBackwardChain(state, t_dWFill, [
            blk.mlpFcWeight, blk.mlpFcBias, blk.mlpProjWeight, blk.mlpProjBias,
        ], { animateFirst: true });
    }

    sceneMoveSlice(state, t_copy,
        { blk: blk.mlpResidual, fixDim: Dim.X, fixIdx: POS, kind: 'grad' },
        { blk: blk.mlpResult, fixDim: Dim.X, fixIdx: POS },
        { symbol: '=' });

    scenePairDot(state, t_dActDemo,
        { blk: blk.mlpResult, fixDim: Dim.X, fixIdx: POS, kind: 'grad' },
        { blk: blk.mlpProjWeight, fixDim: Dim.X, fixIdx: kAlive, kind: 'fwd' },
        { blk: blk.mlpAct, idx: new Vec3(kAlive, POS, 0) },
        { maxPairs: 8 });

    for (let [timer, k] of [[t_aliveDemo, kAlive], [t_deadDemo, kDead]] as const) {
        let idx = new Vec3(k, POS, 0);
        sceneElemMul(state, timer,
            { blk: blk.mlpAct, idx, kind: 'grad' },
            { blk: blk.mlpFc, idx, kind: 'fwd' },
            { blk: blk.mlpFc, idx },
            { label: "gelu'(x)" });
        if (timer.t > 0.15 && timer.t < 1) {
            // 浮層畫 gelu′ 的曲線，並把這一格的 x 標在上面
            drawDataFlow(state, blk.mlpFc, idx);
        }
    }

    scenePairDot(state, t_dLnDemo,
        { blk: blk.mlpFc, fixDim: Dim.Y, fixIdx: POS, kind: 'grad' },
        { blk: blk.mlpFcWeight, fixDim: Dim.Y, fixIdx: cStar, kind: 'fwd' },
        { blk: blk.ln2.lnResid, idx: new Vec3(POS, cStar, 0) },
        { maxPairs: 8 });

    scenePairDot(state, t_dWDemo,
        { blk: blk.mlpFc, fixDim: Dim.X, fixIdx: kAlive, kind: 'grad' },
        { blk: blk.ln2.lnResid, fixDim: Dim.Y, fixIdx: cStar, kind: 'fwd' },
        { blk: blk.mlpFcWeight, idx: new Vec3(kAlive, cStar, 0) },
        { maxPairs: POS + 1 });

    sceneCollapse(state, t_biasDemo,
        { blk: blk.mlpFc, fixDim: Dim.X, fixIdx: kAlive, kind: 'grad' },
        { blk: blk.mlpFcBias, idx: new Vec3(kAlive, 0, 0) },
        { maxCells: POS + 1 });
}
