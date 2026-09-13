import React from 'react';
import { Dim, Vec3 } from "@/src/utils/vector";
import { lerp } from "@/src/utils/math";
import { Phase } from "./Walkthrough";
import { commentary, ITimeInfo, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { argmaxAbs, dupCell, dupSlice, place, plainCell, sceneCollapse, scenePairDot, seg, writeText } from "./BackpropScenes";
import { BackpropCamera, FILL_MOVE, FILL_SHOT } from "./BackpropCamera";
import { cellPos } from "./BackpropAnim";
import { IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { embedInline } from "./Walkthrough01_Prelim";
import { Tex } from "../components/Tex";
import { gradAt } from "../components/GradMath";

export function walkthrough15_BackLayerNorm(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_LayerNorm) {
        return;
    }

    // 中間兩層重複同樣的事，直接跳到第一層的 Layer Norm：反向傳播的倒數第二站
    let block0 = layout.blocks[0];
    let ln1 = block0.ln1;
    let X = layout.residual0;
    let POS = state.gradData?.lossPos ?? 5;
    let { C } = layout.shape;

    setInitialCamera(state, new Vec3(-6.680, 0.000, -65.256), new Vec3(281.000, 9.000, 2.576));
    wt.dimHighlightBlocks = [ln1.lnResid, X];

    let cStar = argmaxAbs(C, c => gradAt(ln1.lnResid, new Vec3(POS, c, 0)));

    commentary(wt, null, 0)`
中間兩層做的事跟剛才一模一樣，我們直接跳到**第一層**的 Layer Norm —— 再往前就是嵌入表了。

Layer Norm 的反向是整趟裡最容易算錯的一段，原因很具體：它的輸出**不只依賴自己那一格**。
前向時，每一格都要減掉整行的平均 μ、再除以整行的標準差 σ。
μ 和 σ 是整行一起算的 —— 動任何一格，整行的輸出都會跟著變。反向就必須把這個牽連還回去。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.6);
    let t_fade = afterTime(null, 1.3);

    breakAfter();
    commentary(wt)`
先看容易的兩塊。

${c_blockRef('β', ln1.lnMu)} 對每個位置都加同一個數，所以它的梯度是把一列 dLN 沿位置加起來。

${c_blockRef('γ', ln1.lnSigma)} 對每個位置乘上歸一化後的值 xn，所以它的梯度是 dLN 那一列和 xn 那一列逐格相乘再相加
（右邊那一排代表 xn）。`;
    breakAfter();

    let t_zoomGB = afterTime(null, 0.8);
    let t_betaDemo = afterTime(null, 3.5, 0.3);
    let t_camGamma = afterTime(null, 0.8);
    let t_gammaDemo = afterTime(null, 4.5, 0.3);
    let t_gbFill = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`d\beta_{c} = \sum_{t} d\text{LN}_{c,t}\qquad d\gamma_{c} = \sum_{t} d\text{LN}_{c,t}\;\hat{x}_{c,t}`} />)}

接著是難的那一塊：穿過歸一化本身，回到 ${c_blockRef('輸入', X)}。看位置 5 那一行，分四拍：

1. 梯度先乘上 γ，得到 g。
2. 扣掉整行 g 的平均 E[g] —— 因為前向減過 μ。
3. 再扣掉 xn · E[g · xn] —— 因為前向除過 σ。
4. 最後除以 σ，落進輸入。`;
    breakAfter();

    let t_zoomX = afterTime(null, 0.8);
    let t_dXDemo = afterTime(null, 9.0, 0.3);
    let t_dXFill = afterTime(null, 2.0);
    let t_zoomOut = afterTime(null, 0.8);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`dX_{c,t} = \frac{1}{\sigma_t}\Big(g_{c} - \overline{g} - \hat{x}_{c,t}\;\overline{g\,\hat{x}}\Big) + dR^{\text{attn}}_{c,t},\qquad g_{c} = \gamma_{c}\,d\text{LN}_{c,t}`} />)}

最後那個「+」是殘差：輸入除了進 Layer Norm，還直接走殘差進了 ${c_blockRef('注意力殘差', block0.attnResidual)}。
一格被兩個地方用到，梯度就是兩條路相加。

中間兩個扣除項合起來的效果很值得記：**Layer Norm 會把梯度裡「整行一起變大」和「整行一起放大」的成分扣掉**。
那兩件事在前向已經被歸一化吃掉了，調了也沒用。梯度只剩下真正會改變「行內相對關係」的部分。
這也是 Layer Norm 讓訓練穩定的原因：它在前向和反向都把尺度問題擋掉了。`;
    breakAfter();

    let t_settle = afterTime(null, 1.6);

    // 每段示範寫成函式：同一個函式拿去畫，也拿去給相機量出它會用到畫面上的哪些地方
    let betaScene = (tm: ITimeInfo) => sceneCollapse(state, tm,
        { blk: ln1.lnResid, fixDim: Dim.Y, fixIdx: cStar, kind: 'grad' },
        { blk: ln1.lnMu, idx: new Vec3(0, cStar, 0) },
        { maxCells: POS + 1 });
    // xn 沒有自己的貼圖；Layer Norm 的前向值是 γ·xn + β，同一列裡跟 xn 只差一個固定的縮放與平移，拿來代表
    let gammaScene = (tm: ITimeInfo) => scenePairDot(state, tm,
        { blk: ln1.lnResid, fixDim: Dim.Y, fixIdx: cStar, kind: 'grad' },
        { blk: ln1.lnResid, fixDim: Dim.Y, fixIdx: cStar, kind: 'fwd' },
        { blk: ln1.lnSigma, idx: new Vec3(0, cStar, 0) },
        { maxPairs: POS + 1 });
    let dXScene = (tm: ITimeInfo) => sceneLnColumn(state, tm, ln1.lnResid, X, ln1.lnSigma, ln1.lnAgg2, block0.attnResidual, POS);

    // 總覽沿用手調的值；特寫由場景實際會畫到的範圍算出來
    let camera = new BackpropCamera(state);
    let overview = camera.fixed(new Vec3(3.4, 0, -76.4), new Vec3(281, 9, 1.5));
    camera.shot(t_moveCamera, overview);
    camera.shot(t_zoomGB, camera.scene('beta', betaScene, t_betaDemo));
    camera.shot(t_camGamma, camera.scene('gamma', gammaScene, t_gammaDemo));
    camera.shot(t_gbFill, camera.blocks('gbFill', [ln1.lnSigma, ln1.lnMu], FILL_SHOT), FILL_MOVE);
    camera.shot(t_zoomX, camera.scene('dX', dXScene, t_dXDemo));
    camera.shot(t_dXFill, camera.blocks('dXFill', [X], FILL_SHOT), FILL_MOVE);
    camera.shot(t_zoomOut, overview);
    camera.apply();

    focusBackwardScene(state, new Set([
        X, ln1.lnAgg1, ln1.lnAgg2, ln1.lnSigma, ln1.lnMu, ln1.lnResid, block0.attnResidual,
    ]), t_fade.t);

    if (t_fade.t > 0) {
        processBackwardChain(state, t_fade, [ln1.lnResid]);
        processBackwardChain(state, t_fade, [block0.attnResidual]);
    }
    if (t_gbFill.t > 0) {
        processBackwardChain(state, t_gbFill, [ln1.lnResid, ln1.lnSigma, ln1.lnMu]);
    }
    if (t_dXFill.t > 0) {
        processBackwardChain(state, t_dXFill, [ln1.lnResid, X]);
    }
    if (t_settle.t > 0) {
        processBackwardChain(state, t_settle, [X]);
    }

    betaScene(t_betaDemo);
    gammaScene(t_gammaDemo);
    dXScene(t_dXDemo);
}

/** 這一章畫在模型裡的字的字級。格子只有 1.5 單位寬，字太小就看不見。 */
const TEXT = 2.2;

/**
 * Layer Norm 對輸入的反向，一整行演四拍：乘 γ、扣平均、扣投影、除 σ，最後加上殘差那一份。
 */
function sceneLnColumn(state: IProgramState, timer: ITimeInfo, ln: IBlkDef, X: IBlkDef, gamma: IBlkDef, sigmaAgg: IBlkDef, resid: IBlkDef, t: number) {
    if (!(timer.t > 0 && timer.t < 1)) {
        return;
    }
    let cell = state.layout.cell;
    let T = timer.t;
    let tG = seg(T, 0.0, 0.2);
    let tMean = seg(T, 0.24, 0.44);
    let tProj = seg(T, 0.48, 0.66);
    let tSig = seg(T, 0.7, 0.82);
    let tLand = seg(T, 0.85, 1.0);

    let lnTl = cellPos(state, ln, new Vec3(t, 0, 0));
    let xTl = cellPos(state, X, new Vec3(t, 0, 0));
    let lift = new Vec3(0, 0, cell * 10);
    let work = lnTl.add(lift).add(new Vec3(cell * 8, 0, 0));

    let g = dupSlice(state, { blk: ln, fixDim: Dim.X, fixIdx: t, kind: 'grad' });
    if (!g) {
        return;
    }
    let gp = lnTl.lerp(work, tG);
    if (tLand > 0) {
        gp = work.lerp(xTl, tLand);
    }
    place(g, gp);
    g.highlight = 0.35;

    // 1. γ 那一行飛到旁邊相乘
    if (tG > 0 && tMean <= 0) {
        let gm = dupSlice(state, { blk: gamma, fixDim: Dim.X, fixIdx: 0, kind: 'fwd' });
        if (gm) {
            let from = cellPos(state, gamma, new Vec3(0, 0, 0));
            let beside = work.add(new Vec3(-cell * 3, 0, 0));
            place(gm, from.lerp(beside, tG));
            if (tG >= 1) {
                writeText(state, beside.add(new Vec3(cell * 2, -cell * 2, cell)), 'x', TEXT);
            }
        }
    }
    if (tG >= 1 && tLand <= 0) {
        writeText(state, work.add(new Vec3(cell * 0.5, -cell * 4, cell)), 'g = γ ‧ dLN', TEXT);
    }

    // 2、3. 一格平均值從上往下掃過整行，每經過一格就減掉
    let colHeight = g.dy;
    let sweep = (label: string, tt: number) => {
        if (!(tt > 0 && tt < 1)) {
            return;
        }
        let m = plainCell(state, ln);
        let top = work.add(new Vec3(cell * 2.5, 0, 0));
        let y = lerp(0, colHeight - cell, tt);
        place(m, top.add(new Vec3(0, y, 0)));
        writeText(state, top.add(new Vec3(cell * 6, y + cell * 0.5, cell)), label, TEXT);
        writeText(state, top.add(new Vec3(-cell * 0.9, y + cell * 0.5, cell)), '—', TEXT);
    };
    sweep('E[g]', tMean);
    sweep('xn ‧ E[g ‧ xn]', tProj);

    // 4. 除以 σ
    if (tSig > 0 && tLand <= 0) {
        let sd = dupCell(state, sigmaAgg, new Vec3(t, 0, 0), 'fwd');
        if (sd) {
            let from = cellPos(state, sigmaAgg, new Vec3(t, 0, 0));
            let to = work.add(new Vec3(0, -cell * 2.5, 0));
            place(sd, from.lerp(to, tSig));
            if (tSig >= 1) {
                writeText(state, to.add(new Vec3(cell * 3.5, cell * 0.5, cell)), '/ σ', TEXT);
            }
        }
    }

    // 殘差那一路也送一份進來
    if (tLand > 0) {
        let r = dupSlice(state, { blk: resid, fixDim: Dim.X, fixIdx: t, kind: 'grad' });
        if (r) {
            let from = cellPos(state, resid, new Vec3(t, 0, 0));
            place(r, from.add(lift).lerp(xTl, tLand));
            if (tLand > 0.5) {
                writeText(state, xTl.add(new Vec3(-cell * 1.6, -cell * 1.5, cell * 2)), '+', TEXT);
            }
        }
    }
}
