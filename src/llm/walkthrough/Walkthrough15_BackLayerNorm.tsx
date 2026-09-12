import React from 'react';
import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";

export function walkthrough15_BackLayerNorm(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_LayerNorm) {
        return;
    }

    let block0 = layout.blocks[0];
    let ln1 = block0.ln1;

    setInitialCamera(state, new Vec3(-6.680, 0.000, -65.256), new Vec3(281.000, 9.000, 2.576));
    wt.dimHighlightBlocks = [ln1.lnResid, layout.residual0];

    commentary(wt, null, 0)`
Layer Norm 的反向是這整趟裡最容易算錯的一段，原因很具體：
它的輸出**不只依賴自己那一格**。

前向時，每一格都要減掉整欄的平均 μ、再除以整欄的標準差 σ。
而 μ 和 σ 是整欄一起算出來的 —— 所以你動任何一格，整欄的輸出都會跟著變。
反向就必須把這個牽連還回去。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t_fade = afterTime(null, 0.8);

    breakAfter();
    commentary(wt)`
先看容易的兩塊：${c_blockRef('γ', ln1.lnSigma)} 和 ${c_blockRef('β', ln1.lnMu)}。

它們是逐元素的縮放與平移，梯度直接了當：

dγ = Σ dLN ⊙ xn　　dβ = Σ dLN

那個 Σ 是**跨所有位置加總** —— 因為同一組 γ、β 被序列裡每一個位置共用。
式子裡的 xn 指的是歸一化之後、還沒乘 γ 之前的值。`;
    breakAfter();

    let t_dGammaBeta = afterTime(null, 2.5);

    breakAfter();
    commentary(wt)`
接著是難的那一塊：怎麼穿過歸一化本身，回到 ${c_blockRef('輸入', layout.residual0)}。

把 μ 和 σ 對輸入的依賴一併微分完、整理化簡，會得到三項：

dx = (γ / σ) ⊙ ( d − E[d] − xn ⊙ E[d ⊙ xn] )

滑鼠移到 ${c_blockRef('Layer Norm', ln1.lnResid)} 上就是這個式子。三項各有意思：

第一項 d 是「這一格自己的責任」。

第二項 E[d] 是**扣掉整欄的平均責任** —— 因為前向減過 μ。

第三項 xn ⊙ E[d ⊙ xn] 是**扣掉與自己方向相關的部分** —— 因為前向除過 σ。`;
    breakAfter();

    let t_dX = afterTime(null, 3.5);

    breakAfter();
    commentary(wt)`
後兩項合起來的效果很值得記：**Layer Norm 會把梯度裡「整欄一起變大」和
「整欄一起放大」的成分扣掉**。

換句話說，它不准反向傳播去調整整欄的平均值和尺度 —— 因為那兩件事在前向就被歸一化吃掉了，
調了也沒用。梯度只剩下真正會改變「欄內相對關係」的那部分。

這也是為什麼 Layer Norm 讓訓練穩定：它同時在前向和反向都把尺度問題擋掉了。

還有一點跟 MLP 對照著看：這裡和 attention 一樣，**整欄是綁在一起的** ——
你在浮層上看到的 E[...] 就是整欄的聚合。MLP 裡沒有這種東西。`;
    breakAfter();

    let t_settle = afterTime(null, 1.0);

    moveCameraTo(state, t_moveCamera, new Vec3(3.4, 0, -76.4), new Vec3(281, 9, 1.5));

    let relevant = new Set([
        layout.residual0,
        ln1.lnAgg1,
        ln1.lnAgg2,
        ln1.lnSigma,
        ln1.lnMu,
        ln1.lnResid,
    ]);
    focusBackwardScene(state, relevant, t_fade.t);

    if (t_dGammaBeta.t > 0) {
        processBackwardChain(state, t_dGammaBeta, [ln1.lnResid, ln1.lnSigma, ln1.lnMu]);
    }
    if (t_dX.t > 0) {
        // 聚合樁（μ、σ）不在梯度路徑上，它們被併進這條式子裡，所以不列入鏈
        processBackwardChain(state, t_dX, [ln1.lnResid, layout.residual0]);
    }
    if (t_settle.t > 0) {
        processBackwardChain(state, t_settle, [layout.residual0]);
    }
}
