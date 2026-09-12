import React from 'react';
import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";

export function walkthrough13_BackProjection(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Projection) {
        return;
    }

    let block0 = layout.blocks[0];
    let heads = block0.heads;

    setInitialCamera(state, new Vec3(-73.167, 0.000, -270.725), new Vec3(293.606, 2.613, 1.366));
    wt.dimHighlightBlocks = [block0.attnOut, ...heads.map(h => h.vOutBlock)];

    commentary(wt, null, 0)`
多頭注意力有個容易被忽略的細節：那幾個 head 其實**從頭到尾沒有互相講過話**。
它們各算各的，最後只是把輸出並排接在一起，再乘上一個投射矩陣。

投射層做的就是這件事 —— 它是唯一讓各個 head 的資訊混在一起的地方。
反向時，它也是唯一負責把責任**切開分回各個 head** 的地方。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t_fade = afterTime(null, 0.8);

    breakAfter();
    commentary(wt)`
梯度從 ${c_blockRef('注意力輸出', block0.attnOut)} 進來（上一章的殘差分給它的那一份）。

投射是個標準的矩陣乘法 O = Wproj · V，所以反向也是標準的兩條：

dWproj = dO ᵀ · V　　dV = Wproj ᵀ · dO

先看權重那一條。${c_blockRef('投射權重', block0.projWeight)} 是
${c_dimRef('C', DimStyle.C)} × ${c_dimRef('C', DimStyle.C)} 的方陣 ——
它學的是「怎麼把三個 head 的意見調配成一個結論」。`;
    breakAfter();

    let t_dProjW = afterTime(null, 3.0);

    breakAfter();
    commentary(wt)`
接著是有意思的那一條。

前向時三個 head 的輸出是「並排接起來」的：head 0 佔了前 ${c_dimRef('A', DimStyle.A)} 個維度，
head 1 佔接下來 A 個，依此類推。**串接在反向就是切開。**

所以 dV 算出來之後，它的前 A 個維度就是 head 0 該拿的、中間 A 個是 head 1 的、
最後 A 個是 head 2 的。不需要任何額外運算 —— 只是把同一塊東西按位置分給三個人。`;
    breakAfter();

    let t_dHeads = afterTime(null, 3.5);

    breakAfter();
    commentary(wt)`
三個 head 各自拿到自己那一份，接下來就會沿著各自的 Q、K、V 往回走 ——
那是下一章「自注意力」的內容。

順帶一提，${c_blockRef('投射偏置', block0.projBias)} 的梯度是把整批位置加起來的結果。
偏置對每個位置貢獻同一個數，所以每個位置的責任都要算到它頭上。`;
    breakAfter();

    let t_dBias = afterTime(null, 1.5);

    moveCameraTo(state, t_moveCamera, new Vec3(-68.2, 0, -282.4), new Vec3(293.6, 2.6, 1.1));

    let relevant = new Set([
        ...heads.map(h => h.vOutBlock),
        block0.projWeight,
        block0.projBias,
        block0.attnOut,
    ]);
    focusBackwardScene(state, relevant, t_fade.t);

    if (t_dProjW.t > 0) {
        processBackwardChain(state, t_dProjW, [block0.attnOut, block0.projWeight]);
    }
    if (t_dHeads.t > 0) {
        // 串接的反向＝切開：一份 dV 按維度區間分給三個 head
        processBackwardChain(state, t_dHeads, [
            block0.attnOut, ...heads.map(h => h.vOutBlock),
        ]);
    }
    if (t_dBias.t > 0) {
        processBackwardChain(state, t_dBias, [block0.attnOut, block0.projBias]);
    }
}
