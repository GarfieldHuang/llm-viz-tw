import React from 'react';
import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";

export function walkthrough14_BackAttention(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_str, c_blockRef, c_dimRef, breakAfter } } = args;
    let { A } = layout.shape;

    if (wt.phase !== Phase.Backward_Attention) {
        return;
    }

    let block0 = layout.blocks[0];
    let head2 = block0.heads[2];

    setInitialCamera(state, new Vec3(-125.258, 0.000, -178.805), new Vec3(294.000, 12.800, 2.681));
    wt.dimHighlightBlocks = [...head2.cubes];

    commentary(wt, null, 0)`
自注意力是整條反向鏈路上最有意思的一段。前向時三條路徑 Q、K、V 匯合成一個輸出，
反向時梯度就要沿著這三條路徑分頭送回去。

畫面上顯示的不再是啟用值，而是**梯度**：每一格的顏色代表「這個數字改變一點點，損失會變多少」。
灰色代表梯度為 0 —— 那不是壞掉，是那個位置對損失完全沒有影響。

滑鼠移到任一格上，浮層顯示的會是**反向**的式子，箭頭也從「誰算出我」改成「誰把梯度交給我」。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t_fade = afterTime(null, 0.8);

    breakAfter();
    commentary(wt)`
先看最單純的一條。輸出是 ${c_blockRef('O = P V', head2.vOutBlock)}，其中
${c_blockRef('P', head2.attnMtxSm)} 是 softmax 之後的注意力權重。
這是一個純矩陣乘法，反向就是轉置相乘：

dV = Pᵀ dO　　dP = dO Vᵀ

也就是說，${c_blockRef('V', head2.vBlock)} 收到的梯度，是把輸出的梯度依照注意力權重「分配」回去 ——
某個位置被關注得越多，它要負的責任就越大。`;
    breakAfter();

    let t_dV = afterTime(null, 3.0);

    breakAfter();
    commentary(wt)`
接著是整段最關鍵的一步：穿過 softmax。

softmax 是逐列做的，所以它的 Jacobian 不是對角矩陣 —— 同一列裡每個位置都會互相牽動。
把 J = diag(p) − p pᵀ 代進去化簡，會得到一個出乎意料乾淨的式子：

dS = P ⊙ ( dP − rowsum(P ⊙ dP) )

那個 rowsum 是**逐列的一個純量**，再廣播回整列。這代表 ${c_blockRef('注意力分數', head2.attnMtx)} 的梯度，
取決於「這一格的 dP」與「整列的加權平均」之間的差。這正是 attention 的反向與 FFN 的反向本質不同的地方：
在 MLP 裡每個元素各算各的，在這裡整列是綁在一起的。`;
    breakAfter();

    let t_dS = afterTime(null, 3.5);

    breakAfter();
    commentary(wt)`
最後回到 ${c_blockRef('Q', head2.qBlock)} 與 ${c_blockRef('K', head2.kBlock)}。
分數是 S = Q Kᵀ / √${c_dimRef('A', DimStyle.A)}，所以

dQ = dS K / √A　　dK = dSᵀ Q / √A

注意一件事：因為損失只看第 5 個位置，${c_blockRef('Q', head2.qBlock)} 只有**第 5 列**有梯度，
但 ${c_blockRef('K', head2.kBlock)} 和 ${c_blockRef('V', head2.vBlock)} 有**前六列**都有梯度。

這個不對稱不是巧合：第 5 個位置只用了它自己的 query，卻去看了位置 0 到 5 的所有 key 和 value。
因果錐在梯度上直接看得見。`;
    breakAfter();

    let t_dQK = afterTime(null, 3.0);

    breakAfter();
    commentary(wt)`
到這裡為止算的都是**中間量**的梯度，它們算完就丟。真正要留下來的是最後這一步：
${c_blockRef('權重', head2.qWeightBlock)} 的梯度。

dWq = dQᵀ · LN　　dWk = dKᵀ · LN　　dWv = dVᵀ · LN

每一格的意思是：「把這個權重調高一點點，損失會變多少」。optimizer 拿走的就是這張表 ——
整個反向傳播跑這一趟，為的就是它。

注意權重的梯度是**整批位置加總**後的結果：同一個 Wq 被六個位置共用，所以六個位置的責任全部疊在同一格上。
中間量的梯度每個位置各自獨立，權重的梯度不是。`;
    breakAfter();

    let t_dW = afterTime(null, 3.0);

    moveCameraTo(state, t_moveCamera, new Vec3(-92.7, 0, -219), new Vec3(286, 12.8, 1.4));

    let relevant = new Set([
        block0.ln1.lnResid,
        ...head2.cubes,
    ]);
    focusBackwardScene(state, relevant, t_fade.t);

    // 每一步明講要填哪些區塊。
    // 這些鏈是照**反向圖**列的，不是照畫面上的擺放順序 ——
    // 舊版用陣列區間掃描，會漏掉 V、又在講 Q K 時把 V 點亮。
    if (t_dV.t > 0) {
        // dO 已知 -> 分頭送給 dV 與 dP
        processBackwardChain(state, t_dV, [head2.vOutBlock, head2.vBlock, head2.attnMtxSm]);
    }
    if (t_dS.t > 0) {
        // dP -> 穿過 softmax -> dS（中間那兩根聚合樁不在路徑上，跳過）
        processBackwardChain(state, t_dS, [head2.attnMtxSm, head2.attnMtx]);
    }
    if (t_dQK.t > 0) {
        // dS -> dQ 與 dK。V 不在這一步，它的梯度在第一步就拿到了。
        processBackwardChain(state, t_dQK, [head2.attnMtx, head2.qBlock, head2.kBlock]);
    }
    if (t_dW.t > 0) {
        // 權重梯度：這一步沒有前置區塊，整條都要跑動畫
        processBackwardChain(state, t_dW, [
            head2.qWeightBlock, head2.kWeightBlock, head2.vWeightBlock,
            head2.qBiasBlock, head2.kBiasBlock, head2.vBiasBlock,
        ], { animateFirst: true });
    }
}
