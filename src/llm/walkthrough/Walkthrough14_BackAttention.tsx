import React from 'react';
import { Dim, Vec3 } from "@/src/utils/vector";
import { lerpSmoothstep } from "@/src/utils/math";
import { Phase } from "./Walkthrough";
import { commentary, ITimeInfo, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { argmaxAbs, range, sceneFanOutWeighted, scenePairDot, sceneSoftmaxRow, shiftToBlock } from "./BackpropScenes";
import { BackpropCamera, FILL_MOVE, FILL_SHOT } from "./BackpropCamera";
import { embedInline } from "./Walkthrough01_Prelim";
import { Tex } from "../components/Tex";
import { fwdAt, gradAt } from "../components/GradMath";
import { IBlkDef, IGptModelLayout } from "../GptModelLayout";

export function walkthrough14_BackAttention(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Attention) {
        return;
    }

    // 看最後一層：這裡的因果結構最乾淨（Q 只有位置 5 有梯度）。越往前的層，位置已經被後面的注意力混過
    let li = layout.blocks.length - 1;
    let blk = layout.blocks[li];
    let HEAD = 2;
    let head = blk.heads[HEAD];
    let { A, C } = layout.shape;
    let POS = state.gradData?.lossPos ?? 5;
    let sqrtA = Math.sqrt(A);
    // 相機位置沿用前向「自注意力」章節調好的值，平移到最後一層
    let cam = (x: number, z: number) => shiftToBlock(layout, li, new Vec3(x, 0, z));

    setInitialCamera(state, cam(-125.258, -178.805), new Vec3(294.000, 12.800, 2.681));
    wt.dimHighlightBlocks = [...head.cubes];

    let sStar = argmaxAbs(POS + 1, s => gradAt(head.attnMtxSm, new Vec3(s, POS, 0)));
    let aStar = argmaxAbs(A, a => gradAt(head.qBlock, new Vec3(POS, a, 0)));
    let cStar = argmaxAbs(C, c => fwdAt(state, blk.ln1.lnResid, new Vec3(POS, c, 0)));

    commentary(wt, null, 0)`
自注意力是整條反向鏈路上最有意思的一段。前向時 Q、K、V 三條路匯合成一個輸出，
反向時梯度要沿著這三條路分頭送回去。

我們看最後一層的第 2 個 head，一步一步把前向倒著演一次。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.6);
    let t_fade = afterTime(null, 1.3);

    breakAfter();
    commentary(wt)`
先回想前向的最後一步：${c_blockRef('輸出 O', head.vOutBlock)} 的第 5 行，是拿
${c_blockRef('注意力權重 P', head.attnMtxSm)} 的第 5 列當權重，把 ${c_blockRef('V', head.vBlock)} 的前六行加權加起來。

倒過來演就是：dO 的第 5 行**複製成六份**，第 s 份乘上 P[5, s]，送回 V 的第 s 行。
仔細看每一份乘完之後的顏色 —— 被關注越多的位置，拿到的梯度越大。`;
    breakAfter();

    let t_zoomV = afterTime(null, 0.8);
    let t_dVDemo = afterTime(null, 6.5, 0.3);
    let t_dVFill = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`dV_{a,s} = \sum_{t} dO_{a,t}\;P_{t,s}`} />)}

同一個乘法，P 那一邊的梯度：P[5, s] 在前向時乘上了 V 的**整個第 s 行**（16 維），
所以它的梯度是 dO 的第 5 行和 V 的第 s 行的點積。`;
    breakAfter();

    let t_camDP = afterTime(null, 0.8);
    let t_dPDemo = afterTime(null, 5.0, 0.3);
    let t_dPFill = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`dP_{t,s} = \sum_{a} dO_{a,t}\;V_{a,s}`} />)}

接著是整段最關鍵的一步：穿過 softmax。

softmax 的分母是**整列的總和**，所以 ${c_blockRef('分數 S', head.attnMtx)} 的任何一格變了，這一列的每一個 P 都會跟著變。
反向就得把整列綁在一起，分三拍：

1. 整列的 P 和 dP 成對相乘、加起來，得到一個數 ρ —— 這一列 dP 的加權平均。
2. 每一格 dP 減掉 ρ。
3. 再乘上自己那一格的 P，落進 S。`;
    breakAfter();

    let t_zoomS = afterTime(null, 0.8);
    let t_dSDemo = afterTime(null, 7.5, 0.3);
    let t_dSFill = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`dS_{t,s} = P_{t,s}\Big(dP_{t,s} - \underbrace{\textstyle\sum_{j} P_{t,j}\,dP_{t,j}}_{\rho_t}\Big)`} />)}

意思是：**比這一列平均更「想被調高」的格子，梯度為正；比平均低的，梯度為負**。
在 MLP 裡每一格各算各的；在這裡整列是綁在一起的 —— 這就是 attention 的反向跟 MLP 本質不同的地方。

最後回到 ${c_blockRef('Q', head.qBlock)} 與 ${c_blockRef('K', head.kBlock)}。
分數是 S = QᵀK / √${embedInline(<>{A}</>)}，又是一個點積：Q 的一格要沿「被它看過的位置 s」加總，K 的一格要沿「看過它的位置 t」加總。`;
    breakAfter();

    let t_zoomQK = afterTime(null, 0.8);
    let t_dQDemo = afterTime(null, 4.5, 0.3);
    let t_camDK = afterTime(null, 0.8);
    let t_dKDemo = afterTime(null, 4.5, 0.3);
    let t_dQKFill = afterTime(null, 2.5);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`dQ_{a,t} = \frac{1}{\sqrt{${A}}}\sum_{s} dS_{t,s}\,K_{a,s}\qquad dK_{a,s} = \frac{1}{\sqrt{${A}}}\sum_{t} dS_{t,s}\,Q_{a,t}`} />)}

注意畫面上的不對稱：${c_blockRef('Q', head.qBlock)} 只有**第 5 行**有梯度，
${c_blockRef('K', head.kBlock)} 和 ${c_blockRef('V', head.vBlock)} 卻是**前六行**都有。

這不是巧合：位置 5 只用了它自己的 query，卻去看了位置 0 到 5 的所有 key 和 value。因果錐在梯度上直接看得見。
（這是最後一層才這麼乾淨。越往前的層，後面的注意力已經把位置混在一起，梯度會散到更多位置。）`;
    breakAfter();

    commentary(wt)`
到這裡算的都是**中間量**的梯度，算完就丟。真正要留下來的是 ${c_blockRef('權重', head.qWeightBlock)} 的梯度。

Q = Wq · LN，所以 Wq 的一格沿位置 t 加總：dQ 的一列逐格乘上 Layer Norm 的一列。
同一個 Wq 被每個位置共用，每個位置的責任全部疊在同一格上。`;
    breakAfter();

    let t_zoomW = afterTime(null, 0.8);
    let t_dWDemo = afterTime(null, 5.0, 0.3);
    let t_dWFill = afterTime(null, 3.0);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`dW^{Q}_{a,c} = \sum_{t} dQ_{a,t}\;\text{LN}_{c,t}`} />)}

K、V 的權重同理。optimizer 拿走的就是這幾張表 —— 整個反向傳播跑這一趟，為的就是它們。`;

    focusBackwardScene(state, new Set([blk.ln1.lnResid, ...head.cubes]), t_fade.t);

    // 先把 head 攤平，之後所有的格子位置（包括相機要對準的地方）都以攤平後的排版為準
    alignHead(layout, li, HEAD, t_fade.t);

    // 每段示範寫成函式：同一個函式拿去畫，也拿去給相機量出它會用到畫面上的哪些地方
    let dVScene = (tm: ITimeInfo) => sceneFanOutWeighted(state, tm, { blk: head.vOutBlock, colIdx: POS },
        range(POS + 1).map(s => ({
            weight: { blk: head.attnMtxSm, idx: new Vec3(s, POS, 0) },
            dest: { blk: head.vBlock, colIdx: s },
        })));
    let dPScene = (tm: ITimeInfo) => scenePairDot(state, tm,
        { blk: head.vOutBlock, fixDim: Dim.X, fixIdx: POS, kind: 'grad' },
        { blk: head.vBlock, fixDim: Dim.X, fixIdx: sStar, kind: 'fwd' },
        { blk: head.attnMtxSm, idx: new Vec3(sStar, POS, 0) },
        { maxPairs: 8 });
    let dSScene = (tm: ITimeInfo) => sceneSoftmaxRow(state, tm, head.attnMtxSm, head.attnMtx, POS);
    let dQScene = (tm: ITimeInfo) => scenePairDot(state, tm,
        { blk: head.attnMtx, fixDim: Dim.Y, fixIdx: POS, kind: 'grad' },
        { blk: head.kBlock, fixDim: Dim.Y, fixIdx: aStar, kind: 'fwd' },
        { blk: head.qBlock, idx: new Vec3(POS, aStar, 0) },
        { maxPairs: POS + 1, suffix: `/ ${sqrtA}` });
    let dKScene = (tm: ITimeInfo) => scenePairDot(state, tm,
        { blk: head.attnMtx, fixDim: Dim.X, fixIdx: sStar, kind: 'grad' },
        { blk: head.qBlock, fixDim: Dim.Y, fixIdx: aStar, kind: 'fwd' },
        { blk: head.kBlock, idx: new Vec3(sStar, aStar, 0) },
        { maxPairs: POS + 1, suffix: `/ ${sqrtA}` });
    let dWScene = (tm: ITimeInfo) => scenePairDot(state, tm,
        { blk: head.qBlock, fixDim: Dim.Y, fixIdx: aStar, kind: 'grad' },
        { blk: blk.ln1.lnResid, fixDim: Dim.Y, fixIdx: cStar, kind: 'fwd' },
        { blk: head.qWeightBlock, idx: new Vec3(cStar, aStar, 0) },
        { maxPairs: POS + 1 });

    // 總覽沿用手調的值；特寫由場景實際會畫到的範圍算出來，所以要在 alignHead 之後
    let camera = new BackpropCamera(state);
    camera.shot(t_moveCamera, camera.fixed(cam(-92.7, -219), new Vec3(286, 12.8, 1.4)));
    camera.shot(t_zoomV, camera.scene('dV', dVScene, t_dVDemo));
    camera.shot(t_dVFill, camera.blocks('dVFill', [head.vBlock], FILL_SHOT), FILL_MOVE);
    camera.shot(t_camDP, camera.scene('dP', dPScene, t_dPDemo));
    camera.shot(t_dPFill, camera.blocks('dPFill', [head.attnMtxSm], FILL_SHOT), FILL_MOVE);
    camera.shot(t_zoomS, camera.scene('dS', dSScene, t_dSDemo));
    camera.shot(t_dSFill, camera.blocks('dSFill', [head.attnMtx], FILL_SHOT), FILL_MOVE);
    camera.shot(t_zoomQK, camera.scene('dQ', dQScene, t_dQDemo));
    camera.shot(t_camDK, camera.scene('dK', dKScene, t_dKDemo));
    camera.shot(t_dQKFill, camera.blocks('dQKFill', [head.qBlock, head.kBlock], FILL_SHOT), FILL_MOVE);
    camera.shot(t_zoomW, camera.scene('dW', dWScene, t_dWDemo));
    camera.shot(t_dWFill, camera.blocks('dWFill', [
        head.qWeightBlock, head.kWeightBlock, head.vWeightBlock,
        head.qBiasBlock, head.kBiasBlock, head.vBiasBlock,
    ], FILL_SHOT), FILL_MOVE);
    camera.apply();

    if (t_fade.t > 0) {
        processBackwardChain(state, t_fade, [head.vOutBlock]);
    }
    if (t_dVFill.t > 0) {
        processBackwardChain(state, t_dVFill, [head.vOutBlock, head.vBlock]);
    }
    if (t_dPFill.t > 0) {
        processBackwardChain(state, t_dPFill, [head.vOutBlock, head.attnMtxSm]);
    }
    if (t_dSFill.t > 0) {
        processBackwardChain(state, t_dSFill, [head.attnMtxSm, head.attnMtx]);
    }
    if (t_dQKFill.t > 0) {
        processBackwardChain(state, t_dQKFill, [head.attnMtx, head.qBlock, head.kBlock]);
    }
    if (t_dWFill.t > 0) {
        processBackwardChain(state, t_dWFill, [
            head.qWeightBlock, head.kWeightBlock, head.vWeightBlock,
            head.qBiasBlock, head.kBiasBlock, head.vBiasBlock,
        ], { animateFirst: true });
    }

    dVScene(t_dVDemo);
    dPScene(t_dPDemo);
    dSScene(t_dSDemo);
    dQScene(t_dQDemo);
    dKScene(t_dKDemo);
    dWScene(t_dWDemo);
}

/**
 * 跟前向章節一樣，把要看的 head 攤平到同一個平面：Q、K、V 上下排開，其他 head 隱藏。
 *
 * 原本的排版裡，一個 head 的 Q、K、V 是沿深度疊著的 —— 從正面看，後面的 V 和飛過去的格子
 * 全被前面的 Q 擋住。前向的 focusSelfAttentionHead 寫死在第 0 層，這裡照同樣的算法套到任一層。
 */
function alignHead(layout: IGptModelLayout, blockIdx: number, headIdx: number, t: number) {
    let block = layout.blocks[blockIdx];
    let head = block.heads[headIdx];

    for (let h = 0; h < block.heads.length; h++) {
        if (h === headIdx) {
            continue;
        }
        for (let cube of block.heads[h].cubes) {
            cube.opacity = Math.min(cube.opacity, lerpSmoothstep(1, 0, t));
        }
    }

    let deltaZ = lerpSmoothstep(0, block.ln1.lnResid.z - head.attnMtx.z, t);
    for (let cube of head.cubes) {
        cube.z += deltaZ;
    }

    let targetZ = block.ln1.lnResid.z;
    let strideY = head.qBlock.dy + layout.margin;
    let baseY = head.qBlock.y;
    let rows = [
        [head.qBlock, head.qWeightBlock, head.qBiasBlock],
        [head.kBlock, head.kWeightBlock, head.kBiasBlock],
        [head.vBlock, head.vWeightBlock, head.vBiasBlock],
    ];
    let offsets = [-strideY * 2, -strideY, 0];
    for (let i = 0; i < rows.length; i++) {
        let y = lerpSmoothstep(rows[i][0].y, baseY + offsets[i], t);
        let z = lerpSmoothstep(rows[i][0].z, targetZ, t);
        for (let cube of rows[i]) {
            cube.y = y;
            cube.z = z;
        }
    }

    // 讓 K 與 Layer Norm 的中線對齊：Layer Norm 之後的所有區塊一起往下挪
    let mid = (b: IBlkDef) => b.y + b.dy / 2;
    let lnIdx = layout.cubes.indexOf(block.ln1.lnResid);
    let yDelta = lerpSmoothstep(0, mid(block.ln1.lnResid) - mid(head.kBlock), t);
    for (let i = lnIdx + 1; i < layout.cubes.length; i++) {
        layout.cubes[i].y += yDelta;
    }
}
