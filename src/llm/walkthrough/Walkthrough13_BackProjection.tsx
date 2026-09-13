import React from 'react';
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, ITimeInfo, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { argmaxAbs, sceneCollapse, sceneMoveSlice, scenePairDot, shiftToBlock } from "./BackpropScenes";
import { BackpropCamera, FILL_MOVE, FILL_SHOT } from "./BackpropCamera";
import { cellPos } from "./BackpropAnim";
import { embedInline } from "./Walkthrough01_Prelim";
import { Tex } from "../components/Tex";
import { gradAt } from "../components/GradMath";

export function walkthrough13_BackProjection(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Projection) {
        return;
    }

    let li = layout.blocks.length - 1;
    let blk = layout.blocks[li];
    let heads = blk.heads;
    let { A, C } = layout.shape;
    let POS = state.gradData?.lossPos ?? 5;
    let cam = (x: number, z: number) => shiftToBlock(layout, li, new Vec3(x, 0, z));

    setInitialCamera(state, cam(-73.167, -270.725), new Vec3(293.606, 2.613, 1.366));
    wt.dimHighlightBlocks = [blk.attnOut, ...heads.map(h => h.vOutBlock)];

    let H = 0;
    let cStar = argmaxAbs(C, c => gradAt(blk.attnOut, new Vec3(POS, c, 0)));
    let aStar = argmaxAbs(A, a => gradAt(heads[H].vOutBlock, new Vec3(POS, a, 0)));
    let iStar = H * A + aStar;

    commentary(wt, null, 0)`
多頭注意力有個容易被忽略的細節：那幾個 head 其實**從頭到尾沒有互相講過話**。
它們各算各的，最後只是把輸出並排接在一起（concat），再乘上一個投射矩陣。

投射層是唯一讓各個 head 的資訊混在一起的地方。
反向時，它也是唯一負責把責任**切開分回各個 head** 的地方。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.6);
    let t_fade = afterTime(null, 1.3);

    breakAfter();
    commentary(wt)`
梯度從 ${c_blockRef('注意力輸出', blk.attnOut)} 進來（上一章殘差分給它的那一份）。

先看 ${c_blockRef('投射權重', blk.projWeight)}。它是 ${c_dimRef('C', DimStyle.C)} × ${c_dimRef('C', DimStyle.C)} 的方陣，
其中一格 W[c, i] 在前向時，每個位置都把 concat 的第 i 維乘進輸出的第 c 維。

所以這一格的梯度沿位置加總：dAttnOut 的第 c 列，逐格乘上 concat 的第 i 列（也就是 head ${embedInline(<>{H}</>)} 輸出的某一列）。`;
    breakAfter();

    let t_camDW = afterTime(null, 0.8);
    let t_dWDemo = afterTime(null, 5.0, 0.3);
    let t_dWFill = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`dW^{\text{proj}}_{c,i} = \sum_{t} d\text{AttnOut}_{c,t}\;\text{concat}_{i,t}`} />)}

反過來看 concat 那一邊。concat 的一格影響過輸出的**每一個通道 c**，
所以它的梯度沿 c 加總：dAttnOut 位置 5 那一行（48 格），逐格乘上權重的第 i 行。`;
    breakAfter();

    let t_camConcat = afterTime(null, 0.8);
    let t_dConcatDemo = afterTime(null, 5.0, 0.3);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`d\text{concat}_{i,t} = \sum_{c} d\text{AttnOut}_{c,t}\;W^{\text{proj}}_{c,i}`} />)}

算出來的 dConcat 是一整條 48 維。前向時三個 head 的輸出是「並排接起來」的：
head 0 佔前 ${c_dimRef('A', DimStyle.A)} 維、head 1 佔接下來 A 維、head 2 佔最後 A 維。

**串接在反向就是切開**：不需要任何運算，只是把同一條按位置切成三段，各還給自己的 head。`;
    breakAfter();

    let t_camSplit = afterTime(null, 0.8);
    let t_splitDemo = afterTime(null, 4.5, 0.3);
    let t_splitFill = afterTime(null, 2.5);

    breakAfter();
    commentary(wt)`
三個 head 各自拿到自己那一段，接下來會沿著各自的 Q、K、V 往回走 —— 那是下一章。

順帶一提，${c_blockRef('投射偏置', blk.projBias)} 對每個位置加同一個數，所以它的梯度是把每個位置加起來。`;
    breakAfter();

    let t_camBias = afterTime(null, 0.8);
    let t_biasDemo = afterTime(null, 3.0, 0.3);
    let t_biasFill = afterTime(null, 1.5);

    // 每段示範寫成函式：同一個函式拿去畫，也拿去給相機量出它會用到畫面上的哪些地方
    let dWScene = (tm: ITimeInfo) => scenePairDot(state, tm,
        { blk: blk.attnOut, fixDim: Dim.Y, fixIdx: cStar, kind: 'grad' },
        { blk: heads[H].vOutBlock, fixDim: Dim.Y, fixIdx: aStar, kind: 'fwd' },
        { blk: blk.projWeight, idx: new Vec3(iStar, cStar, 0) },
        { maxPairs: POS + 1 });
    let dConcatScene = (tm: ITimeInfo) => scenePairDot(state, tm,
        { blk: blk.attnOut, fixDim: Dim.X, fixIdx: POS, kind: 'grad' },
        { blk: blk.projWeight, fixDim: Dim.X, fixIdx: iStar, kind: 'fwd' },
        { blk: heads[H].vOutBlock, idx: new Vec3(POS, aStar, 0) },
        { maxPairs: 8 });
    // 切開：三段先疊成一整條，浮在注意力輸出那一行前面，再分頭飛回各自的 head
    let splitScene = (tm: ITimeInfo) => {
        let stackTl = cellPos(state, blk.attnOut, new Vec3(POS, 0, 0));
        heads.forEach((h, i) => {
            sceneMoveSlice(state, tm,
                { blk: h.vOutBlock, fixDim: Dim.X, fixIdx: POS, kind: 'grad' },
                { blk: h.vOutBlock, fixDim: Dim.X, fixIdx: POS },
                { from: stackTl.add(new Vec3(0, i * A * layout.cell, 0)), delay: i * 0.12 });
        });
    };
    let biasScene = (tm: ITimeInfo) => sceneCollapse(state, tm,
        { blk: blk.attnOut, fixDim: Dim.Y, fixIdx: cStar, kind: 'grad' },
        { blk: blk.projBias, idx: new Vec3(0, cStar, 0) },
        { maxCells: POS + 1 });

    // 三個 head 沿深度方向前後錯開，正面看會疊成一塊，這一章維持原本的斜角
    let OBLIQUE = { azimuth: 293.6, elevation: 2.6 };

    let camera = new BackpropCamera(state);
    camera.shot(t_moveCamera, camera.fixed(cam(-68.2, -282.4), new Vec3(293.6, 2.6, 1.1)));
    camera.shot(t_camDW, camera.scene('dW', dWScene, t_dWDemo, OBLIQUE));
    camera.shot(t_dWFill, camera.blocks('dWFill', [blk.projWeight], { ...FILL_SHOT, ...OBLIQUE }), FILL_MOVE);
    camera.shot(t_camConcat, camera.scene('dConcat', dConcatScene, t_dConcatDemo, OBLIQUE));
    camera.shot(t_camSplit, camera.scene('split', splitScene, t_splitDemo, OBLIQUE));
    camera.shot(t_splitFill, camera.blocks('splitFill', heads.map(h => h.vOutBlock), { ...FILL_SHOT, ...OBLIQUE }), FILL_MOVE);
    camera.shot(t_camBias, camera.scene('bias', biasScene, t_biasDemo, OBLIQUE));
    camera.apply();

    focusBackwardScene(state, new Set([
        ...heads.map(h => h.vOutBlock),
        blk.projWeight,
        blk.projBias,
        blk.attnOut,
    ]), t_fade.t);

    if (t_fade.t > 0) {
        processBackwardChain(state, t_fade, [blk.attnOut]);
    }
    if (t_dWFill.t > 0) {
        processBackwardChain(state, t_dWFill, [blk.attnOut, blk.projWeight]);
    }
    if (t_splitFill.t > 0) {
        processBackwardChain(state, t_splitFill, [blk.attnOut, ...heads.map(h => h.vOutBlock)]);
    }
    if (t_biasFill.t > 0) {
        processBackwardChain(state, t_biasFill, [blk.attnOut, blk.projBias]);
    }

    dWScene(t_dWDemo);
    dConcatScene(t_dConcatDemo);
    splitScene(t_splitDemo);
    biasScene(t_biasDemo);
}
