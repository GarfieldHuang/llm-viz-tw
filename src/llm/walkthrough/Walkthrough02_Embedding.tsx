import { duplicateGrid, splitGrid } from "../Annotations";
import { getBlockValueAtIdx } from "../components/DataFlow";
import { IBlkDef } from "../GptModelLayout";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { lerp } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough02_Embedding(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, tools: { c_str, c_blockRef, c_dimRef, afterTime, cleanup, breakAfter }, layout } = args;
    let render = state.render;

    if (wt.phase !== Phase.Input_Detail_Embedding) {
        return;
    }

    setInitialCamera(state, new Vec3(15.654, 0.000, -80.905), new Vec3(287.000, 14.500, 3.199));
    wt.dimHighlightBlocks = [layout.idxObj, layout.tokEmbedObj, layout.posEmbedObj, layout.residual0];

    commentary(wt)`
    我們之前看到如何使用一個簡單的查詢表將標記(token)映射為一串整數。這些整數，即標記索引(${c_blockRef('_token indices_', state.layout.idxObj, DimStyle.TokenIdx)})，是我們在模型中第一次也是唯一一次看到的整數。從這裡開始，我們將使用浮點數（十進位制數）。

    讓我們來看看第4個標記（token, 索引3）是如何用於生成輸入嵌入（${c_blockRef('_input embedding_', state.layout.residual0)}）的第4列向量的。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t0_splitEmbedAnim = afterTime(null, 0.3);

    breakAfter();

    commentary(wt)`
    我們使用標記索引（本例中為 ${c_str('B', DimStyle.Token)} = ${c_dimRef('1', DimStyle.TokenIdx)}）來選擇左邊${c_blockRef('_標記嵌入矩陣(token embedding matrix)_', state.layout.tokEmbedObj)}的第2列。請注意，我們在這裡使用的是基於 0 的索引，因此第一列的索引為 0。

    這樣就產生了一個大小為 ${c_dimRef('_C_ = 48', DimStyle.C)} 的列向量，我們將其描述為標記嵌入(token embedding)。`;
    breakAfter();

    let t1_fadeEmbedAnim = afterTime(null, 0.3);
    let t2_highlightTokenEmbed = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
    由於我們要檢視的是第 4 個位置（t = 3）上的標記 ${c_str('B', DimStyle.Token)}，因此我們將取${c_blockRef('_位置嵌入矩陣(position embedding matrix)_', state.layout.posEmbedObj)}的第 4 列。

    這也會產生一個大小為 ${c_dimRef('_C_ = 48', DimStyle.C)} 的列向量，我們將其描述為位置嵌入。`;
    breakAfter();

    let t4_highlightPosEmbed = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
    請注意，這些位置嵌入和標記嵌入都是在訓練過程中學習的（用藍色表示）。

    現在我們有了這兩個列向量，只需將它們相加，就能產生另一個大小為 ${c_dimRef('_C_ = 48', DimStyle.C)} 的列向量。`;
    breakAfter();

    let t3_moveTokenEmbed = afterTime(null, 0.8);
    let t5_movePosEmbed = afterTime(null, 0.8);
    let t6_plusSymAnim = afterTime(null, 0.8);
    let t7_addAnim = afterTime(null, 0.8);
    let t8_placeAnim = afterTime(null, 0.8);
    let t9_cleanupInstant = afterTime(null, 0.0);
    let t10_fadeAnim = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
    現在，我們對輸入序列中的所有標記執行同樣的過程，建立一組包含標記值及其位置的向量。
`;

    breakAfter();

    let t11_fillRest = afterTime(null, 5.0);

    breakAfter();

    commentary(wt)`
    請隨意將滑鼠懸停在${c_blockRef('_輸入嵌入（input embedding）_', state.layout.residual0)}矩陣的各個單元格上，檢視計算過程及其來源。

    我們可以看到，對輸入序列中的所有標記執行這一過程會產生一個大小為 ${c_dimRef('_T_', DimStyle.T)} x ${c_dimRef('_C_', DimStyle.C)} 的矩陣。${c_dimRef('_T_', DimStyle.T)} 代表${c_dimRef('_時間_', DimStyle.T)}，也就是說，你可以把序列中稍後的標記看作是時間上稍後的標記。${c_dimRef('_C_', DimStyle.C)} 代表${c_dimRef('_通道_', DimStyle.C)}，但也被稱為 "特徵"、"維度 "或 "嵌入大小"。這個長度 ${c_dimRef('_C_', DimStyle.C)} 是模型的幾個 "超參數 "之一，由設計者在模型大小和效能之間權衡選擇。

    這個矩陣，我們稱之為${c_blockRef('_輸入嵌入（input embedding）_', state.layout.residual0)}，現在可以透過模型向下傳遞了。在本指南中，我們將非常熟悉由長度為 ${c_dimRef('C', DimStyle.C)}  的 ${c_dimRef('T', DimStyle.T)}  列組成的矩陣集合。
    `;

    cleanup(t9_cleanupInstant, [t3_moveTokenEmbed, t5_movePosEmbed, t6_plusSymAnim, t7_addAnim, t8_placeAnim]);
    cleanup(t10_fadeAnim, [t0_splitEmbedAnim, t1_fadeEmbedAnim, t2_highlightTokenEmbed, t4_highlightPosEmbed]);

    moveCameraTo(state, t_moveCamera, new Vec3(7.6, 0, -33.1), new Vec3(290, 15.5, 0.8));

    let residCol: IBlkDef = null!;
    let exampleIdx = 3;
    if ((t0_splitEmbedAnim.t > 0.0 || t10_fadeAnim.t > 0.0) && t11_fillRest.t === 0) {
        splitGrid(layout, layout.idxObj, Dim.X, exampleIdx + 0.5, t0_splitEmbedAnim.t * 4.0);

        layout.residual0.access!.disable = true;
        layout.residual0.opacity = lerp(1.0, 0.1, t1_fadeEmbedAnim.t);

        residCol = splitGrid(layout, layout.residual0, Dim.X, exampleIdx + 0.5, t0_splitEmbedAnim.t * 4.0)!;
        residCol.highlight = 0.3;

        residCol.opacity = lerp(1.0, 0.0, t1_fadeEmbedAnim.t);

    }

    let tokValue = getBlockValueAtIdx(layout.idxObj, new Vec3(exampleIdx, 0, 0)) ?? 1;


    let tokColDupe: IBlkDef | null = null;
    let posColDupe: IBlkDef | null = null;

    if (t2_highlightTokenEmbed.t > 0.0) {
        let tokEmbedCol = splitGrid(layout, layout.tokEmbedObj, Dim.X, tokValue + 0.5, t2_highlightTokenEmbed.t * 4.0)!;

        tokColDupe = duplicateGrid(layout, tokEmbedCol);
        tokColDupe.t = 'i';
        tokEmbedCol.highlight = 0.3;

        let startPos = new Vec3(tokEmbedCol.x, tokEmbedCol.y, tokEmbedCol.z);
        let targetPos = new Vec3(residCol.x, residCol.y, residCol.z).add(new Vec3(-2.0, 0, 3.0));

        let pos = startPos.lerp(targetPos, t3_moveTokenEmbed.t);

        tokColDupe.x = pos.x;
        tokColDupe.y = pos.y;
        tokColDupe.z = pos.z;
    }


    if (t4_highlightPosEmbed.t > 0.0) {
        let posEmbedCol = splitGrid(layout, layout.posEmbedObj, Dim.X, exampleIdx + 0.5, t4_highlightPosEmbed.t * 4.0)!;

        posColDupe = duplicateGrid(layout, posEmbedCol);
        posColDupe.t = 'i';
        posEmbedCol.highlight = 0.3;

        let startPos = new Vec3(posEmbedCol.x, posEmbedCol.y, posEmbedCol.z);
        let targetPos = new Vec3(residCol.x, residCol.y, residCol.z).add(new Vec3(2.0, 0, 3.0));

        let pos = startPos.lerp(targetPos, t5_movePosEmbed.t);

        posColDupe.x = pos.x;
        posColDupe.y = pos.y;
        posColDupe.z = pos.z;
    }

    if (t6_plusSymAnim.t > 0.0 && tokColDupe && posColDupe && t7_addAnim.t < 1.0) {
        for (let c = 0; c < layout.shape.C; c++) {
            let plusCenter = new Vec3(
                (tokColDupe.x + tokColDupe.dx + posColDupe.x) / 2,
                tokColDupe.y + layout.cell * (c + 0.5),
                tokColDupe.z + tokColDupe.dz / 2);

            let isActive = t6_plusSymAnim.t > (c + 1) / layout.shape.C;
            let opacity = lerp(0.0, 1.0, isActive ? 1 : 0);

            let fontOpts: IFontOpts = { color: new Vec4(0, 0, 0, 1).mul(opacity), size: 1.5, mtx: Mat4f.fromTranslation(plusCenter) };
            let w = measureText(render.modelFontBuf, '+', fontOpts);

            drawText(render.modelFontBuf, '+', -w/2, -fontOpts.size/2, fontOpts);
        }
    }

    let origResidPos = residCol ? new Vec3(residCol.x, residCol.y, residCol.z) : new Vec3();
    let offsetResidPos = origResidPos.add(new Vec3(0.0, 0, 3.0));

    if (t7_addAnim.t > 0.0 && tokColDupe && posColDupe) {
        let targetPos = offsetResidPos;
        let tokStartPos = new Vec3(tokColDupe.x, tokColDupe.y, tokColDupe.z);
        let posStartPos = new Vec3(posColDupe.x, posColDupe.y, posColDupe.z);

        let tokPos = tokStartPos.lerp(targetPos, t7_addAnim.t);
        let posPos = posStartPos.lerp(targetPos, t7_addAnim.t);

        tokColDupe.x = tokPos.x;
        tokColDupe.y = tokPos.y;
        tokColDupe.z = tokPos.z;
        posColDupe.x = posPos.x;
        posColDupe.y = posPos.y;
        posColDupe.z = posPos.z;

        if (t7_addAnim.t > 0.95) {
            tokColDupe.opacity = 0.0;
            posColDupe.opacity = 0.0;
            residCol.opacity = 1.0;
            residCol.highlight = 0.0;
            residCol.access!.disable = false;
            residCol.x = targetPos.x;
            residCol.y = targetPos.y;
            residCol.z = targetPos.z;
        }
    }

    if (t8_placeAnim.t > 0.0) {
        let startPos = offsetResidPos;
        let targetPos = origResidPos;
        let pos = startPos.lerp(targetPos, t8_placeAnim.t);
        residCol.x = pos.x;
        residCol.y = pos.y;
        residCol.z = pos.z;
    }

    if (t9_cleanupInstant.t > 0.0 && residCol) {
        residCol.opacity = 1.0;
        residCol.highlight = 0.0;
        residCol.access!.disable = false;
    }

    if (t11_fillRest.t > 0.0) {
        layout.residual0.access!.disable = true;

        let prevInfo = startProcessBefore(state, layout.residual0);
        processUpTo(state, t11_fillRest, layout.residual0, prevInfo);
    }
    // new Vec3(-6.9, 0, -36.5), new Vec3(281.5, 5.5, 0.8)
}
