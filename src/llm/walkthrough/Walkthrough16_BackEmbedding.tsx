import React from 'react';
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, ITimeInfo, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { range, sceneMoveSlice } from "./BackpropScenes";
import { BackpropCamera, FILL_MOVE, FILL_SHOT } from "./BackpropCamera";
import { embedInline } from "./Walkthrough01_Prelim";
import { Tex } from "../components/Tex";
import { getBlockValueAtIdx } from "../components/DataFlow";

const TOKEN_NAMES = ['A', 'B', 'C'];

export function walkthrough16_BackEmbedding(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Embedding) {
        return;
    }

    let POS = state.gradData?.lossPos ?? 5;

    // 輸入序列直接從模型讀，文字裡的次數與位置才不會跟畫面上的輸入對不起來
    let tokens = range(POS + 1).map(t => Math.round(getBlockValueAtIdx(layout.idxObj, new Vec3(t, 0, 0)) ?? 0));
    let seq = tokens.map(v => TOKEN_NAMES[v] ?? '?').join(' ');
    let counts = TOKEN_NAMES
        .map((name, v) => ({ name, pos: tokens.map((tk, t) => tk === v ? t : -1).filter(t => t >= 0) }))
        .filter(c => c.pos.length > 0)
        .sort((a, b) => b.pos.length - a.pos.length);
    let countText = counts.map(c => `"${c.name}" ${c.pos.length} 次（位置 ${c.pos.join('、')}）`).join('，');
    let most = counts[0] ?? { name: '?', pos: [] };

    setInitialCamera(state, new Vec3(15.654, 0.000, -80.905), new Vec3(287.000, 14.500, 3.199));
    wt.dimHighlightBlocks = [layout.tokEmbedObj, layout.posEmbedObj];

    commentary(wt, null, 0)`
這是最後一站。梯度從損失出發，穿過輸出層、三層 transformer、注意力和 Layer Norm，
現在回到了最開頭的兩張查表。

嵌入層的反向很特別，因為前向那一步**根本不是運算** —— 它是查表。
一個運算的反向是另一個運算，但查表的反向是什麼？`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.6);
    let t_fade = afterTime(null, 1.3);

    breakAfter();
    commentary(wt)`
答案是 **scatter-add**：把梯度加回被查到的那一行。

前向時，位置 t 的 token 去 ${c_blockRef('詞嵌入表', layout.tokEmbedObj)} 查了屬於它的那一行。
反向時，這個位置的梯度就整行加回那一行；沒被查到的行，一個字都不會動。

看著前六個位置的梯度一行一行飛回詞嵌入表：同一個 token 出現幾次，那一行就被加幾次。`;
    breakAfter();

    let t_camTok = afterTime(null, 0.8);
    let t_dTokDemo = afterTime(null, 6.0, 0.3);
    let t_dTokFill = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
詞彙表只有 ${c_dimRef('三個 token', DimStyle.n_vocab)}，序列卻有六個位置，所以**同一行會被好幾個位置加到**。
這個例子的輸入是 ${embedInline(<b>{seq}</b>)}：${embedInline(<>{countText}</>)}。
"${embedInline(<>{most.name}</>)}" 那一行就疊了 ${embedInline(<>{most.pos.length}</>)} 份責任。

${embedInline(<Tex block tex={String.raw`dW^{\text{te}}_{c,v} = \sum_{t:\ \text{tok}_t = v} dX_{c,t}`} />)}

這就是 scatter-add 那個 add 的來源 —— 不是覆蓋，是累加。
也解釋了為什麼常見的詞學得快、罕見的詞學得慢：**梯度的累積次數就是出現次數**。`;
    breakAfter();

    commentary(wt)`
${c_blockRef('位置嵌入表', layout.posEmbedObj)} 的情況剛好相反：位置 t 只會查到第 t 行，一對一，不會重複。
所以每一行最多只拿到一份梯度，原封不動地搬過去。`;
    breakAfter();

    let t_camPos = afterTime(null, 0.8);
    let t_dPosDemo = afterTime(null, 4.5, 0.3);
    let t_dPosFill = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
${embedInline(<Tex block tex={String.raw`dW^{\text{pe}}_{c,t} = dX_{c,t}`} />)}

因為這個例子的損失只看位置 5，**位置 6 以後的行全是灰的**：
它們在這次前向裡沒有影響到位置 5（因果遮罩擋住了），自然也沒有責任。`;
    breakAfter();

    let t_end = afterTime(null, 3.2);

    breakAfter();
    commentary(wt)`
到這裡，反向傳播結束了。

沒有下一步了 —— 嵌入表是葉節點，它的梯度就是終點。
從損失那三個數字 [1, 0, −1] 開始，一路分流、相乘、穿過 softmax 和 GELU、
被 Layer Norm 扣掉整行的平均，最後散落回這兩張表上。

整趟走完，模型裡**每一個參數**都拿到了一個數字，回答同一個問題：
「把我調高一點點，損失會變多少？」

optimizer 接手的就是這些數字。反向傳播的工作，到此為止。`;
    breakAfter();

    // 每段示範寫成函式：同一個函式拿去畫，也拿去給相機量出它會用到畫面上的哪些地方
    let tokScene = (tm: ITimeInfo) => tokens.forEach((tok, t) => {
        sceneMoveSlice(state, tm,
            { blk: layout.residual0, fixDim: Dim.X, fixIdx: t, kind: 'grad' },
            { blk: layout.tokEmbedObj, fixDim: Dim.X, fixIdx: tok },
            { symbol: '+', delay: t * 0.1 });
    });
    let posScene = (tm: ITimeInfo) => tokens.forEach((_, t) => {
        sceneMoveSlice(state, tm,
            { blk: layout.residual0, fixDim: Dim.X, fixIdx: t, kind: 'grad' },
            { blk: layout.posEmbedObj, fixDim: Dim.X, fixIdx: t },
            { symbol: '=', delay: t * 0.08 });
    });

    // 總覽沿用手調的值；示範與填色由實際會畫到的範圍算出來
    let camera = new BackpropCamera(state);
    camera.shot(t_moveCamera, camera.fixed(new Vec3(9.5, 0, -48.4), new Vec3(287, 14.5, 1.4)));
    camera.shot(t_camTok, camera.scene('tok', tokScene, t_dTokDemo));
    camera.shot(t_dTokFill, camera.blocks('tokFill', [layout.tokEmbedObj], FILL_SHOT), FILL_MOVE);
    camera.shot(t_camPos, camera.scene('pos', posScene, t_dPosDemo));
    camera.shot(t_dPosFill, camera.blocks('posFill', [layout.posEmbedObj], FILL_SHOT), FILL_MOVE);
    camera.shot(t_end, camera.blocks('end', [layout.tokEmbedObj, layout.residual0, layout.posEmbedObj], FILL_SHOT), FILL_MOVE);
    camera.apply();

    focusBackwardScene(state, new Set([
        layout.idxObj,
        layout.tokEmbedObj,
        layout.posEmbedObj,
        layout.residual0,
    ]), t_fade.t);

    if (t_fade.t > 0) {
        processBackwardChain(state, t_fade, [layout.residual0]);
    }
    if (t_dTokFill.t > 0) {
        processBackwardChain(state, t_dTokFill, [layout.residual0, layout.tokEmbedObj]);
    }
    if (t_dPosFill.t > 0) {
        processBackwardChain(state, t_dPosFill, [layout.residual0, layout.posEmbedObj]);
    }
    if (t_end.t > 0) {
        processBackwardChain(state, t_end, [layout.tokEmbedObj]);
        processBackwardChain(state, t_end, [layout.posEmbedObj]);
    }

    tokScene(t_dTokDemo);
    posScene(t_dPosDemo);
}
