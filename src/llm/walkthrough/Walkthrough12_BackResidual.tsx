import React from 'react';
import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, ITimeInfo, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { sceneMoveBlock, shiftToBlock } from "./BackpropScenes";
import { BackpropCamera } from "./BackpropCamera";
import { embedInline } from "./Walkthrough01_Prelim";
import { BlockText } from '../components/CommentaryHelpers';
import { Tex } from "../components/Tex";

export function walkthrough12_BackResidual(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Residual) {
        return;
    }

    let li = layout.blocks.length - 1;
    let blk = layout.blocks[li];
    // 注意力殘差的另一個加數：上一層的輸出（第 0 層的話就是嵌入）
    let prev = li > 0 ? layout.blocks[li - 1].mlpResidual : layout.residual0;
    let cam = (x: number, z: number) => shiftToBlock(layout, li, new Vec3(x, 0, z));

    setInitialCamera(state, cam(-135.531, -353.905), new Vec3(291.100, 13.600, 5.706));
    wt.dimHighlightBlocks = [blk.attnResidual, blk.mlpResidual, prev];

    commentary(wt, null, 0)`
這一章只講一個運算：**加法**。

聽起來沒什麼，但殘差連接是讓深層網路訓練得起來的關鍵，而理由完全藏在反向這一側。
前向時它平淡無奇 —— 把輸入原封不動加到輸出上。反向時它做的事，是整個架構的命脈。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 2.4);
    let t_fade = afterTime(null, 1.9);

    breakAfter();
    commentary(wt)`
先不要看公式，直接看梯度會發生什麼事。

它在 ${c_blockRef('MLP 殘差', blk.mlpResidual)}。前向時這裡是「MLP 的輸出 + 注意力殘差」，
所以接下來你會看到這一整塊梯度**分裂成兩份一模一樣的**，分頭飛向兩個加數。`;
    breakAfter();

    let t_split1Demo = afterTime(null, 4.0, 0.3);
    let t_split1Fill = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
兩份**完全相同**。沒有縮放、沒有衰減、沒有矩陣乘法。規則就這麼一句：

${embedInline(<Tex block tex={String.raw`c = a + b \;\Rightarrow\; \frac{\partial c}{\partial a} = \frac{\partial c}{\partial b} = 1 \;\Rightarrow\; da = dc,\quad db = dc`} />)}

一份給 ${c_blockRef('MLP 的輸出', blk.mlpResult)} —— 要穿過 MLP 那條迂迴的路，每一步都被權重改造一次。
另一份給 ${c_blockRef('注意力殘差', blk.attnResidual)} —— **直接跳過整個 MLP**。

（注意力殘差其實還會從 Layer Norm 2 那條路再收到一份。一格被兩個地方用到，梯度就是兩份相加 ——
點一下那一格，側邊欄會把兩條路各推一次再加起來。）`;
    breakAfter();

    let t_moveCamera2 = afterTime(null, 1.5);
    let t_split2Demo = afterTime(null, 4.0, 0.3);
    let t_split2Fill = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
再往上一步，同樣的事又發生一次：注意力殘差把梯度複製給
${c_blockRef('注意力輸出', blk.attnOut)} 和 ${c_blockRef('上一層的輸出', prev)}。

${embedInline(<span className='block ml-2 my-1 text-sm'>
        <span className='block'>迂迴那條路：<BlockText blk={blk.mlpResult}>MLP</BlockText> → 乘權重 → GELU → 乘權重 → 梯度被改造好幾次</span>
        <span className='block'>高速公路：<BlockText blk={blk.attnResidual}>殘差</BlockText> → 乘數恆為 1 → 原封不動送到上一層</span>
    </span>)}`;
    breakAfter();

    commentary(wt)`
現在把這件事乘上深度來想。

沒有殘差的話，梯度每經過一層就要被一個權重矩陣乘一次；乘上幾十次之後，
它不是爆炸就是消失 —— 這就是深層網路曾經訓練不起來的原因。

有了殘差，**每一層都有一條乘數恆為 1 的路徑直通底層**。
就算迂迴那條路把梯度縮到近乎零，高速公路上那一份還是完好無缺地送到了。

所以 GPT 能疊到 96 層，靠的不是什麼精巧的初始化，就是這個加號。`;
    breakAfter();

    let t_settle = afterTime(null, 1.6);

    // 整塊飛：只看一條的話，從看得到整層的距離望過去只剩一條線
    let split1Scene = (tm: ITimeInfo) => {
        sceneMoveBlock(state, tm, blk.mlpResidual, blk.mlpResult, { symbol: '=' });
        sceneMoveBlock(state, tm, blk.mlpResidual, blk.attnResidual, { symbol: '=', delay: 0.12 });
    };
    let split2Scene = (tm: ITimeInfo) => {
        sceneMoveBlock(state, tm, blk.attnResidual, blk.attnOut, { symbol: '=' });
        sceneMoveBlock(state, tm, blk.attnResidual, prev, { symbol: '=', delay: 0.12 });
    };

    // 飛的是整塊，一格很小也看得出來，不必跟拍；上方留給填色時的浮層
    let BLOCK_MOVE = { minCellPx: 1, top: 0.2 };

    let camera = new BackpropCamera(state);
    camera.shot(t_moveCamera, camera.scene('split1', split1Scene, t_split1Demo, BLOCK_MOVE));
    camera.shot(t_moveCamera2, camera.scene('split2', split2Scene, t_split2Demo, BLOCK_MOVE));
    camera.apply();

    focusBackwardScene(state, new Set([
        prev, blk.attnOut, blk.attnResidual, blk.mlpResult, blk.mlpResidual,
    ]), t_fade.t);

    if (t_fade.t > 0) {
        processBackwardChain(state, t_fade, [blk.mlpResidual]);
    }
    if (t_split1Fill.t > 0) {
        processBackwardChain(state, t_split1Fill, [blk.mlpResidual, blk.mlpResult, blk.attnResidual]);
    }
    if (t_split2Fill.t > 0) {
        processBackwardChain(state, t_split2Fill, [blk.attnResidual, blk.attnOut, prev]);
    }
    if (t_settle.t > 0) {
        processBackwardChain(state, t_settle, [prev]);
    }

    split1Scene(t_split1Demo);
    split2Scene(t_split2Demo);
}
