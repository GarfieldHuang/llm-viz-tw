import React from 'react';
import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { focusBackwardScene, processBackwardChain } from "./BackpropTools";
import { flyCopies } from "./BackpropAnim";
import { embedInline } from "./Walkthrough01_Prelim";
import { BlockText } from '../components/CommentaryHelpers';

export function walkthrough12_BackResidual(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, breakAfter } } = args;

    if (wt.phase !== Phase.Backward_Residual) {
        return;
    }

    let block0 = layout.blocks[0];

    // 損失只看位置 5，所以拿那一行的某一格來示範才有非零的梯度可看
    let demoIdx = new Vec3(5, 20, 0);

    setInitialCamera(state, new Vec3(-135.531, 0.000, -353.905), new Vec3(291.100, 13.600, 5.706));
    wt.dimHighlightBlocks = [block0.attnResidual, block0.mlpResidual];

    commentary(wt, null, 0)`
這一章只講一個運算：**加法**。

聽起來沒什麼，但殘差連接是讓深層網路訓練得起來的關鍵，而理由完全藏在反向這一側。
前向時它平淡無奇 —— 把輸入原封不動加到輸出上。反向時它做的事，是整個架構的命脈。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 2.4);
    let t_fade = afterTime(null, 1.9);

    breakAfter();
    commentary(wt)`
先不要看公式，直接看一格會發生什麼事。

我從 ${c_blockRef('MLP 殘差', block0.mlpResidual)} 挑一格梯度出來。接下來你會看到它
**分裂成兩個一模一樣的數字**，分頭飛向兩條路。`;
    breakAfter();

    let t_splitDemo = afterTime(null, 5.5);

    breakAfter();
    commentary(wt)`
兩份**完全相同**。沒有縮放、沒有衰減、沒有矩陣乘法。

規則就這麼一句：前向是 c = a + b，那麼不論 a、b 是什麼，c 對兩者的偏導都是 1，所以

da = dc　　db = dc

一份給 ${c_blockRef('MLP Result', block0.mlpResult)} —— 要穿過 MLP 那條迂迴的路，
經過兩個線性層和一個 GELU，每一步都會被權重矩陣改造一次。

另一份給 ${c_blockRef('注意力殘差', block0.attnResidual)} —— **直接跳過整個 MLP**，一步到位。`;
    breakAfter();

    let t_split1 = afterTime(null, 6.0);

    breakAfter();
    commentary(wt)`
再往下一層，同樣的事情又發生一次：注意力殘差把梯度複製給
${c_blockRef('注意力輸出', block0.attnOut)} 和 ${c_blockRef('原始的嵌入', layout.residual0)}。

${embedInline(<div className='ml-2 my-1 text-sm'>
        <div>迂迴那條路：<BlockText blk={block0.mlpResult}>MLP</BlockText> → 乘權重 → GELU → 乘權重 → 梯度被改造好幾次</div>
        <div>高速公路：<BlockText blk={block0.attnResidual}>殘差</BlockText> → 乘數恆為 1 → 原封不動送到底</div>
    </div>)}`;
    breakAfter();

    let t_split2 = afterTime(null, 6.0);

    breakAfter();
    commentary(wt)`
現在把這件事乘上深度來想。

沒有殘差的話，梯度每經過一層就要被一個權重矩陣乘一次；乘上幾十次之後，
它不是爆炸就是消失 —— 這就是深層網路曾經訓練不起來的原因。

有了殘差，**每一層都有一條乘數恆為 1 的路徑直通底層**。
就算迂迴那條路把梯度縮到近乎零，高速公路上那一份還是完好無缺地送到了。

所以 GPT 能疊到 96 層，靠的不是什麼精巧的初始化，就是這個加號。`;
    breakAfter();

    let t_settle = afterTime(null, 2.4);

    moveCameraTo(state, t_moveCamera, new Vec3(-120.9, 0, -365.8), new Vec3(291.1, 13.6, 3.4));

    let relevant = new Set([
        layout.residual0,
        block0.attnOut,
        block0.attnResidual,
        block0.mlpResult,
        block0.mlpResidual,
    ]);
    focusBackwardScene(state, relevant, t_fade.t);

    // 一格的示範：複製成兩份飛走。加法的反向就這麼一件事，用看的比用讀的快。
    if (t_splitDemo.t > 0 && t_split1.t === 0) {
        flyCopies(state, t_splitDemo,
            { blk: block0.mlpResidual, idx: demoIdx },
            [
                { blk: block0.mlpResult, idx: demoIdx },
                { blk: block0.attnResidual, idx: demoIdx },
            ],
            { symbol: '=' });
    }

    // 一分為二：同一份梯度同時落在兩條分支上
    if (t_split1.t > 0) {
        processBackwardChain(state, t_split1, [
            block0.mlpResidual, block0.mlpResult, block0.attnResidual,
        ]);
    }
    if (t_split2.t > 0) {
        processBackwardChain(state, t_split2, [
            block0.attnResidual, block0.attnOut, layout.residual0,
        ]);
    }
    if (t_settle.t > 0) {
        processBackwardChain(state, t_settle, [layout.residual0]);
    }
}
