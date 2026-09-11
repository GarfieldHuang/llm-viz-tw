import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough06_Projection(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { breakAfter, afterTime, c_blockRef, c_dimRef, cleanup } } = args;

    if (wt.phase !== Phase.Input_Detail_Projection) {
        return;
    }

    setInitialCamera(state, new Vec3(-73.167, 0.000, -270.725), new Vec3(293.606, 2.613, 1.366));
    let block = layout.blocks[0];
    wt.dimHighlightBlocks = [...block.heads.map(h => h.vOutBlock), block.projBias, block.projWeight, block.attnOut];

    let outBlocks = block.heads.map(h => h.vOutBlock);

    commentary(wt, null, 0)`

在自注意力（self-attention）過程之後，我們會從每個頭部得到輸出。這些輸出是受 Q 和 K 向量影響而適度混合的 V 向量。

要合併每個頭部的${c_blockRef('輸出向量(output vectors)', outBlocks)}，我們只需將它們堆疊在一起即可。因此，在時間 ${c_dimRef('t = 4', DimStyle.T)} 時，我們將從 3 個長度為 ${c_dimRef('A = 16', DimStyle.A)} 的向量疊加到 1 個長度為 ${c_dimRef('C = 48', DimStyle.C)} 的向量。
`;

    breakAfter();

    let t_fadeOut = afterTime(null, 1.0, 0.5);
    // let t_zoomToStack = afterTime(null, 1.0);
    let t_stack = afterTime(null, 1.0);

    breakAfter();

    commentary(wt)`

值得注意的是，在 GPT 中，頭部 (${c_dimRef('A = 16', DimStyle.A)}) 內向量的長度等於 ${c_dimRef('C', DimStyle.C)}  / num_heads。這確保了當我們將它們重新堆疊在一起時，能得到原來的長度 ${c_dimRef('C', DimStyle.C)}。

在此基礎上，我們進行投影，得到該層的輸出。這是一個簡單的矩陣-向量乘法，以每列為單位，並加上偏置。
`;

    breakAfter();

    let t_process = afterTime(null, 3.0);

    breakAfter();

    commentary(wt)`

現在我們得到了自注意力(self-attention)層的輸出。我們不是直接將這個輸出傳遞到下一個階段，而是將其與輸入嵌入(input embedding)進行元素級相加。這個過程，用綠色垂直箭頭表示，被稱為 _殘差連線（residual connection）_ 或 _殘差路徑（residual pathway）_。
`;

    breakAfter();

    let t_zoomOut = afterTime(null, 1.0, 0.5);
    let t_processResid = afterTime(null, 3.0);

    cleanup(t_zoomOut, [t_fadeOut, t_stack]);

    breakAfter();

    commentary(wt)`
    就像層歸一化一樣，殘差路徑對於在深度神經網路中實現有效學習非常重要。

    現在有了自注意力的結果，我們可以將其傳遞到變換器（transformer）的下一部分：前饋網路(the feed-forward network)。
`;

    breakAfter();

    if (t_fadeOut.active) {
        for (let head of block.heads) {
            for (let blk of head.cubes) {
                if (blk !== head.vOutBlock) {
                    blk.opacity = lerpSmoothstep(1, 0, t_fadeOut.t);
                }
            }
        }
    }

    if (t_stack.active) {
        let targetZ = block.attnOut.z;
        for (let headIdx = 0; headIdx < block.heads.length; headIdx++) {
            let head = block.heads[headIdx];
            let targetY = head.vOutBlock.y + head.vOutBlock.dy * (headIdx - block.heads.length + 1);
            head.vOutBlock.y = lerp(head.vOutBlock.y, targetY, t_stack.t);
            head.vOutBlock.z = lerp(head.vOutBlock.z, targetZ, t_stack.t);
        }
    }

    let processInfo = startProcessBefore(state, block.attnOut);

    if (t_process.active) {
        processUpTo(state, t_process, block.attnOut, processInfo);
    }

    moveCameraTo(state, t_zoomOut, new Vec3(-8.304, 0.000, -175.482), new Vec3(293.606, 2.623, 2.618));

    if (t_processResid.active) {
        processUpTo(state, t_processResid, block.attnResidual, processInfo);
    }
}
