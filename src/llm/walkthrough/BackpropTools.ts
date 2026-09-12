/**
 * 反向章節共用的動畫工具。
 *
 * 這裡刻意**不**沿用「把前向順序倒著走」的做法。
 *
 * 舊版是拿 `layout.cubes` 的陣列索引做區間遞減，但那是前向的擺放順序，
 * 不是反向圖。實際後果是：講 dV 的時候 V 不會亮（掃描根本沒經過它）、
 * 講 dQ dK 的時候 V 反而亮了（它剛好排在中間）、還會花一半時間掃過
 * 兩個純顯示用的聚合樁。
 *
 * 改成由章節**明講**這一步要填哪些區塊。反向圖的分叉與匯流無法用區間表達，
 * 只能列出來。
 */
import { IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { ITimeInfo } from "./WalkthroughTools";
import { dimProps, splitGrid, findSubBlocks } from "../Annotations";
import { Dim, Vec3 } from "@/src/utils/vector";
import { lerp } from "@/src/utils/math";
import { drawDataFlow } from "../components/DataFlow";
import { drawBackwardDependences } from "../components/DataFlowBackward";

export interface IBackpropInfo {
    /** 這一幀正在填的區塊；已經填完則為 chain 的最後一塊 */
    current: IBlkDef | null;
}

/** 打開取值。沒有梯度貼圖的區塊一律不准打開 —— 那會露出前向啟用值卻染成梯度的顏色。 */
function reveal(blk: IBlkDef) {
    if (blk.access && !blk.gradMissing) {
        blk.access.disable = false;
    }
}

/**
 * 依序把 `chain` 裡的區塊填上梯度。
 *
 * 預設 `chain[0]` 是這一步的**起點** —— 它的梯度在上一步就算好了，所以直接整塊亮出來，
 * 不重跑動畫；`chain[1..]` 才是這一步要逐格填出來的。
 * 若這一步沒有前置（例如權重梯度那一鏡），傳 `animateFirst` 讓整條都跑動畫。
 *
 * 時間依格數加權分配，並對點積長度取分數次方，避免大矩陣吃掉整段。
 */
export function processBackwardChain(
    state: IProgramState,
    timer: ITimeInfo,
    chain: IBlkDef[],
    opts?: { animateFirst?: boolean },
): IBackpropInfo {
    if (chain.length === 0) {
        return { current: null };
    }

    let targets: IBlkDef[];
    if (opts?.animateFirst) {
        targets = chain;
    } else {
        // 起點：上一步的結果，直接整塊亮著
        reveal(chain[0]);
        targets = chain.slice(1);
    }

    if (targets.length === 0) {
        return { current: chain[0] };
    }

    let cellCounts = targets.map(a => (a.cx * a.cy) * Math.pow(a.deps?.dotLen ?? 1, 0.25));
    let totalCells = cellCounts.reduce((a, b) => a + b, 0) || 1;

    let currPos = targets.length - 1;
    let subPos = 1;
    let acc = 0;
    for (let n = 0; n < targets.length; n++) {
        let fract = cellCounts[n] / totalCells;
        acc += fract;
        if (timer.t < acc) {
            currPos = n;
            subPos = (timer.t - (acc - fract)) / fract;
            break;
        }
    }
    if (timer.t >= 1.0) {
        currPos = targets.length - 1;
        subPos = 1;
    }

    // 已經填完的整塊亮著
    for (let n = 0; n < currPos; n++) {
        reveal(targets[n]);
    }

    let blk = targets[currPos];

    if (!timer.active) {
        return { current: blk };
    }

    if (timer.t >= 1.0) {
        // 這一步結束：整條鏈都填滿
        for (let b of targets) {
            reveal(b);
        }
        return { current: targets[targets.length - 1] };
    }

    let dim0 = blk.transpose ? Dim.Y : Dim.X;
    let dim1 = blk.transpose ? Dim.X : Dim.Y;
    let { cx } = dimProps(blk, dim0);
    let { cx: cy } = dimProps(blk, dim1);

    let horizPos = lerp(0, cx, subPos);
    let horizIdx = Math.floor(horizPos);
    let vertPos = lerp(0, cy, horizPos - horizIdx);
    let vertIdx = Math.floor(vertPos);
    let blockPos = new Vec3().withSetAt(dim0, horizIdx).withSetAt(dim1, vertIdx);

    // 點亮「把梯度交給這一格的人」，並畫出反向的公式
    drawBackwardDependences(state, blk, blockPos);
    drawDataFlow(state, blk, blockPos);

    for (let label of state.layout.labels) {
        for (let c of label.cubes) {
            if (c === blk) {
                label.visible = 1.0;
            }
        }
    }

    blk.highlight = 0.3;

    if (!blk.gradMissing) {
        let column = splitGrid(state.layout, blk, dim0, horizPos, 0);
        if (column) {
            for (let col of findSubBlocks(blk, dim0, null, horizIdx)) {
                reveal(col);
                col.highlight = 0.1;
            }
            column.highlight = 0.4;
            let curr = splitGrid(state.layout, column, dim1, vertPos, 0);
            for (let sub of findSubBlocks(column, dim1, null, vertIdx)) {
                reveal(sub);
            }
            if (curr) {
                curr.highlight = 0.7;
            }
        }
    }

    return { current: blk };
}

/**
 * 反向章節共用的場景設定：把不相關的區塊淡出，相關的先關掉取值，
 * 之後由 processBackwardChain 逐格打開。
 *
 * 權重區塊也一併關掉 —— 權重的梯度（dWq、dWv……）是整個反向傳播的終點，
 * 值得自己一個鏡頭亮出來，不該一開始就掛在那裡。
 */
export function focusBackwardScene(state: IProgramState, relevant: Set<IBlkDef>, fadeT: number) {
    for (let blk of state.layout.cubes) {
        if (!relevant.has(blk)) {
            blk.opacity = lerp(1.0, 0.15, fadeT);
        }
    }
    for (let blk of relevant) {
        if (blk.access) {
            blk.access.disable = true;
        }
    }
}
