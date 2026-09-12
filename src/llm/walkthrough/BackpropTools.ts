/**
 * 反向章節共用的動畫工具。
 *
 * 與前向的 processUpTo 對稱：processUpTo 由輸入往輸出「逐格填出」啟用值，
 * processDownFrom 則由輸出往輸入「逐格填出」梯度。
 *
 * 一個關鍵觀察：既有的 drawDependences 畫的是「這一格依賴哪些來源」的箭頭，
 * 而梯度正是沿著這些邊反向流回去，所以不需要另外實作反向箭頭 —— 同一組箭頭，
 * 在反向章節裡讀作「這一格的梯度要送去哪裡」。
 */
import { drawDependences } from "../Interaction";
import { IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { ITimeInfo } from "./WalkthroughTools";
import { dimProps, splitGrid, findSubBlocks } from "../Annotations";
import { Dim, Vec3 } from "@/src/utils/vector";
import { lerp } from "@/src/utils/math";
import { drawDataFlow } from "../components/DataFlow";

export interface IBackpropInfo {
    lastBlockIdx: number;
}

/** 取得「有梯度可顯示」的區塊序列（與 processUpTo 一致地排除權重區塊）。 */
function activeBlocksOf(state: IProgramState): IBlkDef[] {
    return state.layout.cubes.filter(a => a.t !== 'w');
}

/** 反向章節起點：從某個區塊開始往回走。 */
export function startBackpropAt(state: IProgramState, block: IBlkDef): IBackpropInfo {
    let activeBlocks = activeBlocksOf(state);
    return { lastBlockIdx: activeBlocks.indexOf(block) + 1 };
}

/**
 * 由 fromBlock 往回（索引遞減）逐格顯示梯度，直到 toBlock。
 * timer.t 由 0 到 1，對應整段反向過程。
 */
export function processDownFrom(
    state: IProgramState,
    timer: ITimeInfo,
    fromBlock: IBlkDef,
    toBlock: IBlkDef,
): IBackpropInfo {
    let activeBlocks = activeBlocksOf(state);

    let startIdx = activeBlocks.indexOf(fromBlock);
    let endIdx = activeBlocks.indexOf(toBlock);
    if (startIdx < 0 || endIdx < 0 || endIdx > startIdx) {
        return { lastBlockIdx: endIdx };
    }

    // 反向：從 startIdx 遞減到 endIdx。每個區塊佔的時間依格數加權，
    // 與前向同樣取分數次方，避免大矩陣吃掉整段時間。
    let order: number[] = [];
    for (let i = startIdx; i >= endIdx; i--) {
        order.push(i);
    }

    let cellCounts = order.map(i => {
        let a = activeBlocks[i];
        return (a.cx * a.cy) * Math.pow(a.deps?.dotLen ?? 1, 0.25);
    });
    let totalCells = cellCounts.reduce((a, b) => a + b, 0) || 1;

    let accCell = 0;
    let currPos = 0;
    let subPos = 0;
    for (let n = 0; n < order.length; n++) {
        let fract = cellCounts[n] / totalCells;
        accCell += fract;
        if (timer.t < accCell) {
            currPos = n;
            subPos = (timer.t - (accCell - fract)) / fract;
            break;
        }
    }
    if (timer.t >= 1.0) {
        currPos = order.length - 1;
        subPos = 1;
    }

    let currIdx = order[currPos];
    let blk = activeBlocks[currIdx];

    let dim0 = blk.transpose ? Dim.Y : Dim.X;
    let dim1 = blk.transpose ? Dim.X : Dim.Y;
    let { cx } = dimProps(blk, dim0);
    let { cx: cy } = dimProps(blk, dim1);

    let horizPos = lerp(0, cx, subPos);
    let horizIdx = Math.floor(horizPos);
    let vertPos = lerp(0, cy, horizPos - horizIdx);
    let vertIdx = Math.floor(vertPos);
    let blockPos = new Vec3().withSetAt(dim0, horizIdx).withSetAt(dim1, vertIdx);

    // 已經走過的區塊（索引比 currIdx 大的，也就是更靠近輸出端）整塊顯示梯度
    for (let n = 0; n < currPos; n++) {
        let b = activeBlocks[order[n]];
        if (b.access) {
            b.access.disable = false;
        }
    }

    if (timer.active && timer.t < 1.0) {
        // 這些箭頭指向「這一格的梯度要送回哪些來源」
        drawDependences(state, blk, blockPos);
        drawDataFlow(state, blk, blockPos);

        for (let label of state.layout.labels) {
            for (let c of label.cubes) {
                if (c === blk) {
                    label.visible = 1.0;
                }
            }
        }

        blk.highlight = 0.3;

        let column = splitGrid(state.layout, blk, dim0, horizPos, 0);
        if (column) {
            for (let col of findSubBlocks(blk, dim0, null, horizIdx)) {
                if (col.access) {
                    col.access.disable = false;
                    col.highlight = 0.1;
                }
            }
            column.highlight = 0.4;
            let curr = splitGrid(state.layout, column, dim1, vertPos, 0);
            for (let sub of findSubBlocks(column, dim1, null, vertIdx)) {
                if (sub.access) {
                    sub.access.disable = false;
                }
            }
            if (curr) {
                curr.highlight = 0.7;
            }
        }
    } else if (timer.active) {
        let b = activeBlocks[endIdx];
        if (b.access) {
            b.access.disable = false;
        }
    }

    return { lastBlockIdx: currIdx };
}

/**
 * 反向章節共用的場景設定：把不相關的區塊淡出，並把相關區塊的取值先關閉，
 * 之後再由 processDownFrom 逐格打開。
 */
export function focusBackwardScene(state: IProgramState, relevant: Set<IBlkDef>, fadeT: number) {
    for (let blk of state.layout.cubes) {
        if (!relevant.has(blk)) {
            blk.opacity = lerp(1.0, 0.15, fadeT);
        }
    }
    for (let blk of relevant) {
        if (blk.access && blk.t !== 'w') {
            blk.access.disable = true;
        }
    }
}
