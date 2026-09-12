/**
 * 反向章節的「飛行格子」動畫。
 *
 * 前向章節（尤其是 Walkthrough04）解釋一個運算的方式，是把格子從矩陣裡
 * 拆出來、在空間中飛到一起、畫上運算符號、再把結果飛進目的地 ——
 * 你是「看到算術發生」的，不需要先讀懂公式。
 *
 * 反向章節原本只有「逐格填色 + 印公式」，那是公式的示意圖，不是演示。
 * 這個模組把同一套手法搬過來：拆格子（splitGrid / splitGridAll）、
 * 複製（duplicateGrid）、移動（setBlkPosition + lerp）、在模型空間裡
 * 畫符號（drawText）。
 *
 * 時間一律用 smoothstep 緩動，動作才不會一開始就衝出去。
 */
import { duplicateGrid, findSubBlocks, splitGrid, splitGridAll } from "../Annotations";
import { cellPosition, getBlkDimensions, IBlkDef, setBlkPosition } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { lerpSmoothstep } from "@/src/utils/math";
import { inverseLerp } from "./Walkthrough04_SelfAttention";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { ITimeInfo } from "./WalkthroughTools";

export interface IAnimArgs {
    state: IProgramState;
}

/** 這一格在模型空間裡的左上角座標。 */
export function cellPos(state: IProgramState, blk: IBlkDef, idx: Vec3): Vec3 {
    let layout = state.layout;
    return new Vec3(
        cellPosition(layout, blk, Dim.X, idx.x),
        cellPosition(layout, blk, Dim.Y, idx.y),
        cellPosition(layout, blk, Dim.Z, idx.z),
    );
}

/** 把單一格從區塊裡拆出來，之後才能單獨移動。 */
export function cellOf(state: IProgramState, blk: IBlkDef, idx: Vec3): IBlkDef | null {
    let layout = state.layout;
    let col = splitGrid(layout, blk, Dim.X, idx.x + 0.5, 0);
    if (!col) {
        return null;
    }
    return splitGrid(layout, col, Dim.Y, idx.y + 0.5, 0) ?? null;
}

/** 在模型空間的某個點畫一個符號（＋、×、＝ 之類）。 */
export function drawSymbolAt(state: IProgramState, pos: Vec3, symbol: string, size = 1.5, color = new Vec4(0, 0, 0, 1)) {
    let mtx = Mat4f.fromTranslation(pos);
    let fontOpts: IFontOpts = { color, size, mtx };
    let w = measureText(state.render.modelFontBuf, symbol, fontOpts);
    drawText(state.render.modelFontBuf, symbol, -w / 2, -fontOpts.size / 2, fontOpts);
}

// ---------------------------------------------------------------------------
// 一、複製分流：一格 -> 多份，原封不動
// ---------------------------------------------------------------------------

/**
 * 把一格複製成多份，各自飛向一個目的地。
 *
 * 這是殘差連接反向的全部內容：加法的反向就是把梯度**原封不動複製**給每條分支。
 * 看到同一個數字分裂成兩個一模一樣的、分頭飛走，比任何公式都清楚。
 * 投射層把梯度切給三個 head 也用這個。
 */
export function flyCopies(
    state: IProgramState,
    timer: ITimeInfo,
    src: { blk: IBlkDef, idx: Vec3 },
    dests: { blk: IBlkDef, idx: Vec3 }[],
    opts?: { symbol?: string },
) {
    let cell = cellOf(state, src.blk, src.idx);
    if (!cell) {
        return;
    }
    cell.highlight = 0.7;

    let from = getBlkDimensions(cell).tl;
    let t = lerpSmoothstep(0, 1, timer.t);

    for (let i = 0; i < dests.length; i++) {
        let d = dests[i];
        let copy = duplicateGrid(state.layout, cell);
        copy.highlight = 0.7;

        // 錯開一點點出發，才看得出是「好幾份」而不是一份
        let stagger = dests.length > 1 ? (i / dests.length) * 0.15 : 0;
        let tt = lerpSmoothstep(0, 1, inverseLerp(stagger, 1, timer.t));

        let to = cellPos(state, d.blk, d.idx);
        setBlkPosition(copy, from.lerp(to, tt));

        // 飛到之後在目的地畫個等號，點明「一模一樣」
        if (opts?.symbol && tt > 0.85) {
            drawSymbolAt(state, to.add(new Vec3(-state.layout.cell * 1.2, state.layout.cell * 0.5, 0)), opts.symbol);
        }
    }

    // 原件留在原地當參照
    void t;
}

// ---------------------------------------------------------------------------
// 二、成對點積：兩條 -> 逐項相乘 -> 相加 -> 結果飛進目的地
// ---------------------------------------------------------------------------

/**
 * 把兩條切片拆成一格一格，成對飛到一起、畫上 ×，再疊起來畫 +，
 * 最後把結果飛進目的地那一格。
 *
 * 權重梯度 dW = Σ_t dOut ‧ In 就是這樣算的。前向的 Q = Wq·LN 用的是同一套演法，
 * 這裡只是把來源換成「梯度」與「前向值」。
 */
export function flyPairDot(
    state: IProgramState,
    timer: ITimeInfo,
    a: { blk: IBlkDef, alongX: boolean, fixed: number },
    b: { blk: IBlkDef, alongX: boolean, fixed: number },
    dest: { blk: IBlkDef, idx: Vec3 },
    opts?: { maxPairs?: number },
) {
    let layout = state.layout;

    let aSlice = sliceOf(state, a);
    let bSlice = sliceOf(state, b);
    if (!aSlice || !bSlice) {
        return;
    }

    let aCells = splitGridAll(layout, aSlice, a.alongX ? Dim.X : Dim.Y);
    let bCells = splitGridAll(layout, bSlice, b.alongX ? Dim.X : Dim.Y);

    let n = Math.min(aCells.length, bCells.length, opts?.maxPairs ?? 999);
    if (n === 0) {
        return;
    }

    // 集合點放在目的地格子的斜上方，避免壓到原本的區塊
    let destPos = cellPos(state, dest.blk, dest.idx);
    let stageTop = destPos.add(new Vec3(-layout.cell * 14, -layout.cell * 2, layout.cell * 2));
    let sumPos = stageTop.add(new Vec3(layout.cell * 1.5, layout.cell * (n + 2) * 1.2, 0));

    // 三個階段：飛到集合點(0~.45) -> 收攏相加(.45~.8) -> 結果飛進目的地(.8~1)
    let tGather = lerpSmoothstep(0, 1, inverseLerp(0.0, 0.45, timer.t));
    let tCollapse = lerpSmoothstep(0, 1, inverseLerp(0.45, 0.8, timer.t));
    let tLand = lerpSmoothstep(0, 1, inverseLerp(0.8, 1.0, timer.t));

    let prevY = 0;
    for (let i = 0; i < n; i++) {
        let pct = n > 1 ? i / (n - 1) : 0;
        // 逐格錯開出發，看得出是一對一對飛過來的
        let startT = 0.45 * (1 - pct);
        let tt = lerpSmoothstep(0, 1, inverseLerp(startT, startT + 0.45, timer.t));

        let aInit = getBlkDimensions(aCells[i]).tl;
        let bInit = getBlkDimensions(bCells[i]).tl;
        let aTarget = stageTop.add(new Vec3(0, i * layout.cell * 1.2, 0));
        let bTarget = stageTop.add(new Vec3(layout.cell * 3, i * layout.cell * 1.2, 0));

        setBlkPosition(aCells[i], aInit.lerp(aTarget, tt));
        setBlkPosition(bCells[i], bInit.lerp(bTarget, tt));

        // 到位了才畫乘號
        if (tt >= 1.0 && tCollapse < 1.0) {
            drawSymbolAt(state, aTarget.lerp(bTarget, 0.5).add(new Vec3(0, layout.cell * 0.5, layout.cell)), '×');
        }

        // 收攏：全部往同一點靠，中間畫加號
        if (tCollapse > 0) {
            let aNow = getBlkDimensions(aCells[i]).tl;
            let bNow = getBlkDimensions(bCells[i]).tl;
            setBlkPosition(aCells[i], aNow.lerp(sumPos, tCollapse));
            setBlkPosition(bCells[i], bNow.lerp(sumPos, tCollapse));

            if (i > 0 && tCollapse < 0.9) {
                let here = getBlkDimensions(aCells[i]).tl;
                drawSymbolAt(state, new Vec3(here.x - layout.cell, (prevY + here.y) / 2, here.z + layout.cell), '+');
            }
            prevY = getBlkDimensions(aCells[i]).tl.y;
        }
    }

    // 結果飛進目的地
    if (tLand > 0) {
        let destCell = cellOf(state, dest.blk, dest.idx);
        if (destCell) {
            destCell.highlight = 0.7;
            let landInit = sumPos;
            setBlkPosition(destCell, landInit.lerp(destPos, tLand));
        }
    }

    void tGather;
}

function sliceOf(state: IProgramState, s: { blk: IBlkDef, alongX: boolean, fixed: number }): IBlkDef | null {
    // alongX = 掃過橫軸 -> 固定的是直軸(y)，切出一整列
    let layout = state.layout;
    let dim = s.alongX ? Dim.Y : Dim.X;
    return splitGrid(layout, s.blk, dim, s.fixed + 0.5, 0) ?? null;
}

// ---------------------------------------------------------------------------
// 三、累加進表格：整條飛進目標的同一條，並看得出是「疊上去」
// ---------------------------------------------------------------------------

/**
 * 把來源的一整條飛進目標的一整條，並在落點畫 `+=`。
 *
 * 嵌入表的反向是 scatter-add：好幾個位置的梯度會落到同一行，是累加不是覆蓋。
 * 讓每個位置的梯度依序飛進去，就看得出「疊上去」這件事。
 */
export function flyAccumulate(
    state: IProgramState,
    timer: ITimeInfo,
    srcs: { blk: IBlkDef, colIdx: number }[],
    dest: { blk: IBlkDef, colIdx: number }[],
) {
    let layout = state.layout;
    let n = Math.min(srcs.length, dest.length);

    for (let i = 0; i < n; i++) {
        // 一個接一個飛，才看得出是累加
        let startT = (i / n) * 0.7;
        let tt = lerpSmoothstep(0, 1, inverseLerp(startT, startT + 0.3, timer.t));
        if (tt <= 0) {
            continue;
        }

        let srcCol = splitGrid(layout, srcs[i].blk, Dim.X, srcs[i].colIdx + 0.5, 0);
        if (!srcCol) {
            continue;
        }

        let copy = duplicateGrid(layout, srcCol);
        copy.highlight = 0.6;

        let from = getBlkDimensions(srcCol).tl;
        let to = cellPos(state, dest[i].blk, new Vec3(dest[i].colIdx, 0, 0));
        setBlkPosition(copy, from.lerp(to, tt));

        if (tt > 0.9) {
            drawSymbolAt(state, to.add(new Vec3(-layout.cell * 1.5, layout.cell * 2, 0)), '+', 1.2);
        }
    }
}

// ---------------------------------------------------------------------------
// 四、逐元素相乘：兩格飛到一起，畫 ×，結果進目的地
// ---------------------------------------------------------------------------

/** GELU、以及任何「這一格乘上那一格」的反向都用這個。 */
export function flyElemMul(
    state: IProgramState,
    timer: ITimeInfo,
    a: { blk: IBlkDef, idx: Vec3 },
    b: { blk: IBlkDef, idx: Vec3 },
    dest: { blk: IBlkDef, idx: Vec3 },
    opts?: { symbol?: string },
) {
    let layout = state.layout;
    let aCell = cellOf(state, a.blk, a.idx);
    let bCell = cellOf(state, b.blk, b.idx);
    if (!aCell || !bCell) {
        return;
    }
    aCell.highlight = 0.7;
    bCell.highlight = 0.7;

    let destPos = cellPos(state, dest.blk, dest.idx);
    let stage = destPos.add(new Vec3(-layout.cell * 6, -layout.cell * 3, layout.cell * 2));

    let tGather = lerpSmoothstep(0, 1, inverseLerp(0.0, 0.55, timer.t));
    let tLand = lerpSmoothstep(0, 1, inverseLerp(0.6, 1.0, timer.t));

    let aTarget = stage;
    let bTarget = stage.add(new Vec3(layout.cell * 3, 0, 0));

    setBlkPosition(aCell, getBlkDimensions(aCell).tl.lerp(aTarget, tGather));
    setBlkPosition(bCell, getBlkDimensions(bCell).tl.lerp(bTarget, tGather));

    if (tGather >= 1.0 && tLand < 1.0) {
        drawSymbolAt(state, aTarget.lerp(bTarget, 0.5).add(new Vec3(0, layout.cell * 0.5, layout.cell)), opts?.symbol ?? '×');
    }

    if (tLand > 0) {
        let mid = aTarget.lerp(bTarget, 0.5);
        setBlkPosition(aCell, mid.lerp(destPos, tLand));
        setBlkPosition(bCell, mid.lerp(destPos, tLand));
        let destCell = cellOf(state, dest.blk, dest.idx);
        if (destCell) {
            destCell.highlight = 0.7;
        }
    }
}

// ---------------------------------------------------------------------------
// 五、整條切片搬家：一整列／一整行飛到另一個區塊的對應位置
// ---------------------------------------------------------------------------

/**
 * 把來源的一整條（列或行）飛到目的地的對應一條，途中畫上符號。
 *
 * 損失端「P − onehot」用這個：機率那一行飛下來，減去正解，落成 dLogits。
 */
export function flySliceTo(
    state: IProgramState,
    timer: ITimeInfo,
    src: { blk: IBlkDef, colIdx: number },
    dest: { blk: IBlkDef, colIdx: number },
    opts?: { symbol?: string },
) {
    let layout = state.layout;
    let col = splitGrid(layout, src.blk, Dim.X, src.colIdx + 0.5, 0);
    if (!col) {
        return;
    }

    let copy = duplicateGrid(layout, col);
    copy.highlight = 0.7;

    let t = lerpSmoothstep(0, 1, timer.t);
    let from = getBlkDimensions(col).tl;
    let to = cellPos(state, dest.blk, new Vec3(dest.colIdx, 0, 0));
    setBlkPosition(copy, from.lerp(to, t));

    if (opts?.symbol && t > 0.35 && t < 0.95) {
        let here = getBlkDimensions(copy).tl;
        drawSymbolAt(state, here.add(new Vec3(-layout.cell * 1.8, layout.cell * 1.5, layout.cell)), opts.symbol);
    }
}
