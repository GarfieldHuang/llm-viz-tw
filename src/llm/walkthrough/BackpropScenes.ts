/**
 * 反向章節的「倒放前向」場景。
 *
 * 前向章節好懂，是因為它把算術演出來：格子從矩陣裡拆出來、飛到一起、畫上 x 與 +、
 * 結果落進目的地。讀者不用先懂公式。
 *
 * 反向傳播的每一步，都是某個前向運算的鏡像，所以這裡的每個場景都是「同一批格子倒著演」：
 *
 *   前向 O = Σ P·V（P 的一列乘上各條 V 再加起來）
 *   反向 dV：dO 那一條複製成好幾份，各乘上 P 的一格，分頭送回各條 V
 *
 * 規則：
 *  - 飛行中的格子一律是**複本**，原本的區塊留在原地，才看得出「從哪來、到哪去」。
 *  - 複本要明講顯示的是梯度還是前向值（showGrad / showForward）——
 *    反向檢視下區塊的取值來源已經換成梯度貼圖，前向值要從 fwdAccess 換回來。
 *  - 複本只是畫面，不帶 deps：否則它們會被當成真的消費者，浮層和側邊欄的推導會多出一條假的路徑。
 *  - 場景只在 0 < t < 1 時畫；t 走完代表結果已經落地，交給 processBackwardChain 把整塊亮起來。
 *  - 畫在模型裡的字只能用字型圖集裡有的字：ASCII 加 Σ γ β σ μ ε ‧ —。
 *    乘號用 x、減號用 —（跟前向章節一樣）；× 與 − 不在圖集裡，畫出來是空白。
 *  - 擺複本一律用 place、寫字一律用 writeText。BackpropCamera 取景時會在量測模式下把場景跑一遍，
 *    靠這兩個函式知道一段動畫會用到畫面上的哪些地方。
 */
import { dimConsts, dimProps } from "../Annotations";
import { IBlkDef, IGptModelLayout, setBlkPosition } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { measureText } from "../render/fontRender";
import { clamp } from "@/src/utils/data";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { BoundingBox3d, Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { ITimeInfo } from "./WalkthroughTools";
import { inverseLerp } from "./Walkthrough04_SelfAttention";
import { cellPos, drawSymbolAt } from "./BackpropAnim";
import { fwdAt } from "../components/GradMath";

/** 運算符號與說明文字的字級。格子只有 1.5 單位寬，字太小在畫面上根本看不見。 */
const SYM = 2.4;
const LABEL = 1.8;

// ---------------------------------------------------------------------------
// 基本工具
// ---------------------------------------------------------------------------

/** 把 [a, b] 這段時間映成 0..1，並做 smoothstep 緩動。 */
export function seg(t: number, a: number, b: number) {
    return lerpSmoothstep(0, 1, inverseLerp(a, b, t));
}

/** 顯示梯度（沒有梯度資料的區塊保持關閉，絕不露出前向值）。 */
export function showGrad(blk: IBlkDef) {
    if (blk.access && !blk.gradMissing) {
        blk.access = { ...blk.access, disable: false };
    }
    blk.subs?.forEach(showGrad);
}

/** 顯示前向值：反向檢視下 access 已換成梯度貼圖，要從 fwdAccess 換回來。 */
export function showForward(blk: IBlkDef) {
    if (blk.fwdAccess) {
        blk.access = { ...blk.fwdAccess, disable: false };
    } else if (blk.access) {
        blk.access = { ...blk.access, disable: false };
    }
    blk.subs?.forEach(showForward);
}

// ---------------------------------------------------------------------------
// 取景用的量測
// ---------------------------------------------------------------------------

/**
 * 量測模式：場景照常算出每個複本與文字的位置，但不畫、不加進 layout，
 * 只把它們占的範圍收進這個框。
 */
let measuring: BoundingBox3d | null = null;

/** 擺放一個複本。量測模式下同時記下它占的範圍。 */
export function place(blk: IBlkDef, pos: Vec3) {
    setBlkPosition(blk, pos);
    if (measuring) {
        measuring.addInPlace(pos);
        measuring.addInPlace(new Vec3(pos.x + blk.dx, pos.y + blk.dy, pos.z + blk.dz));
    }
}

/** 在模型空間寫字，字的中心在 pos。量測模式下只記下字占的範圍。 */
export function writeText(state: IProgramState, pos: Vec3, text: string, size: number) {
    if (!measuring) {
        drawSymbolAt(state, pos, text, size);
        return;
    }
    let fontBuf = state.render?.modelFontBuf;
    let w = fontBuf?.atlas?.faceInfos?.length
        ? measureText(fontBuf, text, { color: new Vec4(0, 0, 0, 1), size })
        : text.length * size * 0.6;
    measuring.addInPlace(new Vec3(pos.x - w / 2, pos.y - size / 2, pos.z));
    measuring.addInPlace(new Vec3(pos.x + w / 2, pos.y + size / 2, pos.z));
}

/**
 * 把一段場景在 0..1 之間取樣執行，回傳每個取樣點上畫面會用到的範圍。
 * run 收到的計時器只有 t 有意義 —— 場景函式本來也只看 t。
 */
export function measureScene(state: IProgramState, run: (timer: ITimeInfo) => void, samples: number) {
    let ts: number[] = [];
    let boxes: BoundingBox3d[] = [];
    for (let i = 0; i <= samples; i++) {
        let s = i / samples;
        let box = new BoundingBox3d();
        measuring = box;
        try {
            run({ name: '', start: 0, duration: 1, wait: 0, t: clamp(s, 0.002, 0.998), active: true });
        } finally {
            measuring = null;
        }
        ts.push(s);
        boxes.push(box);
    }
    return { ts, boxes };
}

export type ValueKind = 'grad' | 'fwd';

function show(blk: IBlkDef, kind: ValueKind) {
    if (kind === 'fwd') {
        showForward(blk);
    } else {
        showGrad(blk);
    }
}

/**
 * 區塊某一段的獨立複本，**完全不改動原本的區塊**。
 *
 * 不能用 splitGrid 來做：它會改寫 blk.subs，而同一幀裡對同一塊切第二次時，
 * 新切出來的子塊會把舊的 subs 一起帶走 —— 渲染時一層套一層，子塊數量指數成長
 * （實測注意力章節幾幀之內衝到兩萬多個，整個頁面卡死）。
 * 這裡照 splitGrid 的算法算出位置與 localMtx，但產生的是一塊全新的方塊。
 */
function subRange(state: IProgramState, blk: IBlkDef, dim: Dim, iStart: number, iEnd: number): IBlkDef {
    let layout = state.layout;
    let { x, sizeX, offX } = dimProps(blk, dim);
    let { vecId, xName, dxName, offXName, sizeXName } = dimConsts(dim);
    let mtx = Mat4f.fromScaleTranslation(
        new Vec3(1, 1, 1).setAt(vecId, (iEnd - iStart) / sizeX),
        new Vec3().setAt(vecId, iStart / sizeX));
    return {
        ...blk,
        deps: undefined,
        subs: undefined,
        rangeOffsetsX: undefined,
        rangeOffsetsY: undefined,
        rangeOffsetsZ: undefined,
        access: blk.access && { ...blk.access },
        localMtx: (blk.localMtx ?? new Mat4f()).mul(mtx),
        [dxName]: (iEnd - iStart) * layout.cell,
        [xName]: x + iStart * layout.cell,
        [offXName]: iStart + offX,
        [sizeXName]: iEnd - iStart,
    };
}

function addCopy(state: IProgramState, d: IBlkDef, kind: ValueKind | null): IBlkDef {
    d.name = '';
    d.opacity = 1;
    if (!measuring) {
        state.layout.cubes.push(d);
    }
    if (kind) {
        show(d, kind);
    }
    return d;
}

/** 某一格的複本。 */
export function dupCell(state: IProgramState, blk: IBlkDef, idx: Vec3, kind: ValueKind): IBlkDef | null {
    if (idx.x < 0 || idx.y < 0 || idx.x >= blk.cx || idx.y >= blk.cy) {
        return null;
    }
    let col = subRange(state, blk, Dim.X, idx.x, idx.x + 1);
    return addCopy(state, subRange(state, col, Dim.Y, idx.y, idx.y + 1), kind);
}

/** 一整條的參照：fixDim = X 表示固定 x、取直的一行；fixDim = Y 表示固定 y、取橫的一列。 */
export interface ISliceRef {
    blk: IBlkDef;
    fixDim: Dim;
    fixIdx: number;
    kind: ValueKind;
}

function sliceInRange(s: { blk: IBlkDef, fixDim: Dim, fixIdx: number }) {
    let n = s.fixDim === Dim.X ? s.blk.cx : s.blk.cy;
    return s.fixIdx >= 0 && s.fixIdx < n;
}

/** 一整條的複本。 */
export function dupSlice(state: IProgramState, s: ISliceRef): IBlkDef | null {
    if (!sliceInRange(s)) {
        return null;
    }
    return addCopy(state, subRange(state, s.blk, s.fixDim, s.fixIdx, s.fixIdx + 1), s.kind);
}

/** 一整條拆成一格一格的複本，連同它們原本的位置。 */
function dupSliceCells(state: IProgramState, s: ISliceRef, maxCells?: number) {
    if (!sliceInRange(s)) {
        return [];
    }
    let slice = subRange(state, s.blk, s.fixDim, s.fixIdx, s.fixIdx + 1);
    let along = s.fixDim === Dim.X ? Dim.Y : Dim.X;
    let count = along === Dim.X ? s.blk.cx : s.blk.cy;
    let n = Math.min(count, maxCells ?? count);
    let out: { dup: IBlkDef, from: Vec3 }[] = [];
    for (let i = 0; i < n; i++) {
        let d = addCopy(state, subRange(state, slice, along, i, i + 1), s.kind);
        out.push({ dup: d, from: new Vec3(d.x, d.y, d.z) });
    }
    return out;
}

/** 沒有數值的佔位格（例如 ρ、ḡ 這種推導裡的中間量），用聚合色畫成單獨一格。 */
export function plainCell(state: IProgramState, like: IBlkDef, t: 'a' | 'i' | 'w' = 'a'): IBlkDef {
    let cell = state.layout.cell;
    let d = addCopy(state, {
        ...like,
        cx: 1, cy: 1, cz: 1,
        dx: cell, dy: cell, dz: cell,
        localMtx: undefined,
        offX: undefined, offY: undefined, offZ: undefined,
        sizeX: undefined, sizeY: undefined, sizeZ: undefined,
        deps: undefined,
        subs: undefined,
        rangeOffsetsX: undefined,
        rangeOffsetsY: undefined,
        rangeOffsetsZ: undefined,
        access: undefined,
        t,
    }, null);
    d.highlight = 0.4;
    return d;
}

const FRONT = 10; // 飛行時浮在區塊前方幾格，免得穿模

function front(state: IProgramState, mul = 1) {
    return new Vec3(0, 0, state.layout.cell * FRONT * mul);
}

function sliceTl(state: IProgramState, blk: IBlkDef, fixDim: Dim, fixIdx: number) {
    return cellPos(state, blk, new Vec3(fixDim === Dim.X ? fixIdx : 0, fixDim === Dim.Y ? fixIdx : 0, 0));
}

function active(timer: ITimeInfo) {
    return timer.t > 0 && timer.t < 1;
}

/** 從 from 浮起來、平移、再降落到 to。 */
function flyPath(from: Vec3, to: Vec3, lift: Vec3, t: number) {
    if (t < 0.25) {
        return from.lerp(from.add(lift), t / 0.25);
    } else if (t < 0.8) {
        return from.add(lift).lerp(to.add(lift), (t - 0.25) / 0.55);
    }
    return to.add(lift).lerp(to, (t - 0.8) / 0.2);
}

// ---------------------------------------------------------------------------
// 一、加權分流（注意力 dV 的倒放）
// ---------------------------------------------------------------------------

export interface IFanOutItem {
    /** 這一份要乘的前向值（例如 P 的一格） */
    weight: { blk: IBlkDef, idx: Vec3 };
    /** 落點：目的區塊的哪一條（直的一行） */
    dest: { blk: IBlkDef, colIdx: number };
}

/**
 * 一條梯度複製成好幾份，每份乘上一個前向值，再分頭落到各自的目的地。
 *
 * 前向的 O = Σ_s P[t,s]·V[s] 是「P 的一列當權重，把好幾條 V 加成一條」；
 * 反向剛好倒過來：dO 那一條分成好幾份，第 s 份乘上 P[t,s]，送回 V 的第 s 條。
 * 乘完之後複本的顏色會跟著變淡或變深 —— 被關注越多的位置，拿到的梯度越大。
 *
 * 複本先在目的地前方**拉開間距排好**（跟前向章節展開 V 的排法一樣），
 * 否則六份緊貼在一起，看起來就只是一塊。
 */
export function sceneFanOutWeighted(state: IProgramState, timer: ITimeInfo, src: { blk: IBlkDef, colIdx: number }, items: IFanOutItem[]) {
    if (!active(timer) || items.length === 0) {
        return;
    }
    let t = timer.t;
    let cell = state.layout.cell;
    let tSplit = seg(t, 0.0, 0.32);
    let tWeight = seg(t, 0.32, 0.52);
    let tMul = seg(t, 0.52, 0.7);
    let tLand = seg(t, 0.74, 1.0);

    let srcTl = sliceTl(state, src.blk, Dim.X, src.colIdx);
    let lift = front(state);
    let baseTl = sliceTl(state, items[0].dest.blk, Dim.X, items[0].dest.colIdx);

    items.forEach((it, i) => {
        let copy = dupSlice(state, { blk: src.blk, fixDim: Dim.X, fixIdx: src.colIdx, kind: 'grad' });
        if (!copy) {
            return;
        }
        let destTl = sliceTl(state, it.dest.blk, Dim.X, it.dest.colIdx);
        let stage = baseTl.add(new Vec3(i * cell * 4, 0, 0)).add(lift);

        // 先浮起來，再錯開一點點出發，才看得出是「好幾份」
        let ts = seg(tSplit, (i / items.length) * 0.35, 1);
        let lifted = srcTl.add(lift);
        let pos = ts < 0.25 ? srcTl.lerp(lifted, ts / 0.25) : lifted.lerp(stage, (ts - 0.25) / 0.75);
        if (tLand > 0) {
            pos = stage.lerp(destTl, tLand);
        }
        place(copy, pos);
        copy.highlight = 0.35;

        let midY = copy.dy / 2 - cell / 2;
        let w = fwdAt(state, it.weight.blk, it.weight.idx);
        if (tMul > 0 && w !== null && copy.access) {
            copy.access = { ...copy.access, scale: copy.access.scale * lerp(1, w, tMul) };
        }

        if (tWeight > 0 && tLand <= 0) {
            let wd = dupCell(state, it.weight.blk, it.weight.idx, 'fwd');
            if (wd) {
                let wFrom = cellPos(state, it.weight.blk, it.weight.idx);
                let beside = stage.add(new Vec3(-cell * 2.2, midY, 0));
                let into = stage.add(new Vec3(0, midY, 0));
                let wp = wFrom.lerp(beside, tWeight);
                if (tMul > 0) {
                    wp = beside.lerp(into, tMul);
                    wd.opacity = 1 - tMul;
                }
                place(wd, wp);
                wd.highlight = 0.6;
                if (tWeight >= 1 && tMul < 0.6) {
                    writeText(state, beside.add(new Vec3(cell * 1.6, cell * 0.5, cell)), 'x', SYM);
                }
            }
        }

        if (tLand > 0.8) {
            writeText(state, destTl.add(new Vec3(cell * 0.5, -cell * 1.8, cell * 2)), '+', SYM);
        }
    });
}

// ---------------------------------------------------------------------------
// 二、成對點積（權重梯度、以及任何「沿一條加總」的矩陣乘法反向）
// ---------------------------------------------------------------------------

/**
 * 兩條拆成一格一格，成對飛到一起畫 x，再收攏畫 +，結果飛進目的地那一格。
 *
 * 跟前向解釋 Q = Wq·LN 的演法一樣，只是來源換了：一邊是梯度，一邊是前向值，
 * 而且加總的方向常常翻面（前向沿 c 加，dWq 沿 t 加）—— 看格子是從哪個方向被拆出來的就知道。
 */
export function scenePairDot(
    state: IProgramState,
    timer: ITimeInfo,
    a: ISliceRef,
    b: ISliceRef,
    dest: { blk: IBlkDef, idx: Vec3 },
    opts?: { maxPairs?: number, suffix?: string },
) {
    if (!active(timer)) {
        return;
    }
    let t = timer.t;
    let cell = state.layout.cell;

    let aCells = dupSliceCells(state, a, opts?.maxPairs);
    let bCells = dupSliceCells(state, b, opts?.maxPairs);
    let n = Math.min(aCells.length, bCells.length);
    if (n === 0) {
        return;
    }

    let rowH = cell * 1.5;
    let destTl = cellPos(state, dest.blk, dest.idx);
    let stage = destTl.add(new Vec3(-cell * 12, -(n * rowH + cell * 3), 0)).add(front(state));
    let sumPos = stage.add(new Vec3(cell * 1.75, n * rowH + cell * 1.5, 0));

    let tGather = seg(t, 0.0, 0.45);
    let tCollapse = seg(t, 0.48, 0.76);
    let tLand = seg(t, 0.78, 1.0);

    for (let i = 0; i < n; i++) {
        let pct = n > 1 ? i / (n - 1) : 0;
        let tt = seg(tGather, pct * 0.45, pct * 0.45 + 0.55);
        let aT = stage.add(new Vec3(0, i * rowH, 0));
        let bT = stage.add(new Vec3(cell * 3.5, i * rowH, 0));

        let ap = aCells[i].from.lerp(aT, tt);
        let bp = bCells[i].from.lerp(bT, tt);
        if (tCollapse > 0) {
            ap = aT.lerp(sumPos, tCollapse);
            bp = bT.lerp(sumPos, tCollapse);
        }
        place(aCells[i].dup, ap);
        place(bCells[i].dup, bp);
        aCells[i].dup.highlight = 0.3;
        bCells[i].dup.highlight = 0.3;
        if (tLand > 0) {
            aCells[i].dup.opacity = 0;
            bCells[i].dup.opacity = 0;
        }

        if (tt >= 1 && tCollapse <= 0) {
            writeText(state, aT.lerp(bT, 0.5).add(new Vec3(cell * 0.5, cell * 0.5, cell)), 'x', SYM * 0.8);
        }
        if (i > 0 && tCollapse > 0 && tCollapse < 0.85) {
            writeText(state, ap.add(new Vec3(-cell * 1.2, -cell * 0.2, cell)), '+', SYM * 0.8);
        }
    }
    // 沒被演出來的格子也要算進去 —— 用「...」提醒還有更多項
    let fullCount = b.fixDim === Dim.X ? b.blk.cy : b.blk.cx;
    if (n < fullCount && tGather >= 1 && tCollapse <= 0) {
        writeText(state, stage.add(new Vec3(cell * 2.25, n * rowH + cell * 0.3, cell)), '...', LABEL);
    }

    if (opts?.suffix && tCollapse > 0.6 && tLand < 0.7) {
        writeText(state, sumPos.add(new Vec3(cell * 4, cell * 0.5, cell)), opts.suffix, LABEL);
    }

    if (tLand > 0) {
        let r = dupCell(state, dest.blk, dest.idx, 'grad');
        if (r) {
            place(r, sumPos.lerp(destTl, tLand));
            r.highlight = 0.7;
        }
    }
}

// ---------------------------------------------------------------------------
// 三、收攏成一格（偏置、β）
// ---------------------------------------------------------------------------

/**
 * 一整條的每一格一起浮起來，再收攏成一格，落進目的地。
 *
 * 偏置對每個位置都加同一個數，所以反向要把每個位置的責任加起來 —— 這就是那個「加起來」。
 */
export function sceneCollapse(
    state: IProgramState,
    timer: ITimeInfo,
    src: ISliceRef,
    dest: { blk: IBlkDef, idx: Vec3 },
    opts?: { maxCells?: number },
) {
    if (!active(timer)) {
        return;
    }
    let t = timer.t;
    let cell = state.layout.cell;
    let cells = dupSliceCells(state, src, opts?.maxCells);
    if (cells.length === 0) {
        return;
    }
    let destTl = cellPos(state, dest.blk, dest.idx);
    let sumPos = destTl.add(front(state, 0.6));

    let tLift = seg(t, 0.0, 0.35);
    let tMerge = seg(t, 0.4, 0.78);
    let tLand = seg(t, 0.8, 1.0);

    cells.forEach((c, i) => {
        let lifted = c.from.add(front(state, 0.6));
        let p = c.from.lerp(lifted, tLift);
        if (tMerge > 0) {
            p = lifted.lerp(sumPos, tMerge);
        }
        place(c.dup, p);
        c.dup.highlight = 0.3;
        if (tLand > 0) {
            c.dup.opacity = 0;
        }
        if (i > 0 && tLift >= 1 && tMerge < 0.3) {
            writeText(state, lifted.add(new Vec3(-cell * 0.2, -cell * 0.3, cell)), '+', SYM * 0.7);
        }
    });

    if (tLand > 0) {
        let r = dupCell(state, dest.blk, dest.idx, 'grad');
        if (r) {
            place(r, sumPos.lerp(destTl, tLand));
            r.highlight = 0.7;
        }
    }
}

// ---------------------------------------------------------------------------
// 四、搬家（殘差的原封不動、串接的切開、嵌入的 scatter-add）
// ---------------------------------------------------------------------------

/** 一整條搬到另一個區塊的對應一條。 */
export function sceneMoveSlice(
    state: IProgramState,
    timer: ITimeInfo,
    src: ISliceRef,
    dest: { blk: IBlkDef, fixDim: Dim, fixIdx: number },
    opts?: { symbol?: string, from?: Vec3, delay?: number },
) {
    if (!active(timer)) {
        return;
    }
    let t = seg(timer.t, opts?.delay ?? 0, 1);
    if (t <= 0) {
        return;
    }
    let copy = dupSlice(state, src);
    if (!copy) {
        return;
    }
    let from = opts?.from ?? sliceTl(state, src.blk, src.fixDim, src.fixIdx);
    let to = sliceTl(state, dest.blk, dest.fixDim, dest.fixIdx);
    place(copy, flyPath(from, to, front(state), t));
    copy.highlight = 0.45;

    if (opts?.symbol && t > 0.7) {
        let cell = state.layout.cell;
        writeText(state, to.add(new Vec3(-cell * 1.6, cell * 0.5, cell * 2)), opts.symbol, SYM);
    }
}

/**
 * 整塊梯度複製一份，飛到另一個形狀相同的區塊。
 *
 * 殘差分流用這個：單獨一條只有 1.5 單位寬，從看得到整層的距離望過去只是一條線，
 * 整塊飛過去才看得出「同一份梯度一分為二」。
 */
export function sceneMoveBlock(state: IProgramState, timer: ITimeInfo, src: IBlkDef, dest: IBlkDef, opts?: { symbol?: string, delay?: number }) {
    if (!active(timer)) {
        return;
    }
    let t = seg(timer.t, opts?.delay ?? 0, 1);
    if (t <= 0) {
        return;
    }
    let copy = addCopy(state, {
        ...src,
        deps: undefined,
        subs: undefined,
        rangeOffsetsX: undefined,
        rangeOffsetsY: undefined,
        rangeOffsetsZ: undefined,
        access: src.access && { ...src.access },
    }, 'grad');
    let from = new Vec3(src.x, src.y, src.z);
    let to = new Vec3(dest.x, dest.y, dest.z);
    place(copy, flyPath(from, to, front(state), t));
    copy.highlight = 0.45;

    if (opts?.symbol && t > 0.7) {
        let cell = state.layout.cell;
        writeText(state, to.add(new Vec3(-cell * 2.5, cell * 2, cell * 2)), opts.symbol, SYM * 1.5);
    }
}

// ---------------------------------------------------------------------------
// 五、softmax 的反向：整列綁在一起
// ---------------------------------------------------------------------------

/**
 * dS[t,s] = P[t,s] · ( dP[t,s] − ρ ),  ρ = Σ_s P[t,s]·dP[t,s]
 *
 * 演法分三拍，工作區放在兩個矩陣的**正下方**（中間夾著聚合樁和箭頭，放那裡會糊成一團）：
 *  1. 這一列的 P 與 dP 飛到 P 的正下方成對相乘，收攏成一格 ρ（這一列 dP 的加權平均）
 *  2. dP 與 ρ 各送一份到 S 的正下方，相減
 *  3. 減完再乘上同一格的 P，飛回 S 的那一列
 *
 * 第 1 拍就是「整列綁在一起」的來源：每一格的結果都用到整列算出來的 ρ。
 */
export function sceneSoftmaxRow(state: IProgramState, timer: ITimeInfo, P: IBlkDef, S: IBlkDef, row: number) {
    if (!active(timer)) {
        return;
    }
    let t = timer.t;
    let cell = state.layout.cell;
    let n = Math.min(row + 1, P.cx);

    let belowY = P.y + P.dy + cell * 3;
    let lift = front(state, 0.3);
    let under = (blk: IBlkDef, s: number, dy: number) => {
        let c = cellPos(state, blk, new Vec3(s, row, 0));
        return new Vec3(c.x, belowY + dy, c.z).add(lift);
    };
    let rhoPos = new Vec3((P.x + P.dx + S.x) / 2 - cell / 2, belowY + cell * 0.9, P.z).add(lift);

    let tPair = seg(t, 0.0, 0.28);
    let tRho = seg(t, 0.3, 0.45);
    let tSub = seg(t, 0.48, 0.72);
    let tLand = seg(t, 0.75, 1.0);

    if (tRho > 0 && tLand < 0.9) {
        place(plainCell(state, P), rhoPos);
    }
    if (tRho > 0.5 && tSub < 0.3) {
        writeText(state, rhoPos.add(new Vec3(cell * 0.5, cell * 2.8, cell)), 'Σ P‧dP', LABEL);
    }

    for (let s = 0; s < n; s++) {
        let idx = new Vec3(s, row, 0);
        let pFrom = cellPos(state, P, idx);
        let sTl = cellPos(state, S, idx);

        // 第 1 拍：P 與 dP 飛到 P 的正下方，成對相乘，再收攏成 ρ
        if (tPair > 0 && tRho < 1) {
            let pd = dupCell(state, P, idx, 'fwd');
            let gd = dupCell(state, P, idx, 'grad');
            if (pd && gd) {
                let pairP = under(P, s, 0);
                let pairG = under(P, s, cell * 1.8);
                let pP = pFrom.lerp(pairP, tPair);
                let pG = pFrom.lerp(pairG, tPair);
                if (tRho > 0) {
                    pP = pairP.lerp(rhoPos, tRho);
                    pG = pairG.lerp(rhoPos, tRho);
                    pd.opacity = gd.opacity = 1 - tRho;
                }
                place(pd, pP);
                place(gd, pG);
                if (tPair >= 1 && tRho <= 0) {
                    writeText(state, pairP.add(new Vec3(cell * 0.5, cell * 1.4, cell)), 'x', SYM * 0.6);
                }
            }
        }

        // 第 2 拍：dP 與 ρ 各送一份到 S 那一格的正下方，相減
        let subG = under(S, s, 0);
        let subR = under(S, s, cell * 1.8);
        if (tSub > 0 && tLand < 0.6) {
            let g2 = dupCell(state, P, idx, 'grad');
            if (g2) {
                place(g2, pFrom.lerp(subG, tSub));
            }
            place(plainCell(state, P), rhoPos.lerp(subR, tSub));
            if (tSub >= 1) {
                writeText(state, subG.add(new Vec3(cell * 0.5, cell * 1.4, cell)), '—', SYM * 0.6);
            }
        }

        // 第 3 拍：乘上 P，結果飛回 S 的那一列
        if (tLand > 0) {
            if (tLand < 0.6) {
                let p3 = dupCell(state, P, idx, 'fwd');
                if (p3) {
                    place(p3, pFrom.lerp(under(S, s, -cell * 1.8), seg(tLand, 0, 0.4)));
                }
                if (tLand > 0.35) {
                    writeText(state, under(S, s, -cell * 0.4).add(new Vec3(cell * 0.5, 0, cell)), 'x', SYM * 0.6);
                }
            }
            if (tLand > 0.55) {
                let res = dupCell(state, S, idx, 'grad');
                if (res) {
                    place(res, subG.lerp(sTl, seg(tLand, 0.55, 1)));
                    res.highlight = 0.6;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 六、逐元素相乘（GELU 的反向）
// ---------------------------------------------------------------------------

/** 兩格飛到目的格的正下方相乘，再落進目的格。放在下方，才不會被那一格上方的浮層公式擋住。 */
export function sceneElemMul(
    state: IProgramState,
    timer: ITimeInfo,
    a: { blk: IBlkDef, idx: Vec3, kind: ValueKind },
    b: { blk: IBlkDef, idx: Vec3, kind: ValueKind },
    dest: { blk: IBlkDef, idx: Vec3 },
    opts?: { label?: string },
) {
    if (!active(timer)) {
        return;
    }
    let t = timer.t;
    let cell = state.layout.cell;
    let ad = dupCell(state, a.blk, a.idx, a.kind);
    let bd = dupCell(state, b.blk, b.idx, b.kind);
    if (!ad || !bd) {
        return;
    }
    let destTl = cellPos(state, dest.blk, dest.idx);
    let stage = destTl.add(new Vec3(-cell * 3, cell * 5, 0)).add(front(state));
    let aT = stage;
    let bT = stage.add(new Vec3(cell * 4, 0, 0));

    let tGather = seg(t, 0.0, 0.5);
    let tLand = seg(t, 0.62, 1.0);

    let ap = cellPos(state, a.blk, a.idx).lerp(aT, tGather);
    let bp = cellPos(state, b.blk, b.idx).lerp(bT, tGather);
    if (tLand > 0) {
        let mid = aT.lerp(bT, 0.5);
        ap = mid.lerp(destTl, tLand);
        bp = mid.lerp(destTl, tLand);
        ad.opacity = bd.opacity = 1 - tLand;
    }
    place(ad, ap);
    place(bd, bp);
    ad.highlight = bd.highlight = 0.4;

    if (tGather >= 1 && tLand <= 0.2) {
        writeText(state, aT.lerp(bT, 0.5).add(new Vec3(cell * 0.5, cell * 0.5, cell)), 'x', SYM);
        if (opts?.label) {
            writeText(state, bT.add(new Vec3(cell * 0.5, cell * 2.4, cell)), opts.label, LABEL);
        }
    }
    if (tLand > 0.6) {
        let r = dupCell(state, dest.blk, dest.idx, 'grad');
        if (r) {
            place(r, destTl);
            r.highlight = 0.7;
        }
    }
}

// ---------------------------------------------------------------------------
// 七、損失端：預測減去答案
// ---------------------------------------------------------------------------

/**
 * 機率那一行的每一格飛下來，減掉 one-hot（正解那一格減 1，其他減 0），落成 dLogits。
 */
export function sceneLossSeed(state: IProgramState, timer: ITimeInfo, probs: IBlkDef, logits: IBlkDef, pos: number, target: number) {
    if (!active(timer)) {
        return;
    }
    let t = timer.t;
    let cell = state.layout.cell;
    let V = logits.cy;

    for (let v = 0; v < V; v++) {
        let tv = seg(t, (v / V) * 0.3, (v / V) * 0.3 + 0.7);
        if (tv <= 0) {
            continue;
        }
        let idx = new Vec3(pos, v, 0);
        let from = cellPos(state, probs, idx);
        let to = cellPos(state, logits, idx);
        let lift = front(state, 0.7);

        if (tv < 0.85) {
            let pd = dupCell(state, probs, idx, 'fwd');
            if (pd) {
                place(pd, tv < 0.6 ? from.lerp(to.add(lift), tv / 0.6) : to.add(lift));
                pd.highlight = 0.5;
            }
        }
        if (tv > 0.45 && tv < 0.9) {
            writeText(state, to.add(lift).add(new Vec3(cell * 3.2, cell * 0.5, cell)), v === target ? '— 1' : '— 0', LABEL);
        }
        if (tv > 0.8) {
            let r = dupCell(state, logits, idx, 'grad');
            if (r) {
                place(r, to.add(lift).lerp(to, seg(tv, 0.8, 1)));
                r.highlight = 0.7;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 章節共用
// ---------------------------------------------------------------------------

/** 絕對值最大的索引。示範用的格子挑梯度最大的，畫面上才看得出東西。 */
export function argmaxAbs(n: number, f: (i: number) => number | null): number {
    let best = 0;
    let bestV = -1;
    for (let i = 0; i < n; i++) {
        let v = f(i);
        if (v !== null && Math.abs(v) > bestV) {
            bestV = Math.abs(v);
            best = i;
        }
    }
    return best;
}

export function range(n: number) {
    let a: number[] = [];
    for (let i = 0; i < n; i++) {
        a.push(i);
    }
    return a;
}

/**
 * 把「對著第 0 層」寫的相機位置，平移到第 blockIdx 層。
 *
 * 每一層的排版完全相同，只差在沿 y 往下疊；相機中心的 z 對應的就是模型的 -y。
 */
export function shiftToBlock(layout: IGptModelLayout, blockIdx: number, v: Vec3) {
    let dy = layout.blocks[blockIdx].ln1.lnResid.y - layout.blocks[0].ln1.lnResid.y;
    return new Vec3(v.x, v.y, v.z - dy);
}
