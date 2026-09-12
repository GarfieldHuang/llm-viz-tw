/**
 * 反向檢視的浮層公式。
 *
 * 這個檔案存在的理由，只有一句話：
 *
 *   **在反向傳播裡，一個區塊的梯度公式來自它前向的「消費者」，不是它的 deps。**
 *
 * `blk.deps` 記的是「我是由誰算出來的」，前向浮層照著畫沒問題。
 * 但梯度是倒著流的 —— 注意力分數 S 的梯度並不是由 Q、K 決定，而是由吃掉它的
 * softmax 決定（dS = P ‧ (dP — Σ P‧dP)）。所以照 `blk.deps.special` 分流，
 * 每一塊都會拿到錯的公式。
 *
 * 因此這裡先把整張依賴圖反轉成「消費者表」，再依消費者的前向運算決定要畫什麼。
 *
 * 字型限制：圖集字元集只有 ASCII 加 `Σ γ β σ μ ε ‧ —`（見 create-font-atlas.jsm），
 * 所以公式一律用 `d` 前綴、`^T` 表示轉置、`‧` 表示逐元素相乘。不能出現 ∂、⊙、上標 T、中文。
 */
import { BlKDepSpecial, IBlkCellDep, IBlkDef, IGptModelLayout, cellPosition } from "../GptModelLayout";
import { IProgramState } from "../Program";
import {
    IDataFlowArgs, backWhiteColor, drawCircle, drawMaths, drawRoundedRect, drawZeroSymbol,
    getBlockValueAtIdx, opColor, weightSrcColor, workingSrcColor, createMapping,
} from "./DataFlow";
import { gradColor, gradName, shortName } from "./GradNames";
import { ITextBlockArgs, TextBlockType, mkTextBlock } from "./TextLayout";
import { splitGridForHighlight, splitGrid, findSubBlocks, dimProps } from "../Annotations";
import { getDepSrcIdx } from "../Interaction";
import { IFontOpts } from "../render/fontRender";
import { addLine2, drawLineSegs, makeLineOpts } from "../render/lineRender";
import { Colors, DimStyle, dimStyleTextShort } from "../walkthrough/WalkthroughTools";
import { isNotNil } from "@/src/utils/data";
import { BoundingBox3d, Dim, Vec3, Vec4 } from "@/src/utils/vector";

// ---------------------------------------------------------------------------
// 依賴圖反轉
// ---------------------------------------------------------------------------

export interface IBlkConsumer {
    /** 前向吃掉這一塊的區塊；梯度就是從它流回來的 */
    consumer: IBlkDef;
    /** 那條邊（含 srcIdxMtx，用來把 src 的格子反推回 dest 的格子） */
    dep: IBlkCellDep;
    kind: 'dot' | 'add';
    /** dot 時為 0 或 1，決定這一塊是被乘的哪一邊（dQ 還是 dK） */
    operandIdx: number;
}

/**
 * 純顯示用的聚合樁：LayerNorm 的 μ/σ、softmax 的 max/sumexp。
 *
 * 它們在反向裡沒有獨立的梯度路徑 —— 數學上會被併進 LayerNorm 與 softmax 的
 * 反向式裡。若不濾掉，注意力分數 S 會被算成「有三個消費者」，
 * 浮層就會變成毫無意義的三路相加。
 */
function isAggStub(blk: IBlkDef): boolean {
    let sp = blk.deps?.special;
    return sp === BlKDepSpecial.LayerNormMu
        || sp === BlKDepSpecial.LayerNormSigma
        || sp === BlKDepSpecial.SoftmaxAggMax
        || sp === BlKDepSpecial.SoftmaxAggExp;
}

let cachedLayout: IGptModelLayout | null = null;
let cachedMap: Map<IBlkDef, IBlkConsumer[]> | null = null;

/**
 * 建立「區塊 -> 它的前向消費者們」的對照表。
 *
 * layout 每幀重建，所以這裡用單格快取：同一幀內重複呼叫不會重算，
 * 換幀時 layout 物件換人，自動失效。
 */
export function getConsumerMap(layout: IGptModelLayout): Map<IBlkDef, IBlkConsumer[]> {
    if (cachedLayout === layout && cachedMap) {
        return cachedMap;
    }

    let map = new Map<IBlkDef, IBlkConsumer[]>();

    function add(src: IBlkDef, entry: IBlkConsumer) {
        let arr = map.get(src);
        if (!arr) {
            arr = [];
            map.set(src, arr);
        }
        arr.push(entry);
    }

    for (let blk of layout.cubes) {
        let deps = blk.deps;
        if (!deps) {
            continue;
        }
        if (deps.dot) {
            for (let i = 0; i < deps.dot.length; i++) {
                let dep = deps.dot[i];
                add(dep.src, { consumer: blk, dep, kind: 'dot', operandIdx: i });
            }
        }
        if (deps.add) {
            for (let i = 0; i < deps.add.length; i++) {
                let dep = deps.add[i];
                add(dep.src, { consumer: blk, dep, kind: 'add', operandIdx: i });
            }
        }
    }

    cachedLayout = layout;
    cachedMap = map;
    return map;
}

/**
 * 真正有梯度路徑的消費者。
 *
 * 兩種要濾掉：
 *  1. 聚合樁 —— 它們被併進 LayerNorm / softmax 的反向式裡，不是獨立路徑。
 *  2. **本身沒有梯度資料的消費者** —— 沒有梯度的人，不可能交給你梯度。
 *
 * 第二條看似瑣碎，但它決定了 Logits 的浮層對不對：Logits 的前向消費者是
 * 最後那個 softmax，可是損失是直接對 logits 定義的（cross_entropy(logits, target)），
 * 梯度從來沒有流經畫面上那塊 softmax，它也就沒有梯度貼圖。
 * 少了這條，Logits 會被當成「softmax 的反向」，畫出一個引用了不存在的 dP 的式子。
 */
export function getRealConsumers(state: IProgramState, blk: IBlkDef): IBlkConsumer[] {
    return getGraphConsumers(state, blk).filter(c => !c.consumer.gradMissing);
}

/**
 * 圖上的消費者（只濾掉聚合樁，不管有沒有梯度資料）。
 *
 * 拿來區分兩種「沒有下游」：真的是圖的末端，還是下游存在、只是它的梯度
 * 沒有被記錄下來。兩者要給使用者不同的說法。
 */
export function getGraphConsumers(state: IProgramState, blk: IBlkDef): IBlkConsumer[] {
    let all = getConsumerMap(state.layout).get(blk) ?? [];
    return all.filter(c => !isAggStub(c.consumer));
}

/**
 * 把 dep 的索引映射反過來用。
 *
 * `srcIdxMtx` 把 dest 的 (x, y, b) 映到 src 的 (x, y, b)，其中 `i` 代表點積的自由軸。
 * 反過來時，任何「沒有被 src 決定」的 dest 維度，代表梯度要沿著那整條累加；
 * 這裡取中間那一格當代表，讓箭頭有地方指。
 */
export function destIdxFromSrc(dep: IBlkCellDep, srcIdx: Vec3, destBlk: IBlkDef): Vec3 {
    let mtx = dep.srcIdxMtx;
    let dest = new Vec3(-1, -1, -1);

    for (let k = 0; k < 3; k++) {          // src 的第 k 個分量
        for (let d = 0; d < 3; d++) {      // 是否來自 dest 的第 d 維
            if (mtx.g(k, d) === 1) {
                dest.setAt(d, srcIdx.getAt(k));
            }
        }
    }

    for (let d = 0; d < 3; d++) {
        if (dest.getAt(d) < 0) {
            let cnt = d === 0 ? destBlk.cx : d === 1 ? destBlk.cy : destBlk.cz;
            dest.setAt(d, Math.floor(cnt / 2));
        }
    }
    return dest;
}

/** 這條邊是否帶點積自由軸（dep 字串裡有 'i'）。 */
function depHasDotAxis(dep: IBlkCellDep): boolean {
    return dep.srcIdxMtx.g(0, 3) === 1 || dep.srcIdxMtx.g(1, 3) === 1 || dep.srcIdxMtx.g(2, 3) === 1;
}

// ---------------------------------------------------------------------------
// 浮層
// ---------------------------------------------------------------------------

function fontOptsOf(args: IDataFlowArgs): IFontOpts {
    return { color: opColor, mtx: args.mtx, size: 16 };
}

/** 這一格的梯度名稱，例如 dS、dWv */
function selfName(args: IDataFlowArgs) {
    return gradName(args.blk.name);
}

/**
 * 反向時，某個運算元在畫面上被掃過的是「橫的一列」還是「直的一行」。
 *
 * 不能寫死。前向與反向的收縮軸不同（例如 LN：前向沿 C 掃是直的，
 * 反向沿 T 掃就變成橫的），寫死一定會有一邊畫錯。
 */
function sliceShape(alongX: boolean | null): { cellX: number, cellY: number } {
    if (alongX === null) {
        return { cellX: 1, cellY: 1 };
    }
    return alongX ? { cellX: 4, cellY: 1 } : { cellX: 1, cellY: 4 };
}

/** 消費者那一側被掃過的軸：沒有被 blk 決定的那一維。 */
function consumerSweepIsX(c: IBlkConsumer): boolean | null {
    let d = freeDestDim(c.dep);
    return d === null ? null : d === Dim.X;
}

/** 點積另一邊被掃過的軸：它身上對應到「消費者自由軸」的那個分量。 */
function otherSweepIsX(c: IBlkConsumer, other: IBlkCellDep | null): boolean | null {
    let free = freeDestDim(c.dep);
    if (!other || free === null) {
        return null;
    }
    let d = free === Dim.X ? 0 : 1;
    let m = other.srcIdxMtx;
    if (m.g(0, d) === 1) return true;    // 掃過 other 的 x
    if (m.g(1, d) === 1) return false;   // 掃過 other 的 y
    return null;
}

/** 算式一行 + 補充說明一行。說明擠在同一行太容易被誤讀成算式的一部分。 */
function twoLines(args: IDataFlowArgs, formula: ITextBlockArgs[], hint: string) {
    let opts = fontOptsOf(args);
    return drawMaths(args, args.center, mkTextBlock({
        opts,
        type: TextBlockType.Stack,
        subs: [
            { type: TextBlockType.Line, subs: formula },
            { type: TextBlockType.Line, subs: [
                { text: hint, opts: { ...opts, size: opts.size * 0.8 }, color: new Vec4(0.62, 0.62, 0.62, 1) },
            ] },
        ],
    }));
}

function note(args: IDataFlowArgs, text: string) {
    return drawMaths(args, args.center, mkTextBlock({
        opts: { ...fontOptsOf(args), size: 14 },
        subs: [{ text, color: new Vec4(0.75, 0.75, 0.75, 1) }],
    }));
}

export function drawDataFlowBackward(args: IDataFlowArgs): BoundingBox3d {
    let { state, blk, destIdx } = args;

    // 被因果遮罩擋掉的格子，梯度恆為 0
    if (blk.deps?.lowerTri && destIdx.x > destIdx.y) {
        return drawZeroSymbol(args);
    }

    // 沒有梯度資料的區塊：明講，不要假裝有東西
    if (blk.gradMissing) {
        return note(args, 'no gradient data');
    }

    if (isAggStub(blk)) {
        return note(args, '(folded into the backward formula)');
    }

    let consumers = getRealConsumers(state, blk);

    if (consumers.length === 0) {
        // 反向圖的起點：梯度不是從下游算來的，是損失直接種下去的。
        if (blk === state.layout.logits) {
            return drawLossSeed(args);
        }
        // 其他沒有下游的區塊：說不出所以然就別亂講
        return note(args, getGraphConsumers(state, blk).length > 0
            ? 'gradient exists, but upstream activations were not recorded'
            : 'no downstream gradient path');
    }

    if (consumers.length > 1) {
        // 前向的一分為多，在反向就是多路相加 —— 這是反向最值得看的結構。
        return drawFanInSum(args, consumers);
    }

    return drawSingleConsumer(args, consumers[0]);
}

function drawSingleConsumer(args: IDataFlowArgs, c: IBlkConsumer): BoundingBox3d {
    let special = c.consumer.deps?.special ?? BlKDepSpecial.None;

    switch (special) {
    case BlKDepSpecial.Softmax:
        return drawSoftmaxBackward(args);
    case BlKDepSpecial.LayerNorm:
        return drawLayerNormBackward(args);
    case BlKDepSpecial.Gelu:
        return drawGeluBackward(args, c);
    case BlKDepSpecial.InputEmbed:
        return drawEmbedScatter(args, c);
    case BlKDepSpecial.Attention:
        return drawAttentionBackward(args, c);
    }

    if (c.kind === 'dot') {
        return drawMatmulBackward(args, c);
    }
    return drawAddBackward(args, c);
}

/** 損失端：dLogits = P — onehot(target) */
function drawLossSeed(args: IDataFlowArgs): BoundingBox3d {
    let opts = fontOptsOf(args);
    return drawMaths(args, args.center, mkTextBlock({
        opts,
        subs: [
            { text: selfName(args) + ' = ', color: gradColor },
            { text: 'P', color: workingSrcColor },
            { text: ' — onehot(target)' },
        ],
    }));
}

/**
 * 多個消費者 = 梯度要把每一條路徑的貢獻加起來。
 *
 * 例：LayerNorm 的輸出同時餵給 Q、K、V 三個矩陣乘法，
 * 所以 dLN = dQ路徑 + dK路徑 + dV路徑。前向的一分為三，反向的三合一。
 */
function drawFanInSum(args: IDataFlowArgs, consumers: IBlkConsumer[]): BoundingBox3d {
    let opts = fontOptsOf(args);

    let subs: ITextBlockArgs[] = [{ text: selfName(args) + ' = ', color: gradColor }];
    let seen = new Set<string>();
    let n = 0;
    for (let c of consumers) {
        let nm = gradName(c.consumer.name);
        if (seen.has(nm)) {
            continue; // 同一個消費者用了這一塊兩次（例如加法的兩個運算元）
        }
        seen.add(nm);
        if (n > 0) {
            subs.push({ text: ' + ' });
        }
        subs.push({ text: nm, color: gradColor });
        n++;
    }
    subs.push({ text: '   (' + n + ' paths)', color: new Vec4(0.7, 0.7, 0.7, 1) });

    return drawMaths(args, args.center, mkTextBlock({ opts, subs }));
}

/** dS = P ‧ ( dP — Σ P ‧ dP )，逐列。整列綁在一起，這是 attention 反向的關鍵。 */
function drawSoftmaxBackward(args: IDataFlowArgs): BoundingBox3d {
    let opts = fontOptsOf(args);

    // softmax 是逐列做的（沿 x），所以那個 Σ 掃的是橫的一整列
    let rowShape = sliceShape(true);

    return drawMaths(args, args.center, mkTextBlock({
        opts,
        subs: [
            { text: selfName(args) + ' = ', color: gradColor },
            { cellX: 1, cellY: 1, color: workingSrcColor },
            { text: ' ‧ ( ' },
            { cellX: 1, cellY: 1, color: gradColor },
            { text: ' — ' },
            {
                type: TextBlockType.Line,
                rectOpts: makeLineOpts({ color: Colors.Aggregates.mul(0.8), mtx: args.mtx, thick: 1.0, dash: 6 }),
                subs: [
                    { text: 'Σ', opts: { ...opts, size: opts.size * 1.5 } },
                    { ...rowShape, color: workingSrcColor },
                    { text: ' ‧ ' },
                    { ...rowShape, color: gradColor },
                ],
            },
            { text: ' )' },
        ],
    }));
}

/** dx = (γ / σ) ‧ ( d — E[d] — xn ‧ E[d ‧ xn] )，同樣整列綁在一起。 */
function drawLayerNormBackward(args: IDataFlowArgs): BoundingBox3d {
    let opts = fontOptsOf(args);

    return drawMaths(args, args.center, mkTextBlock({
        opts,
        subs: [
            { text: selfName(args) + ' = ', color: gradColor },
            {
                type: TextBlockType.Divide,
                subs: [
                    { subs: [{ text: 'γ', color: weightSrcColor }] },
                    { subs: [{ text: 'σ', color: Colors.Aggregates }] },
                ],
            },
            { text: ' ‧ ( ' },
            { cellX: 1, cellY: 1, color: gradColor },
            { text: ' — E[' },
            { cellX: 1, cellY: 3, color: gradColor },
            { text: '] — xn ‧ E[' },
            { cellX: 1, cellY: 3, color: gradColor },
            { text: ' ‧ xn] )' },
        ],
    }));
}

/** 一般矩陣乘法：這一格的梯度 = 消費者那一整條梯度 與 另一個運算元 的點積。 */
function drawMatmulBackward(args: IDataFlowArgs, c: IBlkConsumer): BoundingBox3d {
    let opts = fontOptsOf(args);

    let other = otherDotOperand(c);
    let otherColor = other && other.src.t === 'w' ? weightSrcColor : workingSrcColor;

    // 方塊的長寬要跟畫面上實際被掃過的方向一致（橫的一列 / 直的一行），
    // 這與式子裡的轉置是兩回事，不能互相套用。
    let gradShape = sliceShape(consumerSweepIsX(c));
    let otherShape = sliceShape(otherSweepIsX(c, other));
    let sum = sumAxisLabel(c);

    return twoLines(args, [
        { text: selfName(args) + ' = dot( ', color: gradColor },
        { ...gradShape, color: gradColor },
        { text: ', ' },
        { ...otherShape, color: otherColor },
        { text: ' )' },
    ], (sum ? `Σ ${sum}   ` : '') + matmulMatrixForm(c, args.blk, other));
}

/**
 * 矩陣乘法反向的矩陣形式，轉置放在推導出來的那一邊。
 *
 * 不能用寫死的樣板 —— 轉置該加在哪一邊，取決於「收縮軸落在各張量的第幾個軸」，
 * 而那會隨 blk 是被乘的哪一邊而變。原本寫成 dLogits ‧ LN^T 就是把 ^T 加錯邊，
 * 維度根本不合；正解是 dLogits^T ‧ LN（(n_vocab, t) @ (t, C)）。
 *
 * 規則：矩陣連乘要求收縮軸是左邊的第二軸、右邊的第一軸。
 * 不符合就補一個轉置。結果的軸序若與 blk 的擺放相反，結果本身也要標轉置。
 */
export function matmulMatrixForm(c: IBlkConsumer, blk: IBlkDef, other: IBlkCellDep | null): string {
    let self = gradName(blk.name);
    if (!other) {
        return '';
    }

    let free = freeDestDim(c.dep);
    if (free === null) {
        return '';
    }

    // 收縮軸在梯度張量（消費者）的第幾軸：x -> 需要轉置才能當左邊的第二軸
    let leftT = free === Dim.X;
    // 收縮軸在另一個運算元的第幾軸：y -> 需要轉置才能當右邊的第一軸
    let rightT = otherSweepIsX(c, other) === false;

    // blk 的哪個分量綁到消費者的非收縮軸；綁在 y 代表結果的軸序與 blk 相反
    let boundComp = -1;
    for (let k = 0; k < 3 && boundComp < 0; k++) {
        for (let d = 0; d < 3; d++) {
            if (c.dep.srcIdxMtx.g(k, d) === 1) { boundComp = k; break; }
        }
    }
    let resultT = boundComp === 1;

    let t = (name: string, need: boolean) => name + (need ? '^T' : '');
    return t(self, resultT) + ' = '
        + t(gradName(c.consumer.name), leftT) + ' ‧ ' + t(shortName(other.src.name), rightT);
}

/** 反向要沿著消費者的哪一軸加總（＝沒有被 blk 決定的那一維）。 */
export function sumAxisLabel(c: IBlkConsumer): string {
    let d = freeDestDim(c.dep);
    if (d === null) {
        return '';
    }
    let style = d === Dim.X ? c.consumer.dimX : c.consumer.dimY;
    return style === DimStyle.None ? '' : dimStyleTextShort(style);
}

/** S = Q K^T / √A 的反向：dQ = dS K / √A，dK = dS^T Q / √A。 */
function drawAttentionBackward(args: IDataFlowArgs, c: IBlkConsumer): BoundingBox3d {
    let opts = fontOptsOf(args);
    let other = otherDotOperand(c);

    let gradShape = sliceShape(consumerSweepIsX(c));
    let otherShape = sliceShape(otherSweepIsX(c, other));
    let sum = sumAxisLabel(c);

    return twoLines(args, [
        { text: selfName(args) + ' = ', color: gradColor },
        {
            type: TextBlockType.Divide,
            subs: [
                { subs: [
                    { text: 'dot( ' },
                    { ...gradShape, color: gradColor },
                    { text: ', ' },
                    { ...otherShape, color: workingSrcColor },
                    { text: ' )' },
                ] },
                { subs: [{ type: TextBlockType.Sqrt, subs: [{ text: 'A' }] }] },
            ],
        },
    ], (sum ? `Σ ${sum}   ` : '') + matmulMatrixForm(c, args.blk, other) + ' / √A');
}

/** 殘差／偏置這類純加法：梯度原封不動地傳過去，一個字都不改。 */
function drawAddBackward(args: IDataFlowArgs, c: IBlkConsumer): BoundingBox3d {
    let opts = fontOptsOf(args);

    let isBroadcast = depHasDotAxis(c.dep) || args.blk.cx === 1;

    return drawMaths(args, args.center, mkTextBlock({
        opts,
        subs: [
            { text: selfName(args) + ' = ', color: gradColor },
            isBroadcast ? { text: 'Σ', opts: { ...opts, size: opts.size * 1.5 } } : null,
            { cellX: 1, cellY: 1, color: gradColor },
            { text: '   ' + gradName(c.consumer.name) + (isBroadcast ? ' summed' : ' unchanged'),
              color: new Vec4(0.7, 0.7, 0.7, 1) },
        ].filter(isNotNil) as ITextBlockArgs[],
    }));
}

/** 嵌入表：梯度是 scatter-add，只有被查到的那一列會拿到東西。 */
function drawEmbedScatter(args: IDataFlowArgs, c: IBlkConsumer): BoundingBox3d {
    let opts = fontOptsOf(args);

    return drawMaths(args, args.center, mkTextBlock({
        opts,
        subs: [
            { text: selfName(args) + ' += ', color: gradColor },
            { cellX: 1, cellY: 1, color: gradColor },
            { text: '   scatter-add, only rows that were looked up',
              color: new Vec4(0.7, 0.7, 0.7, 1) },
        ],
    }));
}

function otherDotOperand(c: IBlkConsumer): IBlkCellDep | null {
    let dot = c.consumer.deps?.dot;
    if (!dot || c.kind !== 'dot') {
        return null;
    }
    return dot[c.operandIdx === 0 ? 1 : 0] ?? null;
}

/**
 * GELU 的反向：畫導函數曲線，並在 x 的位置標點。
 *
 * 前向浮層畫的是 gelu 本身；這裡畫 gelu'，因為梯度乘上去的正是這個值。
 * 兩張圖擺在一起，就能看出為什麼負值區的梯度會被壓扁。
 */
function drawGeluBackward(args: IDataFlowArgs, c: IBlkConsumer): BoundingBox3d {
    let { state, center, mtx, blk, destIdx } = args;

    let k = Math.sqrt(2.0 / Math.PI);
    let geluPrime = (x: number) => {
        let u = k * (x + 0.044715 * x * x * x);
        let t = Math.tanh(u);
        return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * 0.044715 * x * x);
    };

    let w = 70;
    let h = 50;

    let tl = center.sub(new Vec3(w / 2, h, 0));
    let br = center.add(new Vec3(w / 2, 0, 0));

    drawRoundedRect(state.render, tl, br, backWhiteColor, mtx, 4);

    let halfW = 3;
    let mappingX = createMapping(tl.x, br.x, -halfW, halfW);
    let mappingY = createMapping(br.y, tl.y, -0.3, 1.3);

    let nPts = 40;
    let pts = new Float32Array(nPts * 3);
    for (let i = 0; i < nPts; i++) {
        let x = -halfW + i * halfW * 2 / (nPts - 1);
        pts[i * 3 + 0] = mappingX(x);
        pts[i * 3 + 1] = mappingY(geluPrime(x));
    }

    let axisLineOpts = makeLineOpts({ color: new Vec4(0.5, 0.5, 0.5, 1), mtx, thick: 1.5 });
    addLine2(state.render.lineRender, new Vec3(tl.x, mappingY(0)), new Vec3(br.x, mappingY(0)), axisLineOpts);
    addLine2(state.render.lineRender, new Vec3(mappingX(0), tl.y), new Vec3(mappingX(0), br.y), axisLineOpts);

    drawLineSegs(state.render.lineRender, pts, makeLineOpts({ color: gradColor, mtx, thick: 3.5 }));

    // 標上目前這一格的 x，看它落在導函數的哪裡
    let srcBlk = c.consumer.deps?.add?.[0]?.src ?? blk.deps?.add?.[0]?.src;
    let srcVal = srcBlk ? getBlockValueAtIdx(srcBlk, destIdx) : null;
    if (isNotNil(srcVal)) {
        drawCircle(state.render, new Vec3(mappingX(srcVal), mappingY(geluPrime(srcVal))), 2, 1, gradColor, mtx);
    }

    return new BoundingBox3d(tl, br);
}

// ---------------------------------------------------------------------------
// 高亮與箭頭
// ---------------------------------------------------------------------------

/**
 * 反向版的 drawDependences。
 *
 * 前向高亮的是「算出我的那些來源」；反向要高亮的是「把梯度交給我的那些消費者」。
 * 這兩組在圖上是不同的格子，不是同一組箭頭換個讀法就好。
 */
export function drawBackwardDependences(state: IProgramState, blk: IBlkDef, idx: Vec3) {
    let layout = state.layout;

    // 與箭頭共用同一份清單，兩者不可能再各說各話。
    // sweepIsX 也直接沿用，所以「公式框裡畫橫的、畫面上就亮橫的一列」。
    for (let target of getBackwardArrowTargets(state, blk, idx)) {
        if (target.sweepIsX !== null) {
            // 沿著整條累加來的，把那一整條都點亮（橫掃就切 Y、直掃就切 X）
            let acrossDim = target.sweepIsX ? Dim.Y : Dim.X;
            let sub = splitGridForHighlight(layout, target.blk, acrossDim, target.idx.getAt(acrossDim));
            if (sub) {
                sub.highlight = 0.5;
            }
            continue;
        }

        let sub = splitGridForHighlight(layout, target.blk, Dim.X, target.idx.x);
        if (!sub) continue;
        sub = splitGridForHighlight(layout, sub, Dim.Y, target.idx.y);
        if (!sub) continue;
        sub = splitGridForHighlight(layout, sub, Dim.Z, target.idx.z);
        if (sub) sub.highlight = 0.5;
    }
}

/** dest 的哪一維沒有被這條邊決定（梯度要沿著它整條累加）。沒有就回 null。 */
function freeDestDim(dep: IBlkCellDep): Dim | null {
    let mtx = dep.srcIdxMtx;
    let determined = [false, false, false];
    for (let k = 0; k < 3; k++) {
        for (let d = 0; d < 3; d++) {
            if (mtx.g(k, d) === 1) {
                determined[d] = true;
            }
        }
    }
    if (!determined[0]) return Dim.X;
    if (!determined[1]) return Dim.Y;
    return null;
}

/**
 * 反向的箭頭該指向哪些格子。
 *
 * 重點：**公式裡出現的每一個東西都要有箭頭**，而消費者只是其中之一。
 * 例如 dWlm = dot( dLogits[...], LN[...] ) 有兩個運算元，但 LN 並不是
 * 權重的消費者 —— 它是消費者那個點積裡的另一邊。只畫消費者的話，
 * 畫面上就會只有一條橘色箭頭，公式卻有兩項，對不起來。
 *
 * 顏色與公式框裡的方塊一致：梯度橘、權重藍、中間值綠。
 */
export function getBackwardArrowTargets(state: IProgramState, blk: IBlkDef, destIdx: Vec3) {
    // sweepIsX：這個運算元在畫面上被掃過的是橫的一列(true)、直的一行(false)，還是單一格(null)。
    // 高亮與公式框裡方塊的長寬都靠它，兩者才不會一個畫橫的一個畫直的。
    let out: { blk: IBlkDef, idx: Vec3, color: Vec4, sweepIsX: boolean | null }[] = [];

    let push = (target: IBlkDef, idx: Vec3, color: Vec4, sweepIsX: boolean | null = null) => {
        if (target.opacity === 0) {
            return;
        }
        out.push({ blk: target, idx, color, sweepIsX });
    };

    let fwdColor = (b: IBlkDef) =>
        b.t === 'w' ? Colors.Weights : b.t === 'a' ? Colors.Aggregates : Colors.Intermediates;

    let consumers = getRealConsumers(state, blk);

    // 圖的起點（logits）：公式是 P — onehot(target)，那個 P 在 softmax 那一塊
    if (consumers.length === 0 && blk === state.layout.logits) {
        let sm = state.layout.logitsSoftmax;
        if (sm) {
            push(sm, destIdx, fwdColor(sm));
        }
        return out;
    }

    for (let c of consumers) {
        let consumerIdx = destIdxFromSrc(c.dep, destIdx, c.consumer);

        // 一定有的那一條：把梯度交給我的人
        push(c.consumer, consumerIdx, gradColor, consumerSweepIsX(c));

        // 點積的另一邊：它在公式裡，但不是消費者，所以要另外補
        if (c.kind === 'dot') {
            let other = otherDotOperand(c);
            if (other) {
                let contraction = blkContractionValue(blk, destIdx, c);
                push(other.src, otherOperandIdx(other, consumerIdx, contraction),
                    fwdColor(other.src), otherSweepIsX(c, other));
            }
        }

        // LayerNorm 的反向式裡有 γ，它也不是消費者
        if (c.consumer.deps?.special === BlKDepSpecial.LayerNorm) {
            let gamma = c.consumer.deps.add?.find(d => d.src.name === 'γ');
            if (gamma) {
                push(gamma.src, getDepSrcIdx(gamma, consumerIdx).srcIdx, fwdColor(gamma.src));
            }
        }
    }

    return out;
}

/** blk 的哪個索引是前向點積的收縮軸，取它的值。 */
function blkContractionValue(blk: IBlkDef, destIdx: Vec3, c: IBlkConsumer): number {
    let m = c.dep.srcIdxMtx;
    for (let k = 0; k < 3; k++) {
        let bound = false;
        for (let d = 0; d < 3; d++) {
            if (m.g(k, d) === 1) bound = true;
        }
        if (!bound && m.g(k, 3) === 1) {
            return k === 0 ? destIdx.x : k === 1 ? destIdx.y : destIdx.z;
        }
    }
    return 0;
}

/** 點積另一邊的代表格：沿消費者的索引取，收縮軸用 blk 那一側的值釘住。 */
function otherOperandIdx(other: IBlkCellDep, consumerIdx: Vec3, contraction: number): Vec3 {
    let m = other.srcIdxMtx;
    let v = m.mulVec4(Vec4.fromVec3(consumerIdx, 0));
    let idx = new Vec3(v.x, v.y, v.z);
    for (let k = 0; k < 3; k++) {
        if (m.g(k, 3) === 1) {
            idx.setAt(k, contraction);
        }
    }
    return idx;
}
