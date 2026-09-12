/**
 * 把滑鼠指到的那一格的算式，寫成側邊欄看得懂的完整文字。
 *
 * 3D 畫面上的浮層受限於字型圖集（只有 ASCII 加 Σ γ β σ μ ε ‧ —），而且用
 * 小方塊代表「某一列／某一行」，看得到形狀卻看不到是誰。例如權重梯度會畫成
 *
 *     dWlm = dot( ▭▭, ▭▭ )  dLogits ‧ LN^T = -1.11
 *
 * 那兩個方塊到底是哪一列、哪一行，畫面上講不出來。這個模組把同一件事展開成
 *
 *     dWlm[c=22, n_vocab=0] = dot( dLogits[n_vocab=0, 所有 t], LNf[c=22, 所有 t] )
 *
 * 分流規則與浮層完全一致：前向看 blk.deps，反向看「消費者」（見 DataFlowBackward）。
 * 兩邊各自產出各自的表現形式，不共用繪圖碼。
 */
import { BlKDepSpecial, IBlkCellDep, IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { getBlockValueAtIdx } from "./DataFlow";
import { getRealConsumers, IBlkConsumer } from "./DataFlowBackward";
import { gradName, shortName } from "./GradNames";
import { DimStyle, dimStyleTextShort } from "../walkthrough/WalkthroughTools";
import { Dim, Vec3 } from "@/src/utils/vector";

export type OperandKind = 'grad' | 'weight' | 'value' | 'agg' | 'const';

export interface IFormulaOperand {
    /** 顯示名稱，例如 dLogits、Q Weights */
    name: string;
    /** 這個運算元用到的是哪一段，例如「第 5 列」「整列（所有 t）」 */
    detail: string;
    kind: OperandKind;
}

export interface IFormulaDesc {
    dir: 'forward' | 'backward';
    /** 這一格的名字，例如 dWlm */
    target: string;
    /** 這一格的座標，例如 c = 22, n_vocab = 0 */
    index: string;
    /** 展開後的算式 */
    expr: string;
    /** 一般式（不帶座標），例如 dWlm = dLogits^T · LNf */
    rule?: string;
    /** 補充說明 */
    note?: string;
    /** expr 是一句說明而不是算式，渲染時不要加等號 */
    plain?: boolean;
    value: number | null;
    operands: IFormulaOperand[];
}

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------

function dimLabel(style: DimStyle) {
    return style === DimStyle.None ? '' : dimStyleTextShort(style);
}

function indexText(blk: IBlkDef, idx: Vec3) {
    let parts: string[] = [];
    let x = dimLabel(blk.dimX);
    let y = dimLabel(blk.dimY);
    if (x) parts.push(`${x} = ${idx.x}`);
    if (y) parts.push(`${y} = ${idx.y}`);
    return parts.join(', ');
}

function kindOf(blk: IBlkDef, grad: boolean): OperandKind {
    if (grad) return 'grad';
    if (blk.t === 'w') return 'weight';
    if (blk.t === 'a') return 'agg';
    return 'value';
}

function nameOf(blk: IBlkDef, grad: boolean) {
    return grad ? gradName(blk.name) : (blk.name || '(未命名)');
}

/** 這條 dep 在來源上取的是一整列、一整行，還是單一格。 */
function depSpan(dep: IBlkCellDep, destIdx: Vec3, blk: IBlkDef): string {
    let mtx = dep.srcIdxMtx;
    let hasDot = mtx.g(0, 3) === 1 || mtx.g(1, 3) === 1 || mtx.g(2, 3) === 1;

    // 找出每個 src 分量是由 dest 的哪一維決定的
    let fixed: string[] = [];
    for (let k = 0; k < 3; k++) {
        for (let d = 0; d < 3; d++) {
            if (mtx.g(k, d) === 1) {
                let style = d === 0 ? blk.dimX : d === 1 ? blk.dimY : DimStyle.None;
                let label = dimLabel(style);
                let v = idxAt(destIdx, d);
                fixed.push(label ? `${label} = ${v}` : `${v}`);
            }
        }
    }

    if (hasDot) {
        let along = dep.src.dimX !== DimStyle.None && mtx.g(0, 3) === 1 ? dimLabel(dep.src.dimX)
                  : dep.src.dimY !== DimStyle.None && mtx.g(1, 3) === 1 ? dimLabel(dep.src.dimY)
                  : '';
        let head = fixed.length ? fixed.join(', ') + ', ' : '';
        return `${head}整條${along ? ` (所有 ${along})` : ''}`;
    }
    return fixed.length ? fixed.join(', ') : '同一格';
}

function idxAt(v: Vec3, d: number) {
    return d === 0 ? v.x : d === 1 ? v.y : v.z;
}

function op(blk: IBlkDef, detail: string, grad: boolean): IFormulaOperand {
    return { name: nameOf(blk, grad), detail, kind: kindOf(blk, grad) };
}

// ---------------------------------------------------------------------------
// 進入點
// ---------------------------------------------------------------------------

export function describeFormula(state: IProgramState, blk: IBlkDef, idx: Vec3): IFormulaDesc | null {
    let grad = state.showGrads;
    let value = getBlockValueAtIdx(blk, idx);

    let base = {
        dir: (grad ? 'backward' : 'forward') as IFormulaDesc['dir'],
        target: nameOf(blk, grad),
        index: indexText(blk, idx),
        value,
    };

    if (grad) {
        return { ...base, ...describeBackward(state, blk, idx) };
    }
    return { ...base, ...describeForward(state, blk, idx) };
}

// ---------------------------------------------------------------------------
// 前向
// ---------------------------------------------------------------------------

type Body = Pick<IFormulaDesc, 'expr' | 'rule' | 'note' | 'operands' | 'plain'>;

function describeForward(state: IProgramState, blk: IBlkDef, idx: Vec3): Body {
    let deps = blk.deps;
    if (!deps) {
        return {
            expr: '這是模型參數，不由其他區塊算出來',
            plain: true,
            operands: [],
            note: blk.t === 'w' ? '訓練時由 optimizer 更新' : undefined,
        };
    }

    if (deps.lowerTri && idx.x > idx.y) {
        return {
            expr: '0（被因果遮罩擋掉）',
            operands: [],
            note: '位置 ' + idx.y + ' 不能看到未來的位置 ' + idx.x,
        };
    }

    let dot = deps.dot;
    let add = deps.add ?? [];

    switch (deps.special) {
    case BlKDepSpecial.InputEmbed: {
        let tok = state.layout.tokEmbedObj;
        let pos = state.layout.posEmbedObj;
        let tokenIdx = getBlockValueAtIdx(state.layout.idxObj, new Vec3(idx.x, 0, idx.z));
        return {
            expr: `${tok.name}[token = ${tokenIdx ?? '?'}, c = ${idx.y}] + ${pos.name}[t = ${idx.x}, c = ${idx.y}]`,
            rule: 'x = Wte[token] + Wpe[t]',
            note: '查表相加，不是矩陣乘法',
            operands: [
                op(tok, `token = ${tokenIdx ?? '?'} 那一列`, false),
                op(pos, `t = ${idx.x} 那一列`, false),
            ],
        };
    }
    case BlKDepSpecial.LayerNorm: {
        let src = add[0]?.src;
        let gamma = add.find(d => d.src.name === 'γ')?.src;
        let beta = add.find(d => d.src.name === 'β')?.src;
        return {
            expr: `( ${src?.name ?? '輸入'}[${indexText(blk, idx)}] − μ ) / σ · γ[c = ${idx.y}] + β[c = ${idx.y}]`,
            rule: 'LN = (x − E[x]) / √(Var[x] + ε) · γ + β',
            note: 'μ 與 σ 是整欄（所有 c）一起算的，所以同欄每一格都互相牽動',
            operands: [
                src ? op(src, `${indexText(blk, idx)}`, false) : null,
                { name: 'μ, σ', detail: `t = ${idx.x} 那一整欄的平均與標準差`, kind: 'agg' as OperandKind },
                gamma ? op(gamma, `c = ${idx.y}`, false) : null,
                beta ? op(beta, `c = ${idx.y}`, false) : null,
            ].filter(Boolean) as IFormulaOperand[],
        };
    }
    case BlKDepSpecial.LayerNormMu:
        return {
            expr: `E[ ${add[0]?.src.name}[t = ${idx.x}, 整欄] ]`,
            rule: 'μ = E[x]',
            operands: add[0] ? [op(add[0].src, `t = ${idx.x} 整欄`, false)] : [],
        };
    case BlKDepSpecial.LayerNormSigma:
        return {
            expr: `√( Var[ ${add[0]?.src.name}[t = ${idx.x}, 整欄] ] + ε )`,
            rule: 'σ = √(Var[x] + ε)',
            operands: add[0] ? [op(add[0].src, `t = ${idx.x} 整欄`, false)] : [],
        };
    case BlKDepSpecial.SoftmaxAggMax:
        return {
            expr: `max( ${add[0]?.src.name}[整列] )`,
            rule: 'm = max(x)',
            note: '先減去最大值，避免 exp 溢位',
            operands: add[0] ? [op(add[0].src, '整列', false)] : [],
        };
    case BlKDepSpecial.SoftmaxAggExp:
        return {
            expr: `Σ exp( ${add[0]?.src.name}[整列] − m )`,
            rule: 'Z = Σ exp(x − m)',
            operands: add[0] ? [op(add[0].src, '整列', false)] : [],
        };
    case BlKDepSpecial.Softmax: {
        let src = add[0]?.src;
        return {
            expr: `exp( ${src?.name}[${indexText(blk, idx)}] − m ) / Z`,
            rule: 'P = exp(x − max(x)) / Σ exp(x − max(x))',
            note: '分母 Z 是整列加總，所以同一列的每一格會互相牽動',
            operands: [
                src ? op(src, indexText(blk, idx), false) : null,
                { name: 'm, Z', detail: '這一列的最大值與 exp 總和', kind: 'agg' as OperandKind },
            ].filter(Boolean) as IFormulaOperand[],
        };
    }
    case BlKDepSpecial.Attention: {
        if (!dot) break;
        return {
            expr: `dot( ${dot[0].src.name}[${depSpan(dot[0], idx, blk)}], ${dot[1].src.name}[${depSpan(dot[1], idx, blk)}] ) / √A`,
            rule: 'S = Q Kᵀ / √A',
            note: '除以 √A 讓分數的變異數不隨頭的維度變大',
            operands: [
                op(dot[0].src, depSpan(dot[0], idx, blk), false),
                op(dot[1].src, depSpan(dot[1], idx, blk), false),
            ],
        };
    }
    case BlKDepSpecial.Gelu: {
        let src = add[0]?.src;
        return {
            expr: `gelu( ${src?.name}[${indexText(blk, idx)}] )`,
            rule: 'gelu(x) = 0.5x (1 + tanh( √(2/π) (x + 0.044715 x³) ))',
            note: '逐格計算，同列其他格不參與',
            operands: src ? [op(src, indexText(blk, idx), false)] : [],
        };
    }
    }

    if (dot) {
        let bias = add[0];
        let expr = `dot( ${dot[0].src.name}[${depSpan(dot[0], idx, blk)}], ${dot[1].src.name}[${depSpan(dot[1], idx, blk)}] )`;
        if (bias) {
            expr += ` + ${bias.src.name}[${depSpan(bias, idx, blk)}]`;
        }
        return {
            expr,
            rule: bias ? 'y = W · x + b' : 'y = W · x',
            operands: [
                op(dot[0].src, depSpan(dot[0], idx, blk), false),
                op(dot[1].src, depSpan(dot[1], idx, blk), false),
                ...(bias ? [op(bias.src, depSpan(bias, idx, blk), false)] : []),
            ],
        };
    }

    if (add.length >= 2) {
        return {
            expr: add.map(d => `${d.src.name}[${depSpan(d, idx, blk)}]`).join(' + '),
            rule: 'c = a + b',
            note: '殘差連接：兩條路徑相加',
            operands: add.map(d => op(d.src, depSpan(d, idx, blk), false)),
        };
    }

    return { expr: '這一塊沒有可展開的算式', plain: true, operands: [] };
}

// ---------------------------------------------------------------------------
// 反向
// ---------------------------------------------------------------------------

function describeBackward(state: IProgramState, blk: IBlkDef, idx: Vec3): Body {
    if (blk.deps?.lowerTri && idx.x > idx.y) {
        return {
            expr: '0（被因果遮罩擋掉的格子，梯度恆為 0）',
            operands: [],
        };
    }
    if (blk.gradMissing) {
        return {
            expr: '這一塊沒有梯度資料',
            plain: true,
            note: '梯度沒有流經這裡，或這個中間量沒有被記錄下來',
            operands: [],
        };
    }

    let consumers = getRealConsumers(state, blk);
    let self = gradName(blk.name);

    if (consumers.length === 0) {
        if (blk === state.layout.logits) {
            return {
                expr: `P[${indexText(blk, idx)}] − onehot(target)[${indexText(blk, idx)}]`,
                rule: 'dL/dlogits = p − y',
                note: 'softmax 與 cross-entropy 合併推導後，中間的 Jacobian 會整個消掉。'
                    + '所以畫面上那塊 softmax 不在梯度路徑上 —— 它沒有梯度可給，'
                    + '損失是直接種在 logits 上的。',
                operands: [
                    { name: 'P', detail: '模型吐出的機率', kind: 'value' },
                    { name: 'onehot(target)', detail: '正解的 one-hot', kind: 'const' },
                ],
            };
        }
        return { expr: '沒有下游把梯度交給這一塊', plain: true, operands: [] };
    }

    if (consumers.length > 1) {
        let names: string[] = [];
        let seen = new Set<string>();
        for (let c of consumers) {
            let n = gradName(c.consumer.name);
            if (!seen.has(n)) { seen.add(n); names.push(n); }
        }
        return {
            expr: names.join(' + '),
            rule: `${self} = ` + names.join(' + '),
            note: `前向時這一塊被 ${names.length} 個地方用到，反向就要把每一條路徑的貢獻加起來。`
                + '前向的一分為多，在反向是多合一。',
            operands: names.map(n => ({ name: n, detail: '其中一條路徑的貢獻', kind: 'grad' as OperandKind })),
        };
    }

    return describeBackwardOne(state, blk, idx, consumers[0], self);
}

function describeBackwardOne(
    state: IProgramState, blk: IBlkDef, idx: Vec3, c: IBlkConsumer, self: string,
): Body {
    let cn = gradName(c.consumer.name);
    let special = c.consumer.deps?.special ?? BlKDepSpecial.None;
    let other = otherDotOperand(c);
    let on = other ? shortName(other.src.name) : '?';

    switch (special) {
    case BlKDepSpecial.Softmax:
        return {
            expr: `P[${indexText(blk, idx)}] · ( ${cn}[${indexText(blk, idx)}] − Σ P·${cn}（整列） )`,
            rule: `${self} = P ⊙ ( ${cn} − rowsum(P ⊙ ${cn}) )`,
            note: 'softmax 是逐列做的，Jacobian 不是對角矩陣，所以整列綁在一起：'
                + '這一格的梯度取決於自己與整列加權平均的差。',
            operands: [
                { name: 'P', detail: `softmax 的輸出，${indexText(blk, idx)} 與整列`, kind: 'value' },
                { name: cn, detail: 'softmax 輸出的梯度，同一格與整列', kind: 'grad' },
            ],
        };
    case BlKDepSpecial.LayerNorm:
        return {
            expr: `γ / σ · ( ${cn}[${indexText(blk, idx)}] − E[${cn}] − xn · E[${cn}·xn] )`,
            rule: `${self} = (γ / σ) ⊙ ( d − E[d] − xn ⊙ E[d ⊙ xn] )`,
            note: '後兩項等於「扣掉整欄的平均責任」與「扣掉與自己方向相關的部分」——'
                + 'Layer Norm 不准反向去調整整欄的平均與尺度，因為前向已經把它們歸一化掉了。'
                + 'xn 是歸一化後、乘 γ 前的值。',
            operands: [
                { name: cn, detail: '這一格，以及整欄的兩個平均', kind: 'grad' },
                { name: 'γ, σ', detail: '縮放參數與該欄標準差', kind: 'weight' },
            ],
        };
    case BlKDepSpecial.Gelu:
        return {
            expr: `${cn}[${indexText(blk, idx)}] · gelu′( ${blk.name}[${indexText(blk, idx)}] )`,
            rule: `${self} = ${cn} ⊙ gelu′(x)`,
            note: 'x 越負，gelu′ 越接近 0 —— 被 GELU 壓掉的神經元，梯度也一起被壓掉。'
                + '逐格計算，同列其他格不參與。',
            operands: [
                { name: cn, detail: '同一格', kind: 'grad' },
                { name: "gelu′", detail: `在 x = 這一格的前向值 上取導數`, kind: 'const' },
            ],
        };
    case BlKDepSpecial.InputEmbed:
        return {
            expr: `${self}[被查到的那一列] += ${cn}[對應的位置]`,
            rule: `${self} = scatter-add(${cn})`,
            note: '查表的反向是 scatter-add：把梯度加回被查到的那一列，沒被查到的列不動。'
                + '同一列可能被好幾個位置加到，所以是累加而不是覆蓋。',
            operands: [{ name: cn, detail: '用到這一列的每一個位置', kind: 'grad' }],
        };
    case BlKDepSpecial.Attention:
        return {
            expr: `dot( ${cn}[${depSpanBack(c, blk, idx)}], ${on}[對應的一整列] ) / √A`,
            rule: `${self} = ${cn} · ${on} / √A`,
            operands: [
                { name: cn, detail: depSpanBack(c, blk, idx), kind: 'grad' },
                other ? op(other.src, '對應的一整列', false) : null,
            ].filter(Boolean) as IFormulaOperand[],
        };
    }

    if (c.kind === 'dot') {
        let sp = dotBackSpans(blk, idx, c, other);
        return {
            expr: sp.sumLabel
                ? `沿 ${sp.sumLabel} 加總：  ${cn}[${sp.gradSpan}] · ${on}[${sp.otherSpan}]`
                : `dot( ${cn}[${sp.gradSpan}], ${on}[${sp.otherSpan}] )`,
            rule: `${self} = ${cn} · ${on}ᵀ`,
            note: blk.t === 'w'
                ? `權重的梯度沿著 ${sp.sumLabel || '整條序列'} 加總 —— 同一個權重被每個位置共用，所以每個位置的責任都要算進來。`
                : undefined,
            operands: [
                { name: cn, detail: sp.gradSpan, kind: 'grad' },
                other ? op(other.src, sp.otherSpan, false) : null,
            ].filter(Boolean) as IFormulaOperand[],
        };
    }

    let broadcast = blk.cx === 1 || depHasDot(c);
    return {
        expr: broadcast
            ? `Σ ${cn}（所有位置）`
            : `${cn}[${indexText(blk, idx)}]`,
        rule: broadcast ? `${self} = Σ ${cn}` : `${self} = ${cn}`,
        note: broadcast
            ? '偏置對每個位置都加同一個數，所以反向要把所有位置的責任加起來。'
            : '加法的反向：梯度原封不動地傳過去，一個字都不改。這就是殘差連接讓深層網路訓練得起來的原因。',
        operands: [{ name: cn, detail: broadcast ? '所有位置' : '同一格', kind: 'grad' }],
    };
}

function depHasDot(c: IBlkConsumer) {
    let m = c.dep.srcIdxMtx;
    return m.g(0, 3) === 1 || m.g(1, 3) === 1 || m.g(2, 3) === 1;
}

function depSpanBack(c: IBlkConsumer, blk: IBlkDef, idx: Vec3): string {
    return dotBackSpans(blk, idx, c, otherDotOperand(c)).gradSpan;
}

/**
 * 把矩陣乘法的反向寫成具體的加總式。
 *
 * 前向 Q[t,a] = Σ_c Wq[c,a] · LN[t,c]，所以 dWq[c,a] = Σ_t dQ[t,a] · LN[t,c]。
 * 注意加總軸**翻面了**：前向沿 c 收縮，反向沿 t 收縮。這裡就是在算那個翻面。
 */
function dotBackSpans(blk: IBlkDef, idx: Vec3, c: IBlkConsumer, other: IBlkCellDep | null) {
    let m = c.dep.srcIdxMtx;

    // blk 的哪個分量是前向點積的自由軸（dep 字串裡的 'i'）
    let blkFreeComp = -1;
    for (let k = 0; k < 3; k++) {
        let bound = false;
        for (let d = 0; d < 3; d++) {
            if (m.g(k, d) === 1) bound = true;
        }
        if (!bound && m.g(k, 3) === 1) blkFreeComp = k;
    }

    // consumer 的哪一維沒被 blk 決定 —— 那就是反向要加總的軸
    let determined = [false, false, false];
    let fixed: string[] = [];
    for (let k = 0; k < 3; k++) {
        for (let d = 0; d < 3; d++) {
            if (m.g(k, d) === 1) {
                determined[d] = true;
                let label = dimLabel(d === 0 ? c.consumer.dimX : c.consumer.dimY);
                let v = idxAt(idx, k);
                fixed.push(label ? `${label} = ${v}` : `${v}`);
            }
        }
    }
    let sumDim = !determined[0] ? 0 : !determined[1] ? 1 : -1;
    let sumLabel = sumDim < 0 ? '' : dimLabel(sumDim === 0 ? c.consumer.dimX : c.consumer.dimY);

    let allSum = sumLabel ? `所有 ${sumLabel}` : '整條';
    let gradSpan = [...fixed, allSum].join(', ');

    // 另一個運算元：沿同一條加總軸，並由 blk 剩下那個索引定位
    let otherSpan = allSum;
    if (other && blkFreeComp >= 0) {
        let label = dimLabel(blkFreeComp === 0 ? blk.dimX : blk.dimY);
        let v = idxAt(idx, blkFreeComp);
        otherSpan = `${label ? `${label} = ${v}` : v}, ${allSum}`;
    }

    return { gradSpan, otherSpan, sumLabel };
}

function otherDotOperand(c: IBlkConsumer): IBlkCellDep | null {
    let dot = c.consumer.deps?.dot;
    if (!dot || c.kind !== 'dot') {
        return null;
    }
    return dot[c.operandIdx === 0 ? 1 : 0] ?? null;
}
