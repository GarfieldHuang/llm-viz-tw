/**
 * 反向傳播的逐步推導：把某一格的梯度，從前向式一路推到數字。
 *
 * 側邊欄原本只給結果（dS = P ⊙ (dP − rowsum(P ⊙ dP))），讀者看不到它是怎麼來的。
 * 這裡對每一條「用到這一格的路徑」都走同樣五步：
 *
 *   ① 前向：這一格被誰用到（寫出消費者的前向式）
 *   ② 動一下：這一格變一點，消費者變多少（單一格的偏導數）
 *   ③ 連鎖律：乘上消費者交回來的梯度
 *   ④ 加總：所有用到這一格的地方都算進來
 *   ⑤ 代入數字
 *
 * 一律用**單一格的純量**推導，不寫矩陣轉置 —— 轉置的方向會隨運算元是左乘還是右乘而變，
 * 之前已經在這裡寫錯過好幾次。下標只用各區塊自己的軸名，順序與畫面一致（列, 行）。
 *
 * ⑤ 的數字是真的從前向與梯度貼圖讀出來、在這裡重算一次的，最後會跟資料裡的梯度比對，
 * 所以推導寫錯會直接在面板上顯示「不一致」，不會默默印出一個好看但錯的式子。
 */
import { BlKDepSpecial, IBlkCellDep, IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { getBlockValueAtIdx } from "./DataFlow";
import { getRealConsumers, IBlkConsumer } from "./DataFlowBackward";
import { shortName } from "./GradNames";
import { DimStyle } from "../walkthrough/WalkthroughTools";
import { Vec3 } from "@/src/utils/vector";

export interface IDerivStep {
    label: string;
    tex: string;
    note?: string;
}

export interface IDerivPath {
    /** 這條路徑的說明，例如「經由 O（第 0 層 · head 2）」 */
    title: string;
    steps: IDerivStep[];
    /** 這條路徑貢獻的梯度；讀不到資料時為 null */
    value: number | null;
}

export interface IDerivation {
    paths: IDerivPath[];
    /** 各路徑相加 */
    total: number | null;
    /** 資料裡存的梯度（PyTorch autograd 算的） */
    stored: number | null;
    /** 多條路徑時，把它們加起來的那一行 */
    sumTex?: string;
}

// ---------------------------------------------------------------------------
// 讀值
// ---------------------------------------------------------------------------

/** 這一格的梯度。沒有梯度資料的區塊回 null，絕不回前向值。 */
export function gradAt(blk: IBlkDef, idx: Vec3): number | null {
    if (blk.gradMissing) {
        return null;
    }
    return getBlockValueAtIdx(blk, idx);
}

/**
 * 這一格的前向值。
 *
 * 反向檢視下 access.src 已經換成梯度貼圖，要先反查回前向貼圖；
 * 沒有被換過的（沒有梯度資料的區塊）本來就指著前向貼圖。
 */
export function fwdAt(state: IProgramState, blk: IBlkDef, idx: Vec3): number | null {
    let acc = blk.access;
    if (!acc) {
        return null;
    }
    let src = state.gradData?.forwardByGrad.get(acc.src) ?? acc.src;
    if (src === acc.src) {
        return getBlockValueAtIdx(blk, idx);
    }
    return getBlockValueAtIdx({ ...blk, access: { ...acc, src } }, idx);
}

// ---------------------------------------------------------------------------
// 符號
// ---------------------------------------------------------------------------

const COLOR = {
    grad: '#c06010',
    weight: '#4646b4',
    value: '#2f7a2f',
    agg: '#8a6d1f',
};

const texNames: { [k: string]: string } = {
    'Q vectors': 'Q',
    'K vectors': 'K',
    'V vectors': 'V',
    'Q Weights': 'W^{Q}',
    'K Weights': 'W^{K}',
    'V Weights': 'W^{V}',
    'Q Bias': 'b^{Q}',
    'K Bias': 'b^{K}',
    'V Bias': 'b^{V}',
    'Attention Matrix': 'S',
    'Attn Matrix Softmax': 'P',
    'V Output': 'O',
    'V Output Combined': '\\text{concat}',
    'Projection Weights': 'W^{\\text{proj}}',
    'Projection Bias': 'b^{\\text{proj}}',
    'Attention Output': '\\text{AttnOut}',
    'Attention Residual': 'R^{\\text{attn}}',
    'Layer Norm': '\\text{LN}',
    'MLP Weights': 'W^{\\text{fc}}',
    'MLP Bias': 'b^{\\text{fc}}',
    'MLP Projection Weights': 'W^{\\text{mlp}}',
    'MLP Projection Bias': 'b^{\\text{mlp}}',
    'MLP': '\\text{Fc}',
    'MLP Activation': '\\text{Gelu}',
    'MLP Result': '\\text{MlpOut}',
    'MLP Residual': 'R^{\\text{mlp}}',
    'Token Embed': 'W^{\\text{te}}',
    'Position Embed': 'W^{\\text{pe}}',
    'Input Embed': 'X',
    'LM Head Weights': 'W^{\\text{lm}}',
    'Logits': 'z',
    'Logits Softmax': 'p',
    'γ': '\\gamma',
    'β': '\\beta',
};

function texName(blk: IBlkDef) {
    return texNames[blk.name] ?? `\\text{${shortName(blk.name)}}`;
}

function kindOf(blk: IBlkDef): keyof typeof COLOR {
    return blk.t === 'w' ? 'weight' : blk.t === 'a' ? 'agg' : 'value';
}

function colored(tex: string, kind: keyof typeof COLOR) {
    return `{\\color{${COLOR[kind]}}${tex}}`;
}

/** 前向量的符號，例如 V_{a,s}。 */
function fsym(blk: IBlkDef, subs: string[]) {
    return colored(texName(blk) + subStr(subs), kindOf(blk));
}

/** 梯度的符號，例如 dV_{7,2}。 */
function gsym(blk: IBlkDef, subs: string[]) {
    return colored('d' + texName(blk) + subStr(subs), 'grad');
}

/** ∂L/∂X 的完整寫法，只在「連鎖律」那一步用，之後一律簡寫成 dX。 */
function partialL(blk: IBlkDef, subs: string[], grad: boolean) {
    let inner = texName(blk) + subStr(subs);
    return `\\frac{\\partial L}{\\partial ${grad ? colored(inner, 'grad') : inner}}`;
}

function subStr(subs: string[]) {
    return subs.length ? `_{${subs.join(',')}}` : '';
}

function num(v: number | null): string {
    if (v === null || !isFinite(v)) {
        return '?';
    }
    if (v === 0) {
        return '0';
    }
    if (Math.abs(v) >= 1e-3) {
        return v.toFixed(4);
    }
    let [m, e] = v.toExponential(2).split('e');
    return `${m}\\times10^{${parseInt(e)}}`;
}

/** 負數在乘式裡要加括號，不然 0.3 · -0.2 讀起來像減法。 */
function paren(v: number | null) {
    let s = num(v);
    return v !== null && v < 0 ? `(${s})` : s;
}

export function formatNum(v: number | null) {
    if (v === null || !isFinite(v)) {
        return '?';
    }
    return Math.abs(v) >= 1e-3 || v === 0 ? v.toFixed(4) : v.toExponential(2);
}

/**
 * 把一串乘積寫成「代入數字」那一行。
 *
 * 項數太多時（例如沿 C = 48 加總）只列絕對值最大的幾項 ——
 * 讀者要看的是「哪幾項主導了結果」，不是 48 個幾乎為 0 的數字。
 */
function termsLine(terms: number[][], wrap?: { pre: string, post: string }) {
    let prods = terms.map(f => f.reduce((a, b) => a * b, 1));
    let total = prods.reduce((a, b) => a + b, 0);
    let n = terms.length;

    if (n === 0) {
        return { tex: '0', total: 0, note: undefined as string | undefined };
    }

    let show: number[];
    let note: string | undefined;
    if (n <= 6) {
        show = terms.map((_, i) => i);
    } else {
        show = prods.map((p, i) => [Math.abs(p), i])
            .sort((a, b) => b[0] - a[0])
            .slice(0, 3)
            .map(a => a[1])
            .sort((a, b) => a - b);
        note = `共 ${n} 項，這裡只列出絕對值最大的 3 項；總和是全部 ${n} 項加起來的。`;
    }

    let body = show.map(i => terms[i].map(paren).join('\\cdot ')).join(' + ');
    if (show.length < n) {
        body += ' + \\cdots';
    }
    let tex = wrap ? `${wrap.pre}\\big(${body}\\big)${wrap.post}` : body;
    return { tex, total, note };
}

// ---------------------------------------------------------------------------
// 軸與下標
// ---------------------------------------------------------------------------

function letterOf(style: DimStyle): string {
    switch (style) {
        case DimStyle.T: return 't';
        case DimStyle.A: return 'a';
        case DimStyle.C: return 'c';
        case DimStyle.C4: return 'k';
        case DimStyle.n_vocab: return 'v';
    }
    return '';
}

interface IAxes { x: string; y: string; }

/** 區塊兩軸的字母。注意力矩陣兩軸都是 T，列叫 t（查詢）、行叫 s（被看的位置）。 */
function blockAxes(blk: IBlkDef): IAxes {
    let x = letterOf(blk.dimX);
    let y = letterOf(blk.dimY);
    if (x && x === y) {
        y = 't';
        x = 's';
    }
    return { x, y };
}

type Role = 'x' | 'y' | 'i' | null;

/** dep 的第 k 個分量是由消費者的哪一維決定：橫軸 x、直軸 y、點積的加總軸 i，或常數。 */
function compRole(dep: IBlkCellDep, k: number): Role {
    let m = dep.srcIdxMtx;
    if (m.g(k, 0) === 1) return 'x';
    if (m.g(k, 1) === 1) return 'y';
    if (m.g(k, 3) === 1) return 'i';
    return null;
}

/** 點積加總軸的字母。與消費者的軸撞名時改用別的字（O = Σ V·P 的加總軸是位置，叫 s）。 */
function sumLetterOf(dot: IBlkCellDep[], ax: IAxes) {
    let d = dot[0];
    let style = DimStyle.None;
    for (let k = 0; k < 2; k++) {
        if (compRole(d, k) === 'i') {
            style = k === 0 ? d.src.dimX : d.src.dimY;
        }
    }
    let l = letterOf(style);
    if (!l || l === ax.x || l === ax.y) {
        l = style === DimStyle.T ? 's' : 'i';
    }
    return l;
}

/** 某個運算元在式子裡的下標字母，畫面順序（直軸在前）。常數分量與不存在的軸略過。 */
function depLetters(dep: IBlkCellDep, ax: IAxes, sumL: string): string[] {
    let out: string[] = [];
    for (let k of [1, 0]) {
        let style = k === 0 ? dep.src.dimX : dep.src.dimY;
        if (style === DimStyle.None) {
            continue;
        }
        let role = compRole(dep, k);
        let l = role === 'x' ? ax.x : role === 'y' ? ax.y : role === 'i' ? sumL : '';
        if (l) {
            out.push(l);
        }
    }
    return out;
}

function fill(letters: string[], vals: Map<string, number>) {
    return letters.map(l => vals.has(l) ? String(vals.get(l)) : l);
}

function srcIdxOf(dep: IBlkCellDep, dx: number, dy: number, i: number) {
    let m = dep.srcIdxMtx;
    let c = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
        c[k] = m.g(k, 0) * dx + m.g(k, 1) * dy + m.g(k, 3) * i;
    }
    return new Vec3(c[0], c[1], c[2]);
}

function isConcatSrc(blk: IBlkDef) {
    return blk.name === 'V Output Combined';
}

const LABEL = {
    fwd: '① 前向：這一格被誰用到',
    nudge: '② 動一下：我變一點，它變多少',
    chain: '③ 連鎖律：乘上它交回來的梯度',
    sum: '④ 加總：所有用到我的格子都算進來',
    plug: '⑤ 代入這一格的數字',
};

// ---------------------------------------------------------------------------
// 進入點
// ---------------------------------------------------------------------------

export function deriveBackward(state: IProgramState, blkIn: IBlkDef, idx: Vec3): IDerivation | null {
    // 懸停目標可能是上一幀的 layout；換成這一幀的同一塊，消費者表才查得到
    let cur = state.layout.cubes[blkIn.idx];
    let blk = cur && cur.name === blkIn.name ? cur : blkIn;

    // 被因果遮罩擋掉的格子，梯度恆為 0，沒有東西可推。
    // 聚合樁（softmax 的 max／exp 總和、Layer Norm 的 μ／σ）沒有獨立的梯度路徑，
    // 它們被併進 softmax 與 Layer Norm 的反向式裡。
    if ((blk.deps?.lowerTri && idx.x > idx.y) || blk.t === 'a') {
        return null;
    }

    let stored = gradAt(blk, idx);

    if (blk === state.layout.logits) {
        let p = lossPath(state, blk, idx);
        return { paths: [p], total: p.value, stored };
    }

    let consumers = getRealConsumers(state, blk);
    if (consumers.length === 0) {
        return null;
    }

    let paths: IDerivPath[] = [];
    for (let c of consumers) {
        let p = pathFor(state, blk, idx, c);
        if (p) {
            paths.push(p);
        }
    }
    if (paths.length === 0) {
        return null;
    }

    let total: number | null = 0;
    for (let p of paths) {
        total = (total === null || p.value === null) ? null : total + p.value;
    }

    let sumTex: string | undefined;
    if (paths.length > 1) {
        let self = gsym(blk, fill(ownLetters(blk), idxVals(blk, idx)));
        sumTex = `${self} = ` + paths.map(p => paren(p.value)).join(' + ') + ` = ${num(total)}`;
    }

    return { paths, total, stored, sumTex };
}

function ownLetters(blk: IBlkDef) {
    let ax = blockAxes(blk);
    return [ax.y, ax.x].filter(a => a);
}

function idxVals(blk: IBlkDef, idx: Vec3) {
    let ax = blockAxes(blk);
    let m = new Map<string, number>();
    if (ax.x) m.set(ax.x, idx.x);
    if (ax.y) m.set(ax.y, idx.y);
    return m;
}

function pathTitle(state: IProgramState, c: IBlkConsumer) {
    let layout = state.layout;
    let where = '';
    let li = layout.blocks.findIndex(b => b.cubes.includes(c.consumer));
    if (li >= 0) {
        let hi = layout.blocks[li].heads.findIndex(h => h.cubes.includes(c.consumer));
        where = `第 ${li} 層` + (hi >= 0 ? ` · head ${hi}` : '');
    } else if (c.consumer === layout.ln_f.lnResid) {
        where = '最後的 Layer Norm';
    }
    return `經由 ${shortName(c.consumer.name)}` + (where ? `（${where}）` : '');
}

function pathFor(state: IProgramState, blk: IBlkDef, idx: Vec3, c: IBlkConsumer): IDerivPath | null {
    let special = c.consumer.deps?.special ?? BlKDepSpecial.None;
    let title = pathTitle(state, c);

    let body: Omit<IDerivPath, 'title'> | null;
    switch (special) {
    case BlKDepSpecial.Softmax:
        body = softmaxPath(state, blk, idx, c);
        break;
    case BlKDepSpecial.LayerNorm:
        body = blk.name === 'γ' ? gammaBetaPath(state, blk, idx, c, true)
            : blk.name === 'β' ? gammaBetaPath(state, blk, idx, c, false)
            : lnInputPath(state, blk, idx, c);
        break;
    case BlKDepSpecial.Gelu:
        body = geluPath(state, blk, idx, c);
        break;
    case BlKDepSpecial.InputEmbed:
        body = embedPath(state, blk, idx, c);
        break;
    default:
        if (c.kind === 'add' && blk.name === 'V Output' && c.consumer.deps?.dot && isConcatSrc(c.consumer.deps.dot[1].src)) {
            body = concatPath(state, blk, idx, c);
        } else if (c.kind === 'dot') {
            body = dotPath(state, blk, idx, c);
        } else {
            body = addPath(state, blk, idx, c);
        }
    }
    return body ? { title, ...body } : null;
}

// ---------------------------------------------------------------------------
// 前向式
// ---------------------------------------------------------------------------

/** 消費者的前向式（字母版，沒有代入任何值）。 */
function forwardTexOf(state: IProgramState, Y: IBlkDef): string {
    let deps = Y.deps!;
    let ax = blockAxes(Y);
    let yl = [ax.y, ax.x].filter(a => a);

    if (deps.dot) {
        let sumL = sumLetterOf(deps.dot, ax);
        let prod = deps.dot.map(d => fsym(d.src, depLetters(d, ax, sumL))).join('\\,');
        let isAttn = deps.special === BlKDepSpecial.Attention;
        let bias = (deps.add ?? []).filter(d => d.src.t === 'w')
            .map(d => ' + ' + fsym(d.src, depLetters(d, ax, sumL))).join('');
        let tex = `${fsym(Y, yl)} = ${isAttn ? `\\frac{1}{\\sqrt{${deps.dotLen}}}` : ''}\\sum_{${sumL}} ${prod}${bias}`;
        if (isConcatSrc(deps.dot[1].src)) {
            let A = state.layout.shape.A;
            tex += `,\\quad \\text{concat}_{hA+a,\\,t} = O^{(h)}_{a,t}\\ (A = ${A})`;
        }
        return tex;
    }

    let add = (deps.add ?? []).filter(d => d.src.t !== 'a');
    return `${fsym(Y, yl)} = ` + add.map(d => fsym(d.src, depLetters(d, ax, ''))).join(' + ');
}

// ---------------------------------------------------------------------------
// 矩陣乘法（含注意力分數）
// ---------------------------------------------------------------------------

function dotPath(state: IProgramState, blk: IBlkDef, b: Vec3, c: IBlkConsumer): Omit<IDerivPath, 'title'> {
    let Y = c.consumer;
    let deps = Y.deps!;
    let dot = deps.dot!;
    let me = c.dep;
    let other = dot[c.operandIdx === 0 ? 1 : 0];
    let ax = blockAxes(Y);
    let sumL = sumLetterOf(dot, ax);
    let isAttn = deps.special === BlKDepSpecial.Attention;
    let dotLen = deps.dotLen ?? 1;
    let scale = isAttn ? 1 / Math.sqrt(dotLen) : 1;
    let scaleTex = isAttn ? `\\frac{1}{\\sqrt{${dotLen}}}` : '';

    // 由「我這一格」決定的量：消費者的哪幾維被釘住、點積加總軸停在哪
    let fixedDest: { x?: number, y?: number } = {};
    let iVal = 0;
    let vals = new Map<string, number>();
    for (let k = 0; k < 2; k++) {
        let role = compRole(me, k);
        let v = k === 0 ? b.x : b.y;
        if (role === 'x') { fixedDest.x = v; vals.set(ax.x, v); }
        if (role === 'y') { fixedDest.y = v; vals.set(ax.y, v); }
        if (role === 'i') { iVal = v; vals.set(sumL, v); }
    }
    let freeDims = (['x', 'y'] as const).filter(d => ax[d] && fixedDest[d] === undefined);
    let freeLetters = freeDims.map(d => ax[d]);

    let yl = [ax.y, ax.x].filter(a => a);
    let meL = depLetters(me, ax, sumL);
    let otherL = depLetters(other, ax, sumL);

    let meSym = fsym(blk, fill(meL, vals));
    let Ysub = fill(yl, vals);
    let otherSym = fsym(other.src, fill(otherL, vals));
    let sumPre = freeLetters.length ? `\\sum_{${freeLetters.join(',')}} ` : '';

    let fixedText = [...(fixedDest.y !== undefined ? [`${ax.y} = ${fixedDest.y}`] : []),
                     ...(fixedDest.x !== undefined ? [`${ax.x} = ${fixedDest.x}`] : [])].join('、');
    let meName = shortName(blk.name);
    let yName = shortName(Y.name);
    let nudgeNote = `${meName} 的這一格只出現在 ${yName} 裡 ${fixedText || '每一'} 的那些格子`
        + (freeLetters.length ? `（每個 ${freeLetters.join('、')} 各一格）` : '')
        + `，偏導就是跟它相乘的那個數；其他格子沒有用到它，偏導是 0。`;

    // 數字
    let readOther = otherReader(state, Y, other);
    let terms: number[][] = [];
    let missing = false;
    forEachDest(Y, freeDims, fixedDest, (dx, dy) => {
        if (deps.lowerTri && dx > dy) {
            return;
        }
        let g = gradAt(Y, new Vec3(dx, dy, 0));
        let o = readOther(srcIdxOf(other, dx, dy, iVal));
        if (g === null || o === null) {
            missing = true;
            return;
        }
        terms.push([g, o]);
    });
    let line = termsLine(terms, isAttn ? { pre: scaleTex, post: '' } : undefined);
    let value = missing ? null : line.total * scale;

    let Ygrad = gsym(Y, Ysub);
    let meGrad = gsym(blk, fill(meL, vals));

    let steps: IDerivStep[] = [
        { label: LABEL.fwd, tex: forwardTexOf(state, Y) },
        {
            label: LABEL.nudge,
            tex: `\\frac{\\partial ${fsym(Y, Ysub)}}{\\partial ${meSym}} = ${scaleTex}${otherSym}`,
            note: nudgeNote,
        },
        {
            label: LABEL.chain,
            tex: `${partialL(blk, fill(meL, vals), false)} = ${sumPre}${partialL(Y, Ysub, false)}\\cdot\\frac{\\partial ${texName(Y)}${subStr(Ysub)}}{\\partial ${texName(blk)}${subStr(fill(meL, vals))}}`,
            note: '以下把 ∂L/∂X 簡寫成 dX。',
        },
        {
            label: LABEL.sum,
            tex: `${meGrad} = ${scaleTex}${sumPre}${Ygrad}\\,${otherSym}`,
            note: freeLetters.length
                ? `沿 ${freeLetters.join('、')} 加總。` + (blk.t === 'w' ? '權重被每個位置共用，所以每個位置的責任都要算進來。' : '')
                : undefined,
        },
        {
            label: LABEL.plug,
            tex: `${meGrad} = ${line.tex} = ${num(value)}`,
            note: line.note,
        },
    ];
    return { steps, value };
}

function forEachDest(Y: IBlkDef, freeDims: ('x' | 'y')[], fixed: { x?: number, y?: number }, fn: (dx: number, dy: number) => void) {
    let xs = freeDims.includes('x') ? range(Y.cx) : [fixed.x ?? 0];
    let ys = freeDims.includes('y') ? range(Y.cy) : [fixed.y ?? 0];
    for (let dy of ys) {
        for (let dx of xs) {
            fn(dx, dy);
        }
    }
}

function range(n: number) {
    let a: number[] = [];
    for (let i = 0; i < n; i++) a.push(i);
    return a;
}

/** 點積另一邊的前向值。concat 沒有自己的貼圖，要從三個 head 的輸出拼回來。 */
function otherReader(state: IProgramState, Y: IBlkDef, other: IBlkCellDep): (idx: Vec3) => number | null {
    if (isConcatSrc(other.src)) {
        let bl = state.layout.blocks.find(b => b.attnOut === Y);
        let A = state.layout.shape.A;
        return (idx) => {
            let h = Math.floor(idx.y / A);
            let head = bl?.heads[h];
            return head ? fwdAt(state, head.vOutBlock, new Vec3(idx.x, idx.y - h * A, 0)) : null;
        };
    }
    return (idx) => fwdAt(state, other.src, idx);
}

// ---------------------------------------------------------------------------
// 加法：殘差（原封不動）與偏置（沿位置加總）
// ---------------------------------------------------------------------------

function addPath(state: IProgramState, blk: IBlkDef, b: Vec3, c: IBlkConsumer): Omit<IDerivPath, 'title'> {
    let Y = c.consumer;
    let ax = blockAxes(Y);
    let me = c.dep;

    let fixedDest: { x?: number, y?: number } = {};
    let vals = new Map<string, number>();
    for (let k = 0; k < 2; k++) {
        let role = compRole(me, k);
        let v = k === 0 ? b.x : b.y;
        if (role === 'x') { fixedDest.x = v; vals.set(ax.x, v); }
        if (role === 'y') { fixedDest.y = v; vals.set(ax.y, v); }
    }
    let freeDims = (['x', 'y'] as const).filter(d => ax[d] && fixedDest[d] === undefined);
    let freeLetters = freeDims.map(d => ax[d]);
    let yl = [ax.y, ax.x].filter(a => a);
    let meL = depLetters(me, ax, '');
    let Ysub = fill(yl, vals);
    let meSub = fill(meL, vals);
    let sumPre = freeLetters.length ? `\\sum_{${freeLetters.join(',')}} ` : '';

    let terms: number[][] = [];
    let missing = false;
    forEachDest(Y, freeDims, fixedDest, (dx, dy) => {
        let g = gradAt(Y, new Vec3(dx, dy, 0));
        if (g === null) {
            missing = true;
            return;
        }
        terms.push([g]);
    });
    let line = termsLine(terms);
    let value = missing ? null : line.total;

    let broadcast = freeLetters.length > 0;
    let meGrad = gsym(blk, meSub);

    return {
        value,
        steps: [
            { label: LABEL.fwd, tex: forwardTexOf(state, Y) },
            {
                label: LABEL.nudge,
                tex: `\\frac{\\partial ${fsym(Y, Ysub)}}{\\partial ${fsym(blk, meSub)}} = 1`,
                note: broadcast
                    ? `偏置對每個 ${freeLetters.join('、')} 都加上同一個數，所以每一格對它的偏導都是 1。`
                    : '加法：我變多少，它就變多少，偏導是 1。',
            },
            {
                label: LABEL.chain,
                tex: `${partialL(blk, meSub, false)} = ${sumPre}${partialL(Y, Ysub, false)}\\cdot 1`,
                note: '以下把 ∂L/∂X 簡寫成 dX。',
            },
            {
                label: LABEL.sum,
                tex: `${meGrad} = ${sumPre}${gsym(Y, Ysub)}`,
                note: broadcast
                    ? `沿 ${freeLetters.join('、')} 加總：每個位置的責任都算到同一個偏置上。`
                    : '只有一格用到我，梯度原封不動地傳過來。這就是殘差連接讓深層網路訓練得起來的原因。',
            },
            { label: LABEL.plug, tex: `${meGrad} = ${line.tex} = ${num(value)}`, note: line.note },
        ],
    };
}

// ---------------------------------------------------------------------------
// 多頭串接再投射：dO^(h) = Σ_c dAttnOut · Wproj
// ---------------------------------------------------------------------------

function concatPath(state: IProgramState, blk: IBlkDef, b: Vec3, c: IBlkConsumer): Omit<IDerivPath, 'title'> | null {
    let Y = c.consumer;
    let layout = state.layout;
    let bl = layout.blocks.find(x => x.attnOut === Y);
    if (!bl) {
        return null;
    }
    let h = bl.heads.findIndex(x => x.vOutBlock === blk);
    let A = layout.shape.A;
    let C = layout.shape.C;
    let a = b.y;
    let t = b.x;
    let iStar = h * A + a;

    let W = bl.projWeight;
    let terms: number[][] = [];
    let missing = false;
    for (let cc = 0; cc < C; cc++) {
        let g = gradAt(Y, new Vec3(t, cc, 0));
        let w = fwdAt(state, W, new Vec3(iStar, cc, 0));
        if (g === null || w === null) {
            missing = true;
            continue;
        }
        terms.push([g, w]);
    }
    let line = termsLine(terms);
    let value = missing ? null : line.total;

    let Oh = (subs: string[]) => colored(`O^{(${h})}${subStr(subs)}`, 'value');
    let dOh = colored(`dO^{(${h})}_{${a},${t}}`, 'grad');

    return {
        value,
        steps: [
            { label: LABEL.fwd, tex: forwardTexOf(state, Y) },
            {
                label: LABEL.nudge,
                tex: `\\frac{\\partial ${fsym(Y, ['c', String(t)])}}{\\partial ${Oh([String(a), String(t)])}} = ${fsym(W, ['c', String(iStar)])}`,
                note: `head ${h} 的第 ${a} 維，在串接後排在第 ${h}·${A} + ${a} = ${iStar} 維。`
                    + `它只影響位置 t = ${t}，而且對每個 c 都乘上投射權重的第 ${iStar} 行。`,
            },
            {
                label: LABEL.chain,
                tex: `\\frac{\\partial L}{\\partial O^{(${h})}_{${a},${t}}} = \\sum_{c} ${partialL(Y, ['c', String(t)], false)}\\cdot ${fsym(W, ['c', String(iStar)])}`,
                note: '以下把 ∂L/∂X 簡寫成 dX。',
            },
            {
                label: LABEL.sum,
                tex: `${dOh} = \\sum_{c} ${gsym(Y, ['c', String(t)])}\\,${fsym(W, ['c', String(iStar)])}`,
                note: '串接的反向就是切開：投射權重的第 i 行把梯度送回 concat 的第 i 維，再按位置分給各個 head。',
            },
            { label: LABEL.plug, tex: `${dOh} = ${line.tex} = ${num(value)}`, note: line.note },
        ],
    };
}

// ---------------------------------------------------------------------------
// softmax：dS = P (dP − Σ P dP)
// ---------------------------------------------------------------------------

function softmaxPath(state: IProgramState, blk: IBlkDef, b: Vec3, c: IBlkConsumer): Omit<IDerivPath, 'title'> {
    let P = c.consumer;
    let t = b.y;
    let sStar = b.x;

    let rowP: number[] = [];
    let rowDP: number[] = [];
    let missing = false;
    for (let s = 0; s <= t && s < P.cx; s++) {
        let p = fwdAt(state, P, new Vec3(s, t, 0));
        let dp = gradAt(P, new Vec3(s, t, 0));
        if (p === null || dp === null) {
            missing = true;
        }
        rowP.push(p ?? 0);
        rowDP.push(dp ?? 0);
    }
    let rho = termsLine(rowP.map((p, i) => [p, rowDP[i]]));
    let pS = rowP[sStar] ?? 0;
    let dpS = rowDP[sStar] ?? 0;
    let value = missing ? null : pS * (dpS - rho.total);

    let Pts = (s: string) => fsym(P, [String(t), s]);
    let Sts = (s: string) => fsym(blk, [String(t), s]);
    let dPts = (s: string) => gsym(P, [String(t), s]);
    let ss = String(sStar);

    return {
        value,
        steps: [
            {
                label: LABEL.fwd,
                tex: `${fsym(P, ['t', 's'])} = \\frac{e^{${fsym(blk, ['t', 's'])}}}{\\sum_{j \\le t} e^{${fsym(blk, ['t', 'j'])}}}`,
                note: '分母是整列的總和，所以 S 的任何一格變了，這一列的每一個 P 都會跟著變。',
            },
            {
                label: LABEL.nudge,
                tex: `\\frac{\\partial ${Pts('s')}}{\\partial ${Sts(ss)}} = ${Pts('s')}\\,\\big(\\delta_{s,${ss}} - ${Pts(ss)}\\big)`,
                note: `s = ${ss} 時是自己被推高（分子變大）；s ≠ ${ss} 時是被分母搶走一份，所以帶負號。δ 在兩個下標相同時是 1，否則是 0。`,
            },
            {
                label: LABEL.chain,
                tex: `${partialL(blk, [String(t), ss], false)} = \\sum_{s \\le ${t}} ${partialL(P, [String(t), 's'], false)}\\cdot ${Pts('s')}\\,\\big(\\delta_{s,${ss}} - ${Pts(ss)}\\big)`,
                note: '同一列裡每一個 P 都受影響，所以要沿整列加總。以下把 ∂L/∂X 簡寫成 dX。',
            },
            {
                label: LABEL.sum,
                tex: `${gsym(blk, [String(t), ss])} = ${Pts(ss)}\\Big(${dPts(ss)} - \\underbrace{\\textstyle\\sum_{s \\le ${t}} ${Pts('s')}\\,${dPts('s')}}_{\\rho_{${t}}}\\Big)`,
                note: `把 δ 那一項拆出來、其餘提出 ${'P'}[${t},${ss}]。ρ 是這一列 dP 的加權平均（權重就是 P）：比平均大的格子梯度為正，比平均小的為負。`,
            },
            {
                label: LABEL.plug,
                tex: `\\rho_{${t}} = ${rho.tex} = ${num(rho.total)}\\qquad `
                    + `${gsym(blk, [String(t), ss])} = ${paren(pS)}\\cdot\\big(${paren(dpS)} - ${paren(rho.total)}\\big) = ${num(value)}`,
                note: rho.note,
            },
        ],
    };
}

// ---------------------------------------------------------------------------
// Layer Norm
// ---------------------------------------------------------------------------

const LN_EPS = 1e-5;

interface ILnCol {
    xhat: number[];
    sigma: number;
    ok: boolean;
}

function lnColumn(state: IProgramState, X: IBlkDef, t: number, C: number): ILnCol {
    let xs: number[] = [];
    let ok = true;
    for (let c = 0; c < C; c++) {
        let v = fwdAt(state, X, new Vec3(t, c, 0));
        if (v === null) ok = false;
        xs.push(v ?? 0);
    }
    let mu = xs.reduce((a, b) => a + b, 0) / C;
    let v2 = xs.reduce((a, b) => a + (b - mu) * (b - mu), 0) / C;
    let sigma = Math.sqrt(v2 + LN_EPS);
    return { xhat: xs.map(x => (x - mu) / sigma), sigma, ok };
}

function lnForwardTex(LN: IBlkDef, X: IBlkDef, gamma: IBlkDef | undefined, beta: IBlkDef | undefined) {
    let g = gamma ? fsym(gamma, ['c']) : '\\gamma_c';
    let bb = beta ? fsym(beta, ['c']) : '\\beta_c';
    return `\\begin{aligned}`
        + `${fsym(LN, ['c', 't'])} &= ${g}\\,\\hat{x}_{c,t} + ${bb},\\qquad \\hat{x}_{c,t} = \\frac{${fsym(X, ['c', 't'])} - \\mu_t}{\\sigma_t} \\\\`
        + `\\mu_t &= \\frac{1}{C}\\sum_{c} ${fsym(X, ['c', 't'])},\\qquad \\sigma_t = \\sqrt{\\frac{1}{C}\\sum_{c}\\big(${fsym(X, ['c', 't'])} - \\mu_t\\big)^2 + \\varepsilon}`
        + `\\end{aligned}`;
}

function lnParts(c: IBlkConsumer) {
    let add = c.consumer.deps?.add ?? [];
    let X = add[0]?.src;
    let gamma = add.find(d => d.src.name === 'γ')?.src;
    let beta = add.find(d => d.src.name === 'β')?.src;
    return { X, gamma, beta };
}

function lnInputPath(state: IProgramState, X: IBlkDef, b: Vec3, c: IBlkConsumer): Omit<IDerivPath, 'title'> | null {
    let LN = c.consumer;
    let { gamma, beta } = lnParts(c);
    if (!gamma) {
        return null;
    }
    let C = LN.cy;
    let t = b.x;
    let cStar = b.y;

    let col = lnColumn(state, X, t, C);
    let g: number[] = [];
    let ok = col.ok;
    for (let cc = 0; cc < C; cc++) {
        let d = gradAt(LN, new Vec3(t, cc, 0));
        let gm = fwdAt(state, gamma, new Vec3(0, cc, 0));
        if (d === null || gm === null) ok = false;
        g.push((d ?? 0) * (gm ?? 0));
    }
    let gMean = g.reduce((a, bb) => a + bb, 0) / C;
    let gxMean = g.reduce((a, bb, i) => a + bb * col.xhat[i], 0) / C;
    let xs = col.xhat[cStar];
    let value = ok ? (g[cStar] - gMean - xs * gxMean) / col.sigma : null;

    let cs = String(cStar);
    let ts = String(t);
    let Xs = fsym(X, [cs, ts]);
    let xh = (cc: string) => `\\hat{x}_{${cc},${ts}}`;

    return {
        value,
        steps: [
            { label: LABEL.fwd, tex: lnForwardTex(LN, X, gamma, beta) },
            {
                label: LABEL.nudge,
                tex: `\\frac{\\partial ${xh('c')}}{\\partial ${Xs}} = \\frac{1}{\\sigma_{${ts}}}\\Big(\\delta_{c,${cs}} - \\frac{1}{C} - \\frac{${xh('c')}\\,${xh(cs)}}{C}\\Big)`,
                note: '三項各有來源：δ 是 X 自己出現在分子；−1/C 是它被算進平均 μ；最後一項是它被算進標準差 σ。'
                    + '所以動這一格，同一行（同一個 t）的每一個 x̂ 都會變。',
            },
            {
                label: LABEL.chain,
                tex: `${partialL(X, [cs, ts], false)} = \\sum_{c} ${partialL(LN, ['c', ts], false)}\\cdot ${fsym(gamma, ['c'])}\\cdot\\frac{\\partial ${xh('c')}}{\\partial ${Xs}}`,
                note: 'LN = γ·x̂ + β，所以 ∂LN/∂x̂ = γ。以下把 ∂L/∂X 簡寫成 dX，並令 g_c = γ_c · dLN_{c,t}。',
            },
            {
                label: LABEL.sum,
                tex: `${gsym(X, [cs, ts])} = \\frac{1}{\\sigma_{${ts}}}\\Big(g_{${cs}} - \\underbrace{\\tfrac{1}{C}\\textstyle\\sum_c g_c}_{\\bar g} - ${xh(cs)}\\cdot\\underbrace{\\tfrac{1}{C}\\textstyle\\sum_c g_c\\,${xh('c')}}_{\\overline{g\\hat x}}\\Big)`,
                note: '把三項分開加總。ḡ 是扣掉「整行一起變大」的份；最後一項是扣掉「整行一起放大」的份 —— 這兩件事前向已經被歸一化吃掉，調了也沒用。',
            },
            {
                label: LABEL.plug,
                tex: `\\sigma_{${ts}} = ${num(col.sigma)},\\ g_{${cs}} = ${num(g[cStar])},\\ \\bar g = ${num(gMean)},\\ ${xh(cs)} = ${num(xs)},\\ \\overline{g\\hat x} = ${num(gxMean)}`
                    + ` \\\\ ${gsym(X, [cs, ts])} = \\frac{${paren(g[cStar])} - ${paren(gMean)} - ${paren(xs)}\\cdot${paren(gxMean)}}{${num(col.sigma)}} = ${num(value)}`,
            },
        ],
    };
}

function gammaBetaPath(state: IProgramState, blk: IBlkDef, b: Vec3, c: IBlkConsumer, isGamma: boolean): Omit<IDerivPath, 'title'> | null {
    let LN = c.consumer;
    let { X, gamma, beta } = lnParts(c);
    if (!X) {
        return null;
    }
    let C = LN.cy;
    let T = LN.cx;
    let cStar = b.y;
    let cs = String(cStar);

    let terms: number[][] = [];
    let ok = true;
    for (let t = 0; t < T; t++) {
        let d = gradAt(LN, new Vec3(t, cStar, 0));
        if (d === null) ok = false;
        if (isGamma) {
            let col = lnColumn(state, X, t, C);
            ok = ok && col.ok;
            terms.push([d ?? 0, col.xhat[cStar]]);
        } else {
            terms.push([d ?? 0]);
        }
    }
    let line = termsLine(terms);
    let value = ok ? line.total : null;
    let me = gsym(blk, [cs]);

    return {
        value,
        steps: [
            { label: LABEL.fwd, tex: lnForwardTex(LN, X, gamma, beta) },
            {
                label: LABEL.nudge,
                tex: isGamma
                    ? `\\frac{\\partial ${fsym(LN, [cs, 't'])}}{\\partial ${fsym(blk, [cs])}} = \\hat{x}_{${cs},t}`
                    : `\\frac{\\partial ${fsym(LN, [cs, 't'])}}{\\partial ${fsym(blk, [cs])}} = 1`,
                note: `只有第 ${cs} 列用到 ${isGamma ? 'γ' : 'β'}[${cs}]，但每個位置 t 都用到。`,
            },
            {
                label: LABEL.chain,
                tex: `${partialL(blk, [cs], false)} = \\sum_{t} ${partialL(LN, [cs, 't'], false)}\\cdot ${isGamma ? `\\hat{x}_{${cs},t}` : '1'}`,
                note: '以下把 ∂L/∂X 簡寫成 dX。',
            },
            {
                label: LABEL.sum,
                tex: `${me} = \\sum_{t} ${gsym(LN, [cs, 't'])}${isGamma ? `\\,\\hat{x}_{${cs},t}` : ''}`,
                note: '同一組 γ、β 被序列裡每一個位置共用，所以沿位置加總。',
            },
            { label: LABEL.plug, tex: `${me} = ${line.tex} = ${num(value)}`, note: line.note },
        ],
    };
}

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------

const GELU_K = Math.sqrt(2 / Math.PI);

export function gelu(x: number) {
    return 0.5 * x * (1 + Math.tanh(GELU_K * (x + 0.044715 * x * x * x)));
}

export function geluPrime(x: number) {
    let u = GELU_K * (x + 0.044715 * x * x * x);
    let th = Math.tanh(u);
    return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * GELU_K * (1 + 3 * 0.044715 * x * x);
}

function geluPath(state: IProgramState, Fc: IBlkDef, b: Vec3, c: IBlkConsumer): Omit<IDerivPath, 'title'> {
    let G = c.consumer;
    let x = fwdAt(state, Fc, b);
    let dG = gradAt(G, b);
    let gp = x === null ? null : geluPrime(x);
    let value = x === null || dG === null || gp === null ? null : dG * gp;
    let sub = fill(ownLetters(Fc), idxVals(Fc, b));
    let x0 = fsym(Fc, sub);

    return {
        value,
        steps: [
            {
                label: LABEL.fwd,
                tex: `${fsym(G, ['t', 'k'])} = \\text{gelu}\\big(${fsym(Fc, ['t', 'k'])}\\big),\\qquad \\text{gelu}(x) = \\tfrac{1}{2}x\\Big(1 + \\tanh\\big(\\sqrt{2/\\pi}\\,(x + 0.044715\\,x^3)\\big)\\Big)`,
                note: '逐格計算，同一列的其他格不參與。',
            },
            {
                label: LABEL.nudge,
                tex: `\\frac{\\partial ${fsym(G, sub)}}{\\partial ${x0}} = \\text{gelu}'\\big(${x0}\\big)`,
                note: 'x 越負，gelu′ 越接近 0：被 GELU 壓掉的神經元，梯度也一起被壓掉。',
            },
            {
                label: LABEL.chain,
                tex: `${partialL(Fc, sub, false)} = ${partialL(G, sub, false)}\\cdot\\text{gelu}'\\big(${x0}\\big)`,
                note: '以下把 ∂L/∂X 簡寫成 dX。',
            },
            {
                label: LABEL.sum,
                tex: `${gsym(Fc, sub)} = ${gsym(G, sub)}\\cdot\\text{gelu}'\\big(${x0}\\big)`,
                note: '只有同一格用到我，不用加總。',
            },
            {
                label: LABEL.plug,
                tex: `${x0} = ${num(x)},\\quad \\text{gelu}'(${num(x)}) = ${num(gp)} \\\\ ${gsym(Fc, sub)} = ${paren(dG)}\\cdot ${paren(gp)} = ${num(value)}`,
            },
        ],
    };
}

// ---------------------------------------------------------------------------
// 嵌入：查表的反向是 scatter-add
// ---------------------------------------------------------------------------

const TOKEN_LABELS = ['A', 'B', 'C'];

function embedPath(state: IProgramState, blk: IBlkDef, b: Vec3, c: IBlkConsumer): Omit<IDerivPath, 'title'> {
    let layout = state.layout;
    let X = c.consumer;
    let cStar = b.y;
    let cs = String(cStar);
    let fwd = `${fsym(X, ['c', 't'])} = ${fsym(layout.tokEmbedObj, ['c', '\\text{tok}_t'])} + ${fsym(layout.posEmbedObj, ['c', 't'])}`;

    if (blk === layout.posEmbedObj) {
        let t = b.x;
        let g = gradAt(X, new Vec3(t, cStar, 0));
        let me = gsym(blk, [cs, String(t)]);
        return {
            value: g,
            steps: [
                { label: LABEL.fwd, tex: fwd, note: '位置 t 只會查到位置表的第 t 行。' },
                { label: LABEL.nudge, tex: `\\frac{\\partial ${fsym(X, [cs, String(t)])}}{\\partial ${fsym(blk, [cs, String(t)])}} = 1` },
                { label: LABEL.chain, tex: `${partialL(blk, [cs, String(t)], false)} = ${partialL(X, [cs, String(t)], false)}`, note: '以下把 ∂L/∂X 簡寫成 dX。' },
                { label: LABEL.sum, tex: `${me} = ${gsym(X, [cs, String(t)])}`, note: '一個位置對一行，不會重複，所以不用加總。' },
                { label: LABEL.plug, tex: `${me} = ${num(g)}` },
            ],
        };
    }

    let v = b.x;
    let hits: number[] = [];
    let terms: number[][] = [];
    let ok = true;
    for (let t = 0; t < layout.idxObj.cx; t++) {
        let tok = getBlockValueAtIdx(layout.idxObj, new Vec3(t, 0, 0));
        if (tok === null || Math.round(tok) !== v) {
            continue;
        }
        let g = gradAt(X, new Vec3(t, cStar, 0));
        if (g === null) ok = false;
        hits.push(t);
        terms.push([g ?? 0]);
    }
    let line = termsLine(terms);
    let value = ok ? line.total : null;
    let me = gsym(blk, [cs, String(v)]);
    let label = TOKEN_LABELS[v] ?? String(v);

    return {
        value,
        steps: [
            { label: LABEL.fwd, tex: fwd, note: `token "${label}" 出現在位置 ${hits.join('、') || '（沒有）'}，每次都查到詞嵌入表的第 ${v} 行。` },
            {
                label: LABEL.nudge,
                tex: `\\frac{\\partial ${fsym(X, [cs, 't'])}}{\\partial ${fsym(blk, [cs, String(v)])}} = \\mathbb{1}[\\text{tok}_t = ${v}]`,
                note: '查到我的位置偏導是 1，其他位置是 0。',
            },
            {
                label: LABEL.chain,
                tex: `${partialL(blk, [cs, String(v)], false)} = \\sum_t ${partialL(X, [cs, 't'], false)}\\cdot\\mathbb{1}[\\text{tok}_t = ${v}]`,
                note: '以下把 ∂L/∂X 簡寫成 dX。',
            },
            {
                label: LABEL.sum,
                tex: `${me} = \\sum_{t:\\ \\text{tok}_t = ${v}} ${gsym(X, [cs, 't'])}`,
                note: '這就是 scatter-add：同一行被查了幾次，就累加幾份。出現越多次的 token，拿到的梯度越多。',
            },
            { label: LABEL.plug, tex: `${me} = ${line.tex} = ${num(value)}`, note: `t = ${hits.join(', ')}` },
        ],
    };
}

// ---------------------------------------------------------------------------
// 損失端：dz = p − onehot
// ---------------------------------------------------------------------------

function lossPath(state: IProgramState, z: IBlkDef, b: Vec3): IDerivPath {
    let layout = state.layout;
    let pos = state.gradData?.lossPos ?? 5;
    let y = state.gradData?.lossTarget ?? 2;
    let v = b.y;
    let t = b.x;
    let pBlk = layout.logitsSoftmax;
    let ys = String(y);
    let ps = String(pos);
    let yLabel = TOKEN_LABELS[y] ?? ys;

    let fwd = `${fsym(pBlk, ['v', ps])} = \\frac{e^{${fsym(z, ['v', ps])}}}{\\sum_{j} e^{${fsym(z, ['j', ps])}}},\\qquad L = -\\log ${fsym(pBlk, [ys, ps])}`;

    if (t !== pos) {
        return {
            title: '損失只看位置 ' + pos,
            value: 0,
            steps: [
                { label: '① 前向：損失怎麼算出來', tex: fwd, note: `這個範例的損失只問位置 t = ${pos}，正解是 "${yLabel}"（第 ${y} 個 token）。` },
                { label: '② 這一格有沒有出現在式子裡', tex: `\\frac{\\partial L}{\\partial ${fsym(z, [String(v), String(t)])}} = 0`, note: `位置 ${t} 的 logits 完全沒有進到損失裡，所以沒有責任。` },
            ],
        };
    }

    let p = fwdAt(state, pBlk, new Vec3(t, v, 0));
    let onehot = v === y ? 1 : 0;
    let value = p === null ? null : p - onehot;
    let vs = String(v);
    let me = gsym(z, [vs, ps]);

    return {
        title: '損失端：梯度從這裡種下去',
        value,
        steps: [
            { label: '① 前向：損失怎麼算出來', tex: fwd, note: `正解是 "${yLabel}"（y = ${y}）。模型其實深信是 "A"，這是刻意設的反事實標的，梯度才看得見。` },
            {
                label: '② 把 log 拆開',
                tex: `L = -${fsym(z, [ys, ps])} + \\log\\sum_{j} e^{${fsym(z, ['j', ps])}}`,
                note: 'log(a/b) = log a − log b，而 log e^z = z。',
            },
            {
                label: '③ 對 z 微分',
                tex: `\\frac{\\partial L}{\\partial ${fsym(z, ['v', ps])}} = -\\mathbb{1}[v = ${ys}] + \\frac{e^{${fsym(z, ['v', ps])}}}{\\sum_{j} e^{${fsym(z, ['j', ps])}}}`,
                note: '第一項只有 v 是正解時才有；第二項是 log-sum-exp 的導數，剛好就是 softmax。',
            },
            {
                label: '④ 整理：預測減去答案',
                tex: `${gsym(z, ['v', ps])} = ${fsym(pBlk, ['v', ps])} - \\mathbb{1}[v = ${ys}]`,
                note: 'softmax 的 Jacobian 在這裡整個消掉了，所以畫面上那塊 softmax 沒有梯度 —— 損失是直接種在 logits 上的。',
            },
            {
                label: LABEL.plug,
                tex: `${me} = ${paren(p)} - ${onehot} = ${num(value)}`,
                note: v === y ? '正解那一格：機率不夠高，要往上推（梯度為負）。' : '不是正解：機率越高越要往下壓（梯度為正）。',
            },
        ],
    };
}
