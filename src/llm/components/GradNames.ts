import { Vec4 } from "@/src/utils/vector";

/**
 * 反向檢視共用的名稱與配色。
 *
 * 獨立成一個小模組，是為了讓 DataFlow 與 DataFlowBackward 都能取用，
 * 而不必互相 import（那會形成循環）。
 */

/** 梯度專用色：與前向的綠（中間值）／藍（權重）區隔，一眼看得出現在讀的是反向的量。 */
export const gradColor = new Vec4(0.95, 0.55, 0.25, 1);

/**
 * 把 layout 裡的區塊名縮成公式裡用得動的短名。
 *
 * 字型圖集只有 512x256，且字元集僅 ASCII 加上 `Σ γ β σ μ ε ‧ —`
 * （見 create-font-atlas.jsm）。所以這裡一律回傳 ASCII 短名，
 * 不能出現中文、∂、⊙、上標 T。
 */
const shortNames: { [k: string]: string } = {
    'Q vectors': 'Q',
    'K vectors': 'K',
    'V vectors': 'V',
    'QKV vectors': 'QKV',
    'Q Weights': 'Wq',
    'K Weights': 'Wk',
    'V Weights': 'Wv',
    'QKV Weights': 'Wqkv',
    'Q Bias': 'bq',
    'K Bias': 'bk',
    'V Bias': 'bv',
    'Attention Matrix': 'S',
    'Attn Matrix Softmax': 'P',
    'V Output': 'O',
    'V Output Combined': 'O',
    'Projection Weights': 'Wproj',
    'Projection Bias': 'bproj',
    'Attention Output': 'AttnOut',
    'Attention Residual': 'Resid',
    'Layer Norm': 'LN',
    'LN Agg: μ, σ': 'LNagg',
    'SM Agg': 'SMagg',
    'MLP Weights': 'Wfc',
    'MLP Bias': 'bfc',
    'MLP Projection Weights': 'Wmlp',
    'MLP Projection Bias': 'bmlp',
    'MLP Activation': 'Gelu',
    'MLP Result': 'MlpOut',
    'MLP Residual': 'Resid',
    'MLP': 'Fc',          // layout 把「升維後的 fc 輸出」命名為 MLP，但解說裡叫它 Fc
    'Token Embed': 'Wte',
    'Position Embed': 'Wpe',
    'Input Embed': 'Emb',
    'LM Head Weights': 'Wlm',
    'Logits': 'Logits',
    'Logits Softmax': 'Probs',
    'Tokens': 'Tok',
};

/** 取區塊的短名；沒登記的就把空白去掉，至少不會撐爆版面。 */
export function shortName(name: string): string {
    if (!name) {
        return '?';
    }
    if (shortNames[name]) {
        return shortNames[name];
    }
    // 未登記：取每個字的首字母，'Layer Norm Resid' -> 'LNR'
    let words = name.split(/\s+/).filter(a => a.length > 0);
    if (words.length > 1) {
        return words.map(w => w[0]).join('');
    }
    return name;
}

/** 梯度的顯示名：dQ、dWv、dS …… */
export function gradName(name: string): string {
    return 'd' + shortName(name);
}
