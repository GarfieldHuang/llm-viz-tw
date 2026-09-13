/**
 * 反向傳播視覺化的資料層。
 *
 * 設計上刻意做成「純加法」，不修改任何前向程式：
 *
 *  1. 梯度數值由 gen_grad_data.py 以 PyTorch 預先算好，存成 public/gpt-nano-sort-grads.json。
 *  2. 對於前向模型用到的「每一張 texture」，建立一張同尺寸的梯度 texture。
 *     GptModel.ts 的 validate() 已經證明：PyTorch 張量的 row-major flat 順序，
 *     與對應 texture 的 flat layout 完全一致，因此不需要任何 reshape。
 *  3. 要顯示梯度時，只把 block 的 access.src 換成梯度 texture。
 *     access.mat 內含的欄位偏移（Q/K/V 共用 qkvOutput、多頭共用 attnMatrix 等）
 *     會自動沿用，不必逐個 block 處理。
 *  4. Program.ts 每一幀都會重新產生 layout，所以這個換置是暫時的，
 *     下一幀自動復原 —— 前向章節不可能被污染。
 */
import { IGptModelLink } from "./GptModel";
import { IBlkDef, IGptModelLayout } from "./GptModelLayout";
import { IBufferTex, createBufferTex, writeToBufferTex } from "@/src/utils/renderPhases";
import { ITensorSet } from "@/src/utils/tensor";

export interface IGradData {
    /** 前向 texture -> 梯度 texture */
    texByForward: Map<IBufferTex, IBufferTex>;
    /**
     * 梯度 texture -> 前向 texture。
     * 反向檢視會把 access.src 換成梯度，但推導時同一格的前向值也要讀得到
     * （dV 要乘上 P、dS 要用到 P 本身），所以要能反查回去。
     */
    forwardByGrad: Map<IBufferTex, IBufferTex>;
    /** 前向 texture -> 顯示縮放（梯度量級與啟用值差很多，必須各自正規化） */
    scaleByForward: Map<IBufferTex, number>;
    /** 供解說文字取用的原始張量集 */
    tensors: ITensorSet;
    /** 產生梯度時所用的損失設定 */
    lossPos: number;
    lossTarget: number;
    loss: number;
}

/** 依 |g| 的分佈挑一個顯示縮放，讓梯度在色帶上落在看得清楚的範圍。 */
function pickScale(data: Float32Array): number {
    let maxAbs = 0;
    for (let i = 0; i < data.length; i++) {
        let a = Math.abs(data[i]);
        if (a > maxAbs) maxAbs = a;
    }
    return maxAbs > 0 ? 1.0 / maxAbs : 1.0;
}

function makeGradTex(
    gl: WebGL2RenderingContext,
    forward: IBufferTex | undefined | null,
    tensorName: string,
    tensors: ITensorSet,
    out: Map<IBufferTex, IBufferTex>,
    scales: Map<IBufferTex, number>,
) {
    if (!forward) {
        return;
    }
    let t = tensors[tensorName];
    if (!t) {
        return;
    }
    let src = t.toFloat32Array();
    let need = forward.width * forward.height * forward.channels;

    // 尺寸必須完全吻合才寫入；不吻合寧可不顯示，也不要畫出對不上的數字。
    let buf: Float32Array;
    if (src.length === need) {
        buf = src;
    } else if (src.length < need) {
        buf = new Float32Array(need);
        buf.set(src);
    } else {
        console.warn(`[backprop] ${tensorName} 尺寸過大 (${src.length} > ${need})，略過`);
        return;
    }

    let tex = createBufferTex(gl, forward.width, forward.height, forward.channels);
    writeToBufferTex(gl, tex, buf);
    // 浮層要讀得到數字：getBlockValueAtIdx 走的是 localBuffer，不是 GPU 貼圖。
    // 不設這個，反向的公式後面永遠只有一個空的等號。
    tex.localBuffer = buf;
    out.set(forward, tex);
    scales.set(forward, pickScale(src));
}

export function createGradData(gl: WebGL2RenderingContext, model: IGptModelLink, tensors: ITensorSet): IGradData {
    let texByForward = new Map<IBufferTex, IBufferTex>();
    let scaleByForward = new Map<IBufferTex, number>();

    let add = (fwd: IBufferTex | undefined | null, name: string) =>
        makeGradTex(gl, fwd, name, tensors, texByForward, scaleByForward);

    // --- embedding ---
    add(model.vocabEmbed.weight, 'd_transformer.wte.weight');
    add(model.posEmbed.weight, 'd_transformer.wpe.weight');
    add(model.add.output, 'd_x');

    // --- 每一個 transformer block：中間量與權重的梯度 ---
    // gen_grad_data.py 現在對所有層都 retain_grad。第 0 層沿用無前綴的張量名，
    // 其餘各層是 b1. / b2. 前綴（詳解章節都在第 0 層，所以它保留原名）。
    for (let i = 0; i < model.blocks.length; i++) {
        let b = model.blocks[i];
        if (!b) continue;
        let p = i === 0 ? 'd_' : `d_b${i}.`;

        // 中間量
        add(b.ln_1.output, `${p}ln1`);
        add(b.attn.qkvOutput, `${p}qkv`);
        add(b.attn.attnMatrix, `${p}att`);
        add(b.attn.attnMatrixSoftmax, `${p}attSm`);
        add(b.attn.scaledVectors, `${p}y`);
        add(b.attn.proj.output, `${p}yProj`);
        // 必須用 attn.output，不是 attn.add.output：GptModelWasm 的 createAttentionLayer
        // 建了兩個 add 層，output 指向第一個，add 欄位是另一個沒人用的。
        // 同步程式把注意力殘差寫進 attn.output，layout 讀的也是它。
        add(b.attn.output, `${p}attnResid`);
        add(b.ln_2.output, `${p}ln2`);
        add(b.mlp.fcLayer.output, `${p}fc`);
        add(b.mlp.mlpGelu, `${p}gelu`);
        add(b.mlp.projLayer.output, `${p}mlp`);
        add(b.mlp.addLayer.output, `${p}mlpResid`);

        // 權重
        add(b.attn.qkvWeight, `d_transformer.h.${i}.attn.c_attn.weight`);
        add(b.attn.qkvBias, `d_transformer.h.${i}.attn.c_attn.bias`);
        add(b.attn.proj.weight, `d_transformer.h.${i}.attn.c_proj.weight`);
        add(b.attn.proj.bias, `d_transformer.h.${i}.attn.c_proj.bias`);
        add(b.ln_1.normWeight, `d_transformer.h.${i}.ln_1.weight`);
        add(b.ln_1.normBias, `d_transformer.h.${i}.ln_1.bias`);
        add(b.ln_2.normWeight, `d_transformer.h.${i}.ln_2.weight`);
        add(b.ln_2.normBias, `d_transformer.h.${i}.ln_2.bias`);
        add(b.mlp.fcLayer.weight, `d_transformer.h.${i}.mlp.c_fc.weight`);
        add(b.mlp.fcLayer.bias, `d_transformer.h.${i}.mlp.c_fc.bias`);
        add(b.mlp.projLayer.weight, `d_transformer.h.${i}.mlp.c_proj.weight`);
        add(b.mlp.projLayer.bias, `d_transformer.h.${i}.mlp.c_proj.bias`);
    }

    // --- 最後的 layer norm 與輸出頭 ---
    add(model.ln_f.output, 'd_ln_f');
    add(model.ln_f.normWeight, 'd_transformer.ln_f.weight');
    add(model.ln_f.normBias, 'd_transformer.ln_f.bias');
    add(model.lm_head.output, 'd_lm_head');
    add(model.lm_head.weight, 'd_lm_head.weight');

    console.log(`[backprop] 已建立 ${texByForward.size} 張梯度貼圖，loss=${(tensors as any).loss}`);

    // 梯度必須是對「畫面上這條輸入」算的。之前資料用錯序列，前向值與梯度完全對不上，
    // 而畫面上看起來一切正常 —— 所以這裡明確比對，不一致就大聲講。
    let gradIdx: number[] | undefined = (tensors as any).inputIdx;
    let viewIdx = model.inputBuf;
    if (gradIdx && viewIdx && viewIdx.length >= gradIdx.length) {
        let same = gradIdx.every((v, i) => Math.round(viewIdx[i]) === v);
        if (!same) {
            console.error(`[backprop] 梯度資料的輸入 ${JSON.stringify(gradIdx)} 與畫面上的輸入 ${JSON.stringify(Array.from(viewIdx))} 不同，推導會對不上`);
        }
    }

    let forwardByGrad = new Map<IBufferTex, IBufferTex>();
    for (let [fwd, grad] of texByForward) {
        forwardByGrad.set(grad, fwd);
    }

    return {
        texByForward,
        forwardByGrad,
        scaleByForward,
        tensors,
        lossPos: (tensors as any).lossPos ?? 5,
        lossTarget: (tensors as any).lossTarget ?? 2,
        loss: (tensors as any).loss ?? 0,
    };
}

/**
 * 把 layout 切換成「顯示梯度」。
 * 只動 access（每幀重建，故為暫時性），不動幾何、不動 deps。
 * 沒有對應梯度的 block 會關閉取值，寧可顯示為空白，也不要畫出誤導的數字。
 */
export function applyGradView(layout: IGptModelLayout, grads: IGradData | null) {
    if (!grads) {
        return;
    }
    for (let cube of layout.cubes) {
        applyToBlk(cube, grads);
    }
}

function applyToBlk(blk: IBlkDef, grads: IGradData) {
    // 先記住前向的取值：反向動畫常常要讓一格顯示前向值（dV 要乘上 P、dQ 要乘上 K）
    if (blk.access && !blk.fwdAccess) {
        blk.fwdAccess = blk.access;
    }
    // 聚合樁（softmax 的 max／exp 總和）跟 P 共用同一張貼圖的 r/g 通道。
    // 若照樣換成梯度貼圖，它們會顯示 dP 的數字，看起來像是自己有梯度 —— 其實沒有。
    if (blk.t === 'a') {
        if (blk.access) {
            blk.access = { ...blk.access, disable: true };
        }
        blk.gradMissing = true;
        return;
    }
    if (blk.access) {
        let gradTex = grads.texByForward.get(blk.access.src);
        if (gradTex) {
            // 梯度的數值範圍與啟用值完全不同，沿用前向的 scale 會整片看起來是 0，
            // 因此改用該張量自己的 1/max|g| 正規化。
            let scale = grads.scaleByForward.get(blk.access.src) ?? 1.0;
            blk.access = { ...blk.access, src: gradTex, scale };
            blk.gradMissing = false;
        } else {
            // 注意：這裡只關掉取值，src 仍指向前向貼圖。
            // 任何人若把 disable 轉回 false，露出來的會是前向啟用值卻染成梯度的顏色，
            // 因此打上 gradMissing 讓動畫知道這一塊永遠不准打開。
            blk.access = { ...blk.access, disable: true };
            blk.gradMissing = true;
        }
    } else {
        blk.gradMissing = true;
    }
    if (blk.subs) {
        for (let sub of blk.subs) {
            applyToBlk(sub, grads);
        }
    }
}
