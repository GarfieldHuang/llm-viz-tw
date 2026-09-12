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

    // --- 只有第 0 個 transformer block 有逐層的中間量梯度（詳解章節都在這一層）---
    let b0 = model.blocks[0];
    if (b0) {
        add(b0.ln_1.output, 'd_ln1');
        add(b0.attn.qkvOutput, 'd_qkv');
        add(b0.attn.attnMatrix, 'd_att');
        add(b0.attn.attnMatrixSoftmax, 'd_attSm');
        add(b0.attn.scaledVectors, 'd_y');
        add(b0.attn.proj.output, 'd_yProj');
        add(b0.attn.add.output, 'd_attnResid');
        add(b0.ln_2.output, 'd_ln2');
        add(b0.mlp.fcLayer.output, 'd_fc');
        add(b0.mlp.mlpGelu, 'd_gelu');
        add(b0.mlp.projLayer.output, 'd_mlp');
        add(b0.mlp.addLayer.output, 'd_mlpResid');

        // 權重的梯度
        add(b0.attn.qkvWeight, 'd_transformer.h.0.attn.c_attn.weight');
        add(b0.attn.qkvBias, 'd_transformer.h.0.attn.c_attn.bias');
        add(b0.ln_1.normWeight, 'd_transformer.h.0.ln_1.weight');
        add(b0.ln_1.normBias, 'd_transformer.h.0.ln_1.bias');
        add(b0.ln_2.normWeight, 'd_transformer.h.0.ln_2.weight');
        add(b0.ln_2.normBias, 'd_transformer.h.0.ln_2.bias');
        add(b0.attn.proj.bias, 'd_transformer.h.0.attn.c_proj.bias');
        add(b0.mlp.fcLayer.bias, 'd_transformer.h.0.mlp.c_fc.bias');
        add(b0.mlp.projLayer.bias, 'd_transformer.h.0.mlp.c_proj.bias');
        add(b0.attn.proj.weight, 'd_transformer.h.0.attn.c_proj.weight');
        add(b0.mlp.fcLayer.weight, 'd_transformer.h.0.mlp.c_fc.weight');
        add(b0.mlp.projLayer.weight, 'd_transformer.h.0.mlp.c_proj.weight');
    }

    add(model.lm_head.output, 'd_lm_head');
    add(model.lm_head.weight, 'd_lm_head.weight');

    console.log(`[backprop] 已建立 ${texByForward.size} 張梯度貼圖，loss=${(tensors as any).loss}`);

    return {
        texByForward,
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
