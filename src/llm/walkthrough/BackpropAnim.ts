/**
 * 反向章節動畫的最底層工具：定位一格、把一格拆出來、在模型空間裡寫符號。
 *
 * 真正的演法（加權分流、成對點積、softmax 整列反向……）在 BackpropScenes.ts。
 */
import { cellPosition, IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";

/** 這一格在模型空間裡的左上角座標。 */
export function cellPos(state: IProgramState, blk: IBlkDef, idx: Vec3): Vec3 {
    let layout = state.layout;
    return new Vec3(
        cellPosition(layout, blk, Dim.X, idx.x),
        cellPosition(layout, blk, Dim.Y, idx.y),
        cellPosition(layout, blk, Dim.Z, idx.z),
    );
}

/** 在模型空間的某個點畫一個符號（＋、×、＝ 之類）。 */
export function drawSymbolAt(state: IProgramState, pos: Vec3, symbol: string, size = 1.5, color = new Vec4(0, 0, 0, 1)) {
    let mtx = Mat4f.fromTranslation(pos);
    let fontOpts: IFontOpts = { color, size, mtx };
    let w = measureText(state.render.modelFontBuf, symbol, fontOpts);
    drawText(state.render.modelFontBuf, symbol, -w / 2, -fontOpts.size / 2, fontOpts);
}
