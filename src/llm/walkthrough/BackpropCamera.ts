/**
 * 反向章節的相機：依場景實際會畫到的範圍取景，並讓使用者在播放中也能自己移動相機。
 *
 * 前向章節的相機值，是作者播一次動畫、把相機挪到剛好框住，再把數字貼進程式的。
 * 反向的特寫原本是對準目的格、手填偏移和 zoom：來源區塊與起飛那一段常常在畫面外，
 * 也不管視窗的寬高比。這裡改成算出來：
 *
 *  1. 量測：場景函式在 0..1 之間取樣執行（BackpropScenes 的 measureScene，不畫、不加進 layout），
 *     得到每個時間點所有複本與文字占的範圍。
 *  2. 取景：用跟渲染同一套投影把範圍投到螢幕上，二分搜尋 zoom，讓範圍剛好落進安全框。
 *  3. 跟拍：整段框進來之後一格小於 minCellPx，就只框「這個時間點附近」出現的東西，
 *     中心與 zoom 沿時間平滑，相機跟著格子從來源一路移到目的地；格子散得很開時暫時拉遠。
 *
 * 寫入相機的時機跟前向的 moveCameraTo 一樣（播放中、或拖時間軸時才寫），另外：
 *  - 使用者在某個鏡頭裡動過相機，這個鏡頭剩下的時間都不再覆寫；
 *  - 下一個鏡頭從相機實際所在的位置接手，不會跳回去；
 *  - 時間往回拖，相機交還給腳本。
 *
 * 前向章節仍然用 WalkthroughTools 的 moveCameraTo；這個檔案只給反向章節用。
 */
import { cameraToMatrixView, ICamera, ICameraPos } from "../Camera";
import { IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { clamp } from "@/src/utils/data";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { BoundingBox3d, Vec3 } from "@/src/utils/vector";
import { ITimeInfo } from "./WalkthroughTools";
import { measureScene } from "./BackpropScenes";

// 以下要跟 Camera.ts 一致：垂直視角、每單位 zoom 的相機距離、近遠平面
const FOV_DEG = 40;
const DIST_PER_ZOOM = 200;
const NEAR = 100;
const FAR = 10000000;
const TAN_HALF_FOV = Math.tan(FOV_DEG / 2 * Math.PI / 180);

/** 這個投影把深度映到 [0, 1]，實際裁掉的是離相機約 NEAR / 2 以內的東西；每一角至少要離這麼遠 */
const MIN_DEPTH = 60;
const MIN_ZOOM = 0.3;
const MAX_ZOOM = 50;

/** 左右與下方保留的比例（占畫布的寬或高）；上方由各鏡頭決定 */
const MARGIN_SIDE = 0.05;
const MARGIN_BOTTOM = 0.06;

/** 每段場景取樣幾次；跟拍時一個取樣點框住前後多少時間（占整段的比例）；平滑的寬度（取樣點數） */
const SAMPLES = 48;
const FOLLOW_WINDOW = 0.06;
const SMOOTH_SIGMA = 2.5;
/** 跟拍時最遠拉到一格幾個像素。飛行中的格子旁邊沒有字，小一點沒關係，但整批都要看得到 */
const FOLLOW_MIN_CELL_PX = 3;

const EPS_EXACT = { pos: 1e-3, deg: 1e-3, zoom: 1e-5 };
const EPS_LOOSE = { pos: 0.5, deg: 0.5, zoom: 0.02 };

/** Camera.ts 的 modelMtx：模型的 (x, y, z) 對到世界的 (x, -z, -y) */
const MODEL_MTX = (() => {
    let m = new Mat4f();
    m[5] = 0;
    m[6] = -1;
    m[9] = -1;
    m[10] = 0;
    return m;
})();

export interface IShotOpts {
    /** 方位角、仰角（度）。預設正面、略為仰視：複本浮在區塊前方，斜著看會偏到別的區塊上 */
    azimuth?: number;
    elevation?: number;
    /** 整段框進來之後一格至少要有幾個像素，做不到就改成跟拍 */
    minCellPx?: number;
    /** 一格最多放大到幾個像素：範圍很小的時候，不要把鏡頭貼上去 */
    maxCellPx?: number;
    /** 畫布上方保留的比例：浮層公式會畫在格子上方 */
    top?: number;
}

/** 逐格填色的鏡頭：填色時格子上方會跳出公式浮層，上面多留一些 */
export const FILL_SHOT: IShotOpts = { top: 0.25, maxCellPx: 14 };
/** 填色鏡頭用填色計時器的前 40% 移過去 */
export const FILL_MOVE = 0.4;

interface IFitOpts {
    azimuth: number;
    elevation: number;
    minCellPx: number;
    maxCellPx: number;
    top: number;
}

function fitOpts(o?: IShotOpts): IFitOpts {
    return {
        azimuth: o?.azimuth ?? 270,
        elevation: o?.elevation ?? -6,
        minCellPx: o?.minCellPx ?? 7,
        maxCellPx: o?.maxCellPx ?? 24,
        top: o?.top ?? 0.08,
    };
}

export interface ICameraTrack {
    /** 此刻這個鏡頭要的相機位置 */
    at(): ICameraPos;
}

interface IShot {
    timer: ITimeInfo;
    track: ICameraTrack;
    move: number;
}

interface ICameraState {
    /** 上一次寫進相機的值；相機跟它不一樣，代表使用者動過 */
    lastWritten: ICameraPos | null;
    /** 使用者在第幾個鏡頭裡動過相機（-1 表示還沒進任何鏡頭） */
    override: { shot: number } | null;
    /** 起點改用「相機當時實際位置」的鏡頭 */
    captured: Map<number, ICameraPos>;
    /** 算好的取景；版面或畫布大小變了就重算 */
    tracks: Map<string, { sig: string, keys: ICameraPos[] }>;
}

function cameraState(state: IProgramState): ICameraState {
    let wt = state.walkthrough;
    wt.phaseTransitiveData ??= {};
    wt.phaseTransitiveData.backpropCamera ??= { lastWritten: null, override: null, captured: new Map(), tracks: new Map() };
    return wt.phaseTransitiveData.backpropCamera;
}

/**
 * 一個章節的相機腳本。每一幀：依時間先後登記所有鏡頭，再呼叫一次 apply()。
 * apply() 要在場景把複本加進 layout 之前呼叫，取景量到的才是乾淨的版面。
 */
export class BackpropCamera {
    private shots: IShot[] = [];
    private sig: string | null = null;

    constructor(private state: IProgramState) { }

    /** 固定位置（總覽鏡頭沿用手調的值） */
    fixed(center: Vec3, angle: Vec3): ICameraTrack {
        let pos: ICameraPos = { center, angle };
        return { at: () => pos };
    }

    /** 把幾個區塊整塊框進來（逐格填色時用） */
    blocks(key: string, blocks: IBlkDef[], opts?: IShotOpts): ICameraTrack {
        return {
            at: () => this.evaluate(`blocks:${key}`, 0, () => {
                let box = new BoundingBox3d();
                for (let b of blocks) {
                    box.addInPlace(new Vec3(b.x, b.y, b.z));
                    box.addInPlace(new Vec3(b.x + b.dx, b.y + b.dy, b.z + b.dz));
                }
                return [fitBox(this.state, box, fitOpts(opts), null).pos];
            }),
        };
    }

    /** 跟著一段場景取景：整段框得進來就固定不動，框不進來就跟拍。timer 是那段場景自己的計時器 */
    scene(key: string, run: (timer: ITimeInfo) => void, timer: ITimeInfo, opts?: IShotOpts): ICameraTrack {
        return {
            at: () => this.evaluate(`scene:${key}`, timer.t, () =>
                sceneKeys(this.state, measureScene(this.state, run, SAMPLES), fitOpts(opts))),
        };
    }

    /** 登記一個鏡頭：timer 開始後，用它的前 move（比例）移過去，之後一直跟著 track */
    shot(timer: ITimeInfo, track: ICameraTrack, move = 1) {
        this.shots.push({ timer, track, move });
    }

    apply() {
        let state = this.state;
        let wt = state.walkthrough;
        let cam = state.camera;
        let cs = cameraState(state);
        let shots = [...this.shots].sort((a, b) => a.timer.start - b.timer.start);

        if (wt.time < wt.prevTime) {
            // 時間往回拖：相機交還給腳本
            cs.override = null;
            cs.captured.clear();
            cs.lastWritten = null;
        }

        let cur = -1;
        for (let i = 0; i < shots.length; i++) {
            if (shots[i].timer.active) {
                cur = i;
            }
        }

        if (cs.lastWritten && !samePos(cam, cs.lastWritten, EPS_EXACT)) {
            // 相機跟上一次寫進去的不一樣：使用者拖曳或滾輪動過了
            cs.override = { shot: cur };
            cs.lastWritten = null;
        }

        if (cur < 0) {
            return;
        }
        let shot = shots[cur];
        let scripted = cur > 0 ? shots[cur - 1].track.at() : (wt.cameraInitial ?? posOf(cam));

        if (cs.override) {
            if (cs.override.shot >= cur) {
                // 使用者在這個鏡頭裡動過相機：剩下的時間都不覆寫
                return;
            }
            // 使用者在前面的鏡頭動過相機：這個鏡頭從相機現在的位置接手
            cs.override = null;
            cs.captured.set(cur, posOf(cam));
        } else if (wt.running && wt.prevTime <= shot.timer.start && !cs.captured.has(cur) && !samePos(cam, scripted, EPS_LOOSE)) {
            // 播放中剛跨進這個鏡頭，相機卻不在腳本以為的位置（暫停時被移動過，或視窗大小變了）
            cs.captured.set(cur, posOf(cam));
        }

        if (!wt.running && wt.time === wt.prevTime) {
            return;
        }

        let src = cs.captured.get(cur) ?? scripted;
        let p = shot.move > 0 ? lerpSmoothstep(0, 1, shot.timer.t / shot.move) : 1;
        let pos = lerpPos(src, shot.track.at(), p);
        cam.center = pos.center;
        cam.angle = pos.angle;
        cs.lastWritten = posOf(pos);
    }

    private evaluate(key: string, s: number, compute: () => ICameraPos[]): ICameraPos {
        let cs = cameraState(this.state);
        let sig = this.signature();
        let entry = cs.tracks.get(key);
        if (!entry || entry.sig !== sig) {
            entry = { sig, keys: compute() };
            cs.tracks.set(key, entry);
        }
        return evalKeys(entry.keys, s) ?? this.state.walkthrough.cameraInitial ?? posOf(this.state.camera);
    }

    /** 版面與畫布大小的指紋。區塊移動（例如注意力章節把 head 攤平）或視窗縮放，取景都要重算 */
    private signature() {
        if (this.sig === null) {
            let h = 0;
            let cubes = this.state.layout.cubes;
            for (let c of cubes) {
                h = (h * 31 + c.x * 1.3 + c.y * 1.7 + c.z * 2.3 + c.dx + c.dy * 0.7) % 1e9;
            }
            let size = this.state.render.size;
            this.sig = `${cubes.length}:${h.toFixed(3)}:${size.x}x${size.y}`;
        }
        return this.sig;
    }
}

/** 一段場景的取景：整段框得進來就只有一個位置；框不進來就每個取樣點一個位置（跟拍） */
function sceneKeys(state: IProgramState, m: { ts: number[], boxes: BoundingBox3d[] }, opts: IFitOpts): ICameraPos[] {
    let union = new BoundingBox3d();
    for (let b of m.boxes) {
        union.combineInPlace(b);
    }
    if (union.empty) {
        return [];
    }
    let whole = fitBox(state, union, opts, null);
    if (whole.cellPx >= opts.minCellPx) {
        return [whole.pos];
    }

    let n = m.ts.length;
    let near = (i: number, radius: number) => {
        let box = new BoundingBox3d();
        for (let j = 0; j < n; j++) {
            if (Math.abs(m.ts[j] - m.ts[i]) <= radius + 1e-9) {
                box.combineInPlace(m.boxes[j]);
            }
        }
        return box;
    };

    // 跟拍：每個取樣點只框自己附近那一小段時間裡出現的東西
    let fitted = m.ts.map((_, i) => {
        let box = near(i, FOLLOW_WINDOW);
        return box.empty ? null : fitBox(state, box, opts, FOLLOW_MIN_CELL_PX).pos;
    });
    // 附近還沒有東西出現的取樣點（例如後面幾份還沒出發），沿用最近一個有東西的
    let last = fitted.find(f => f !== null)!;
    let raw: ICameraPos[] = [];
    for (let f of fitted) {
        last = f ?? last;
        raw.push(last);
    }

    // 中心與 zoom 沿時間平滑，鏡頭才不會一格一格跳
    let cx = gaussian(raw.map(k => k.center.x));
    let cy = gaussian(raw.map(k => k.center.y));
    let cz = gaussian(raw.map(k => k.center.z));
    let logZoom = gaussian(raw.map(k => Math.log(k.angle.z)));

    // 平滑會讓鏡頭落後格子：逐點確認相鄰取樣點之間出現的東西都框得進來，框不下就再拉遠
    let f = framer(state, opts);
    let zCap = Math.max(f.zNear, f.zoomForCellPx(FOLLOW_MIN_CELL_PX));
    let zooms = raw.map((_, i) => {
        let z = Math.exp(logZoom[i]);
        let box = near(i, 1 / (n - 1));
        return box.empty ? z : f.search(boxCorners(box), new Vec3(cx[i], cy[i], cz[i]), z, Math.max(z, zCap));
    });
    // 拉遠的地方往前後攤開一點，鏡頭才不會突然一縮；只往大取，框得進來的仍然框得進來
    let spread = gaussian(zooms.map(z => Math.log(z))).map(z => Math.exp(z));

    return raw.map((k, i) => ({
        center: new Vec3(cx[i], cy[i], cz[i]),
        angle: new Vec3(k.angle.x, k.angle.y, Math.max(zooms[i], spread[i])),
    }));
}

/** 投影一個範圍、找 zoom 的工具。投影跟 Camera.ts 渲染時用的是同一套 */
function framer(state: IProgramState, opts: IFitOpts) {
    let size = state.render.size;
    let cell = state.layout.cell;
    let aspect = size.x / size.y;
    let persp = Mat4f.fromPersp(FOV_DEG, aspect, NEAR, FAR);
    let safe = {
        x0: -1 + 2 * MARGIN_SIDE,
        x1: 1 - 2 * MARGIN_SIDE,
        y0: -1 + 2 * MARGIN_BOTTOM,
        y1: 1 - 2 * opts.top,
    };
    /** 在相機注視點的距離上，一格剛好 px 個像素時的 zoom；cellPx 是反過來 */
    let zoomForCellPx = (px: number) => cell * size.y / (2 * TAN_HALF_FOV * px * DIST_PER_ZOOM);
    let cellPx = (zoom: number) => cell * size.y / (2 * TAN_HALF_FOV * zoom * DIST_PER_ZOOM);

    /** 範圍投影到螢幕上（NDC）的邊界，以及有沒有哪一角離相機太近 */
    let project = (corners: Vec3[], at: Vec3, zoom: number) => {
        let { lookAt } = cameraToMatrixView({ center: at, angle: new Vec3(opts.azimuth, opts.elevation, zoom) } as ICamera);
        let m = persp.mul(lookAt).mul(MODEL_MTX);
        let r = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity, tooNear: false, lookAt };
        for (let p of corners) {
            let w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
            if (w < MIN_DEPTH) {
                r.tooNear = true;
                continue;
            }
            let x = (m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12]) / w;
            let y = (m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13]) / w;
            r.x0 = Math.min(r.x0, x);
            r.x1 = Math.max(r.x1, x);
            r.y0 = Math.min(r.y0, y);
            r.y1 = Math.max(r.y1, y);
        }
        return r;
    };
    let fits = (corners: Vec3[], at: Vec3, zoom: number) => {
        let r = project(corners, at, zoom);
        return !r.tooNear && r.x0 >= safe.x0 && r.x1 <= safe.x1 && r.y0 >= safe.y0 && r.y1 <= safe.y1;
    };
    /** 從 zFrom 往遠處找框得進安全框的最小 zoom（離得越近 zoom 越小）；到 zCap 還框不下就停在 zCap */
    let search = (corners: Vec3[], at: Vec3, zFrom: number, zCap: number) => {
        if (fits(corners, at, zFrom)) {
            return zFrom;
        }
        let lo = zFrom;
        let hi = zFrom * 2;
        while (hi < zCap && !fits(corners, at, hi)) {
            lo = hi;
            hi *= 2;
        }
        if (hi >= zCap) {
            if (!fits(corners, at, zCap)) {
                return zCap;
            }
            hi = zCap;
        }
        for (let i = 0; i < 20; i++) {
            let mid = Math.sqrt(lo * hi);
            if (fits(corners, at, mid)) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return hi;
    };

    return { aspect, safe, zNear: Math.max(MIN_ZOOM, zoomForCellPx(opts.maxCellPx)), zoomForCellPx, cellPx, project, search };
}

interface IFit {
    pos: ICameraPos;
    /** 在相機注視點的距離上，一格有幾個像素 */
    cellPx: number;
}

/**
 * 找一個相機位置，讓 box 的八個角投影後都落在安全框內，而且盡量近。
 * farCellPx 不是 null 時，最遠只拉到一格剩這麼多像素（框不下的部分就讓它出去）。
 */
function fitBox(state: IProgramState, box: BoundingBox3d, opts: IFitOpts, farCellPx: number | null): IFit {
    let c = box.center();
    let look = new Vec3(c.x, -c.z, -c.y);
    let size = state.render.size;
    if (!(size.x > 1 && size.y > 1)) {
        return { pos: { center: look, angle: new Vec3(opts.azimuth, opts.elevation, 2) }, cellPx: 0 };
    }

    let f = framer(state, opts);
    let corners = boxCorners(box);
    let zCap = farCellPx === null ? MAX_ZOOM : Math.max(f.zNear, f.zoomForCellPx(farCellPx));
    let zoom = f.search(corners, look, f.zNear, zCap);
    for (let iter = 0; iter < 3; iter++) {
        // 透視會讓範圍投影後偏向一邊：把它移回安全框的正中間，再找一次 zoom
        let r = f.project(corners, look, zoom);
        if (!(r.x0 <= r.x1)) {
            break;
        }
        let halfH = zoom * DIST_PER_ZOOM * TAN_HALF_FOV;
        let offX = (r.x0 + r.x1) / 2 - (f.safe.x0 + f.safe.x1) / 2;
        let offY = (r.y0 + r.y1) / 2 - (f.safe.y0 + f.safe.y1) / 2;
        let right = new Vec3(r.lookAt[0], r.lookAt[4], r.lookAt[8]);
        let up = new Vec3(r.lookAt[1], r.lookAt[5], r.lookAt[9]);
        look = look.add(right.mul(offX * halfH * f.aspect)).add(up.mul(offY * halfH));
        zoom = f.search(corners, look, f.zNear, zCap);
    }

    return {
        pos: { center: look, angle: new Vec3(opts.azimuth, opts.elevation, zoom) },
        cellPx: f.cellPx(zoom),
    };
}

function boxCorners(b: BoundingBox3d) {
    let out: Vec3[] = [];
    for (let x of [b.min.x, b.max.x]) {
        for (let y of [b.min.y, b.max.y]) {
            for (let z of [b.min.z, b.max.z]) {
                out.push(new Vec3(x, y, z));
            }
        }
    }
    return out;
}

/** 沿取樣點做高斯平滑（兩端只用得到的那一側） */
function gaussian(values: number[]): number[] {
    let r = Math.ceil(SMOOTH_SIGMA * 3);
    return values.map((_, i) => {
        let wSum = 0;
        let sum = 0;
        for (let j = Math.max(0, i - r); j <= Math.min(values.length - 1, i + r); j++) {
            let w = Math.exp(-((j - i) ** 2) / (2 * SMOOTH_SIGMA ** 2));
            wSum += w;
            sum += values[j] * w;
        }
        return sum / wSum;
    });
}

function evalKeys(keys: ICameraPos[], s: number): ICameraPos | null {
    if (keys.length === 0) {
        return null;
    }
    if (keys.length === 1) {
        return keys[0];
    }
    let x = clamp(s, 0, 1) * (keys.length - 1);
    let i = Math.min(Math.floor(x), keys.length - 2);
    return lerpPos(keys[i], keys[i + 1], x - i);
}

/** 方位角走最短的方向；zoom 在對數上內插，拉近拉遠的速度才均勻 */
function lerpPos(a: ICameraPos, b: ICameraPos, t: number): ICameraPos {
    return {
        center: a.center.lerp(b.center, t),
        angle: new Vec3(
            a.angle.x + angleDelta(a.angle.x, b.angle.x) * t,
            lerp(a.angle.y, b.angle.y, t),
            Math.exp(lerp(Math.log(a.angle.z), Math.log(b.angle.z), t))),
    };
}

/** 從方位角 a 轉到 b 最短的角度，落在 [-180, 180) */
function angleDelta(a: number, b: number) {
    return ((b - a) % 360 + 540) % 360 - 180;
}

function samePos(a: ICameraPos, b: ICameraPos, eps: { pos: number, deg: number, zoom: number }) {
    return a.center.dist(b.center) <= eps.pos
        && Math.abs(angleDelta(a.angle.x, b.angle.x)) <= eps.deg
        && Math.abs(a.angle.y - b.angle.y) <= eps.deg
        && Math.abs(Math.log(a.angle.z / b.angle.z)) <= eps.zoom;
}

function posOf(p: ICameraPos): ICameraPos {
    return { center: p.center.clone(), angle: p.angle.clone() };
}
