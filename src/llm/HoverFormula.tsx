import React, { useState } from 'react';
import clsx from 'clsx';
import { useProgramState } from './Sidebar';
import { describeFormula, IFormulaOperand } from './components/FormulaText';
import { deriveBackward, formatNum, IDerivation } from './components/GradMath';
import { Tex } from './components/Tex';
import s from './HoverFormula.module.scss';

/**
 * 側邊欄底部的「這一格怎麼算出來的」面板。
 *
 * 前向：3D 浮層受字型圖集限制，只能用小方塊代表「某一列／某一行」，這裡把同一個算式展開成文字。
 *
 * 反向：只給結果式讀者看不懂它從哪來，所以改成逐步推導 ——
 * 從前向式出發，對這一格偏微分、套連鎖律、加總、代入數字，最後跟資料裡的梯度對答案。
 */
export const HoverFormula: React.FC = () => {
    let progState = useProgramState();
    let pinned = progState.display.pinnedTarget;
    let target = pinned ?? progState.display.hoverTarget;

    // 展開了哪幾條路徑。換一格就回到預設（只展開第一條）。
    let [openState, setOpenState] = useState<{ key: string, open: number[] }>({ key: '', open: [0] });

    if (!target) {
        return <div className={clsx(s.panel, s.idle)}>
            把滑鼠移到 3D 畫面的任一格上，這裡會顯示它完整的算式。點一下格子可以把它固定在這裡。
        </div>;
    }

    let desc = describeFormula(progState, target.mainCube, target.mainIdx);
    if (!desc) {
        return null;
    }

    let backward = desc.dir === 'backward';
    let deriv = backward ? deriveBackward(progState, target.mainCube, target.mainIdx) : null;

    let key = `${desc.dir}:${target.mainCube.idx}:${target.mainCube.name}:${target.mainIdx.x},${target.mainIdx.y}`;
    let open = openState.key === key ? openState.open : [0];

    function toggle(i: number) {
        let next = open.includes(i) ? open.filter(a => a !== i) : [...open, i];
        setOpenState({ key, open: next });
    }

    function unpin() {
        progState.display.pinnedTarget = null;
        progState.markDirty();
    }

    return <div className={clsx(s.panel, deriv && s.tall)}>
        {pinned && <div className={s.pinBar}>
            <span>已固定這一格</span>
            <button className={s.unpin} onClick={unpin}>取消固定</button>
        </div>}

        <div className={s.head}>
            <span className={clsx(s.dirTag, backward ? s.backward : s.forward)}>
                {backward ? '反向' : '前向'}
            </span>
            <span className={clsx(s.target, backward && s.gradName)}>{desc.target}</span>
            {desc.index && <span className={s.index}>[{desc.index}]</span>}
            {desc.value !== null && desc.value !== undefined && (
                <span className={s.headValue}>= {desc.value.toFixed(4)}</span>
            )}
        </div>

        {deriv ? <Derivation deriv={deriv} open={open} onToggle={toggle} /> : <>
            <div className={s.caption}>3D 畫面上那個式子的完整版</div>

            <div className={s.exprRow}>
                {!desc.plain && <span className={s.eq}>=</span>}
                <span className={clsx(s.expr, desc.plain && s.plainExpr)}>{desc.expr}</span>
            </div>

            {desc.rule && <div className={s.rule}>
                {desc.rule.split('\n').map((line, i) => <div key={i}>{line}</div>)}
            </div>}

            {desc.operands.length > 0 && <div className={s.operands}>
                <div className={s.operandsLabel}>浮層上的小方塊，對應到：</div>
                {desc.operands.map((o, i) => <Operand key={i} op={o} />)}
            </div>}

            {desc.note && <div className={s.note}>{desc.note}</div>}
        </>}
    </div>;
};

const Derivation: React.FC<{
    deriv: IDerivation,
    open: number[],
    onToggle: (i: number) => void,
}> = ({ deriv, open, onToggle }) => {
    let multi = deriv.paths.length > 1;

    return <div className={s.deriv}>
        <div className={s.caption}>
            {multi
                ? `這一格被 ${deriv.paths.length} 個地方用到。每條路徑各推一次，最後加起來。`
                : '從前向式一步一步推到這一格的梯度'}
        </div>

        {deriv.paths.map((p, i) => {
            let isOpen = !multi || open.includes(i);
            return <div key={i} className={s.path}>
                <div className={clsx(s.pathHead, multi && s.clickable)} onClick={multi ? () => onToggle(i) : undefined}>
                    {multi && <span className={s.caret}>{isOpen ? '▾' : '▸'}</span>}
                    <span className={s.pathTitle}>{p.title}</span>
                    <span className={s.pathValue}>{formatNum(p.value)}</span>
                </div>
                {isOpen && p.steps.map((st, j) => <div key={j} className={s.step}>
                    <div className={s.stepLabel}>{st.label}</div>
                    <Tex block tex={st.tex} className={s.stepTex} />
                    {st.note && <div className={s.stepNote}>{st.note}</div>}
                </div>)}
            </div>;
        })}

        {deriv.sumTex && <div className={clsx(s.step, s.sumStep)}>
            <div className={s.stepLabel}>⑥ 所有路徑相加</div>
            <Tex block tex={deriv.sumTex} className={s.stepTex} />
        </div>}

        <CheckLine total={deriv.total} stored={deriv.stored} />
    </div>;
};

/** 推導算出來的數字，跟 PyTorch autograd 存下來的梯度對答案。 */
const CheckLine: React.FC<{ total: number | null, stored: number | null }> = ({ total, stored }) => {
    if (total === null || stored === null) {
        return <div className={clsx(s.check, s.checkUnknown)}>有資料讀不到，這一格無法驗算。</div>;
    }
    let diff = Math.abs(total - stored);
    let ok = diff <= 1e-5 + 2e-3 * Math.abs(stored);
    return <div className={clsx(s.check, ok ? s.checkOk : s.checkBad)}>
        {ok ? '✓ ' : '✗ '}
        照上面的式子算出 {formatNum(total)}，PyTorch 反向傳播存下來的是 {formatNum(stored)}
        {ok ? '，一致。' : `，差了 ${diff.toExponential(2)}。`}
    </div>;
};

const Operand: React.FC<{ op: IFormulaOperand }> = ({ op }) => {
    return <div className={s.operand}>
        <span className={clsx(s.opName, s['kind_' + op.kind])}>{op.name}</span>
        <span className={s.opDetail}>{op.detail}</span>
    </div>;
};
