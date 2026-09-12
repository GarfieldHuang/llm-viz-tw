import React from 'react';
import clsx from 'clsx';
import { useProgramState } from './Sidebar';
import { describeFormula, IFormulaOperand } from './components/FormulaText';
import s from './HoverFormula.module.scss';

/**
 * 側邊欄底部的「這一格怎麼算出來的」面板。
 *
 * 3D 浮層受字型圖集限制，只能用小方塊代表「某一列／某一行」，看得到形狀
 * 卻看不出是誰。這裡把同一個算式展開成完整文字，前向與反向都有。
 */
export const HoverFormula: React.FC = () => {
    let progState = useProgramState();
    let target = progState.display.hoverTarget;

    if (!target) {
        return <div className={clsx(s.panel, s.idle)}>
            把滑鼠移到 3D 畫面的任一格上，這裡會顯示它完整的算式
        </div>;
    }

    let desc = describeFormula(progState, target.mainCube, target.mainIdx);
    if (!desc) {
        return null;
    }

    let backward = desc.dir === 'backward';

    return <div className={s.panel}>
        <div className={s.head}>
            <span className={clsx(s.dirTag, backward ? s.backward : s.forward)}>
                {backward ? '反向' : '前向'}
            </span>
            <span className={clsx(s.target, backward && s.gradName)}>{desc.target}</span>
            {desc.index && <span className={s.index}>[{desc.index}]</span>}
        </div>

        {/* 講明白這一塊跟 3D 浮層是同一個式子，只是把小方塊展開成名字 */}
        <div className={s.caption}>3D 畫面上那個式子的完整版</div>

        <div className={s.exprRow}>
            {!desc.plain && <span className={s.eq}>=</span>}
            <span className={clsx(s.expr, desc.plain && s.plainExpr)}>{desc.expr}</span>
        </div>

        {desc.value !== null && desc.value !== undefined && (
            <div className={s.valueRow}>
                <span className={s.eq}>=</span>
                <span className={s.value}>{desc.value.toFixed(4)}</span>
            </div>
        )}

        {desc.rule && <div className={s.rule}>{desc.rule}</div>}

        {desc.operands.length > 0 && <div className={s.operands}>
            <div className={s.operandsLabel}>浮層上的小方塊，對應到：</div>
            {desc.operands.map((o, i) => <Operand key={i} op={o} />)}
        </div>}

        {desc.note && <div className={s.note}>{desc.note}</div>}
    </div>;
};

const Operand: React.FC<{ op: IFormulaOperand }> = ({ op }) => {
    return <div className={s.operand}>
        <span className={clsx(s.opName, s['kind_' + op.kind])}>{op.name}</span>
        <span className={s.opDetail}>{op.detail}</span>
    </div>;
};
