import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough08_Transformer(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Transformer) {
        return;
    }

    setInitialCamera(state, new Vec3(-135.531, 0.000, -353.905), new Vec3(291.100, 13.600, 5.706));

    let c0 = commentary(wt, null, 0)`
    這就是一個完整的變換器(transformer)模組！

    它們構成了任何 GPT 模型的主體，並且會重複多次，一個區塊的輸出會輸入到下一個區塊，繼續剩餘路徑。

    在深度學習中，很難確切地說出這些層中的每一層都在做什麼，但我們有一些大致的想法：較早的層往往
    專注於學習較低階別的特徵和模式，而較後的層則學習識別和理解更高階別的抽象和關係。
    在自然語言處理中，低層可能學習語法、句法和簡單的詞彙關聯，而高層可能捕捉更復雜的語義關係、
    話語結構和上下文相關的含義。
    `;

}
