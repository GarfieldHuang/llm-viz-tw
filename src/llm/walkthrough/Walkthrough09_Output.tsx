import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough09_Output(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Output) {
        return;
    }

    setInitialCamera(state, new Vec3(-20.203, 0.000, -1642.819), new Vec3(281.600, -7.900, 2.298));

    let c0 = commentary(wt, null, 0)`

最後，我們來到模型的末端。最後變換器模組的輸出經過層歸一化處理，然後我們使用線性變換（矩陣乘法），這次沒有偏置。

這種最終轉換將我們的每個列向量從長度 C 轉換為長度 nvocab。因此，它實際上為我們的每一列中的每個詞彙生成一個分數。
這些分數有一個特殊的名稱：logits。

"logits"這個名字來源於 "log-odds"，即每個標記（token）的機率的對數。之所以使用 "對數"，
是因為我們接下來應用的 softmax 會進行指數運算，將其轉換為 "機率 "或機率。

為了將這些分數轉換為良好的機率，我們對它們進行 softmax 操作。現在，對於每一列，
我們都有一個模型分配給詞彙中每個單詞的機率。

在這個特定的模型中，它已經有效地學習瞭如何對三個字母進行排序這一問題的所有答案，
因此機率在很大程度上傾向於正確答案。

當我們對模型進行時間步進時，我們會使用上一列的機率來決定下一個要新增到序列中的標記（token）。
例如，如果我們已經向模型提供了 6 個標記，我們就會使用第 6 列的輸出機率。

這一列的輸出是一系列機率，我們實際上必須從中選出一個作為下一個機率。我們的做法是 "從分佈中取樣"。
也就是說，我們按照機率加權隨機選擇一個標記。例如，機率為 0.9 的標記將在 90% 的情況下被選中。

不過，這裡還有其他選擇，比如總是選擇機率最高的標記。

我們還可以使用溫度參數來控制分佈的 "平滑度"。溫度越高，分佈越均勻，溫度越低，分佈越集中在機率最高的標記上。

在應用 softmax 之前，我們先用溫度除以 logits（線性變換的輸出）。由於 softmax 中的指數化會對較大的數字產生較大影響，因此將所有數字拉近會減少這種影響。
`;

}
