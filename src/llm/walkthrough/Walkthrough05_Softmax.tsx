import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough05_Softmax(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Softmax) {
        return;
    }

    setInitialCamera(state, new Vec3(-24.350, 0.000, -1702.195), new Vec3(283.100, 0.600, 1.556));

    let c0 = commentary(wt, null, 0)`

Softmax操作不僅在前面的部分中作為自注意力的一部分使用，而且也會出現在模型的最後。

它的目的是將一個向量的值歸一化，使其總和為 1.0。然而，這並不像除以總和那麼簡單。相反，每個輸入值都要先進行指數化處理。

  a = exp(x_1)

這樣做的效果是使所有值都為正。有了指數化值的向量後，我們就可以用每個值除以所有值的總和。
這將確保所有數值之和為 1.0。由於所有指數化值都是正值，我們知道得出的值將介於 0.0 和 1.0 之間，
這就為原始值提供了一個機率分佈。

這就是 softmax 的原理：簡單地將數值指數化，然後除以總和。

不過，還有一個小問題。如果輸入值很大，那麼指數化後的值也會很大。我們最終會用一個很大的數除以一個很大的數，
這可能會導致浮點運算出現問題。

Softmax 運算的一個有用特性是，如果我們在所有輸入值上新增一個常數，結果將是相同的。
因此，我們可以找到輸入向量中的最大值，然後將其從所有值中減去。這樣就能確保最大值為 0.0，
從而使 softmax 在數值上保持穩定。

讓我們來看看自注意力層中的 softmax 操作。我們的輸入向量是自注意力矩陣的一行（但只到對角線）。

與層歸一化一樣，我們有一箇中間步驟來儲存一些聚合值以保持流程效率。

對於每一行，我們都會儲存該行的最大值以及移位值和指數值的總和。
然後，為了生成相應的輸出行，我們可以執行一小套操作：減去最大值、指數化和除以總和。

為什麼叫 "softmax"？這種操作的 "硬 "版本稱為 argmax，簡單地說，就是找到最大值，
將其設為 1.0，並將所有其他值設為 0.0。相比之下，softmax操作則是該操作的 "更柔和 "版本。
由於softmax涉及指數運算，最大值被強調並推向 1.0。同時仍保持所有輸入值的機率分佈。
這樣就能獲得更細緻的表示，不僅能捕捉到最有可能的選項，還能捕捉到其他選型的相對可能性。
`;

}
