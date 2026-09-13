# LLM 視覺化（繁體中文版）

GPT 類大語言模型的 **3D 互動視覺化**：一個 token 從進入模型到吐出機率分佈，
每一步矩陣運算都看得到，並附逐步導覽解說。

原作 [llm-viz](https://github.com/bbycroft/llm-viz) by [Brendan Bycroft](https://bbycroft.net/)（MIT）
→ 簡體翻譯 [llm-viz-cn](https://github.com/czhixin/llm-viz-cn) by AI探索官
→ **本版：繁體化 + 台灣術語校正 + GitHub Pages 靜態部署**

## 專案內容

展示的是一個可運作的 GPT 風格網路，也就是 OpenAI 在 GPT-2、GPT-3 中使用的網路拓撲。

預設載入的是一個帶有真實權重的極小模型（nano-gpt，85,000 個參數），
任務是把 A、B、C 三個字母的短列表排序 —— 出自 Andrej Karpathy 的
[minGPT](https://github.com/karpathy/minGPT) 範例模型。

渲染器本身支援任意大小的網路，也能處理 GPT-2 規模，但那份權重未內含（需要 100MB 以上）。

## 導覽章節

| # | 章節 | # | 章節 |
|---|---|---|---|
| 00 | 簡介 | 05 | Softmax |
| 01 | 預備知識 | 06 | 投影 (Projection) |
| 02 | 嵌入 (Embedding) | 07 | MLP |
| 03 | 層歸一化 (LayerNorm) | 08 | Transformer |
| 04 | 自注意力 (Self-Attention) | 09 | 輸出 |

### 反向傳播

| # | 章節 | # | 章節 |
|---|---|---|---|
| 10 | 損失與 dLogits | 14 | 自注意力 (Self-Attention) |
| 11 | MLP 與 GELU | 15 | 層歸一化 (LayerNorm) |
| 12 | 殘差分流 | 16 | 嵌入 (Embedding) |
| 13 | 投射 (Projection) | | |

每個反向運算都用「把前向動畫倒著演」的方式呈現：格子從矩陣裡飛出來、相乘、加總、落進梯度。
點一下 3D 畫面裡的任一格，側邊欄會從前向式出發，逐步推導（偏微分 → 連鎖律 → 加總 → 代入數字）到那一格的梯度，
並與 PyTorch autograd 的結果對答案。

梯度資料由 `gen_grad_data.py` 產生（需要 PyTorch，不需要 minGPT）。輸入必須與畫面上的前向模型相同；
腳本會先逐張比對手寫前向與 minGPT 的中間值，再驗證反向公式，全部通過才寫出 `public/gpt-nano-sort-grads.json`。

## 本地執行

```
yarn install
yarn dev
```

開 <http://localhost:3002>

## 部署到 GitHub Pages

推上 `main` 後由 `.github/workflows/deploy.yml` 自動建置並發布。
Repo 需在 **Settings → Pages → Source** 選 **GitHub Actions**。

`basePath` 由 workflow 依 repo 名稱自動注入，不必手動改設定。

## 與上游的差異

- 全站介面與導覽解說繁體化（`s2twp` 轉換 + 人工術語校正）
- 修正 OpenCC 在技術文件上的誤轉：`引數→參數`、`關注力→注意力`、`對映→映射`、`擴充套件→擴展`
- 改為 Next.js 靜態匯出（`output: 'export'`），可部署於純靜態主機
- 資產路徑改走 `src/utils/assetPath.ts`，支援掛在子路徑下
- 內含 `public/native.wasm`（上游為 build 產物，未進 repo）
- 移除上游附帶的 RISC-V CPU 模擬與流體模擬子專案（與 LLM 無關，且含靜態匯出不支援的 API route）

## 已知限制

3D 畫面內的標籤（`LayerNorm`、`Q/K/V`、`μ`、`σ` 等）維持英文。
那些字是用 MSDF 字型圖集畫在 WebGL canvas 上的，圖集僅 512×256 且不含任何中日韓字形；
要支援中文需重製多頁圖集，體積會膨脹數 MB。
實務上這些是不該翻譯的技術術語，維持原文反而較佳。

## 授權

MIT，沿用上游 [bbycroft/llm-viz](https://github.com/bbycroft/llm-viz)。
