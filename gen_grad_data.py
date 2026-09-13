"""
產生反向傳播視覺化所需的梯度資料 -> public/gpt-nano-sort-grads.json

設計原則：
  * 輸入必須與畫面上的前向模型完全相同。GptModel.ts 與 GptModelWasm.ts 寫死的是
    [2, 1, 0, 1, 1, 2, 0, 0, 0, 0, 0]（C B A B B C）。
    gen_test_data.py 用的是另一條驗證序列（A A C B A B），**不能照抄** ——
    之前照抄過，結果梯度與畫面上的前向值根本不是同一次前向，側邊欄的逐步推導全部驗算失敗。
  * 不需要 minGPT：前向照 minGPT 的 GPT 定義用純 PyTorch 寫出來，權重從
    public/gpt-nano-sort-model.json 還原（該檔已含完整 state_dict）。
    寫完先拿 gen_test_data.py 用 minGPT 跑出來的中間值（gpt-nano-sort-t0-partials.json）
    逐張比對，一模一樣才繼續。
  * 損失刻意採用「反事實標的」：這個 nano-gpt 在此範例上已經完全收斂
    （位置 5 的預測是 "A"，機率 ≈ 1.0），用正確標的算出來的梯度全都趨近 0，
    畫面上會是一片空白。因此在位置 t=5 問：「如果正解其實是 C 呢？」
    這會得到乾淨且看得見的梯度，且 dL/dlogits = p - y ≈ [1, 0, -1]，好講也好驗證。

用法：
    python gen_grad_data.py
"""
import argparse
import base64
import json
import math
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

# 畫面上的輸入。改這裡之前，先確認 GptModel.ts / GptModelWasm.ts 也一起改了。
VIZ_IDX = [2, 1, 0, 1, 1, 2, 0, 0, 0, 0, 0]

LOSS_POS = 5      # 提示詞的最後一個位置，也就是要生出第一個排序結果的地方
LOSS_TARGET = 2   # 'C'。模型其實深信是 'A'(0)，故意給相反的標的
LN_EPS = 1e-5     # minGPT 用 nn.LayerNorm 的預設值


def decode_tensor(v):
    raw = base64.b64decode(v['data'])
    np_dtype = {'torch.float32': '<f4', 'torch.int64': '<i8'}[v['dtype']]
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(v['shape'])
    return torch.from_numpy(arr.copy())


def load_tensor_json(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    config = d.pop('config', None)
    tensors = {k: decode_tensor(v) for k, v in d.items() if isinstance(v, dict) and 'data' in v}
    return config, tensors


def tensor_to_json(t):
    t = t.detach().contiguous().to(torch.float32)
    return {
        "shape": list(t.shape),
        "dtype": "torch.float32",
        "data": base64.b64encode(t.numpy().tobytes()).decode(),
    }


def new_gelu(x):
    """minGPT 的 NewGELU（與 GPT-2 相同的 tanh 近似）。"""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def forward(params, masks, cfg, idx, retain):
    """
    照 minGPT 的 GPT.forward 逐步展開，並把每個中間量存下來。
    retain=True 時對中間量 retain_grad，反向之後才拿得到它們的梯度。

    命名沿用既有的 Backprop.ts 對接：第 0 層無前綴，其餘各層 b1. / b2.。
    """
    n_head = cfg['n_head']
    C = cfg['n_embd']
    n_layer = cfg['n_layer']
    B, T = idx.shape
    hs = C // n_head
    captured = {}

    def cap(name, t):
        if retain and t.requires_grad:
            t.retain_grad()
        captured[name] = t
        return t

    pos = torch.arange(0, T, dtype=torch.long).unsqueeze(0)
    tok_emb = cap('tok_emb', F.embedding(idx, params['transformer.wte.weight']))
    pos_emb = cap('pos_emb', F.embedding(pos, params['transformer.wpe.weight']))
    h = cap('x', tok_emb + pos_emb)

    for i in range(n_layer):
        prefix = '' if i == 0 else f'b{i}.'

        def p(name, i=i):
            return params[f'transformer.h.{i}.{name}']

        def c(name, t, prefix=prefix):
            return cap(prefix + name, t)

        ln1 = c('ln1', F.layer_norm(h, (C,), p('ln_1.weight'), p('ln_1.bias'), eps=LN_EPS))
        qkv = c('qkv', F.linear(ln1, p('attn.c_attn.weight'), p('attn.c_attn.bias')))
        q_, k_, v_ = qkv.split(C, dim=2)
        q = c('q', q_.view(B, T, n_head, hs).transpose(1, 2))
        k = c('k', k_.view(B, T, n_head, hs).transpose(1, 2))
        v = c('v', v_.view(B, T, n_head, hs).transpose(1, 2))

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(masks[i][:, :, :T, :T] == 0, float('-inf'))
        att = c('att', att)
        attSm = c('attSm', F.softmax(att, dim=-1))
        y = c('y', (attSm @ v).transpose(1, 2).contiguous().view(B, T, C))
        yProj = c('yProj', F.linear(y, p('attn.c_proj.weight'), p('attn.c_proj.bias')))

        attnResid = c('attnResid', h + yProj)
        ln2 = c('ln2', F.layer_norm(attnResid, (C,), p('ln_2.weight'), p('ln_2.bias'), eps=LN_EPS))
        fc = c('fc', F.linear(ln2, p('mlp.c_fc.weight'), p('mlp.c_fc.bias')))
        gelu = c('gelu', new_gelu(fc))
        mlp = c('mlp', F.linear(gelu, p('mlp.c_proj.weight'), p('mlp.c_proj.bias')))
        h = c('mlpResid', attnResid + mlp)
        # 區塊輸出另外取一個名字，方便視覺化端直接對接
        captured[f'block{i}'] = h

    ln_f = cap('ln_f', F.layer_norm(h, (C,), params['transformer.ln_f.weight'], params['transformer.ln_f.bias'], eps=LN_EPS))
    logits = cap('lm_head', F.linear(ln_f, params['lm_head.weight']))
    cap('probs', F.softmax(logits, dim=-1))
    return captured, logits


def check_against_mingpt(params, masks, cfg, partials_path):
    """手寫的前向必須與 minGPT 逐張相同，否則後面算的梯度沒有意義。"""
    _, ref = load_tensor_json(partials_path)
    idx = ref['idx'].to(torch.long)
    with torch.no_grad():
        mine, _ = forward(params, masks, cfg, idx, retain=False)

    print(f'手寫前向 vs minGPT（{partials_path}，batch={idx.shape[0]}）:')
    ok = True
    for name, r in ref.items():
        if name == 'idx' or name not in mine:
            continue
        m = mine[name].to(torch.float32)
        r = r.to(torch.float32)
        if m.shape != r.shape:
            print(f'  FAIL {name:10} shape {tuple(m.shape)} vs {tuple(r.shape)}')
            ok = False
            continue
        finite = torch.isfinite(r)
        same_mask = torch.equal(finite, torch.isfinite(m))
        err = (m[finite] - r[finite]).abs().max().item() if finite.any() else 0.0
        good = same_mask and err < 1e-4
        ok = ok and good
        print(f"  {'OK ' if good else 'FAIL'} {name:10} max|diff|={err:.2e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-json', default='public/gpt-nano-sort-model.json')
    ap.add_argument('--partials-json', default='public/gpt-nano-sort-t0-partials.json')
    ap.add_argument('--out', default='public/gpt-nano-sort-grads.json')
    args = ap.parse_args()

    cfg, state = load_tensor_json(args.model_json)
    n_layer = cfg['n_layer']

    # attn.bias 是因果遮罩（buffer），不是參數
    masks = [state[f'transformer.h.{i}.attn.bias'] for i in range(n_layer)]
    params = {k: v.to(torch.float32).clone().requires_grad_(True)
              for k, v in state.items() if not k.endswith('.attn.bias')}

    if not check_against_mingpt(params, masks, cfg, args.partials_json):
        raise SystemExit('手寫前向與 minGPT 不一致，不寫出資料')
    print()

    T = cfg['block_size']
    B = 1   # 視覺化端固定以 B=1 執行
    idx = torch.tensor([VIZ_IDX], dtype=torch.long)
    assert idx.shape == (B, T)

    captured, logits = forward(params, masks, cfg, idx, retain=True)

    probs = F.softmax(logits[0, LOSS_POS], dim=-1)
    pred = int(probs.argmax())
    print(f'輸入 {VIZ_IDX}，位置 {LOSS_POS} 的預測 = {pred}（p = {probs.tolist()}）')
    if pred == LOSS_TARGET:
        raise SystemExit('模型在這個位置本來就預測對了，反事實標的就不成立，請換一個 LOSS_TARGET')

    target = torch.full((B,), LOSS_TARGET, dtype=torch.long)
    loss = F.cross_entropy(logits[:, LOSS_POS, :], target)
    print(f'loss = {loss.item():.6f}  (t={LOSS_POS}, target={LOSS_TARGET})')
    loss.backward()

    print()
    print('驗證梯度（PyTorch autograd vs 手推公式）:')
    if not verify(captured, params, cfg, idx):
        raise SystemExit('梯度驗證失敗，不寫出資料')

    out = {}
    missing = []
    for name, t in captured.items():
        if t.grad is None:
            missing.append(name)
            continue
        out['d_' + name] = tensor_to_json(t.grad)
    for name, prm in params.items():
        if prm.grad is None:
            missing.append(name)
            continue
        out['d_' + name] = tensor_to_json(prm.grad)

    extra = {
        'config': {**cfg, 'B': B},
        'inputIdx': VIZ_IDX,
        'loss': loss.item(),
        'lossKind': 'counterfactual-single-position-crossentropy',
        'lossPos': LOSS_POS,
        'lossTarget': LOSS_TARGET,
        'note': 'gradients w.r.t. every captured activation and parameter, for the same input the '
                'visualization runs (inputIdx). The model has converged on this example, so a correct '
                'target would give near-zero gradients; we deliberately use a counterfactual target so '
                'the error signal is visible.',
    }
    payload = {**extra, **out}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1)

    print(f'\n寫出 {args.out}（{os.path.getsize(args.out):,} bytes，{len(out)} 個梯度張量）')
    if missing:
        print('沒有梯度的項目（不在損失的計算路徑上）:', missing)


def verify(captured, params, cfg, idx):
    """把 PyTorch autograd 的結果，跟教科書上的 attention / embedding 反向公式逐元素比對。
    畫出來的動畫再漂亮，數值錯了就沒有意義，所以這一步是必要的。"""
    ok = True

    def cmp(name, a, b, tol=2e-5):
        nonlocal ok
        err = (a - b).abs().max().item()
        scale = max(a.abs().max().item(), 1e-12)
        rel = err / scale
        good = rel < tol
        ok = ok and good
        print(f"  {'OK ' if good else 'FAIL'} {name:38} max|diff|={err:.3e}  rel={rel:.2e}")

    B, T = idx.shape
    C = cfg['n_embd']
    n_head = cfg['n_head']
    hs = C // n_head

    q, k, v = captured['q'], captured['k'], captured['v']
    attSm, y = captured['attSm'], captured['y']
    att = captured['att']

    dY = y.grad.view(B, T, n_head, hs).transpose(1, 2)

    cmp("dV = P^T dO", attSm.transpose(-2, -1) @ dY, v.grad)
    cmp("dP = dO V^T", dY @ v.transpose(-2, -1), attSm.grad)

    P, dP = attSm, attSm.grad
    cmp("dS = P o (dP - rowsum(P o dP))", P * (dP - (P * dP).sum(dim=-1, keepdim=True)), att.grad)

    O = attSm @ v
    cmp("rowsum(P o dP) == rowsum(O o dO)", (P * dP).sum(dim=-1), (O * dY).sum(dim=-1))

    scale = 1.0 / math.sqrt(hs)
    dS = att.grad
    cmp("dQ = dS K / sqrt(d)", dS @ k * scale, q.grad)
    cmp("dK = dS^T Q / sqrt(d)", dS.transpose(-2, -1) @ q * scale, k.grad)

    # Layer Norm：dX = (g - mean(g) - xhat * mean(g * xhat)) / sigma，g = gamma * dLN
    x_in = captured['x']
    ln1 = captured['ln1']
    gamma = params['transformer.h.0.ln_1.weight']
    mu = x_in.mean(dim=-1, keepdim=True)
    sigma = torch.sqrt(((x_in - mu) ** 2).mean(dim=-1, keepdim=True) + LN_EPS)
    xhat = (x_in - mu) / sigma
    g = ln1.grad * gamma
    dX_ln = (g - g.mean(dim=-1, keepdim=True) - xhat * (g * xhat).mean(dim=-1, keepdim=True)) / sigma
    # x 同時被 ln1 與殘差用到，所以 dX = LN 那一路 + 殘差那一路
    cmp("dX = LN path + residual path", dX_ln + captured['attnResid'].grad, x_in.grad)

    # embedding：scatter-add
    dTok = captured['tok_emb'].grad
    dWte_manual = torch.zeros_like(params['transformer.wte.weight'])
    dWte_manual.index_add_(0, idx.reshape(-1), dTok.reshape(-1, C))
    cmp("d_wte = scatter-add(d_tok_emb)", dWte_manual, params['transformer.wte.weight'].grad)

    # 因果結構：loss 只看 t=5，所以 t>5 的位置梯度必須是 0
    tail = dTok[:, LOSS_POS + 1:, :].abs().max().item()
    good = tail < 1e-12
    ok = ok and good
    print(f"  {'OK ' if good else 'FAIL'} {'causal: d_tok_emb[t>5] == 0':38} max={tail:.3e}")

    print("\n  => 全部通過" if ok else "\n  => 有項目不符，請檢查")
    return ok


if __name__ == '__main__':
    main()
