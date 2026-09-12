"""
產生反向傳播視覺化所需的梯度資料 -> public/gpt-nano-sort-grads.json

設計原則：
  * 完全不修改 gen_test_data.py 與既有的前向資料，前向教學不受影響。
  * 權重直接從 public/gpt-nano-sort-model.json 還原（該檔已含完整 state_dict），
    因此不需要 minGPT 的 model.pt。
  * 輸入 idx 與 gen_test_data.py 完全相同，梯度才對得上前向視覺化顯示的數值。
  * 損失刻意採用「反事實標的」：這個 nano-gpt 在此範例上已經完全收斂
    （每個位置的預測機率都是 1.0），用正確標的算出來的梯度全都趨近 0，
    畫面上會是一片空白 —— 這正說明了「模型答對時沒有東西需要修正」。
    因此我們在位置 t=5 問一個反事實問題：「如果正解其實是 C 而不是模型
    深信不疑的 A，誤差會怎麼往回傳？」這會得到乾淨且看得見的梯度，
    且 dL/dlogits = p - y = [1, 0, -1]，好講也好驗證。

用法：
    python gen_grad_data.py --mingpt <minGPT repo 路徑>
"""
import argparse
import base64
import json
import math
import os
import sys

import torch
from torch.nn import functional as F


def load_model_json(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    config = d.pop('config')
    state = {}
    for k, v in d.items():
        raw = base64.b64decode(v['data'])
        dtype = {'torch.float32': torch.float32, 'torch.int64': torch.int64}[v['dtype']]
        np_dtype = {'torch.float32': '<f4', 'torch.int64': '<i8'}[v['dtype']]
        import numpy as np
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(v['shape'])
        state[k] = torch.from_numpy(arr.copy()).to(dtype)
    return config, state


def tensor_to_json(t):
    t = t.detach().contiguous().to(torch.float32)
    return {
        "shape": list(t.shape),
        "dtype": "torch.float32",
        "data": base64.b64encode(t.numpy().tobytes()).decode(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mingpt', required=True, help='minGPT repo 路徑（含 mingpt/ 套件）')
    ap.add_argument('--model-json', default='public/gpt-nano-sort-model.json')
    ap.add_argument('--out', default='public/gpt-nano-sort-grads.json')
    args = ap.parse_args()

    sys.path.insert(0, args.mingpt)
    from mingpt.model import GPT
    from mingpt.utils import set_seed
    set_seed(3407)

    config, state = load_model_json(args.model_json)

    mc = GPT.get_default_config()
    mc.model_type = config['model_type']
    mc.vocab_size = config['vocab_size']
    mc.block_size = config['block_size']
    model = GPT(mc)
    model.load_state_dict(state)
    model.eval()   # 關閉 dropout：梯度才是確定性的，可重現

    n_head, n_embd = mc.n_head, mc.n_embd
    T = mc.block_size
    # 視覺化端固定以 B=1 執行（Program.ts 的 shape.B 與 LayerView 的 initModel 都是 1），
    # 梯度張量必須同樣是 B=1，否則寫進 texture 時尺寸對不上。
    B = 1

    # 與 gen_test_data.py 的第一列輸入完全相同 —— 也就是畫面上顯示的那條序列
    idx = torch.tensor([[0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.long)

    # ---- 前向（逐層捕捉，並 retain_grad 以便取得中間量的梯度）----
    captured = {}

    def cap(name, t):
        t.retain_grad()
        captured[name] = t
        return t

    pos = torch.arange(0, T, dtype=torch.long).unsqueeze(0)
    tok_emb = cap('tok_emb', model.transformer.wte(idx))
    pos_emb = cap('pos_emb', model.transformer.wpe(pos))
    x = cap('x', tok_emb + pos_emb)

    # 逐層展開，每一層的中間量都 retain_grad。
    # 原本只對第 0 層這麼做，其餘各層直接 block(h) 帶過 —— 結果視覺化裡
    # 滑到後面幾層的區塊只有權重有梯度，中間量一律空白。
    # 第 0 層沿用無前綴的名字（verify() 與既有的 Backprop.ts 對接都靠它），
    # 其餘各層加上 b1. / b2. 前綴。
    def run_block(block, x, prefix):
        def c(name, t):
            return cap(prefix + name, t)

        ln1 = c('ln1', block.ln_1(x))

        qkv = c('qkv', block.attn.c_attn(ln1))
        q_, k_, v_ = qkv.split(n_embd, dim=2)
        q = c('q', q_.view(B, T, n_head, n_embd // n_head).transpose(1, 2))
        k = c('k', k_.view(B, T, n_head, n_embd // n_head).transpose(1, 2))
        v = c('v', v_.view(B, T, n_head, n_embd // n_head).transpose(1, 2))

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(block.attn.bias[:, :, :T, :T] == 0, float('-inf'))
        att = c('att', att)
        attSm = c('attSm', F.softmax(att, dim=-1))
        y_ = attSm @ v
        y = c('y', y_.transpose(1, 2).contiguous().view(B, T, n_embd))
        yProj = c('yProj', block.attn.c_proj(y))

        attnResid = c('attnResid', x + yProj)
        ln2 = c('ln2', block.ln_2(attnResid))
        fc = c('fc', block.mlp.c_fc(ln2))
        gelu = c('gelu', block.mlp.act(fc))
        mlp = c('mlp', block.mlp.c_proj(gelu))
        mlpResid = c('mlpResid', attnResid + mlp)
        return mlpResid

    h = x
    for i, block in enumerate(model.transformer.h):
        prefix = '' if i == 0 else f'b{i}.'
        h = run_block(block, h, prefix)
        # 區塊輸出另外取一個名字，方便視覺化端直接對接
        captured[f'block{i}'] = h

    ln_f = cap('ln_f', model.transformer.ln_f(h))
    logits = cap('lm_head', model.lm_head(ln_f))
    probs = cap('probs', F.softmax(logits, dim=-1))

    # ---- 損失：位置 LOSS_POS 的反事實 cross-entropy ----
    LOSS_POS = 5              # 提示詞的最後一個位置，也就是要生出第一個排序結果的地方
    LOSS_TARGET = 2           # 'C'。模型其實深信是 'A'(0)，故意給相反的標的
    target = torch.full((B,), LOSS_TARGET, dtype=torch.long)
    loss = F.cross_entropy(logits[:, LOSS_POS, :], target)
    print(f'loss = {loss.item():.6f}  (t={LOSS_POS}, target={LOSS_TARGET})')

    model.zero_grad(set_to_none=True)
    loss.backward()

    # ---- 數值正確性驗證：對照教科書公式 ----
    captured['_idx'] = idx
    print()
    print('驗證梯度（PyTorch autograd vs 手推公式）:')
    if not verify(captured, model, mc, B, T):
        raise SystemExit('梯度驗證失敗，不寫出資料')

    # ---- 收集梯度 ----
    out = {}
    missing = []
    for name, t in captured.items():
        if name.startswith('_'):
            continue
        if t.grad is None:
            missing.append(name)
            continue
        out['d_' + name] = tensor_to_json(t.grad)
    for name, p in model.named_parameters():
        if p.grad is None:
            missing.append(name)
            continue
        out['d_' + name] = tensor_to_json(p.grad)

    extra = {
        'config': {**mc.to_dict(), 'B': B},
        'loss': loss.item(),
        'lossKind': 'counterfactual-single-position-crossentropy',
        'lossPos': LOSS_POS,
        'lossTarget': LOSS_TARGET,
        'note': 'gradients w.r.t. every captured activation and parameter. '
                'The model has converged on this example, so a correct target would give '
                'near-zero gradients; we deliberately use a counterfactual target so the '
                'error signal is visible.',
    }
    payload = {**extra, **out}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1)

    print(f'\n寫出 {args.out}（{os.path.getsize(args.out):,} bytes，{len(out)} 個梯度張量）')
    if missing:
        print('沒有梯度的項目:', missing)
    return captured, model, loss




def verify(captured, model, mc, B, T):
    """把 PyTorch autograd 的結果，跟教科書上的 attention / embedding 反向公式逐元素比對。
    畫出來的動畫再漂亮，數值錯了就沒有意義，所以這一步是必要的。"""
    import math
    ok = True

    def cmp(name, a, b, tol=2e-5):
        nonlocal ok
        err = (a - b).abs().max().item()
        scale = max(a.abs().max().item(), 1e-12)
        rel = err / scale
        good = rel < tol
        ok = ok and good
        print(f"  {'OK ' if good else 'FAIL'} {name:38} max|diff|={err:.3e}  rel={rel:.2e}")

    n_head = mc.n_head
    hs = mc.n_embd // n_head

    q, k, v = captured['q'], captured['k'], captured['v']
    att, attSm, y = captured['att'], captured['attSm'], captured['y']

    # dO：把 y 的梯度 (B,T,C) 還原成每個 head 的 (B,nh,T,hs)
    dY = y.grad.view(B, T, n_head, hs).transpose(1, 2)

    # dV = P^T dO
    dV_manual = attSm.transpose(-2, -1) @ dY
    cmp("dV = P^T dO", dV_manual, v.grad)

    # dP = dO V^T
    dP_manual = dY @ v.transpose(-2, -1)
    cmp("dP = dO V^T", dP_manual, attSm.grad)

    # dS = P * (dP - rowsum(P*dP))
    P, dP = attSm, attSm.grad
    dS_manual = P * (dP - (P * dP).sum(dim=-1, keepdim=True))
    cmp("dS = P o (dP - rowsum(P o dP))", dS_manual, att.grad)

    # FlashAttention 恆等式：rowsum(P o dP) == rowsum(O o dO)
    O = (attSm @ v)
    lhs = (P * dP).sum(dim=-1)
    rhs = (O * dY).sum(dim=-1)
    cmp("rowsum(P o dP) == rowsum(O o dO)", lhs, rhs)

    # dQ = dS K / sqrt(hs) ; dK = dS^T Q / sqrt(hs)
    scale = 1.0 / math.sqrt(hs)
    dS = att.grad
    cmp("dQ = dS K / sqrt(d)", dS @ k * scale, q.grad)
    cmp("dK = dS^T Q / sqrt(d)", dS.transpose(-2, -1) @ q * scale, k.grad)

    # embedding：scatter-add
    idx_used = captured['_idx']
    dTok = captured['tok_emb'].grad                       # (B, T, C)
    dWte_manual = torch.zeros_like(model.transformer.wte.weight)
    dWte_manual.index_add_(0, idx_used.reshape(-1), dTok.reshape(-1, mc.n_embd))
    cmp("d_wte = scatter-add(d_tok_emb)", dWte_manual, model.transformer.wte.weight.grad)

    # 因果結構：loss 只看 t=5，所以 t>5 的位置梯度必須是 0
    tail = dTok[:, 6:, :].abs().max().item()
    good = tail < 1e-12
    ok = ok and good
    print(f"  {'OK ' if good else 'FAIL'} {'causal: d_tok_emb[t>5] == 0':38} max={tail:.3e}")

    print("\n  => 全部通過" if ok else "\n  => 有項目不符，請檢查")
    return ok

if __name__ == '__main__':
    main()
