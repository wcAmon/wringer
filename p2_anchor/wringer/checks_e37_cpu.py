"""E37 CPU 三檢(prereg_e37_reservoir cpu_checks_plan;全 CPU 免 GPU)。

①旁路代數:W_eff 恆等式(off/soft/hard 一律 +γ·BA/r)、Δγ 計量精確
  (R_eff 差 ≡ Δγ·BA)、凍結後 BA 零梯度 + 位元不變。
②排水 toy:凍結水庫 + 兩格階梯在 3元 模組上——薄格(γ=0.5)不觸發、
  殘量自動累積至全放(γ=0)觸發翻碼(D5/D8 累積機制)、absorb 高、
  回滲不可能(BA 前後位元相同)。
③生命週期:γ=0 extract ≡ k2_weight;res 未載入 forward 逐位 = 舊行為;
  ResBypass 合併導出位元正確(W + BA/r)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.checks_e37_cpu
"""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer.comp import greedy_comp_grid
from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.res_fill import ResBypass
from p2_anchor.wringer.stq import SoftGrid, STQParam

torch.manual_seed(20260806)
M, K, nB = 8, 16, 2
C3 = 0.0


def mk_param(use_comp=True, c=C3, n=3):
    grid = SoftGrid(c=c, device="cpu", n=n)
    a0 = torch.full((M, nB, 1), 0.1)
    p = STQParam(a0, grid, r=4, c=c, learn_theta=False, use_comp=use_comp)
    return p, grid


def mk_weight(grid):
    """格點上的權重(材化態:u 恰為碼值)。"""
    idx = torch.randint(0, grid.vals.shape[0], (M, nB, K // nB))
    v = grid.vals[idx]
    a0 = torch.full((M, nB, 1), 0.1)
    return (a0 * v).reshape(M, K).clone(), v


def check1():
    p, grid = mk_param()
    W, _ = mk_weight(grid)
    _ = p(W)                                   # lazy init(mode off 仍先跑)
    p.mode = "soft"
    p.s = 3.0
    _ = p(W)
    delta = torch.randn(M, K) * 0.03
    base_soft = p(W).clone()
    p.mode = "off"
    base_off = p(W).clone()
    p.mode = "hard"
    base_hard = p(W).clone()
    p.load_reservoir(delta, torch.eye(K), 1, gamma=0.7)   # BA/r = delta
    for mode, s, base in (("off", 0.0, base_off), ("soft", 3.0, base_soft),
                          ("hard", 0.0, base_hard)):
        p.mode, p.s = mode, s
        got = p(W)
        assert torch.allclose(got, base + 0.7 * delta, atol=1e-6), \
            f"①W_eff 恆等式 FAIL @{mode}"
    # Δγ 計量:R_eff(γ2)−R_eff(γ1) = (γ1−γ2)·BA(W_hard 不動時)
    r1 = (1.0 - 0.7) * delta
    p.gamma = 0.3
    r2 = (1.0 - 0.3) * delta
    assert torch.allclose(r2 - r1, 0.4 * delta, atol=1e-7), "①Δγ 計量 FAIL"
    # 凍結:soft 前向 backward,res 無梯度、位元不變
    b0 = p.res_B.clone()
    p.mode, p.s = "soft", 3.0
    W_req = W.clone().requires_grad_(False)
    out = p(W_req)
    out.square().sum().backward()
    assert p.res_B.grad is None and p.res_A.grad is None, "①凍結梯度 FAIL"
    assert torch.equal(p.res_B, b0), "①凍結位元 FAIL"
    assert p.lora_A.grad is not None, "①載體梯度應存在"
    print("① 旁路代數 PASS(W_eff 三模式恆等、Δγ 計量、凍結零梯度)")


def check2():
    p, grid = mk_param()
    W, v0 = mk_weight(grid)
    p.mode, p.s = "soft", 3.0
    _ = p(W)                                   # lazy init(off 會早退)
    # 水庫內容 = 少數 donor 位置的整格步(碼可全吸收)
    delta = torch.zeros(M, K)
    donors = [(0, 0), (1, 3), (2, 8), (3, 12)]
    a = 0.1
    for (mm, kk) in donors:
        u_cur = float(W.reshape(M, nB, K // nB)[mm, kk // (K // nB),
                                                kk % (K // nB)] / a)
        step = 1.0 if u_cur < 0.5 else -1.0    # 朝異胞方向整格步
        delta[mm, kk] = a * step
    p.load_reservoir(delta, torch.eye(K), 1, gamma=1.0)
    Xc = torch.randn(64, K)
    Xc[:, 0] = Xc[:, 3] * 0.9 + Xc[:, 0] * 0.1   # 相關欄(H 非對角)
    H = Xc.T @ Xc

    @torch.no_grad()
    def r_eff():
        u_, al_ = p._u(W)
        from p2_anchor.wringer.ktier import round_with_mids
        _, _, vh = round_with_mids(u_, p.theta_eff(), grid.vals,
                                   grid.t1, grid.t2)
        return W + (1.0 - p.gamma) * delta - (al_ * vh).reshape(M, K)

    n_full = float(((delta @ H) * delta).sum())      # 滿虧欠基準
    # 格1(γ=0.5):薄格——孤立欄半步不過整格判準;相關欄可經 H 聚合先觸發
    p.gamma = 0.5
    log1 = greedy_comp_grid(p, W, r_eff(), H, rounds=3, cap=0.5)
    # 格2(γ=0):未吸收殘量自動累積成全步 → 觸發其餘翻碼(D5/D8)
    p.gamma = 0.0
    log2 = greedy_comp_grid(p, W, r_eff(), H, rounds=3, cap=0.5)
    total = log1["flips"] + log2["flips"]
    assert total >= len(donors) - 1, \
        f"②兩格合計觸發 FAIL:{log1['flips']}+{log2['flips']}"
    Rf = r_eff()
    n_final = float(((Rf @ H) * Rf).sum())
    assert n_final < 0.2 * n_full, \
        f"②終端水位 FAIL:{n_final:.3g} ≥ 0.2×{n_full:.3g}"
    # 回滲不可能:水庫位元不變
    assert torch.equal(p.res_B, delta), "②回滲 FAIL(BA 被改動)"
    # 碼確實移動:extract 與初始碼比對
    T1, T2, al, wh = p.extract(W)
    v_now = (T1.float() + p.c * T2.float())
    moved = int((v_now != v0).sum())
    assert moved >= len(donors) - 1, f"②碼移動 FAIL:moved={moved}"
    print(f"② 排水 toy PASS(格1 flips={log1['flips']} + 格2 "
          f"flips={log2['flips']},終端水位 {n_final/max(n_full,1e-30):.2%} "
          f"of 滿虧欠,回滲=0,moved={moved})")


def check3():
    # γ=0 extract ≡ k2_weight
    p, grid = mk_param()
    W, _ = mk_weight(grid)
    p.mode, p.s = "soft", 3.0
    _ = p(W)
    p.load_reservoir(torch.randn(M, K) * 0.05, torch.eye(K), 1, gamma=0.0)
    T1, T2, al, wh = p.extract(W)
    w_rt = k2_weight(T1, T2, al, p.c)
    assert torch.equal(wh, w_rt), "③extract≡k2_weight FAIL"
    # γ≠0 時 extract 必須斷言擋下
    p.gamma = 0.4
    try:
        p.extract(W)
        raise AssertionError("③殘水 extract 未被擋下")
    except AssertionError as e:
        assert "排空" in str(e), f"③斷言訊息異常:{e}"
    # res 未載入 = 舊行為逐位
    pa, _ = mk_param()
    pb, _ = mk_param()
    pb_state_src = pa                          # 同權重同 seed 路徑
    for q in (pa, pb):
        q.mode = "hard"
        _ = q(W)
    pb.lora_A.data.copy_(pa.lora_A.data)
    pb.lora_B.data.copy_(pa.lora_B.data)
    assert torch.equal(pa(W), pb(W)), "③res=None 舊行為 FAIL"
    # ResBypass 合併導出位元正確
    lin = nn.Linear(K, M, bias=False)
    W0 = lin.weight.detach().clone()
    b = ResBypass(M, K, r=4, device="cpu")
    b.B.data.normal_(0, 0.1)
    parametrize.register_parametrization(lin, "weight", b)
    w_eff = lin.weight.detach().clone()
    parametrize.remove_parametrizations(lin, "weight",
                                        leave_parametrized=False)
    lin.weight.data.copy_(w_eff)
    assert torch.allclose(lin.weight, W0 + (b.B @ b.A) / b.r, atol=1e-6), \
        "③serve_merge 位元 FAIL"
    print("③ 生命週期 PASS(extract≡k2_weight、殘水斷言、res=None 舊行為、"
          "合併導出)")


if __name__ == "__main__":
    check1()
    check2()
    check3()
    print("E37 CPU 三檢全 PASS")
