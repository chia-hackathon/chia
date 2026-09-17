# rvv_baseline_failures.json — S2 排除清單
- machine_vfcvt_f_x_v-0：pristine Saturn 對 stock Spike 就有 1-ulp 差異（rvv_baseline 實測 9/10）。
- machine_vsetvl-0：IME v0.9.0 規定 IME-legal 的 vsetvli 必須選非零 lambda 寫進 vtype[62:60]（preserve-or-initialize），stock Spike 不認識 IME 永遠是 0，所以此測試在 stock Spike 下不可能通過；vtype 的 IME 欄位語意改由 S3 以 IME Spike 模型 lockstep 判定（9/12 決定）。
- machine_vsetivli-0（9/14 加入）：與 machine_vsetvl-0 同一類。A3 修正後的樹在 S2 抽樣 150 支只剩這一支失敗，分歧是 `spike x14 = 0x5` vs `DUT x14 = 0x3000000000000005`，即 vtype[62:60] 的 lambda=0b011（VLEN=256/SEW=8 的最大合法值）；stock Spike 不認識 IME 欄位永遠回 0。凡是把 vtype 值寫回純量暫存器比對的 vsetvl 家族測試都會這樣，S3 以 IME Spike 模型判定。
- machine_vsetvli-0（9/17 加入）：與 machine_vsetvl-0／vsetivli-0 同類，分歧特徵相同（vtype[62:60] 的 lambda）。證據：`titan_runs/a3_verify/results.json` 的 `s2_full` —— A3 修正後跑完整 841 支，839 支執行、只有 vsetivli-0 與 vsetvli-0 兩支失敗，兩者都是這一類。先前只加了 vsetivli-0，因為 150 支抽樣沒抽到 vsetvli-0。
