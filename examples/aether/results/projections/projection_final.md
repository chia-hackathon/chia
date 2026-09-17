# Llama-3.2-1B 最終投影：batch decode (N=1 / 16 / 32) + prefill + roofline

CHIA 內迴圈六輪已結束，所有 kernel 判定到達實際上限（見 `out/loop/FINAL_REPORT.md`）。
本文件是**最後一份投影**：把 round5 的 measured-cycle 資料庫外推到 batched decode，
回答「kernel 已經沒空間了，系統層還剩什麼」。

產生方式（新工具，本任務新增）：

```
python -m loop.llama_batch_project --S 512 \
    --measured out/llama-profile/measured_cycles_round5.json \
    --batches 1,2,4,8,16,32,64
python -m loop.llama_batch_project --S 512 \
    --measured out/llama-profile/measured_cycles_round5.json \
    --batches 1,16,32 --lmhead-tile
python -m loop.llama_project --scenario prefill --N 64 \
    --measured out/llama-profile/measured_cycles_round5.json \
    --device int8_gemv=gemmini --device lm_head_gemv=gemmini
```

原始輸出：`out/llama-profile/proj_final_batch_decode.txt`、
`proj_final_batch_decode_lmheadtile.txt`、`proj_final_prefill.txt`。

**為什麼要新寫一支工具**：`loop/llama_project.py` 沒有 batch-decode 模式
(`--scenario` 只有 `decode`/`prefill`，`--S` 是 decode 的 KV 長度、`--N` 是 prefill
每趟 token 數；N=16 的 GEMV 量測在 measured JSON 裡被標成 "documentation only"，
`MeasuredDB.load` 一個 kernel 只取一筆 `Measurement`)。這點在
`projection_round5.md` 的 (e) 節已經確認過，當時是用手算補的。
`loop/llama_batch_project.py` 直接複用 `llama_project.project()` 的 costed op
清單，因此 **B=1 欄位逐 cycle 重現** `projection_round5.md` 的 159,733,828
cyc/token / 6.260 tok/s（工具自己會印 cross-check 行）。

---

## 1. 成本模型與假設

| 類別 | 隨 batch B 如何縮放 | 依據 |
|---|---|---|
| `int8_gemv`（Q/K/V/O + MLP 投影，16 層）| **不乘 B**，改乘 tile 比值 `tile(B)/tile(1)` | 權重從 DRAM 讀一次，B 條序列的 activation 共用。tile 量測：N=1 132,424 cyc、N=16 142,458 cyc（M=512,K=2048） |
| `lm_head_gemv` | 同上（乘同一個 tile 比值） | 假設：lm_head 沒有自己的 N=16 量測，沿用 GEMV 的 batch 行為（+7.6% cycles 換 16 倍算術） |
| `attn_scores` / `attn_pv` / `softmax` | **× B** | 每條序列有自己的 KV cache，無法共用；S=512 固定 |
| `rmsnorm` / `rope` / `silu_mul` / `add` / `embedding` | **× B** | 每條序列自己的 activation |
| roofline | GEMV：`(權重 bytes + B×(activation in + accumulator out)) / 8`；其餘 `B ×` 單序列 roofline | mbus 8 B/cycle（2026-09-09 修正值） |

- Tile 模型：兩個實測點 (1, 132,424) 與 (16, 142,458)，斜率 **668.9 cycle/每多一個
  column**，其餘 B 為**線性外推**。所以 **B=32（tile 153,161 cyc）是估計值，不是量測**，
  表中以 `*` 標記。B=2/4/8/64 同理。
- 時脈 1.00 GHz（`measured_cycles_round5.json` 的 `clock_hz`），與前五輪所有 tok/s 同基準。
- `attn_*` 在 B=16 直接乘 16，是**保守假設**：真實實作可把 16 條序列的 Q 併成一個
  16×64 的 GEMM 去打 Gemmini，score/PV 的 K/V cache 讀取量不變但算術密度變高，
  實際會比 ×16 好。沒有量測，所以不敢寫進表裡（見 §4 的「下一步」）。

---

## 2. 主表：decode（S=512）、prefill、roofline

`tok/s agg` = 整個 batch 每秒產出的 token 總數；`tok/s seq` = 單一序列自己感受到的
生成速率（latency 面）。

| 情境 | tile cyc | step cycles | cycles/token | **tok/s (aggregate)** | **tok/s (per-seq)** | **roofline tok/s (8 B/cyc, aggregate)** | 距 roofline |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode **N=1** | 132,424 | 159,733,828 | 159,733,828 | **6.260** | 6.260 | 6.352 | 1.01x |
| decode **N=16** | 142,458 | 296,540,142 | 18,533,759 | **53.956** | 3.372 | 79.283 | 1.47x |
| decode **N=32**\* | 153,161\* | 442,466,877 | 13,827,090 | **72.322**\* | 2.260\* | 128.438 | 1.78x |
| prefill (N=64 tokens/pass) | — | — | 7,390,000 | **135.312**† | — | 209.620 | 1.5x |

\* N=32 的 GEMV tile 為線性外推（估計值）。
† prefill 仍帶 `lm_head_gemm` 未量測的 `roofline × 4` placeholder（見
`projection_round4.md` 的 "Important caveat"）；扣掉該 placeholder 加倍的 artifact
後的 apples-to-apples 值約 **157.2 tok/s**（6.36M cyc/token）。這個數字從 round4
起未再變動，round5/round6 都沒碰 prefill 路徑的 kernel。

完整的 batch 掃描（含 2/4/8/64）：

| B | step cycles | cycles/token | tok/s agg | tok/s seq | roofline agg | GEMV 佔比 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 159,733,828 | 159,733,828 | 6.260 | 6.260 | 6.352 | 94.8% |
| 2\* | 168,854,248 | 84,427,124 | 11.845 | 5.922 | 12.470 | 90.1% |
| 4\* | 187,095,090 | 46,773,773 | 21.380 | 5.345 | 24.052 | 82.1% |
| 8\* | 223,576,774 | 27,947,097 | 35.782 | 4.473 | 44.909 | 70.1% |
| 16 | 296,540,142 | 18,533,759 | 53.956 | 3.372 | 79.283 | 54.9% |
| 32\* | 442,466,877 | 13,827,090 | 72.322 | 2.260 | 128.438 | 39.6% |
| 64\* | 734,320,348 | 11,473,755 | 87.155 | 1.362 | 133.474 | 27.2% |

相對 B=1：**B=16 → 8.62x 總吞吐、0.539x 單序列速率；B=32 → 11.55x / 0.361x；
B=64 → 13.92x / 0.218x**。

### lm_head 基準的敏感度

主表的 `lm_head_gemv` 沿用 round2–round5 一貫的 **n1-basis 換算**
（`projection_round4.md`：`ratio = 634,507 / 740,101 = 0.857325`，
`132,653 × 0.857325 = 113,727` cycles @ 1,048,576 MAC → 全 lm_head 28.49M cyc/token）。
這個基準其實**比自己的 roofline 還快**（28.49M vs 32.90M，等效 9.2 B/cycle），
物理上不可能——它是把 M=512 小 tile 的表現線性外推到 128,256 列的樂觀假象。

用 lm_head **自己那顆 M=2048,K=2048 tile 的實測值**（634,507 cycles / 4 MiB 權重
= **6.61 B/cycle**，符合「大權重全冷 DRAM 只有 6.6–7.4 B/cycle」的硬體事實）改算：
`634,507 × (128,256/2,048) = 39,736,001` cyc/token。

| 情境 | n1-basis（主表） | lm_head 自身 tile 基準（6.61 B/cyc） |
|---|---:|---:|
| decode N=1 tok/s | 6.260 | **5.849** |
| decode N=16 tok/s agg | 53.956 | **51.840** |
| decode N=32 tok/s agg\* | 72.322 | **70.256** |

也就是最終 decode 速度的合理區間是 **N=1: 5.85–6.26 tok/s、N=16: 51.8–54.0 tok/s
(aggregate)、N=32: 70.3–72.3 tok/s (aggregate，估計)**。差異全部來自 lm_head 一顆 op。

---

## 3. 為什麼 decode 被 mbus 8 B/cycle 綁死，kernel 層已無空間

**(a) decode 每個 token 就是要把整個模型權重搬過 mbus 一次。**
`llama_model.py` 的 op 清單算出來：decode 一個 token 讀 **1,247,031,296 B**、寫
5,471,232 B，其中 GEMV + lm_head 的權重就佔 **1,235,746,816 B**（= 總參數減掉
norm 參數，`llama_model.self_test()` 有這條恆等式斷言）。int8 權重、一個 token 用
一次、零重用。1.236 GB ÷ 8 B/cycle = **154.5M cycles**，這就是地板。
`llama_project.py` 的 roofline 報告直接印出：`memory-only 156.56M cyc/tok (6.39 tok/s)
@ 8 B/cycle`，而且 **20 個 operator 裡有 18 個在 roofline 上是 memory-bound**。

**(b) 這條 8 B/cycle 是硬體常數，不是調參數。** Gemmini 的權重/激活 DMA 走
rocket-chip 的 mbus（`MemoryBusParams(beatBytes = 8)`），實體上就是 8 B/cycle；
2026-09-09 之前專案誤用 16 B/cycle，修正後所有 memory-bound roofline 加倍
（n1/n16 gemv 65,536 → 131,072；lmhead 262,144 → 524,288）。更糟的是大權重**全冷
DRAM** 時連 8 都達不到：lmhead tile 實測 **6.6 B/cycle**（agent 自己在
`out/loop/20260909-200541-dae1/agent_07.txt` 推導出「冷流速率 r ≈ 6.43 B/cycle…
634,507 就是它的底線；8 B/cycle 的 mbus 名義帶寬在 DRAMSim2 冷流上不可達」）。

**(c) kernel 層的搜尋已經撞到這條線。** 六輪之後：

- `int8_gemv` (N=1) = 132,424 cycles vs 131,072 roofline = **1.010x**，換算成
  **7.92 B/cycle，約 mbus 峰值的 99%**。round4 只用 2 個 iteration 探針確認它沒動，
  round3 花了 $29 也只是重現同一個數字。
- `int8_gemv` (N=16) = 142,458 vs 131,072 = **1.087x**；round6 又燒 4 個 iteration
  $4.92，兩次嘗試都退步，**沒有新 best**。
- `lm_head` tile = 634,507 vs 524,288 = 1.210x，但那 1.21x 不是 kernel 寫得差，
  而是 6.61 冷流 B/cycle vs 8 名義 B/cycle 的差距（8/6.61 = 1.21，剛好對上）。
- 整體 decode：projected 159.73M vs roofline 157.43M，`llama_project.py` 自己報
  **headroom 1.0x**。就算把每顆 kernel 打到 roofline，decode 也只從 6.260 →
  6.352 tok/s（+1.5%）。

**(d) 所以唯一的槓桿是 batch。** 增加 batch 不會多搬一個 byte 的權重
（B 條序列共用同一次權重串流），只多搬 B 份 activation（每層每 token 幾 KB）——
從 N=1 到 N=16，GEMV tile 只多花 **7.6%** 的 cycle 卻做了 **16 倍**的算術。
這就是表 2 裡 aggregate 吞吐 6.26 → 53.96 tok/s（**8.62x**）的全部來源。
Amdahl 的方向在這裡是反過來的：不是讓 kernel 更快，而是讓同一批 bytes 服務更多 token。

---

## 4. batch 的代價與 break-even

**(a) 收益遞減，而且是被 attention 吃掉的。** 每個 decode step 的組成（cycles）：

| kernel | B=1 | 佔比 | B=16 | 佔比 | B=32\* | 佔比 | B=16 時距 roofline |
|---|---:|---:|---:|---:|---:|---:|---:|
| int8_gemv | 122.89M | 76.9% | 132.20M | 44.6% | 142.13M | 32.1% | ~1.01x |
| attn_scores | 4.22M | 2.6% | **67.44M** | **22.7%** | 134.89M | 30.5% | **4.02x** (16.78M) |
| attn_pv | 2.46M | 1.5% | **39.44M** | **13.3%** | 78.87M | 17.8% | **2.35x** (16.78M) |
| lm_head_gemv | 28.49M | 17.8% | 30.65M | 10.3% | 32.95M | 7.5% | — |
| softmax | 0.80M | 0.5% | 12.74M | 4.3% | 25.48M | 5.8% | 3.04x (4.19M) |
| silu_mul | 0.50M | 0.3% | 7.94M | 2.7% | 15.88M | 3.6% | 3.79x (2.10M) |
| rmsnorm / add / rope / embedding | 0.39M | 0.2% | 6.13M | 2.1% | 12.26M | 2.8% | ~2.5x |

B=1 時 GEMV 佔 94.8%，batch 後掉到 54.9% (B=16) / 39.6% (B=32)。
**新的瓶頸是 attention**：`attn_scores` + `attn_pv` + `softmax` 在 B=16 佔 40.3%，
而且這三顆分別是 roofline 的 4.02x / 2.35x / 3.04x——**與 N=1 的 GEMV 不同，
這裡真的還有空間**（六輪期間它們只在 round1–round3 被碰過，之後就沒再優化）。
所以 aggregate 吞吐才會從 8.62x (B=16) 只長到 11.55x (B=32) 和 13.92x (B=64)。

**(b) 單序列延遲的 break-even。** 每多一倍 batch，單序列速率如下（tok/s per-seq）：

| B | 1 | 2\* | 4\* | 8\* | 16 | 32\* | 64\* |
|---|---:|---:|---:|---:|---:|---:|---:|
| tok/s per-seq | 6.260 | 5.922 | 5.345 | 4.473 | 3.372 | 2.260 | 1.362 |
| vs B=1 | 1.000 | 0.946 | 0.854 | 0.714 | 0.539 | 0.361 | 0.218 |
| tok/s aggregate | 6.260 | 11.845 | 21.380 | 35.782 | 53.956 | 72.322 | 87.155 |
| 每倍 batch 的邊際吞吐 | — | +89% | +81% | +67% | +51% | +34% | +21% |

判讀：

- **B ≤ 4 幾乎免費**：單序列只慢 5–15%，吞吐 1.9–3.4 倍。如果有任何延遲 SLO，
  這是無腦該做的區間。
- **B=16 是「甜蜜點」**：8.62x 吞吐，代價是單序列砍半（6.26 → 3.37 tok/s）。
  這也是唯一除了 N=1 之外**有實測 kernel**（round5 的 142,458）的點。
- **B=32 以上開始不划算**：吞吐邊際收益跌到 +34% / +21%，單序列速率掉到
  2.26 / 1.36 tok/s。而且 B≥32 的 tile cycle 是外推估計，B=32 的 KV cache 也要
  32 × 512 × 8 heads × 64 × 2 (K+V) = 16.8 MB int8，已經超過任何合理的 on-chip 容量。
- **break-even 條件**：若要求單序列 ≥ 5 tok/s，最大 batch 約 **B=4–5**；
  若要求 ≥ 3 tok/s，可到 **B=16–20**；若純粹追求吞吐且不管延遲，B=32–64 仍在成長
  但已逼近 **~90–100 tok/s 的漸近線**（attention 的 ×B 成長最終會吃掉全部收益）。

**(c) 下一步（如果還有 loop）**：不是再調 N=1 GEMV，而是
(1) 實測 batched attention（把 B 條序列的 Q 併成 GEMM 打 Gemmini，可能比本文的
保守 ×B 假設好很多），(2) 補上從未量測的 `lm_head_gemm`（prefill 最大的未知數），
(3) 實測 N=32 GEMV tile 以取代線性外推。

---

## 5. 數字出處

| 數字 | 出處 |
|---|---|
| n1 GEMV 132,424 cyc | `out/loop/ledger.md` (c) 表 `llama-q8-gemv-gemmini-n1` round2b 起；run_id 20260909-200611-94c0 |
| n16 GEMV 142,458 cyc | `out/loop/ledger.md` round5/round6；run_id 20260911-065653-8115 |
| lmhead tile 634,507 cyc、ratio 0.857325 | `out/llama-profile/projection_round4.md` "lm_head_gemv_gemmini conversion"；run_id 20260909-200541-dae1 |
| decode B=1 159.73M cyc / 6.260 tok/s / roofline 157.43M | `out/llama-profile/proj_final_batch_decode.txt`、`projection_round5.md` (a) |
| prefill 7.39M cyc / 135.312 tok/s / roofline 209.620 | `out/llama-profile/proj_final_prefill.txt` |
| batch 表全部欄位 | `out/llama-profile/proj_final_batch_decode.txt`（`loop/llama_batch_project.py` 產生） |
| lm_head 6.61 B/cycle 替代基準 | `out/llama-profile/proj_final_batch_decode_lmheadtile.txt` |
| mbus 8 B/cycle | `out/loop/README.md` "Roofline correction (2026-09-09)"；`projection_round3.md`（`MemoryBusParams(beatBytes = 8)`） |
| 冷 DRAM 6.6–7.4 B/cycle | `out/loop/20260909-200541-dae1/agent_01.txt`（6.60 B/cycle）、`agent_07.txt`（≈6.43 B/cycle，「634,507 就是它的底线」） |
| n1 GEMV 達 7.92 B/cycle ≈ mbus 99% | `out/llama-profile/projection_round4.md` (d) |
| decode 每 token 1.247 GB read / 權重 1.236 GB | `loop/llama_model.py` `totals(model_ops(...))`；self-test 的 "GEMV MACs == total params - norm params" 恆等式 |
