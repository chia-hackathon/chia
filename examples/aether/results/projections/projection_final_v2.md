# Llama-3.2-1B 最終投影 v2：2026-09-12 審計修正後

本文件取代 `projection_final.md` 作為目前權威投影，**不刪除舊檔**（舊檔標成
「已知偏樂觀，見本檔」）。修正原因：2026-09-12 審計發現 `projection_final.md`
的 N=1 decode 6.26 tok/s、lm_head 6 B/cycle 假設偏樂觀，且全文預設 1 GHz，
沒有把 RTL 實際跑在 500 MHz mbus 的情況列出來。

工具改動（本任務新增，見「改動檔案」一節）：
`loop/llama_project.py` 與 `loop/llama_batch_project.py` 新增
`--gemv-bytes-per-cycle` / `--lmhead-bytes-per-cycle` / `--cold {upper,lower}`，
讓 `int8_gemv`（decode GEMV）與 `lm_head_gemv` 直接用「冷 DRAM 實測 B/cycle」
costing（`cycles = op.macs / rate`），取代原本「用 measured_cycles.json 裡的
tile 做 MAC 線性外推」的作法——後者對 lm_head 隱含外推出 ~9.2 B/cycle，
超過 mbus 8 B/cycle 的物理上限，不可能發生。`--clock-ghz` 兩支工具都已存在
（預設 1.0），本次未改動其語意，只是拿來跑 0.5 GHz。

---

## 1. 四項修正的依據

| # | 修正 | 舊值（`projection_final.md`） | 新值 | 出處 |
|---|---|---|---|---|
| 1 | N=1 decode GEMV 的「暖」量測含 harness 殘留 | tile 132,424 cyc（含約 200 KiB 暖 L2 尾巴） | 冷態升序實測 **149,757 cyc = 7.00 B/cycle**（上界） | `out/loop/20260911-115840-5f95/agent_04.txt`（「≈9k 的尾段懲罰完全由 harness 留下的髒行狀態決定」）；`out/loop/round3-llama-q8-gemv-gemmini-n1.log` 系列的升序 K 量測 |
| 2 | lm_head 用 n1-basis MAC 外推得到不可能的速率 | `measured_cycles_round5.json` 的 `lm_head_gemv_gemmini`：113,727 cyc / 1,048,576 MAC = **9.22 B/cycle**，超過 mbus 8 B/cycle 上限 | 用 lm_head **自己的** 4 MiB tile 量測：634,507 cyc / 4,194,304 B = **6.61 B/cycle**，直接乘 128,256×2,048 的完整 lm_head | `out/llama-profile/measured_cycles_round5.json`（`lm_head_gemv_gemmini` 條目）；`out/loop/20260909-200541-dae1/agent_01.txt`、`agent_07.txt`（4 MiB tile 634,507 cyc 的推導） |
| 3 | 冷 DRAM 速率下界 | （未建模） | agent 對 stride-2048 權重流做曲線擬合：升序 K 的 +17,104 cyc 反推暖尾 ≈192 KiB，扣除後 **冷流速率 r ≈ 6.43 B/cycle** | `out/loop/20260909-200541-dae1/agent_07.txt`：「634,507 = 192K/16 + (4M−192K)/r ⇒ 冷流速率 r ≈ 6.43 B/cycle」 |
| 4 | 時脈假設 1 GHz，RTL 實際 500 MHz | 全文只列 1 GHz | 兩欄並列：**1 GHz（樂觀假設）** 與 **500 MHz（RTL 實測時脈）** | `out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:216-220`：「Clock sbus_0/pbus_0/fbus_0/mbus_0/cbus_0: using diplomatically specified frequency of 500.0」——bus 全 500 MHz，1 GHz 從未被 elaborate 驗證過 |

`--gemv-bytes-per-cycle 7.00`（上界，來自修正 1）、`--gemv-bytes-per-cycle 6.43`
（下界，來自修正 3）、`--lmhead-bytes-per-cycle 6.61`（來自修正 2，兩個 GEMV
情境共用同一個 lm_head 值，因為沒有分別測過上/下界）；`--cold upper` /
`--cold lower` 是這兩組的捷徑（見兩支工具 `--help`）。

---

## 2. 主表：N=1（暖/冷上/冷下）、N=16、N=32、prefill，@1 GHz 與 @500 MHz

N=16 / N=32 用 `--cold upper` / `--cold lower` 重跑
`loop/llama_batch_project.py`；prefill 用 `int8_gemm` / `lm_head_gemm`（compute-bound
GEMM，不受本次 GEMV/lm_head 冷態修正影響，兩個時脈欄位之外的數字與
`projection_final.md` 完全一致，見下方 cross-check）。

| 情境 | cycles/token | tok/s @ 1 GHz | tok/s @ 500 MHz | roofline tok/s @1GHz (8B/c) | 備註 |
|---|---:|---:|---:|---:|---|
| **N=1，暖態舊值**（`projection_final.md`，供對照） | 159.73M | 6.260 | 3.130 | 6.352 | 沿用 tile 132,424 cyc（含暖 L2 尾巴），**偏樂觀** |
| **N=1，冷態上界**（GEMV 7.00 B/c，lm_head 6.61 B/c） | 187.10M | 5.345 | 2.673 | 6.352 | `--cold upper`；GEMV 用冷態升序自身實測 |
| **N=1，冷態下界**（GEMV 6.43 B/c，lm_head 6.61 B/c） | 199.43M | 5.014 | 2.507 | 6.352 | `--cold lower`；GEMV 用 agent_07 擬合的冷流速率 |
| N=16（冷態上界，聚合） | 20.37M | 49.082 | 24.541 | 79.283 | tile N=16 由冷態 N=1 依原本 warm 的 N1→N16 成長比例（1.0757x）外推，**假設**：沒有分別測過冷 N=16 |
| N=16（冷態下界，聚合） | 21.20M | 47.164 | 23.582 | 79.283 | 同上，用下界 rate |
| N=32（冷態上界，聚合，估計） | 14.82M | 67.493 | 33.747 | 128.438 | GEMV tile 對 N 仍是線性外推（`TileModel.estimated`），未量測 |
| N=32（冷態下界，聚合，估計） | 15.26M | 65.523 | 32.762 | 128.438 | 同上 |
| prefill（N=64/pass） | 7.39M† | 135.312 | 67.656 | 209.620 | GEMM compute-bound，**不受冷態 GEMV 修正影響**，數字與舊檔一致（cross-check 見下） |

† 仍含未量測的 `lm_head_gemm` placeholder（`roofline × 4`），與舊檔相同的已知限制。

**Cross-check（工具自我一致性）**：
```
$ python -m loop.llama_project --scenario decode --S 512 --measured out/llama-profile/measured_cycles_round5.json \
    --device int8_gemv=gemmini --device lm_head_gemv=gemmini --cold upper --no-ops
projected   : 187.10M cycles/token  ->     5.345 tokens/s      # 與批次工具 B=1 欄位逐 cycle 相同
$ python -m loop.llama_batch_project --S 512 --measured out/llama-profile/measured_cycles_round5.json \
    --batches 1,16,32 --cold upper
B=1 cross-check vs llama_project.py decode: 187,104,977 cyc/tok projected, batch model says 187,104,977
$ python -m loop.llama_project --scenario prefill --N 64 --measured out/llama-profile/measured_cycles_round5.json \
    --device int8_gemv=gemmini --device lm_head_gemv=gemmini --cold upper --no-ops
projected   : 7.39M cycles/token  ->   135.312 tokens/s        # --cold 對 prefill 完全無影響（用的是 gemm，非 gemv）
```

不加 `--cold` / `--gemv-bytes-per-cycle` 時兩支工具的輸出與修正前逐 bit 相同
（`--self-test` 全通過），預設行為未被破壞。

---

## 3. kernel 層剩餘空間（審計結論，非本次新測）

decode 在 N=1 冷態下 GEMV/lm_head 已佔 ~95% 的 cycle，其餘 kernel 空間很小；
以下是審計認定「理論上還能擠」的三個方向，供下一輪 loop 參考：

| 方向 | 預估端到端增益 | 說明 |
|---|---:|---|
| Gemmini / Saturn 重疊執行 | **+4–5%** | 目前投影是純序列加總（`llama_project.py` 逐 op 相加），完全沒有建模 Gemmini 串流權重與 Saturn 算 attn/softmax/norm 之間的重疊；但發現 (7)（`out/loop/20260908-025820-f7c4/agent_04.txt`）已證明**記憶體流量的重疊是淨負**（Gemmini 與 core 共用同一條 tile-egress/L2/DRAM 路徑），所以只有「Saturn 的純算術部分」能疊到 Gemmini 的 DMA 等待窗口上，空間有限 |
| Saturn 四顆 kernel（attn_scores、attn_pv、softmax、silu_mul）打 2x | **+2%** | 這四顆已經在 1.01x–1.34x roofline 附近（見 `FINAL_REPORT.md` §2），繼續優化 kernel 本身已經很難再快；2x 只有可能來自架構層改動（例如 LMUL 提升、向量管線加寬），且它們在 N=1 冷態下的合計 cycle 佔比本來就只有 ~30%，Amdahl 上限本就不高 |
| in-flight 請求數匹配 L2 MSHR 數 | **0–3%** | §3(4) 的 rows/cmd 曲線（`out/loop/20260909-200541-dae1/agent_07.txt`、`agent_12.txt`）顯示 in-flight 請求數與吞吐量是**非單調**關係（6→649k、8→632k、16→687k cycle）；调到 MSHR 數的甜蜜點理論上有救，但曲線本身已經很平，上限不高 |

**合計理論空間 ≈ +6–10%**，遠小於 batch（N=1→N=16 是 +862%）——
再次印證 §5「kernel 層沒有空間了，唯一槓桿是 batch」的結論，只是把「沒有空間」
的數字從舊報告的「0%」修正為「還有個位數 %，且都需要架構層改動」。

---

## 4. 硬體突破表：只有「mbus 128-bit + MSHR 24」同時發生才有效

**[2026-09-19 訂正：本節標題與下表的預測已被 `out/hw-sweep/` 的實測推翻——單獨加寬 mbus 就拿到 98.6% 的收益，MSHR 幾乎無關，見本節結尾的訂正框。]**

單獨改動任何一項都不會有實質收益，理由列在下表：

| 改動 | 單獨效果 | 為什麼要兩個一起改 |
|---|---|---|
| mbus 加寬到 128-bit（beatBytes 8→16，理論 16 B/cycle @ 500 MHz） | **幾乎無效果**（審計結論：真正瓶頸是 L2 MSHR 佔用/bank 衝突，不是 mbus 頻寬本身——見 §5 對「DRAMSim2 冷流」誤判的修正） | 只要同一 bank 的 in-flight 請求數仍被 MSHR 卡住，加寬匯流排只是讓每筆請求「更快被送出」，送出後照樣在 MSHR 佇列裡排隊 |
| L2 MSHR 12 → 24 | **受限於 mbus 位寬**：MSHR 變多允許更多請求同時在飛，但如果 mbus 本身只有 8 B/cycle，同時在飛的請求只是搶同一條窄通道，總吞吐不變，只是延遲隱藏變好（對 tail latency 有幫助，對穩態頻寬幫助有限） | 同上，互為前提 |
| **兩者同時改** | 才可能讓穩態頻寬真正逼近 mbus 的名義峰值（8→16 B/cycle），因為此時「送得出去」與「同時能有多少筆在飛」都不再是瓶頸 | — |

出處：mbus 500 MHz、beatBytes=8 見 `out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:216-220`
與 `out/loop/FINAL_REPORT.md` §3(1)；L2 MSHR/bank 衝突是實際瓶頸（而非
DRAMSim2 的「冷流」）見 `out/loop/20260909-200541-dae1/agent_04.txt`
（stride-2048 下同一 bank 的 8/16 個請求堆積、MSHR 數不夠即串行化）與
`out/loop/20260909-052323-9d90/agent_01.txt`（模擬器從未帶 `+dramsim`，跑的是
預設 `mm_magic_t` 無時序 DRAM 模型，所以「DRAMSim2 冷流」這個說法本身不成立，
真正的計時來源是 rocket-chip 的 L2/mbus 模型，見本檔 §5 對 `FINAL_REPORT.md`
78–80 行的修正）。

> **【2026-09-19 訂正，第 5 條修正 — 本節整節被推翻】** 上表「只有兩者同時改
> 才有效」的預測是錯的，而且錯的方向具啟發性。`out/hw-sweep/` 對這四個設計點
> 做了直接、獨立的 RTL 量測（不是曲線推論）：**單獨加寬 mbus（8→16 B/cycle）
> 就拿到兩項合計增益的 98.6%**（n1 冷態 5.67→8.92，滿分 9.05 B/cycle；lmhead
> 冷態 6.61→8.87，滿分 8.89 B/cycle）；**單獨加深 L2 MSHR（12→24）幾乎無效**
> （n1 冷態僅 +6.5%，5.67→6.04 B/cycle；lmhead 冷態僅 +0.01%，634,507→634,580
> cycles）。機制：control 本身暖態量測 7.92 B/cycle 對 8 B/cycle 的 mbus 已是
> 99.0% roofline——mbus 早已飽和，根本沒有「MSHR 形狀」的餘裕可回收；本節據以
> 立論的 in-flight 請求數掃描，量到的其實是窄匯流排上的排隊壅塞，不是 MSHR
> 不足。**正確結論：decode 受限於 memory bus 寬度，L2 MSHR 在兩種寬度下都不是
> 瓶頸。** 加寬後兩個 kernel 收斂到同一個新天花板（8.87–9.05 B/cycle，新匯流排
> 的 56–57%），代表限制已經移到 Gemmini 的 StreamReader / L2 佔用而非匯流排
> 本身。詳見 `out/hw-sweep/README.md`（含四點量測與驗證方法）、
> `out/paper/hardware_sweep.csv`、`out/paper/methodology.md` Correction 5。

---

## 5. 改動檔案

- `loop/llama_project.py`：`project()` 新增 `gemv_bytes_per_cycle` /
  `lmhead_bytes_per_cycle` 參數；CLI 新增 `--gemv-bytes-per-cycle`、
  `--lmhead-bytes-per-cycle`、`--cold {upper,lower}`；`--clock-ghz`
  已存在（預設 1.0），未改動語意。預設行為（不帶新旗標）逐 bit 不變，
  `--self-test` 全數通過。
- `loop/llama_batch_project.py`：`run()` / CLI 新增同名旗標，內部把
  `gemv_bytes_per_cycle` / `lmhead_bytes_per_cycle` 轉傳進
  `llama_project.project()`，取代原本只用 warm tile 做比例外推的作法；
  B=1 欄位在帶 `--cold` 時與 `llama_project.py` 的獨立投影逐 cycle 一致
  （見 §2 的 cross-check）。`--clock-ghz` 已存在（預設 1.0），未改動語意。
- `out/llama-profile/projection_final_v2.md`（本檔，新增）。
- `out/loop/FINAL_REPORT.md`（更新，見「2026-09-12 審計修正」節）。
- `docs/aether-board-2026-09-13.html`（新增，`docs/aether-board-2026-09-11.html`
  的另存版本，round36 節末尾加審計修正小節）。
