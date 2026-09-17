# CHIA 內迴圈最終報告（round1 – round6）

> **結論一句話**：六輪、196 次 iteration、$149.09、38 小時 wall-clock 之後，
> 八顆 kernel 全部收斂到實際硬體上限（GEMV 已達 mbus 8 B/cycle 的 99%，**暖態**，
> 見下方「2026-09-12 審計修正」），
> **kernel 層沒有空間了**；decode 的唯一槓桿是 batch——
> N=1 **6.26 tok/s（暖態，見審計修正）** → N=16 53.96 tok/s（8.62x aggregate）。

- 資料來源：`out/loop/ledger.json` / `ledger.md`（權威成本與 iteration 數，
  由 `loop/ledger.py` 從 `loop/aether.db` 的 `iters` 表重算，不讀 `runs` rollup 欄）、
  `out/loop/iterations.md`（每次 iteration 的 agent note）、
  `out/llama-profile/projection_round{2,3,4,5}.md` 與
  `out/llama-profile/projection_final.md`（投影）。
- 時脈一律 1.00 GHz；roofline 一律用修正後的 **8 B/cycle** mbus。

---

## 1. 六輪摘要

| 輪次 | 日期（UTC） | wall-clock | kernel（runs） | iterations (ok/fail) | 成本 | 主要成果 |
|---|---|---:|---|---:|---:|---|
| **round1** | 2026-09-07 04:31 → 08:17 | 3h46m | n1-gemv, attn-scores, attn-pv, silu-mul, softmax（5） | 35 (33/2) | $20.6599 | 首次全面搜尋，一輪就吃掉大部分紅利：softmax 21,906→1,752 (12.5x)、attn-scores 51,280→8,574 (6.0x)、silu-mul 361,260→66,811 (5.4x)、attn-pv 22,856→4,853 (4.7x)、n1-gemv 150,339→132,653（已達 1.012x roofline） |
| **round2** | 2026-09-07 14:21 → 14:57 | 36m | 同上 + lmhead（6） | 17 (16/1) | $7.3727 | **被 Claude session-limit 429 中斷**，6 個 run 只有 4 個跑到 iteration。唯一新東西是首次引入 lmhead kernel（740,101→691,515）；silu-mul→50,256、softmax→1,630 |
| **round2b** | 2026-09-08 02:58 → 07:19 | 4h21m | 6 個 kernel 全部 `--seed best` 重啟 | 41 (39/2) | $31.2473 | round2 的續跑：n1-gemv→**132,424**（此後五輪未再被超越）、lmhead→659,142、attn-scores→**8,233**、silu-mul→35,774、softmax→1,568 |
| **round3** | 2026-09-08 20:24 → 09-09 09:58 | 13h34m | + q8-gemm（9 runs，含 2 次重啟） | 57 (54/3) | $61.8682 | 最貴也最關鍵的一輪：新增 prefill 的 q8-gemm 146,737→138,892；lmhead→635,908；attn-pv→**4,814**、silu-mul→**31,024**、softmax→**1,555**。兩顆 GEMV 被 429 打斷後人工重啟。輪末發現 **roofline 帶寬 bug（16→8 B/cycle）**，所有「距 roofline」比值重算 |
| **round4** | 2026-09-09 20:05 → 09-10 04:50 | 8h44m | lmhead(12), q8-gemm(6), n1(2 探針) | 26 (25/1) | $14.3770 | 第一輪完全在修正後 8 B/cycle 基準下量測與分析。lmhead→**634,507**（只快 1.002x，`LQ8_B_ROWS` 4→16 的假設沒兌現）、q8-gemm→**137,246**；n1 探針確認 132,424 不動。三個 run 全部 FINISHED，無中斷 |
| **round5** | 2026-09-11 06:56 → 11:19 | 4h22m | n16-gemv（1） | 11 (11/0) | $7.9986 | 攻 batched decode 的 N=16 GEMV tile（從未優化過）：168,367→**142,458**（1.182x 改善，1.087x roofline）。harness 從本輪開始每個 iteration 存 `simlog_NN.txt` |
| **round6** | 2026-09-11 11:58 → 14:20 | 2h21m | n16-gemv 續跑（1） | 6 (6/0) | $4.9184 | 4 次 iteration 全部退步（688,048 / 1,992,395 …），**best 維持 142,458，無新紀錄** → 判定 n16 亦已到頂，迴圈結束 |
| *(smoke)* | 2026-09-06 19:37 → 19:48 | 11m | softmax（1） | 3 (3/0) | $0.6494 | 迴圈本身的煙霧測試，softmax 21,906→4,312 |

**累計（含 smoke，以 `out/loop/ledger.json` 為準）：32 runs、196 iterations
（187 ok / 9 fail，成功率 95.4%）、$149.0915、37.97 小時 wall-clock。**
不含 smoke 的 round1–round6 為 31 runs / 193 iterations / $148.4421。
（與 `projection_round5.md` 的累計 $144.1731 一致：$144.1731 + round6 $4.9184 = $149.0915。）

---

## 2. 每顆 kernel 的最終結果

cycles 皆為單次 kernel 呼叫；roofline = `max(compute, memory)`，memory 用 8 B/cycle。

| kernel | baseline | final | roofline (floor) | baseline/final 倍數 | final/roofline | 累計成本 | 判定理由 |
|---|---:|---:|---:|---:|---:|---:|---|
| `llama-q8-gemv-gemmini-n1` | 150,339 | **132,424** | 131,072 | 1.135x | **1.010x** | $48.9947 | 實測 7.92 B/cycle = mbus 峰值 99%，round3 花 $29 重現同一數字、round4 探針再確認未動——純 DMA-bound，無空間 |
| `llama-q8-gemv-gemmini-n16` | 168,367 | **142,458** | 131,072 | 1.182x | 1.087x | $12.9170 | round5 一輪吃完紅利，round6 四次嘗試全退步、零新 best |
| `llama-q8-gemv-gemmini-lmhead` | 740,101 | **634,507** | 524,288 | 1.166x | 1.210x | $23.2817 | 那 1.21x 不是 kernel 差，是**冷 DRAM 只有 6.61 B/cycle vs 名義 8**（8/6.61 = 1.21 剛好對上）；round4 12 次 iteration 只再拿到 1.002x |
| `llama-q8-gemm` (prefill) | 146,737 | **137,246** | 131,072 | 1.069x | 1.047x | $12.2686 | compute-bound，兩輪優化後只剩 4.7% 差距，且不受 mbus 修正影響 |
| `llama-attn-scores-int8` | 51,280 | **8,233** | 6,144 | 6.229x | 1.340x | $16.6475 | round2b 之後 round3 四次 iteration 全部沒進步；Saturn 單一 in-order 向量算術管線，beat 模型已對齊 |
| `llama-attn-pv-int8` | 22,856 | **4,814** | 4,096 | 4.748x | 1.175x | $11.6115 | 同上，round3 只再擠出 39 cycles |
| `llama-silu-mul` | 361,260 | **31,024** | 30,720 | 11.645x | **1.010x** | $10.1468 | 距地板僅 1%，Cephes exp 鏈已完全 LMUL=4 管線化 |
| `llama-softmax` | 21,906 | **1,555** | 1,472 | 14.087x | 1.056x | $13.2237 | 距地板 5.6%，從 round3 起未再改善 |

整體：八顆 kernel 的**幾何平均加速 3.08x**（最好 14.1x，最差 1.07x），
且**七顆落在 roofline 的 1.01x–1.34x 之間**。

端到端投影（`out/llama-profile/projection_final.md`；**這整張表都是暖態，
偏樂觀，見下方「2026-09-12 審計修正」與 `out/llama-profile/projection_final_v2.md`
的冷態版本**）：

| 情境 | cycles/token | tok/s | roofline tok/s (8 B/cyc) | headroom |
|---|---:|---:|---:|---:|
| decode N=1 (S=512, **暖態**) | 159.73M | **6.260**（暖態，見審計修正） | 6.352 | **1.0x** |
| decode N=16 (aggregate，**暖態**) | 18.53M | 53.956 | 79.283 | 1.47x |
| decode N=32 (aggregate，估計，**暖態**) | 13.83M | 72.322 | 128.438 | 1.78x |
| prefill (N=64/pass) | 7.39M† | 135.312† | 209.620 | 1.5x |

冷態修正版（N=1，`--cold upper/lower`）：5.345 / 5.014 tok/s @1GHz，
2.673 / 2.507 tok/s @500MHz（RTL 實際時脈）；完整表見
`out/llama-profile/projection_final_v2.md` §2。

† 仍含未量測的 `lm_head_gemm` placeholder（`roofline × 4`）；apples-to-apples 約 157.2 tok/s。

---

## 3. 關鍵發現

**(1) Roofline 帶寬 bug：16 → 8 B/cycle（2026-09-09）。**
Gemmini 的權重/激活流量走 rocket-chip 的 mbus（`MemoryBusKey => MemoryBusParams(beatBytes = 8)`），
實體只有 8 B/cycle，不是先前假設的 16。修正讓所有 memory-bound roofline 加倍
（n1/n16 gemv 65,536→131,072、lmhead 262,144→524,288；q8-gemm 不變，它是 compute-bound），
也把「還有 2x 空間」的錯覺變成「已經貼著地板」。
出處：`out/loop/README.md` "Roofline correction (2026-09-09)"、
`out/llama-profile/projection_round3.md` 的「2026-09-09 roofline 修正」節。

**(2) 大權重全冷 DRAM 只有 6.6–7.4 B/cycle，連 8 都達不到——**
**【2026-09-12 審計修正】原因不是「DRAMSim2 冷流」，是 L2 MSHR/bank 衝突。**
lmhead tile 實測 **6.61 B/cycle**（`out/loop/20260909-200541-dae1/agent_01.txt`），
agent 當時自行推導出「冷流速率 r ≈ 6.43 B/cycle……634,507 就是它的底線；
8 B/cycle 的 mbus 名義帶寬在 DRAMSim2 冷流上不可達」（`.../agent_07.txt`），
**但這個「DRAMSim2 冷流」的說法本身站不住腳**：`out/loop/20260909-052323-9d90/agent_01.txt`
確認模擬器啟動指令從未帶 `+dramsim`／`+dramsim_ini_dir`
（`loop/nodes.py:156-168`、`chia/chipyard/verilator_run_node.py:513-519`），
跑的是預設的無時序 DRAM 模型 `mm_magic_t`，根本沒有 DRAMSim2 時序模型在跑，
所以不存在「DRAMSim2 冷流」這件事。真正卡住吞吐的是 **L2 的 MSHR 佔用與
bank 衝突**：`out/loop/20260909-200541-dae1/agent_04.txt` 指出 stride-2048 的
權重流下，同一條 mvin 命令的 8/16 個請求全部落在同一個（≥512 B 交織的）L2
bank 上，超過該 bank 的 MSHR 數就必須串行化——這才是「4 行×64 B 是唯一可行
形狀」「634,507 是底線」的物理原因，與 DRAMSim2 完全無關。這解釋了 lmhead
為何卡在 1.21x「roofline」而其實已到頂；詳見
`out/llama-profile/projection_final_v2.md` §4。
相對地，1 MiB 的 n1 tile 因為部分命中 L2，跑到 **7.92 B/cycle（峰值 99%）**
（`projection_round4.md` (d)）——但那個 7.92 B/cycle 本身也混了約 200 KiB 的
harness 暖 L2 尾巴，冷態升序實測是 **149,757 cyc = 7.00 B/cycle**，見下方
「2026-09-12 審計修正」一節。

**(3) LoadController `nCmds = 2` 與 XactTracker 的 16 個槽。**
Gemmini 的 `StreamReader` 固定 `max_in_flight_mem_reqs = 16`
（`out/loop/20260908-025820-f7c4/agent_06.txt`），而 LoadController 的
`nCmds = max_in_flight_mem_reqs/DIM + 1 = 2`
（`out/loop/20260909-052323-9d90/agent_01.txt`）——**同時只有 2 條 mvin 命令能展開**。
這是整個 DMA 行為的主因：`out/loop/20260911-065653-8115/agent_05.txt` 指出
「LoadController 只允许 2 条命令同时打开」，1 行的 mvin 因此退化到約 **1 B/cycle**。

**(4) rows/cmd 曲線：每條 mvin 搬幾行才划算。**
實測（lmhead，stride-2048 權重流）：**16 行 695k、8 行 690k、chunk 置換 −1.4k、
cols>64 觸發硬體 assert、mvout 交錯 +400、rows=1/2 受主機發射率限制**，結論是
「4 行 × 64 B 是 stride-2048 權重流唯一可行點，634,507 就是底線」
（`out/loop/20260909-200541-dae1/agent_07.txt`）。
在飛請求數曲線另一組點：6→649k、8→632k、16→687k（`.../agent_12.txt`）——
**非單調**，太多在飛請求反而讓 DRAM row buffer 打架。

**(5) warm L2 的「尾巴」。**
同一份權重的第二趟會明顯變快，且 harness 留下的髒行狀態會決定尾段成本：
「(d) 幾乎逐 loop 復現了 (s)（141,213 vs 142,892；5.85k→10.7k 的同一條曲線）……
那 ≈9k 的尾段懲罰完全由 harness 留下的『主機寫入產生的髒行』狀態決定」
（`out/loop/20260911-115840-5f95/agent_04.txt`）。
這也是 round6 四次嘗試全退步的原因之一：它在調的已經是量測雜訊層級的東西。

**(6) 主機 RoCC 命令發射成本 ≈ 10.7 cycle/命令（不是原先假設的 20–52）。**
探針解碼：`822,414 − ~646.5k ≈ 176k / 16,384 條 ≈ **10.7 周期/命令**`
（`out/loop/20260909-200541-dae1/agent_12.txt`，該檔明寫「教訓：主機發射成本約
10–11 周期而非 20」）。在 lmhead 這種 ~33,000 條 RoCC 命令的 kernel 裡，
636k 周期內每 19 周期一條，**已經貼著主機發射率**（`.../agent_01.txt`）。

**(7) Saturn 分攤（co-execution）不可行。**
試過讓 Gemmini 串流 896 KiB、Saturn 用 unit-stride RVV 從 L2 讀剩下的 128 KiB 暖尾巴：
結果「**比 Gemmini 自己串流整個 1 MiB 還慢**（132,653）……在這顆 SoC 上，
主機的記憶體流量——即使是 L2 *命中*——付出的代價大約等於它省下的」
（`out/loop/20260908-025820-f7c4/agent_04.txt`）；
`out/loop/20260907-043120-a7d0/agent_08.txt` 下同樣結論：
「Gemmini 與 core 共用同一條 tile-egress/L2/DRAM 路徑，掃描期間**任何**主機記憶體流量都是淨負」。
`out/coexist/` 只證明 coexist config 能 build/cosim 乾淨（mismatches=0），
不含吞吐效益數據——「不可行」這個結論來自上面兩個 loop 探針。

**(8) harness printf 修正（2026-09-11，round5 之前落地）。**
問題：`printf` 走 HTIF 成本高到會吃掉整個量測，探針 iteration 只能刻意放棄分數
（「被計分的 `Cycles taken` 會因兩次運行 + flush + printf 而達到 ~600k，這是有意犧牲的」，
`out/loop/20260909-200611-94c0/agent_01.txt`；同 run `agent_02.txt`：
「實測 768,849 與 EXPECTED 差 ~170k……printf 經 HTIF 更貴」），
而探針印出來的數字又不會進 DB，跑完就消失。

**修正內容**（三處，round5 launch 前落地）：
1. `loop/loop.py` 每個 iteration 把模擬器 log 另存為
   `out/loop/<run_id>/simlog_NN.txt`（探針輸出因此永久留存，不再只活在對話裡）；
2. `loop/llm.py` 新增 `_benchmark_stdout_tail()`（第 212 行，由第 284 行呼叫），
   讓**通過的** iteration 的 feedback 也附上
   `=== benchmark stdout (tail) ===` 區塊（帶 `PROBE` 前綴行）——
   在此之前只有失敗的 iteration 看得到 stdout；
3. 測試 `loop/tests/test_llm_feedback.py`（4 passed）。

**效果對照**：round5 的 run `20260911-065653-8115`，`agent_04/05/09/10.txt`
可以直接引用 simlog 的 `PROBE` 數字（例如 `T2−T3 = 11,207`、`A2048_cold 5,593`），
數字有結構化落地；round4 以前的探針則全部遺失——例如
`20260908-202413-e8e8` 的 iter 1、`20260909-200541-dae1` 的 iter 2/7/8/11，
DB 只留 `fitness`，原始數字只能事後 grep `agent_NN.txt` 的敘述撈回。
`out/loop/iterations.md` 多處（218/222/234 行）在解讀結果時仍明確扣掉 "− printf"。

---

## 4. 失敗與教訓

| # | 事件 | 影響 | 教訓 |
|---|---|---|---|
| 1 | **Claude session-limit 429 中斷** | round2 六個 run 開跑 36 分鐘就全被打斷（17 iterations，$7.37 幾乎浪費）；round3 的 n1（4 iters 後）與 lmhead（3 iters 後）各被打斷一次 | 靠 `--seed best` 重啟可以無損續跑（round2b / round3 的兩個 restart run 都是這樣救回來的），但**必須把 seed 機制當成一級公民設計**，而不是事後補救 |
| 2 | **watchdog 時區誤判** | round3 的 watchdog 把 API 回的 "resets 5pm America/Los_Angeles" 當成本地時間解析，自動重啟時間全錯，最後由人工在 2026-09-09T13:23+08:00 手動重啟兩個 run | 任何解析外部時間字串的自動化，時區必須顯式化；`rounds.json` 的 round3 note 有完整記錄 |
| 3 | **根碟塞爆與搬遷** | Ray 的 temp/spill 目錄放在本機碟上把根碟塞爆，2026-09-09 緊急搬到 `/share1`（`out/loop/migrate-ray-to-share1.sh`，另有十餘份 `out/chia-up-*migrate*.log`） | 長時間跑 Verilator + Ray 的專案，spill 目錄一開始就要指到大容量共享碟 |
| 4 | **探針數字遺失（round4 以前）** | 多次探針 iteration 的關鍵數字只活在 `agent_NN.txt` 的敘述裡（例如 rows/cmd 曲線的 6/8/16 三點、10.7 cycle/命令的解碼），DB 的 `iters` 只存 `fitness`，探針因 printf 污染還被記成「退步」；`20260908-202413-e8e8` iter 1、`20260909-200541-dae1` iter 2/7/8/11 皆如此 | **已於 2026-09-11 round5 前修正**（見關鍵發現 (8)）：`loop/loop.py` 存 `simlog_NN.txt`、`loop/llm.py` 的 `_benchmark_stdout_tail` 讓 passing iteration 也看得到 `PROBE` 輸出。round4 以前的數字只能靠事後 grep `agent_*.txt` 撈回（本報告即是） |
| 5 | **iters 表沒有逐 iteration timestamp** | `ledger.md` 的 per-iteration duration 只能用 kernel 檔的 mtime 反推，目錄被 touch/複製就失真 | 已在 `ledger.md` 開頭列為已知限制 |
| 6 | **成本分布極不均** | round3 一輪就燒掉 $61.87（41.5% 的總成本）卻只換到 1.0003x 的 decode 端到端改善；n1-gemv 一顆 kernel 累計 $48.99 而它從 round2b 之後就再也沒動過 | **收斂偵測應該是自動的**：連續 N 次 iteration 無新 best 就該停。round4 的「2-iteration 探針」策略（$1.83 確認一顆 kernel 沒動）才是正確做法 |

---

## 5. 系統層結論：batch 是唯一剩下的槓桿

完整分析見 `out/llama-profile/projection_final.md`，工具為新增的
`loop/llama_batch_project.py`（`llama_project.py` 沒有 batch-decode 模式）。

**(a) decode 為什麼被 mbus 綁死。** 每個 token 就是要把整個模型的 int8 權重
搬過 mbus 一次：**1,235,746,816 B 權重（總共 1.247 GB read）、零重用**。
1.236 GB ÷ 8 B/cycle ≈ 154.5M cycles 就是地板；`llama_project.py` 報
`memory-only 156.56M cyc/tok`，**20 個 operator 有 18 個在 roofline 上是 memory-bound**。
實測 159.73M 對 157.43M roofline = **headroom 1.0x**：就算把每顆 kernel 都打到
理論地板，decode 也只從 6.260（暖態，見審計修正）→ 6.352 tok/s（+1.5%）。

**(b) batch 為什麼有效。** 增加 batch 不會多搬一個 byte 的權重（B 條序列共用同一次
權重串流），只多搬 B 份 activation。實測：從 N=1 到 N=16，GEMV tile 只多花
**7.6% 的 cycle（132,424 → 142,458）卻做了 16 倍的算術**。

**下表為暖態，見「2026-09-12 審計修正」與 `projection_final_v2.md` 的冷態版本
（N=1 冷態：5.345 / 5.014 tok/s @1GHz，上/下界）。**

| batch | step cycles | cycles/token | **tok/s aggregate** | **tok/s per-seq** | roofline agg | GEMV 佔比 |
|---:|---:|---:|---:|---:|---:|---:|
| **1**（暖態） | 159,733,828 | 159,733,828 | **6.260**（暖態，見審計修正） | 6.260 | 6.352 | 94.8% |
| 4\* | 187,095,090 | 46,773,773 | 21.380 | 5.345 | 24.052 | 82.1% |
| **16** | 296,540,142 | 18,533,759 | **53.956** | 3.372 | 79.283 | 54.9% |
| **32\*** | 442,466,877 | 13,827,090 | **72.322** | 2.260 | 128.438 | 39.6% |
| 64\* | 734,320,348 | 11,473,755 | 87.155 | 1.362 | 133.474 | 27.2% |

\* GEMV tile cycles 為 N=1/N=16 兩點的線性外推（斜率 668.9 cyc/column），**估計值**。

**(c) break-even。** B≤4 幾乎免費（單序列只慢 5–15%，吞吐 3.4 倍）；
**B=16 是甜蜜點**（8.62x 吞吐、單序列砍半，且是唯一有實測 kernel 的 batch 點）；
B=32 之後邊際收益跌到 +34% / +21%，且 KV cache 已達 16.8 MB。
若單序列需 ≥5 tok/s，最大 batch 約 4–5；需 ≥3 tok/s 可到 16–20。

**(d) batch 之後瓶頸會換人。** B=1 時 GEMV 佔 94.8%，B=16 時只剩 54.9%，
`attn_scores + attn_pv + softmax` 反而升到 **40.3%**——而這三顆分別是 roofline 的
**4.02x / 2.35x / 3.04x**，和 N=1 的 GEMV（1.01x）完全不同，**是真的還有空間**。
如果還有下一輪 loop，該做的是：(1) 實測 batched attention（把 B 條序列的 Q 併成
GEMM 打 Gemmini，可能遠優於本投影保守的 ×B 假設）、(2) 補上從未量測的
`lm_head_gemm`（prefill 最大未知數）、(3) 實測 N=32 GEMV tile 以取代線性外推。

---

## 6. 檔案索引

| 內容 | 路徑 |
|---|---|
| 權威成本 / iteration 帳本 | `out/loop/ledger.md`、`out/loop/ledger.json`（`loop/ledger.py` 產生） |
| 每次 iteration 的 agent note | `out/loop/iterations.md`、`iterations.json`（`loop/journal.py` 產生） |
| 輪次定義 | `out/loop/rounds.json` |
| 逐輪投影 | `out/llama-profile/projection_round{2,3,4,5}.md` |
| **最終投影（batch）** | `out/llama-profile/projection_final.md`、`proj_final_*.txt` |
| **batch 投影工具（新增）** | `loop/llama_batch_project.py` |
| 每個 run 的原始對話與 kernel | `out/loop/<run_id>/agent_NN.txt`、`kernel_NN.{c,h}`、`simlog_NN.txt`（round5 起） |
| **審計修正後投影（新增）** | `out/llama-profile/projection_final_v2.md` |

---

## 7. 2026-09-12 審計修正

`projection_final.md` 與本報告正文（round1–6，2026-09-07~11 完成）的
6.26 tok/s（N=1 decode）系統性偏樂觀。完整推導與新投影表見
`out/llama-profile/projection_final_v2.md`；本節只列修正要點。

**四項修正：**

1. **n1 GEMV 的「暖」量測含約 200 KiB harness 暖 L2 尾巴。** 132,424 cyc
   （round2b 至今的 best，即 §2 表中的 `llama-q8-gemv-gemmini-n1`）不是純冷
   態數字；冷態升序實測 **149,757 cyc = 7.00 B/cycle**（上界）。
2. **lm_head 原本的 n1-basis 外推得到 9.22 B/cycle，超過 mbus 8 B/cycle 的
   物理上限，不可能發生。** 改用 lm_head 自己的 4 MiB tile 量測：
   634,507 cyc = **6.61 B/cycle**，直接乘上完整 128,256×2,048 形狀。
3. **agent_07 對 stride-2048 權重流做過曲線擬合，給出冷流速率下界
   r ≈ 6.43 B/cycle**（`out/loop/20260909-200541-dae1/agent_07.txt`），
   之前沒有被拿來當投影參數用。
4. **時脈假設 1 GHz 從未驗證；RTL elaborate 顯示 bus 全部跑在 500 MHz**
   （`out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:216-220`：
   sbus/pbus/fbus/mbus/cbus 全部 "frequency of 500.0"）。本報告與
   `projection_final.md` 只列了 1 GHz 一欄。

**工具改動：** `loop/llama_project.py`、`loop/llama_batch_project.py` 新增
`--gemv-bytes-per-cycle` / `--lmhead-bytes-per-cycle` / `--cold {upper,lower}`，
讓 GEMV/lm_head 直接用冷態 B/cycle costing，取代原本的 tile 線性外推；
`--clock-ghz`（預設 1.0）兩支工具原本就有。不加新旗標時輸出與修正前逐 bit
相同（`--self-test` 全通過）。

**新投影（N=1，@1GHz / @500MHz）：**

| | cycles/token | tok/s @1GHz | tok/s @500MHz |
|---|---:|---:|---:|
| 暖態舊值 | 159.73M | 6.260 | 3.130 |
| 冷態上界（GEMV 7.00, lm_head 6.61 B/c） | 187.10M | 5.345 | 2.673 |
| 冷態下界（GEMV 6.43, lm_head 6.61 B/c） | 199.43M | 5.014 | 2.507 |

N=16 / N=32 / prefill 的完整表（同樣兩個時脈欄位）在
`projection_final_v2.md` §2。**§3(2) 的「DRAMSim2 冷流」說法已確認錯誤並
在原文就地修正**：模擬器從未帶 `+dramsim`，跑的是無時序的 `mm_magic_t`，
真正瓶頸是 L2 的 MSHR 佔用與 bank 衝突（`out/loop/20260909-200541-dae1/agent_04.txt`）。

**kernel 層剩餘空間（審計結論，供下一輪參考，非新測）：** Gemmini/Saturn
重疊執行 +4–5%、Saturn 四顆已近 roofline 的 kernel（attn_scores/attn_pv/
softmax/silu_mul）打 2x 架構改動 +2%、in-flight 請求數匹配 L2 MSHR 數
0–3%——合計 ≈ +6–10%，遠小於 batch 的 +862%（N=1→N16）。**硬體突破**只有
mbus 加寬到 128-bit **同時** L2 MSHR 12→24 才有效，單獨改任一項都會被另一項
卡住（推導見 `projection_final_v2.md` §4）。

**下一步（Round 7 建議）：** 層級融合 kernel（把 Gemmini 權重串流與 Saturn
attn/softmax/norm 的純算術部分疊在一起跑，而非序列相加）+ in-flight 請求數
探針（掃 MSHR 數附近的甜蜜點，取代目前非單調、只能事後解讀的 rows/cmd 曲線）。
