# Plan A — 誠實的時間線（已執行完畢，2026-09-23）

> **文件狀態。** 這份文件是累加寫成的：前段是 2026-09-22 擬定的原始計畫，
> 後段是 09-23 逐項稽核後的更正。**更正一律覆蓋前段**，順序為
> 「已確認的量測結果」→「命名決定」→「tex 盤點」→「DB 盤點的修正」→「汙染稽核」。
> 重算已於 09-23 完成，本文所有數字均已換成 **168 traces / 40M 視窗** 的實測值。
> 尚未完成的工作集中在文末「剩餘工作」一節。
> ablation arm 一節（`design_noref.md`）**已放棄**，理由見該節開頭。

## Context

目前 `paper/vfs_report.tex` 的實證主體只有 **參數搜尋那一條**（rule-based mutation，`out_gen50`），
結論停在「VFS 的吞吐量主導是一個 regime，不是度量的本質」，並在 §6 把
「an LLM that proposes designs 能不能離開那個 flank」列為 **future work**（第 433–438 行）。

從那之後，design arm（`out_design_llm`，`arm=main`，LLM 直接寫 HARCOM source）
已經跑完 gens 11–34，並且**確實離開了 flank**：0.9256 → 0.9895，7 cells → 18 cells。
也就是說，論文自己提出的 future work 已經有答案了。

先前討論過「分成三到四條平行 arm」的寫法。那個寫法不成立，因為 design arm 的
gen 15（加入得獎者資料的那一代）接的是 gen-14 archive，而 gen-14 archive 是
**同一條 design arm 在沒有得獎資料時做出來的**（best 0.9838、latency 1/1、T 8.97，
已經跨出 flank）。把 gen15 改標成 gen11 會讓「有資訊」那條看起來一開始就在帶子裡，
掩蓋掉整篇論文要量的那個效果。

Plan A 改用**單一時間線**：同一個 gen-10 樹幹分出兩支，其中一支中途換了 prompt。
所有數字都是現有資料，沒有混淆，而且敘事比平行 arm 更強——
資訊是在 agent 已經跨出 flank **之後**才加入的，加入之後二十代只再前進 0.0057。

但單靠時間線有一個必須自認的弱點：prompt 在 gen 15 換過，所以階段 3 與階段 4
的差異分不開「得獎資料的效果」和「邊際報酬遞減」。
因此本計畫在時間線之外**加跑一條 ablation arm**（下方「Ablation arm」一節）：
從**同一份** gen-14 archive 分岔，用無得獎資料的 prompt 跑 gens 15–25，
與既有的有資料分支正面對照。這同時修掉論文 §6 自己承認的限制
（"compares two archives observationally rather than through an ablation"，第 444 行）。

### 一個必須先釐清的命名錯誤

論文和先前的討論把 `out_gen50` 叫作 "params arm"，但 DB 裡它的 `arm` 欄位是 **`offline`**
（rule-based 參數突變）。真正 `arm=params` 的是 `out_params/`：另一條**獨立**的 126-trace
run，從 `gen000_seed` 從頭開始，只跑到 gen 14（10 cells, best 0.9227），
**不是從 gen-10 樹幹分出來的**，因此不能當成對照組。

本計畫一律用 DB 的名字：**offline arm**（`out_gen50`）與 **design arm**（`out_design_llm`）。
`out_params` 不進論文，或只當作 seeding 的附註。

---

## 論文的新骨幹：四個階段，同一個樹幹

全部取自 `archive_snapshots` 表（`db.py:94`，每代一列 JSON blob）。

**以下為重算後的最終數字**（168 traces / 40M；offline 那一列是 126 traces，
以 bridge 換算，理由見「Bridge」一節）。

| 階段 | 來源 | 世代 | cells | best VFS | best variant | prompt |
|---|---|---|---|---|---|---|
| 1 樹幹 | `out_gen50` (offline) | 1–10 | 7 | **0.9200** | gen007_2 | — |
| 2 parameter 分支 | `out_gen50` | 11–39 | 7（29 代不變，且**與樹幹同格**） | 0.9288（gen 16 達成，其後 23 代僅 +5.8e-5） | gen016_2 → gen037_2 | — |
| 3 design 分支，**無**得獎資料 | `out_design_llm` | 11–14 | 7 → 14 | **0.9817** | gen014_0 | `design.md` @ 9a8c672 |
| 4 design 分支，**有**得獎資料 | `out_design_llm` | 15–34 | 14 → 18 | **0.9895** | gen029_1 | + `prompts/cbpng_winners/` (2bb9128) |

重算**沒有改變任何一格**：三份 archive 的 cell 數、最佳設計、排序全部與重算前相同。

階段 3 結束時（gen-14 archive，14 格）的實際內容，重算後：

```
cell       variant     gen   VFS      T       P1/P2
(2,1,1)    gen014_0    14    0.9817   8.858   1/1   ← clean prompt 下的最佳
(3,1,1)    gen014_2    14    0.9770   8.897   1/1
(4,1,1)    gen012_0    12    0.9723   9.245   1/1
(1,1,1)    gen012_1    12    0.9613   8.582   1/1
(1,0,0)    gen014_1    14    0.9576   8.170   0/0
(0,1,1)    gen013_1    13    0.9218   8.335   1/1
(2,3,6)    gen011_0    11    0.0798   0.325   28/28  ← 退化，佔住無法觸及的格
--- 以下七格是樹幹原封不動繼承的 ---
(4,1,2)    gen007_2     7    0.9200   6.800   1/2
(4,1,3)    gen009_0     9    0.8481   5.594   1/3
(5,1,2)    gen006_2     6    0.7822   5.825   1/2
(5,1,3)    gen010_2    10    0.7388   4.309   1/3
(4,2,2)    gen001_0     1    0.7011   3.786   2/2
(5,2,2)    gen003_0     3    0.6985   3.790   2/2
(5,2,3)    gen003_2     3    0.6644   3.611   2/3
```

樹幹的七個 elite 全部 P2 ≥ 2、T 3.6–6.8，也就是**全部躺在 flank 上**，
而且到 gen 34 為止一個都沒被擠掉。design arm 在四代內新增的六格**全部** P2 ≤ 1、
T ≥ 8.17——它不是把樹幹推上去，而是在樹幹旁邊另外開了一片。

階段 4 的停滯（`archive_snapshots`）：gen 19 起 cells 卡在 18，
best 0.9874 → 0.9880 (gen 24) → 0.9895 (gen 29) → gen 34 無變化。

**關鍵 prompt 時間點（已由 git 證實）**：`prompts/design.md` 的初版（9a8c672, 2026-09-09）
完全沒有得獎者資料，而且方向是錯的（"spending P2 latency to buy accuracy — the least
explored part of the space"）。得獎者資料在 2bb9128（2026-09-19）才進 prompt，
對應 sweep 3（2026-09-18T21:08），也就是 gen 15。

---

## 量測債：必須重跑的部分  ✅ 已完成（2026-09-23）

> **結果摘要。** 實際執行的是下文「四份 archive」的縮減版（Option D）：
> design 側重算到 168/40M，offline 側留在 126 traces 用 bridge 換算。
> 去重後 **16 份 source × 168 traces = 2688 次 CBP-NG run**，耗時 **5 小時 32 分**
> （非原估的 3–5 小時，因為 3 路並行而非 16 路）。
> 輸出在 `bp_evolve_data/rescore/out168_40m/`（每份含 per-trace counters，
> 之後做 depth sweep 或子集分析都不必再模擬），
> 重建的 archive 在 `bp_evolve_data/rescore/archives168_40m/`。
> **沒有任何 elite 換格、碰撞或遺失。**

重算前，數字橫跨三種量測條件，不能放在同一張表：

| 資料 | traces | 視窗 |
|---|---|---|
| offline arm 全部（含樹幹） | 126 | run-to-end (`BPE_SIM_INSTRUCTIONS` 預設 2e9) |
| design arm gens ≤ 18 | 168 | run-to-end |
| design arm gens 19+ | 168 | 40M |

（sweep 3 在 gen 14 resume 時已把 archive 重算到 168 traces——
gen007_2 從 0.9256 掉到 0.9185 就是這次重算的結果——但換 40M 視窗時
**沒有重算**，見 `launch_design_local.sh:97` 的 notes 字串。）

要統一到 **168 traces + 40M 視窗**，需要重算四份 archive：

1. 樹幹：`resume_gen10_archive.json`（7 elites）
2. offline gen-39：`out_gen50/bp_evolve.db`，`sweep_id=4, generation=39`（7 elites，只存在 SQLite 裡）
3. design gen-14：`out_design_llm/bp_evolve.db`，`sweep_id=2, generation=14`（14 elites）
4. design gen-34：`out_design_llm/bp_evolve.db`，`sweep_id=17, generation=34`（18 elites，其中 13 個是 run-to-end）

去重後約 28 個不同的 HARCOM source × 168 traces ≈ 4700 次 CBP-NG run @ 40M，
`cbp_ng` pool 16 槽，估 **3–5 小時**。純 CPU，不佔 Claude session。

重算會讓 elite 換 cell 或互相碰撞，cell 數可能變動——這本身就是要報告的結果，不是錯誤。

**實際執行時的兩處偏離：**

1. 第 2 項（offline gen-39）**沒有重算**。它的七個 elite 全是樹幹參數的後代，
   重算它等於把 design arm 已經在 168 下量好的數字反過來拉回舊的小樣本。
   改用 bridge 換算，並在論文中標明其量測條件。
2. design arm gens 19+ 的 elite（gen028_2、gen029_1、gen031_0/1/2、gen034_1）
   **不需要重算**——DB 的 sweep notes 明確寫著
   `40M from gen 19, run-to-end before, archive not re-scored`，
   它們原本就在目標條件下量測。`rebin.py` 以 `NATIVE_FROM_GEN = 19` 放行這六個，
   其餘一律要求有重算結果，缺了就拒絕寫檔。

---

## 程式改動  ✅ 已完成

> `--rescore` 已加入 `bp_evolve_loop.py`（`rescore()` 在 :1245 附近）。
> **但最後沒有用它跑這次重算**：`_rescore_archive` 會在 cell 碰撞時丟掉落敗者的
> metrics，而離線重新 binning 正需要那些數字；它也會等所有 future 收齊才輸出，
> 中途中斷就全沒了。改用 scratchpad 的 `rescore_sources.py`：
> 直接對 **source 去重**後逐份評分、逐份原子寫檔（`.tmp` + `os.replace`）、可續跑，
> 再由 `rebin.py` 離線重建 archive。兩支腳本都不進 repo。

只有一處，約 30 行。

**`bp_evolve_loop.py`** — 加一個獨立的 `--rescore` 入口。

現成可重用的零件：
- `_rescore_archive(archive, traces, cbp_node, args) -> Archive`（`bp_evolve_loop.py:586`）
  已經做完全部的事：從 `elite.source` 重建 `Variant`、build、`run_fn_tier0`、
  透過 `result_mapper_fn`（`evaluator.py:813`）重算 EPI/P1/P2/VFS，
  並用 `Archive.descriptor()`（`archive.py:83`）**重新推導 cell**。回傳新的 `Archive`，
  不改動輸入、不寫檔。
- `score_held_out`（`bp_evolve_loop.py:1245`）是現成的 standalone 模式樣板：
  從 `--archive` 讀檔、起 cbp node、跑任意 trace list、印表格。

改動內容：
- 新增 argparse flag `--rescore PATH`（輸出路徑），沿用既有的 `--archive`（`:1512`）當輸入。
- 在 `main()` 裡仿照 `score_held_out` 的分支：載入 archive（`Archive.from_json`, `archive.py:185`）、
  建 cbp node、呼叫 `_rescore_archive(archive, inner, cbp_node, args)`、
  把 `fresh.to_json()` 寫到輸出路徑。
- 不要碰 `_rescore_archive` 本身，也不要碰它在 `:353` 的 resume 呼叫點。

執行環境（視窗只能用環境變數，沒有 flag——`constants.py:114`）：
```
BPE_SIM_INSTRUCTIONS=40000000 BPE_HELD_OUT_FRACTION=0 \
  python -u bp_evolve_loop.py --rescore OUT.json --archive IN.json --inner-traces 168 ...
```

另外需要一個小的一次性腳本（放 scratchpad，不進 repo）把
`archive_snapshots.archive_json` 從 SQLite 撈出來寫成 `.json`。

---

## 論文改動（`paper/vfs_report.tex`，513 行）

### A. 結構

| 動作 | 位置 |
|---|---|
| 摘要重寫：從「這是一份 audit」改成「audit + 它自己提出的測試的答案」 | 40–60 |
| Introduction 的三點貢獻改成四點，第四點是 design arm 跨出 flank | 62–99 |
| §2 The Design Loop：現在描述的是單一 loop，要說明樹幹 + 兩個分支，以及 prompt 在 gen 15 改過 | 101–147 |
| **新增一節「Can an Agent Leave the Flank?」**，放在 §4（evaluators）之後、§5（bounds）之前；主體是上面那張四階段表 | 新增於 ~368 |
| §6 Discussion：433–438 的 future work 段落改寫成回指新那一節 | 433–438 |
| §7 Conclusion 加一句：離開 flank 需要的是演算法改變，不是搜尋預算 | 447–457 |

### B. 已經錯的句子（必須改，不是選擇性的）

| 行 | 現況 | 問題 |
|---|---|---|
| 441 | "our best score (0.9288) remains below Pallan's and every winner's; this paper is an audit, not an entry" | 已不成立（0.9895 高於 Pallan 的 0.9345，接近 Dang 0.9742 以上） |
| 443–444 | "Only $2\times2\times2$ of the behavior grid's 144 cells are reachable" | 被 18 cells 推翻 |
| 47, 145, 258, 277 | "116" 設計 | 應為重算後的實際評估數 |
| 143 | "40 generations of three variants each" | 與新的世代結構不符 |
| 289–291 | 100 (H≥6) + 16 (H≤5) = 116 的分割 | 必須重新推導 |
| 285, 324, 330, 389 | "seven" elites | 要改成分階段的 cell 數 |
| 326 | "over 21 pairs" = C(7,2) | 隨 elite 數改變 |
| 280 | gen037_2 稱為 "the best design" | 只是 offline arm 的最佳 |

### C. 表與圖

- **Table I `tab:elast`（185–208）**：新增三列 `gen011_2` / `gen014_0` / `gen029_1`。
  彈性用已驗證的方法算（對 `vfs.vfs(ipc, cpi, epi)` 取有限差分；
  此法在 gen037_2 上完全重現 0.9288 / +0.278 / −0.135 / −0.024）。
  已算出的值：gen029_1 VFS 0.9895、T 8.73、e_T **+0.013**；
  對照 MORSL −0.008、Fan −0.015、offline best **+0.278**。
  重算後要重新產生。
- **Fig. 2 `fig:tcurve`（210–244）**：全部座標 inline，無外部資料檔。
  - 229–230「ours」加入 design arm 的點（T ≈ 8.7–9.0 區），並區分 offline / design 兩種 marker。
  - 236–237 兩條 T=3.61 / T=6.87 的灰虛線目前用來框住搜尋範圍，必須延伸或改成兩組括號
    （offline 的範圍 vs design 的範圍）——這張圖現在講的就是主結果。
- **Table II `tab:tau`（329–347）**：目前兩欄是「gen 10 archive」與「gen 39 archive」，
  後者其實是 offline arm。重新標示，並考慮加第三欄 design gen-34 archive。
- **Table III `tab:bounds`（404–417）**：bounds arm，不受影響。
- **§4.1（300–318）**：40 promoted / 39 gem5 designs 都是 offline arm 的 Tier 1/2，加上標籤即可。

### D. 必須揭露的事項（新增到 §6 Limitations）

1. **prompt 中途改過**：得獎者資料在 gen 15 進入 prompt，所以階段 3 與階段 4
   不是隨機分配的對照，而是同一條 lineage 的前後兩段。這是 Plan A 的核心限制，要寫清楚。
2. **跨研究比較只是示意**：`prompts/design.md:135` 已註明得獎者的絕對 VFS 來自不同的
   trace sample，"do not compare directly"。Fig. 2 中我們的點與得獎者的點不是正面比較。
3. **scratch 目錄洩漏造成的近似重複**（2026-09-23 稽核修正，見文末「汙染稽核」節）：
   三組為**近似**重複而非完全相同；其中 gen022_2 **進了 archive**，
   把自己的雙胞胎 gen021_0 擠出 cell (3,1,1)。
4. **量測條件**：說明重算後全部統一到 168 traces / 40M，以及重算前的混合狀態。

---

## Ablation arm：從 gen-14 分岔的無得獎資料分支  ❌ 已放棄

> **放棄理由（2026-09-23）。** 這條 arm 需要 24–36 小時的 LLM 世代，
> 而 `llm` pool 只有一槽；重算本身又用掉了 5.5 小時。
> 距 9/24 AOE（= 2026-09-25 19:59 台北）已不足以安全完成，
> 一旦中途失敗就沒有第二次機會。
>
> **更重要的是它不再必要。** 原本設計它是為了證明「跨出 flank 與得獎者資料無關」，
> 但 gens 11–14 本來就是在 clean prompt 下跑的，而重算確認
> `gen014_0` 在**完全沒看過得獎者資料**的情況下就已經達到 P1/P2 = 1/1、VFS 0.9817，
> 高於 parameter arm 跑完 39 代的 0.9288。這個反事實不需要新實驗。
>
> 代價是 §6 的 "observationally rather than through an ablation" 必須留著。
> 這是誠實的限制陳述，不是缺陷。以下設計保留給審稿人要求時使用。

### 設計

| | 既有分支（已跑完） | 新分支 |
|---|---|---|
| 起點 | gen-14 archive（14 cells, best gen014_0 0.9838） | **同一份**，重算到 168/40M 之後 |
| 世代 | 15–25 | 15–25 |
| prompt | 完整 `design.md`（301 行，含得獎資料） | `design_noref.md`（約 190 行） |
| 結果 | 18 cells, best gen024_1 0.9880 | ? |

每代 3 個變體、平行度 3、168 traces、`BPE_SIM_INSTRUCTIONS=40000000`。
11 代 × 3 = **33 個 LLM session**，估 24–36 小時 wall clock。

### `design_noref.md` 怎麼做

**不能**直接用 `prompts/design.md@9a8c672`。**注意（2026-09-24 更正）：
那個 commit 不是 gens 11–14 當時的版本**——commit 日期是 9/19 16:54，而
design arm 是 9/18 跑的；commit 日期無法證明更早那次執行用了什麼 prompt。
實際送出的 prompt 要看 transcript（見文末「已明確排除」的理由更正）：971 行，
含得獎者整節。`design_noref.md` 必須從那份**實際** prompt 刪掉得獎者一節與
`gshareN_ahead`，而且容器不能掛 CBP-NG checkout。
用它等於同時削掉對照組的**能力**而不只是**資訊**，差異就無法歸因於得獎資料。

正確做法是從今天的 301 行版本刪掉得獎衍生的部分：

| 刪除 | 行 | 理由 |
|---|---|---|
| 「The levers that worked for the winners…」整段 | 121–128 | 點名 MORSL 4.8 MPKI / Fan 5.2 MPKI 及其手法 |
| `## What the CBP-NG 2025 winners did` + `${REFERENCE_SOURCE}` | 133–189 | 直接的得獎資料 |
| `## What the archive keeps not doing` | 190–259 | 含 MORSL 的三個量測方法、rank-in-tag |
| `## One winning design in detail` + `${WINNER_DESIGN}` | 260–269 | 每個變體輪播一篇得獎論文 |

**保留 97–120**：那段的 VFS 峰值分析（T = 8.96 為最佳、1 MPKI ≈ +0.015 VFS）
是從我們自己的 archive 量出來的，不是得獎資料。刪除後約 190 行。

`bp_evolve_loop.py` 需要能指定 prompt 檔（目前路徑是寫死的），或直接用環境變數切換。

### 前置條件（不可省略）

1. **重算必須先做完。** 兩條分支要從同一份、以相同條件量測的 gen-14 archive 出發。
   現在那份是 168 traces / run-to-end，新分支會跑 40M。
   因此「量測債」一節的重算是本 arm 的 blocker，不是可選項。
   既有分支的 gens 15–18（run-to-end）也必須一起重算，否則它自己內部就不一致。
2. **`clear_agent_scratch` 要先進去。** `agents.py` 裡那個未提交的改動修的正是
   LLM container 的 `/tmp` 洩漏——gen021_0 / gen022_2 那組近似重複就是這樣來的。
   如果對照組也發生，這條 arm 就報廢了。
3. 寫到**新的 out dir**（例如 `out_design_noref/`），不要碰 `out_design_llm/`。
4. `--seed` 設成與 sweep 3 相同，讓 gen 15 的 parent 選擇一致
   （archive 一分歧之後就沒意義了，但 gen 15 免費拿到）。

### 這條 arm 進論文之後的影響

- 階段 4 從「觀察」升級成「ablation」，是論文的主結果之一。
- **gens 11–14 不能和新分支合併計算**——它們跑的是第三種 prompt（93 行版）。
  在論文裡它們只能當獨立的歷史觀察：「最早的逃離發生在沒有任何得獎資料時」。
- **n = 1 的警告必須寫進去**：每條分支只有一個 lineage、每代 3 個變體。
  若最後差距是 0.988 vs 0.985，那在雜訊範圍內。
  因此要同時報 **cell 數**與**每代三個變體的 VFS 分布**，不能只報 best。
- §6 的「observationally rather than through an ablation」那句可以刪掉。

---

## 執行順序

1. ✅ 撈出 archive JSON（SQLite → `rescore/in/`）。
2. ✅ 加 `--rescore` flag（最後改用 scratchpad 腳本，見「程式改動」）。
3. ✅ 先以樹幹確認輸出格式與 trace 數生效。
4. ✅ 16 份 source 重算（5 小時 32 分）。
5. ❌ ablation arm — 已放棄，見該節。
6. ⬜ 用重算結果重新算 Table I 彈性、Fig. 2 座標、Table II 分母、評估次數計數。
7. ❌ 隨 5 一併放棄；§6 的 "observationally rather than through an ablation" 保留。
8. ⬜ `pdflatex vfs_report` 兩次（本機未安裝 texlive，尚未編譯驗證過任何一次）。

---

## 驗證

- ✅ **重算路徑正確性**：`gen007_2` 重算後為 **0.9200**，落在預期的 0.9185 附近
  而非 126-trace 的 0.9256——trace 數確實生效。十六份輸出全部回報 `n_traces = 168`。
- ✅ **binning 邏輯未被破壞**：`rebin.py --verify` 用各 archive 自己存的數字重建，
  完整重現原本的 7 / 14 / 18 格與各自的最佳設計，證明離線 binning 與 `archive.py` 一致。
- ✅ **id 歧義**：`variant_id` 的 PK 是 `(sweep_id, variant_id)`，並非全域唯一。
  已確認兩份 design archive 共有的 10 個 id 其 source hash 全部相同，
  16 個重算目標無歧義。
- ✅ **彈性方法**：先前已對 gen037_2 重跑有限差分，重現 0.9288 / +0.278 / −0.135 / −0.024。
- ⬜ **內部一致性掃描**：改完後 grep tex 中的 `116`、`seven`、`21 pairs`、`0.9288`、`gen037`、
  `2\times2\times2`、`144`，確認沒有殘留。
- ⬜ **編譯**：`pdflatex vfs_report` 兩次，檢查無 undefined reference，頁數仍在 A³ 的上限內。
  **本機沒有 texlive，至今一次都沒編譯過**，這是目前最大的未驗證風險。
- ❌ ablation arm 的兩項檢查隨該 arm 一併放棄。

---

## 不在範圍內

- 不重跑既有的任何世代（樹幹、offline arm、design arm gens 11–34 都只重新計分，
  不重新產生設計）。唯一新產生設計的是 ablation arm 的 gens 15–25。
- 不做把 design gens 15–29 改標成 gens 11–25 的「三條平行 arm」寫法——
  起點混淆的理由見 Context。
- 不 commit 目前 working tree 裡未提交的三個檔案
  （`agents.py`、`bp_evolve_loop.py`、`prompts/design.md`）——等你指示。

---

# 已確認的量測結果（2026-09-23，重算進行中）

以下數字全部來自現有資料，不需要等重算；重算只會把它們移到同一個條件上，
不會改變結論的方向。

## 官方視窗確定為 40M

`cbp-ng/docs/example_and_reference_predictor_results.csv` 的 168 條 trace，
每一條的 `instructions` 都**正好是 40,000,000**，沒有任何一條超過 41M。
所以 40M 是主辦方的量測條件，run-to-end 是我們自己的偏離。
這排除了「全部改報 run-to-end 以省時間」那條路。

## 可用的 anchor（本來以為有七個，實際只有五個）

用我們的 `vfs.py` 在 depth 9、168 traces @40M 下重算主辦方的 CSV：

| predictor | T | VFS | MPKI | EPI | P1/P2 |
|---|---|---|---|---|---|
| gshareN | 7.837 | 0.9179 | 9.073 | 66.6 | 1/1 |
| bimodalN | 8.468 | 0.8937 | 11.057 | 55.0 | 1/1 |
| tage | 5.769 | 0.8732 | 5.490 | 1245.5 | 1/2 |
| gshare | 5.863 | 0.8429 | 7.434 | 669.6 | 1/2 |
| bimodal | 5.875 | 0.7792 | 11.573 | 595.3 | 1/1 |

`reference`（0.9729）與 `never_taken`（0.4410）**不可用**：CSV 對這兩列的
`energy_per_instruction` 是 0，VFS 無法把它們當設計排名——`vfs.py:238` 自己的
selftest 就因為這個把兩者排除。先前把 0.9729 當 anchor 是錯的。

這五個 anchor 反而更有價值，理由見下一節。

## T 的分界是結構性的，且由 P2 latency 決定

29 個 elite（樹幹 7 + offline 7 + LLM 寫的 15，去重後）按 T 排序，
在 **T ≈ 6.9 一刀切開，零例外**：

| | 數量 | P2 | 產生它的 operator |
|---|---|---|---|
| T < 6.9 | 16 | 全部 ≥ 2（值為 2, 3, 28） | rule 與 LLM 都有 |
| T > 6.9 | 13 | 全部 ≤ 1（值為 0, 1） | **只有 LLM** |

最寬的空帶是 **T ∈ (6.866, 7.703)**，下緣 gen037_2（offline 最佳），
上緣 gen018_0。兩個 operator 都沒有任何設計落在裡面。

**主辦方的參考設計證實同一條線**：gshareN (T 7.84) 與 bimodalN (T 8.47) 都是
P1/P2 = 1/1，落在空帶上方；tage (5.77) 與 gshare (5.86) 都是 1/2，落在下方。
這不是我們資料的假象。

論文應該把主張從「design arm 跨出 flank」改成更精確的
「**P2 latency 是結構變數；P2 ≥ 2 把 T 卡在 6.87 以下，rule-based operator
在 39 代內從未產生過 P2 ≤ 1 的設計，LLM operator 第一次嘗試就產生了**」。

## 「design arm 的設計全部在高 T」不成立

15 個 LLM 寫的 elite 裡有兩個留在低 T 帶：
- `gen011_0`：T 0.325、P1/P2 28/28、VFS 0.0797——壞掉的離群值。
- `gen016_2`：T 4.386、P1/P2 2/2、VFS 0.7047——**一個貨真價實的低吞吐設計**。

空帶的主張不受影響，但任何「全部/每一個 design arm 設計」的量詞都要拿掉。

## Bridge：126 → 168 traces 是均勻的比例偏移

同樣 7 個樹幹 source，兩種 trace 數，皆 run-to-end：

| variant | VFS@126 | VFS@168 | Δ | 相對 |
|---|---|---|---|---|
| gen001_0 | 0.7067 | 0.6998 | −0.0069 | −0.98% |
| gen003_0 | 0.7042 | 0.6973 | −0.0070 | −0.99% |
| gen003_2 | 0.6704 | 0.6631 | −0.0074 | −1.10% |
| gen006_2 | 0.7855 | 0.7800 | −0.0054 | −0.69% |
| gen007_2 | 0.9256 | 0.9185 | −0.0071 | −0.77% |
| gen009_0 | 0.8545 | 0.8464 | −0.0081 | −0.95% |
| gen010_2 | 0.7444 | 0.7374 | −0.0070 | −0.94% |

**相對偏移 −0.917% ± 0.127%**，跨 VFS 0.66–0.93 七個設計。
這是 offline arm 可以留在 126 條、用量測出來的修正值銜接的依據。
第三段（168 run-to-end → 168 @40M）由正在跑的同一批樹幹 source 提供。

## 彈性方法再次驗證

對 `vfs.vfs(ipc, cpi, epi)` 取對稱有限差分（h = 1e-4），
gen037_2 得到 VFS 0.9288、e_T +0.278、e_CPI −0.135、e_EPI −0.024，
與論文 Table I 第 203 行逐位相同。

---

# 命名決定（tex 盤點發現的衝突，必須先解決）

論文第 387 行寫 `All three arms converge to the same optimum.`——
這裡的 "arm" 指的是 §V 的**三條 bounds 控制臂**（LLM-bounds / rule-bounds /
fixed-bounds），與我們要新增的 "design arm" 是完全不同的東西。
L382「Both arms」、L390「Both adaptive arms」、L394「Where the arms differ」、
L395–396「the LLM arm」全部是同一個衝突點。

決定：
- §V 的三條一律改稱 **bounds arms**（LLM-bounds、rule-bounds、fixed-bounds），
  L387 改寫為「All three bounds arms converge to the same optimum」並明確限定
  範圍是 bounds 實驗，不是整篇研究。
- 樹幹分出的兩支一律稱 **parameter arm**（= DB 的 `arm=offline`，`out_gen50`）
  與 **design arm**（`out_design_llm`）。不使用 "params arm"（那是 `out_params`）。

# tex 盤點：先前沒列到、但會被推翻的主張

按嚴重度排序，全部必須改寫而不只是換數字：

1. **L96–97**：`Candidate designs are deliberately produced by a non-LLM operator,
   so that what we measure is the objective rather than a model's priors.`
   這是整篇論文宣告的方法論原則，而 design arm 正是 LLM 產生設計。
   Introduction 的第三點貢獻（L95–99）必須重寫。這是最深的結構問題。
2. **L141**：`the operator can resize the predictor but never change its algorithm.`
   §II 必須先改成描述兩個 operator，才談得上更新數字。
3. **L241–242 / L256–257**：`The dotted lines bracket every design our search
   produced` / `Every design our search produced lies in T∈[3.61,6.87]`——
   被 13 個 T > 6.9 的設計推翻。
4. **L90–91**：`the flank on which every parameter-search design lies`——
   加上 "parameter-search" 限定後其實**仍然成立**，是全文少數可以原樣留下的句子。
5. **L284 / L279**：`the most accurate design we found (mpki 5.084)` 與
   `mpki varies from 5.08 to 10.08`——族群極值與範圍，隨 design arm 改變。
6. **L285 / L372–373**：`all seven final elites sit at LOGLB=7` /
   `Every elite sits at the LOGLB ceiling`——design arm 的 elite 根本沒有
   LOGLB template bound。這同時抽掉了 §V 存在的動機句（L374–375）。
7. **L305 / L309**：`all 40 promoted designs` / `Across the 39 designs it
   evaluated`——若 Tier-1/2 沒跑過 design arm，必須明講這兩句只描述 parameter arm。
8. **L364–365**：`every elite loses a nearly constant 0.0287±0.0019`——
   held-out 的均勻偏移，結構不同的 design arm 最可能打破它。
9. **算術鏈**：116 (L47/145/258/277)、L143 隱含的 120、L305 的 40 promoted、
   L309 的 39 gem5——四個數字互相牽連，只改 116 會讓鏈不一致。
10. **Table II 的 disc. 欄（L340–344）**以 21 對為分母，elite 數一改就要全部重算，
    不是等比例縮放。
11. **序數詞**：L88「three contributions」、L92「two checks」、
    L118「three evaluation tiers」、L98–99「connects three toolchains」——
    插入新節後要手動改，`\ref` 不會幫忙。

Label 清單（10 個）與 float 行號範圍見盤點原始報告；
`eq:speedup` (L165)、`sec:res` (L294)、`sec:bounds` (L370) 三個 label 從未被引用。
在 L368 插入新節會把 §V→§VI、§VI→§VII、§VII→§VIII，
`tab:bounds` 由 Table III 降為 Table IV。

---

# DB 盤點的修正（2026-09-23，必須覆蓋前文）

## 「116」的真正來源，以及它為什麼是錯的

116 **不是** offline arm 評估過的設計數。`out_gen50/bp_evolve.db` 在 gens 11–39
只有 90 筆 variant（87 個相異設計）。

116 是 offline archive 在**第 34 代**的 `cells + rejected`（7 + 109），
也就是從 gen 0 累積到 gen 34 所「看過並裁決過」的設計數。
論文的表跑到 gen 39，對應的數字是 **138**（131 rejected + 7 archived）。
這個數字顯然是在 sweep 跑到 gen 34 時抄下來就沒再更新。

此外 116/138 都**重複計算了被重跑的 gen 14**（sweep 3 與 sweep 4 各跑了一次
gen 14，archive 兩批都算進去），所以它是「評估事件數」而非相異設計數。

誠實的寫法：
- parameter arm，gens 11–39：**90 個提案、90 個 build 成功、90 個跑完、18 個進過
  archive（87 個相異設計）**；含樹幹的累積裁決事件數 138。
- design arm，gens 11–34：**71 個提案、71 個 build 成功、69 個跑完（2 個 run-time
  失敗，皆在 gen 20）、33 個進過 archive（69 個相異設計）**，曾經進過 archive 的
  相異 elite 共 40 個。
- design arm 的 `rejected` 計數器在 sweep 2→3 的 resume（gen 14）被**歸零**，
  所以它的 `cells+rejected`（gen 34 時為 48）與 offline 的 116/138 **不可比**。

## parameter arm 的停滯從第 16 代就開始，不是第 39 代

三個相異的 best VFS，全部：

| 世代 | best variant | 精確 VFS |
|---|---|---|
| 11–15 | gen007_2 | 0.92563531 |
| 16–36 | **gen016_2** | 0.92878204 |
| 37–39 | gen037_2 | 0.92884003 |

gen 11→39 總共買到 **+0.00320**；gen 16 之後的 23 代只買到 **+0.000058**，
在論文報的四位小數下完全看不見。論文 L145「improved by only 0.35% over the last
26 generations」要改成更強也更準的說法：**停滯始於第 16 代，其後 23 代的增益是 6e-5**。

七個樹幹 elite 到**第 22 代就被全部換掉**，gens 23–39 的 17 代（約 51 個設計）
只是在自己的後代之間換手，四位小數下毫無變化。

**注意不要和後文「parameter arm 的七個 cell 與樹幹的七個完全相同」混淆**：
換掉的是**佔據者**，不是**格子**。29 代的參數搜尋把每一格的住戶都換成自己的後代，
卻一個新格子都沒開出來——這兩件事一起講才是完整的圖像：
搜尋在原地變強了一點點（+0.0032），但行為空間的覆蓋完全沒有擴張。

## 樹幹不在 out_gen50 裡（更正前文）

`out_gen50/bp_evolve.db` 最早的 snapshot 與 variant 都是 **gen 11**。
樹幹在 `out_run1/bp_evolve.db`（sweep 1, gens 0–12）與
`out_resume/bp_evolve.db`（sweep 1, gens 3–10），
gen-10 archive 本身是檔案 `resume_gen10_archive.json`（7 cells，rejected 29）。

兩個 DB 都不是單一 sweep：offline 有 4 個 sweep，design 有 **17** 個。
兩個 DB 的 **gen 14 都有兩份 snapshot**，來自不同 sweep 且內容不同。

## design arm 的十八格，有八格不是 agent 做的

gen001_0、gen003_0、gen003_2、gen006_2、gen007_2、gen009_0、gen010_2 七個樹幹設計
從 gen 11 一路存活到 gen 34 從未被動過；加上 gen011_0（VFS 0.0797 的退化設計，
佔著一個沒別人到得了的格子，25 份 snapshot 全程在位）——
**十八格裡有八格是繼承來的，不到一半是 agent 的成果**。

cell 數是覆蓋度指標，不是成就計數，論文必須這樣寫，否則 7→18 會被誤讀。

## design arm 內部也有量測條件斷層

sweep 2→3 的 resume 在 gen 14 把 archive 從 126 重算到 168，
best VFS 因此**下降** 0.9838 → 0.9811（同一個 gen014_0）。
所以 design arm 的曲線在 gen 14 不連續，前後兩半不同尺度。
每一份 snapshot 內部的 n_traces 是齊一的（沒有混合），斷層只在 gen 14 這一刀。

## Tier 1／Tier 2 從未跑過 design arm

design DB 只有 tier 0 的結果，完全沒有 tier-1／tier-2 列。
所以論文 L305「all 40 promoted designs」與 L309「Across the 39 designs」
**只描述 parameter arm**，必須明講。

## 跨 arm 的 variant id 碰撞（嚴重）

兩條 arm 各自獨立產生了同名但**完全不同**的設計：

| id | parameter arm | design arm |
|---|---|---|
| `gen014_1` | VFS 0.7073, T 3.816, P1/P2 2/2 | VFS 0.9562, T 8.155, P1/P2 0/0 |
| `gen016_2` | VFS 0.9288, T 6.870, P1/P2 1/2（**該 arm 的最佳**） | VFS 0.7047, T 4.386, P1/P2 2/2 |

論文裡每個 variant id 都必須標註所屬 arm。重算腳本已經防了這一點
（`SOURCE_FILES` 只取 design archives），但寫作時極易出錯。

另：design arm 的 gen031_1／gen031_2 的 struct name 是
`gen032_nostall`／`gen032_pcfold`，與記錄的世代不符。


---

# 汙染稽核（2026-09-23，覆蓋 D.3 與前文所有「重複」描述）

## 先前說法的兩處錯誤

1. **沒有任何 byte-identical 重複。** 71 份 design arm source 的 SHA-256 全部相異。
   先前寫的「gen022_2 = gen021_0」等三組是**近似**重複，差 1–7 行。
2. **先前說「三組皆未進 archive」是錯的。** A 組進了，而且造成一次實質重複的 cell 佔領。

## 三組近似重複

| 組 | 成員 | 差異 | 是否進 archive |
|---|---|---|---|
| A | gen021_0 / gen022_2 | **一行** `fanout` hint；VFS 相差 7e-7 | **是** |
| B | gen033_0 / gen033_1 | 7 行 | 否 |
| C | gen032_1 / gen033_2 | 註解與小改；VFS/EPI/MPKI **bit-identical** | 否 |

A 組是必須寫進論文的案例：`gen022_2` 以 **+7.02e-7** 的名目增益，
把 `gen021_0` 從 cell (3,1,1) 擠下去——同一個 cell 被實質相同的預測器贏了兩次。
記錄的 parent 還不同（gen009_0 vs gen010_2），所以這不是正常的親子繼承。

## 洩漏機制與指紋

不是 loop 的檔案系統，是 **LLM container 的 `/tmp` 生命期長於 session**。
舊版 `design.md` 只說 "Copy it somewhere under `/tmp` to work in"，
沒有告訴 agent 別的目錄樹不屬於它。

指紋：**五個 variant 宣告了後面世代的 struct 名稱**——
gen029_0→`gen030_0`、gen031_1→`gen032_nostall`、gen031_2→`gen032_pcfold`、
gen032_1→`gen033_0`、gen033_2→`gen034_ahead`。agent 讀到了上一輪 session 的遺留檔案。

## 修正後的計數（取代前文所有 71 / 90 / 87）

| | 先前 | 修正後 |
|---|---|---|
| design arm 相異設計 | 71 proposed / 69 ran | **68 相異** |
| offline arm 相異基因型 | 90 ran / 87 distinct id | **84 相異** |

offline arm 的 6 次冗餘評估與 LLM 無關：**resume 時 mutation RNG 重播**，
3 組各 3 次，`template_args` 相同、VFS bit-identical。
只有第一次進過 archive，故 **offline archive 本身未受汙染**。

## 修補尚未完整（必須揭露）

`bp_evolve_loop.py:437` 的 guard 是
`if args.serial_variants and args.arm != "offline":`。
`serial_variants` 是前提，所以**並行執行的 sweep 仍無保護**
（sweep 18 自 gen 26 起為並行）。

---

# 重算結果（2026-09-23 完成，168 traces / 40M，全部以此為準）

16 份 source、2688 次 CBP-NG run、5 小時 32 分。
輸出 `bp_evolve_data/rescore/out168_40m/`，重建 archive `…/archives168_40m/`。

| variant | arm | VFS | T | EPI | MPKI | P1/P2 |
|---|---|---|---|---|---|---|
| gen014_0 | design (clean) | **0.9817** | 8.858 | 374 | 6.058 | 1/1 |
| gen014_2 | design (clean) | 0.9770 | 8.897 | 531 | 6.244 | 1/1 |
| gen012_0 | design (clean) | 0.9723 | 9.245 | 1128 | 5.971 | 1/1 |
| gen012_1 | design (clean) | 0.9613 | 8.582 | 200 | 7.422 | 1/1 |
| gen014_1 | design (clean) | 0.9576 | 8.170 | 183 | 8.199 | **0/0** |
| gen013_1 | design (clean) | 0.9218 | 8.335 | **84** | 9.445 | 1/1 |
| gen007_2 | 樹幹 | **0.9200** | 6.800 | 1372 | 5.627 | 1/2 |
| gen018_0 | design | 0.8954 | 7.721 | 88 | **11.267** | **0/0** |
| gen009_0 | 樹幹 | 0.8481 | 5.594 | 1480 | 5.358 | 1/3 |
| gen006_2 | 樹幹 | 0.7822 | 5.825 | 1609 | 9.282 | 1/2 |
| gen010_2 | 樹幹 | 0.7388 | 4.309 | 1710 | **5.265** | 1/3 |
| gen016_2 | design | 0.7064 | 4.391 | 87 | 10.495 | 2/2 |
| gen001_0 | 樹幹 | 0.7011 | 3.786 | 1343 | 5.576 | 2/2 |
| gen003_0 | 樹幹 | 0.6985 | 3.790 | 1530 | 5.574 | 2/2 |
| gen003_2 | 樹幹 | 0.6644 | 3.611 | 1786 | 5.467 | 2/3 |
| gen011_0 | design（退化） | 0.0798 | 0.325 | 395 | 6.835 | 28/28 |

（gens 19+ 的六個 elite——gen028_2 0.9855、gen029_1 **0.9895**、gen031_0 0.9722、
gen031_1 0.9864、gen031_2 0.9876、gen034_1 0.9782——原本就在此條件下量測，未重算。）

## 三段 bridge 全部實測完畢

| 段 | 變化 | 樣本 |
|---|---|---|
| 126 → 168 traces（跑到底） | **−0.92% ± 0.13%** | 7 個樹幹 source |
| 168 跑到底 → 168 @ 40M | **+0.20% ± 0.04%** | 16 個 source |

視窗的影響比 trace 數小五倍，離散度小一個數量級。
**這證成了「offline arm 留在 126 traces、以 bridge 換算」的決定**，
也讓「量測條件混用」的疑慮可以用兩個數字交代完畢。

低 T 設計（P2 ≥ 2）的偏移是 +0.16 ~ +0.28%；高 T 設計（P2 ≤ 1）只有 +0.04 ~ +0.06%。
**高吞吐量設計對視窗長度幾乎免疫**——它們沒有長延遲路徑會在視窗邊界被截斷。

## 斷層在統一條件下仍然零例外

空白帶 **T ∈ (6.866, 7.721)**，寬度 0.855。

**先前這裡寫成 (6.800, 7.721) 是錯的。**下緣不是重算後的 `gen007_2`（T 6.800），
而是 parameter arm 的 `gen037_2`（T 6.8662，P2 = 2）——它在 126-trace 條件下的 T
比重算後的樹幹最佳更高。這句話在 tex 裡涵蓋 **both arms**，所以必須取
「P2 ≥ 2 之中 T 最大者」，也就是 6.866。上緣是 `gen018_0` 的 7.7206（重算前 7.703）。
tex L433 已改成 `(6.87, 7.72)`。零例外：P2 ≥ 2 的全部 ≤ 6.866，P2 ≤ 1 的全部 ≥ 7.721。

**另一個先前的錯誤**：曾寫「組織方的 `gshareN` (T = 7.837) 正好落在空白帶裡」。
不對——7.837 > 7.721，它在帶子**上方**，和 `bimodalN` (8.47) 一樣。
tex 原本就寫成 "sit above the band"，是對的。不過結論仍然成立：
那條帶是「我們這條 lineage 沒走過的區域」，不是不可達區，論文不能寫成 unreachable。

## 重算沒有推翻任何結論

三份 archive 的 cell 數（7 / 14 / 18）、最佳設計、排序全部不變，
沒有任何 elite 換格、碰撞或遺失。

## 兩個由重算新確立的事實

1. **parameter arm 的七個最終 cell 和樹幹的七個完全相同。**
   跑了 29 代，一個新 cell 都沒開出來；design arm 開出 11 個樹幹沒有的。
   全部三條線的 cell 聯集是 **18 / 144**。
2. **重算確認了 DB 盤點那一節的數字：樹幹設計七個 + 退化的 `gen011_0` = 八格。**
   寫錯的是 tex：草稿原本寫成「八格是樹幹設計，第九格是 gen011_0」，
   等於把八格重複計算了一次。已改為「七格是樹幹的整份 archive，第八格是 gen011_0」。

## 適合直接寫進 §V 的兩組對照

- **`gen007_2` vs `gen009_0`**（同一條樹幹）：`gen009_0` 的 MPKI 更低（5.358 vs 5.627），
  VFS 卻低 0.072。差別只在 P2 = 3 而非 2。**同一條 lineage 上，更準的設計分數更低。**
- **`gen016_2` vs `gen018_0`**（同為 design arm）：EPI 幾乎相同（87 vs 88）、
  MPKI 幾乎相同（10.5 vs 11.3），只有 P1/P2 是 2/2 對 0/0，VFS 差 **0.189（27%）**。
  **能量與準確度受控，延遲是唯一變因。**

兩組都不跨 arm、不跨量測條件，最難被質疑。

---

# 剩餘工作（截至 2026-09-23）

## 已完成

- 重算 16 份 source 並重建三份 archive。
- `paper/vfs_report.tex`：新增 §V `Can an Agent Leave the Flank?`（`\label{sec:design}`），
  §VI 以後順延，`tab:bounds` 成為 Table IV。**所有 `[RS]` 佔位已填實，歸零。**
- tex 命名衝突修正：§VI 的三條 arm 一律改稱 **bounds arms**；
  `parameter arm` / `design arm` 保留給樹幹分岔的兩支。
- tex 範圍限定：L284–285、L364、L372 等只成立於 parameter arm 的句子已加上限定詞。
- §6 兩段改寫：future work 改為回指 §V 並收窄結論
  （「靠拿掉一個 pipeline stage，不是靠預測得更準」）；
  limitations 移除已被推翻的 "remains below Pallan's and every winner's"，
  改為不宣稱勝負的寫法。
- 汙染稽核結論已寫入 plan.md 與 §V 的 limitations。

## 待辦（2026-09-23 全部完成）

1. ✅ **編譯驗證**——本機有 `tectonic`（`~/miniconda3/bin/tectonic`），不需要 texlive。
   `tectonic -X compile vfs_report.tex --outdir <dir>`。
   目前**編譯乾淨**：無 error、無 undefined reference、無 overfull box。
   編譯過程抓到一個真 bug：§V 引用了 `tab:stages` 但那張表從未寫進 tex。已補上。
2. ✅ **Table I `tab:elast`**：加入三列 design arm
   （`gen029_1` 0.9895 / +0.013、`gen014_0` 0.9817 / +0.036、`gen018_0` 0.8954 / +0.244）。
   方法先對 `gen037_2` 與 `gen029_2` 重跑，精確重現 0.9288 / +0.278 / −0.135 / −0.024
   與 0.6750 / +0.677 / −0.138 / −0.040 才套用。
   `ours, best` / `ours, low T` 改標為 `param. gen037_2` / `param. gen029_2`，
   caption 標明 design 列是 168 traces / 40M、parameter 列是 126 traces。
   **核心結果**：`gen029_1` 的 ∂T = +0.013，與 reference (+0.008)、MORSL (−0.007)、
   Fan (−0.015) 同一個 regime；parameter arm 是 +0.278。
3. ✅ **Fig. 2 `fig:tcurve`**：加入 design arm 14 個點（菱形 marker），
   parameter arm 補到 6 個點並改標為 `parameter arm`，
   3.61 / 6.87 兩條虛線保留但 caption 改為「框住 parameter arm」，
   另加一條灰色陰影帶標出 T ∈ (6.87, 7.72) 這個空帶。
   degenerate 的 `gen011_0`（T 0.33）超出座標範圍，已在 caption 註明。
   高度 3.9 cm → 4.4 cm，legend 改兩欄。
4. ✅ **Table II `tab:tau`**：兩欄重新標示為 `trunk, gen. 10` / `param., gen. 39`，
   並**新增第三欄 `design, gen. 34`**（18 elites，153 pairs）。
   方法先重現既有兩欄（+1.000…+0.619 與 +0.905…+1.000）才套用。
   新結果：design archive 是三者中最不穩的，D=3 時 τ = +0.608、D=30 時 +0.752。
   這正是論文自己的機制所預測的——MPKI 散佈 trunk 1.81× / param 1.09× / design 2.14×。
   已加一段把兩者連起來（illumination 與 depth robustness 互相拉扯）。
5. ✅ **評估次數的計數：`116` 是對的，不用改。**
   先前「必須重新推導」的判斷有誤。116 = tier-0 有分數的相異 variant_id 數
   （`out_full` + `out_resume` + `out_gen50`，重複 id 取較晚的 sweep，
   因為 sweep 4 重做 gen 14、`out_resume` 重做 gen 3）。
   在這個母體下 §III 每一個數字都精確重現：
   r(VFS,T) = +0.949、r(VFS,MPKI) = −0.113、MPKI 5.08–10.08、
   H ≥ 6 的 100 個平均 5.43 最大 5.97、H ≤ 5 的 16 個平均 9.47 最小 9.00。
   **§III 完全不用改。**
   （注意 `gen014_2` 在 sweep 3 與 sweep 4 下是兩個不同的設計共用同一個 id，
   sweep 3 那個 H = 8 但 MPKI 9.08；取錯代表元會破壞 H ≥ 6 的 5.97 上界。）
   design arm 的 `71 提出 / 69 執行 / 2 個 run 失敗 / 33 曾進 archive` 經 DB 驗證全部正確。
6. ✅ **內部一致性掃描**：`2\times2\times2` 與 `21 pairs` 已不存在；
   所有 `seven` 都已正確限定；無殘留的重算前數值
   （`0.9838` / `0.9811` / `0.9256` / `0.9185` / `8.97` 全部歸零）。
   `L295` 的 "The best design, gen037_2" 改為 "This arm's best design"。

## 這一輪另外修掉的事

- **§V 的 prompt 邊界論證大幅補強。** 原本只寫「before it was shown any winner's work」。
  實際查證：gens 11–14 當時的 `design.md` 是 9a8c672 版，**93 行、"ahead" 出現 0 次、
  零得獎者字樣**；ahead-pipelining 是 2bb9128 才進 prompt（170 行、5 行提到）。
  也就是說 agent 跨出 flank 時**根本沒被告知 ahead-pipelining 存在**，比原本的說法強得多。
- **邊界有獨立於 prompt 檔的證據。** 2bb9128 同時把 traces 從 126 換成 168，
  而 DB 裡 gens 11–14 全是 126、gen 15 起全是 168，切換點正好在 14|15。
  （`launch_design_local.sh:97` 的 sweep notes 是單一硬編碼模板，每個 sweep 寫同一串字，
  所以 sweep 1 的 note 提到得獎者 digest **不能當證據**，那是後來才寫進模板的。）
- **「加入得獎者資料後四代 best VFS 原地不動」**已寫進 §V：
  gens 15–18 的 best_variant 一直是 `gen014_0`，第一次進步在 gen 19。
- 摘要與 Introduction 補上 design arm：貢獻從三點改為四點。

## 未決（需要使用者裁示）

- **是否 commit。** 目前未提交：`agents.py`、`bp_evolve_loop.py`、`prompts/design.md`、
  `plan.md`（本檔）、`paper/vfs_report.tex`、`paper/section_design_arm.tex.draft`。
  分支 `bp-evolve`。**未經指示不得提交。**
- ✅ **頁數（2026-09-23 解決）。** 上限確認為 **4 頁（不含 reference）**。
  `vfs_report.tex` 正文是 6 頁，量測後確認**沒有任何單一元件值一整頁**
  （單砍 §IV、§VI、Fig. 1、Fig. 2 或任一張表，都還是 6 頁）。
  產出 **`paper/vfs_report_4p.tex`**（正文剛好 4 頁，bib 從第 5 頁開始，
  0 undefined reference、0 overfull box），做法是：
  1. 刪掉 §VI「An LLM on the Search Space」與 `tab:bounds`（bounds arm 整條貢獻），
     連同摘要最後一句與 Introduction 的第四點貢獻；
  2. 刪掉 Fig. 1（`loop.pdf` 全寬流程圖），內容改寫成 §II 開頭的一段文字；
  3. §III/§IV/§V 散文壓縮約 20%（26.4k → 約 21k 字元），數據一個都沒動；
  4. Table II 拿掉 D = 9（恆為 +1.000）與 D = 20 兩列；
     Table III 把三個 group header 列折成一個 Arm 欄；
  5. Discussion 與 Conclusion 併成一節；§V 的第一個 subsection 標題拿掉；
  6. 版面：float 間距縮到 8pt、表格改 `\scriptsize`、Fig. 2 高度 4.4 → 4.0 cm、
     圖例從左上移到右下（原本蓋住得獎者與 design arm 的點）。
  `vfs_report.tex`（6 頁版）原封不動保留，兩個檔案都在 `paper/`。
- **§6 limitations 對得獎者的措辭。** 目前寫成「達到得獎者的量級，但 trace sample 不同，
  不是正面比較」，並保留「本文量測一個 loop，不是參賽作品」的定位。
  要更強或更弱都只需改那一句。

## 已明確排除

- ablation arm（`design_noref.md`）——使用者於 2026-09-23 裁示不跑，時間全給論文。
  除了時間不足（完整對照窗口 gens 15–29 要 45–50 小時，只剩 57.6 小時，
  而這條 arm 歷史上每 5–10 小時失敗一次需要人工重啟；且汙染防護只在
  `--serial-variants` 下生效，而那是每代 3.7 小時的慢模式）。

  **理由更正（2026-09-24）。** 這裡原本寫「已非必要：gens 11–14 本來就是
  clean prompt，`gen014_0`（0.9817）就是反事實」。**那是錯的。** 稽核 gen-11
  當時實際送出的 prompt（`~/.claude-bpevolve/projects/.../*.jsonl`，971 行）
  發現其中已含 301 行的 "What the CBP-NG 2025 winners did"，寫明
  ahead-pipelining、index-early/select-late 配方，以及 `gshareN_ahead` 的
  原始碼；loop 自己的 `sweeps.note` 也記了這件事。所以 gens 11–14 **不是**
  clean prompt，`gen014_0` 不是反事實，「連續四代 best VFS 沒動」也不能當
  ablation 的替代品（那四代看到的是同一份得獎資料的擴充版，不是有無之別）。

  正確的理由是：這條 ablation **本來就不足以**回答問題。要乾淨對照必須
  (a) 從 prompt 拿掉得獎者整節與 `gshareN_ahead`，而且 (b) 容器裡不能掛
  CBP-NG checkout（agent 可以直接讀參考預測器的原始碼）。只做 (a) 不夠。
  兩者都做就是 hermetic gen-11 replay，時間不夠，因此放棄。論文改為主張
  **operator reach**（誰能表達這個改動），而不是 rediscovery——那才是這個
  loop 實際測到的東西；§6 的 "observationally rather than through an
  ablation" 必須留著。
- gem5 重跑。
- `prompts/design.md:135` 的 caveat **不改**：沒有證據顯示得獎者的評估集等於我們的
  168 條公開 trace，放寬它只是用另一個過度自信的說法取代原本的。
