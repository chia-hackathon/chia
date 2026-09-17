# Titan — 用 CHIA agentic loop 在 Saturn 向量單元上實作 RISC-V IME

Titan 是一條跑在 CHIA 上的 agentic loop：由 LLM agent 在 Chipyard/Saturn 樹上
實作 RISC-V **IME（Integrated Matrix Extension, Zvvm 家族）v0.9.0**，目標是
edge-LLM 的 **INT8 GEMM** ——「C += A × Bᵀ」的 tile MAC 與 2D tile load/store 直接
進向量單元，省掉軟體重排。Loop 自己建置 RTL、產生測試、判定、回饋、再迭代，直到收斂。

設計上最關鍵的一點是**每個 judge 都早於被它審判的對象**（見 `titan_loop.py` 檔頭）：
`rvv_ref.py` 是人先寫好的、riscv-vector-tests 是 Saturn 原本就附的、Spike IME 模型
在 RTL 之前先收斂。三個 judge 都不是實作者自己的產物。

## 1. 已實作 / 未實作的指令（8 / 15）

完整取捨與理由見 `docs/round3_design.md`（§1 全 15 條調查、§2 批次決策）。

| 已完成（8） | 群組 | 輪次 |
|---|---|---|
| `vmmacc.vv` | Zvvmm | r1 |
| `vqmmacc.vv` | Zvvmm | r2 |
| `vwmmacc.vv`、`v8wmmacc.vv` | Zvvmm | r3 |
| `vmtl.v`、`vmts.v` | Zvvmtls | r1 |
| `vmttl.v`、`vmtts.v` | Zvvmttls | r3 |

Zvvmm（整數矩陣 MAC，W=1/2/4/8）與 Zvvmtls/Zvvmttls（順序保留與轉置的 tile
load/store）都已完整關閉。

**延後的 7 條全部屬於 Zvvfmm**：`vfmmacc.vv`、`vfwmmacc.vv`、`vfqmmacc.vv`、
`vf8wmmacc.vv`、`vfwimmacc.vv`、`vfqimmacc.vv`、`vf8wimmacc.vv`。理由（`round3_design.md` §2.5–2.6）：
對「INT8 GEMM」這個目標買不到東西；需要一整套新東西（FP MAC array、`altfmt` 格式解碼、
`frm`/`fflags`、G/psm/rnd 累加與捨入模型、MX（E8M0）scale 解碼、OFP8/OFP4 子字組格式），
而這些都落在共用的 int/fp issue path —— 正是第一輪 bug 群聚的地方（見「已知限制」§2）。
後三條 MX 整數輸入形式（`vf*immacc`）是穿著整數 opcode 的 FP-accumulate 指令，在 funct6
0x39/0x3a/0x3b 上只以 `vm` 一個 bit 與 `vwmmacc`/`vqmmacc`/`v8wmmacc` 區分，因此繼承同一個
延後決定；本輪只實作 `vm=1`。

## 2. 五個驗證階段與最終數字

| 階段 | 內容 | judge | r15 結果 | 出處 |
|---|---|---|---|---|
| M | Zvvm 進 Spike，依規格的 SAIL | S1 的成對程式 | 模型種子 80/80 | `docs/ledger.md`（r15 列） |
| S1 directed | 成對的 IME / RVV-1.0 程式，每輪 80 支 | `rvv_ref.py` | **directed 80/80** | `docs/r15_full_verify.json` → `directed.n_tests=80`, `counts.pass=80` |
| S1 gate | 收斂時的完整 directed 閘門，548 個 geometry | `rvv_ref.py` | **548/548** | `docs/ledger.md`（r15 列）；`docs/round3_design.md` §6「S1 gate: 244 → **548**」 |
| S2 regression | Saturn 自帶的 riscv-vector-tests 全套 | 測試自帶 + stock Spike | **837 ran / 0 failing** | `docs/r15_full_verify.json` → `s2_full.n_ran=837`, `n_failing=0`, `failing=[]` |
| S3 stress | 隨機 tile geometry，與 IME Spike 模型 lockstep | stage M 的模型 | **64/64** | `docs/ledger.md`（r15 列） |

S2 的 837 是全套 841 扣掉 **4 支有文件記錄的排除**（`docs/r15_full_verify.json` 的
`baseline.subtracted`，逐支理由見 `rvv_baseline_failures.README.md`）：
`machine_vfcvt_f_x_v-0` 是 pristine Saturn 對 stock Spike 本來就有的 1-ulp 差異（與 IME 無關）；
`machine_vsetvl-0` / `vsetivli-0` / `vsetvli-0` 則因為 IME v0.9.0 要求 IME-legal 的 vsetvli 把
非零 lambda 寫進 `vtype[62:60]`，stock Spike 不認識 IME 永遠回 0，凡是把 vtype 值寫回純量
暫存器比對的測試在 stock Spike 下**不可能**通過；這些欄位改由 S3 以 IME Spike 模型 lockstep 判定。

## 3. 怎麼跑

```bash
# 1. 叢集：從範本做出自己的 cluster.yaml（cluster.yaml 已被 .gitignore）
cp examples/titan/cluster.yaml.example examples/titan/cluster.yaml
$EDITOR examples/titan/cluster.yaml      # 填 <HEAD_IP> / <SHARED_ROOT> / <HOME>
chia up examples/titan/cluster.yaml

# 2. 從零跑整條 loop
chia job submit -- python examples/titan/titan_loop.py

# 3. 續跑：拿前一輪收斂的模型與 RTL 當起點（第三輪就是這樣跑的）
chia job submit -- python examples/titan/titan_loop.py \
    --model-seed docs/r6_stageM_spike_model_sail.diff \
    --rtl-diff   docs/r13_round2.diff
```

`--model-seed` 是「套用舊模型後**仍然**跑 model agent」（失敗是重點，不是錯誤），
`--model-diff` 則只重建重判、跳過 agent；`--rtl-diff` 只取 diff 裡非 riscv-isa-sim 的
hunks，套用後先跑 attempt 0，讓 agent 的第一個 prompt 帶著真實結果。

常用 `TITAN_*` 環境變數（全部定義與預設值在 `constants.py`）：

| 變數 | 預設 | 作用 |
|---|---|---|
| `TITAN_CHIPYARD_PATH` | `/home/ray/chipyard` | 容器內的 Chipyard 樹 |
| `TITAN_INSNS` | `all` | 指令範圍：`one` / `two` / `three` / `all` 或逐條 mnemonic；`one` 可跑位元相同的回歸 |
| `TITAN_VLEN` / `TITAN_DLEN` | `256` / `128` | Saturn 組態 |
| `TITAN_SYNTH_CONFIG` / `TITAN_BASELINE_CONFIG` | `TitanV256D128ShuttleConfig` / … | 建置與對照組態 |
| `TITAN_RVV_TESTS_DIR` / `TITAN_REGRESSION_BASELINE` | 見 `constants.py` | riscv-vector-tests 位置與 S2 排除清單 |
| `TITAN_LOG_ROOT` | `$TMPDIR/titan` | run 產物根目錄 |
| `TITAN_MAX_ITERS` / `TITAN_MODEL_MAX_ITERS` | `60` / `25` | 迭代上限 |
| `TITAN_LLM_MODEL` / `TITAN_LLM_TIMEOUT_SECONDS` | `claude-opus-4-7` / `1800` | agent 設定 |
| `TITAN_CLI_BACKGROUND_MS` | `1800000` | Claude CLI 的 MCP 背景化門檻；設 0 = 完全不背景化 |

## 4. 規格放置

`constants.py:31` 定義 `SPEC_DIR = EXAMPLE_DIR / "specs" / "ime"`，`titan_loop.py:1636`
把它掛成 agent 的 `SpecTool`。這個目錄裡 `instructions.json` 與 `extract_spec.py` 是本專案
產生的，**已收錄**（JSON 標頭：`"spec": "Zvvm Family of Integrated Matrix Extensions"`,
`"version": "0.9.0"`, `"commit": "5a2d0f65"`），`ime_encodings.py` 在 import 時就讀它。
但 `specs/ime/integrated-matrix-v0.9.0.adoc` 是**第三方 RISC-V 規格原文，本 repo 不轉散布**——
請自行取得 IME v0.9.0（commit `5a2d0f65`）的 AsciiDoc，用**原檔名**放進該路徑。少了它套件
仍可 import，但 agent 讀不到規格正文，loop 跑不出有意義的結果。

## 5. 分支與 framework patch

本分支 `titan` 基於 **`c58eef4`** —— 每一次 Titan run 都用的那個 framework commit。其上只帶
**一個** framework patch：`chia/models/claude.py` 加上向後相容的 `disallowed_tools` /
`max_turns`（對應 CLI 的 `--disallowedTools` / `--max-turns`），用來禁掉 `Task` 工具，避免
agent 生出成本與編輯都不在帳上的 sub-agent。見本分支的第一個 commit。

## 6. 已知限制（誠實版）

1. **沒有真正的 PPA。** 這個部署裡沒有安裝任何合成工具，也沒有 PDK
   （genus/yosys/openroad 全缺、`import hammer` 失敗、沒有 sky130 collateral），
   `cluster.yaml` 的 `vlsi` 節點 `num_workers: 0` 且 image 未填。
   `docs/ppa_results.json` 裡**沒有**任何 cell area、WNS、達成頻率或功耗數字；那是對
   Chisel/firtool 既有流程輸出的 SystemVerilog 做**靜態結構比較**的 proxy，方法與每一條
   規則見 `docs/ppa_method.md`（§1 逐項缺件證據、§4 規則）。沒有捏造任何 PPA 數字。
2. **`.vf` 排程敏感性是 margin move，不是已證明的根因。**
   A/B 實驗（`docs/report_data.md` §5.6d）確認把 `MatrixMultiplyPipe` 的
   `post_write_stall` 拿掉（A3，`io.stall := valid`）能讓 3/3 `.vf` 測試通過，
   `docs/r12_rtl_a3.diff` 就是這個修正。但真正被消除的是**餘裕**：
   matrix FU 被無條件加進 `integerFUs`，在 `VectorIssueStructure.Shared` 下
   int sequencer 會擋住 fp sequencer，而 `.vf` 的純量運算元比其他指令晚一級到達
   —— 這個結構耦合仍在，只是目前不再被觸發。同一份檔案 §5.6c 也記錄 file-level
   bisect 的失敗（8 arm 有 7 arm build_failed，完全沒定位出東西）。
3. **S2 排除了 4 支 RVV 測試。** 理由見上面第 2 節與 `rvv_baseline_failures.README.md`：
   一支是與 IME 無關的既有 1-ulp 差異，三支是 stock Spike 結構上不可能通過的
   vtype-lambda 家族（改由 S3 的 IME 模型 lockstep 判定）。沒有一支是因為「修不好」而拿掉的。

## 7. `docs/` 索引

| 檔案 | 是什麼 |
|---|---|
| `r6_stageM_spike_model_sail.diff` | Stage M 的 Spike IME 模型（依 SAIL 重寫後的版本） |
| `r12_rtl_a3.diff` | **第一輪設計，4 條指令**（A3 修正後的 RTL + 模型） |
| `r13_round2.diff` | **第二輪**，加入 `vqmmacc.vv` |
| `r15_round3.diff` | **第三輪，最終 8 條指令設計** |
| `round3_design.md` | 全 15 條指令調查、第三輪批次決策、依賴的 SAIL 行號、編碼與實作 |
| `r15_full_verify.json` | 最終驗證結果（directed 80/80、S2 full 837/0、排除清單） |
| `ledger.md` | 全部 19 次 run 的時間 / 成本 / 結果（累計 46h11m、$837.13） |
| `report_data.md` | 全語料庫的逐項證據與根因分析 |
| `ppa_method.md` / `ppa_results.json` | 結構性 RTL proxy 比較的方法與結果（非真實 PPA） |
| `report.html` | 已發布的完整報告 |
