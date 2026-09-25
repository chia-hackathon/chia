#!/usr/bin/env python3
"""Everything that has to work before the Titan loop is worth starting.

Nothing in `examples/titan/` has ever run on a cluster.  The self-tests prove
the generators are internally consistent; they prove nothing about the
toolchain, the images, or the CHIA paths this example is the first to walk.
This script walks them, in dependency order, and stops at the first failure
with a message that says what to do about it.

    chia job submit -- python examples/titan/preflight.py
    chia job submit -- python examples/titan/preflight.py --from 5

Two checks here are worth more than the rest.

**Check 4** is the first time a RISC-V assembler has ever seen a program this
example generated.  Everything upstream of it is Python agreeing with itself.

**Check 7** expects a *failure*, and fails if it gets a pass.  It runs a
directed program on an unmodified Saturn, which has no IME: the first
`.insn` must trap.  A pass there would mean the program is not testing what
it claims to, and that is worth catching before sixty iterations of an agent
are graded by it.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Callable, List

#: Scratch on the worker nodes. $TMPDIR is set to /share1/.../node_tmp by
#: cluster.yaml on every node; the root filesystem is full.
PREFLIGHT_WORK_DIR = os.path.join(tempfile.gettempdir(), "titan-preflight")

# --------------------------------------------------------------------------
# check registry
# --------------------------------------------------------------------------

@dataclass
class Result:
    ok: bool
    detail: str = ""
    remedy: str = ""
    seconds: float = 0.0


CHECKS: List[tuple] = []
_state: dict = {}


def check(number: int, title: str):
    def wrap(fn: Callable[[], Result]):
        CHECKS.append((number, title, fn))
        return fn
    return wrap


def ok(detail: str = "") -> Result:
    return Result(True, detail)


def bad(detail: str, remedy: str = "") -> Result:
    return Result(False, detail, remedy)


# --------------------------------------------------------------------------
# lazily-built state
#
# `--from N` skips the checks that would have populated what check N needs,
# so each dependency is fetched through a helper that builds it if it is not
# already there.  Without this, resuming at any check but the first crashes
# on a KeyError -- which is exactly what happened the first time.
# --------------------------------------------------------------------------

def _suite():
    if "suite" not in _state:
        import ime_tests
        from constants import VLEN
        _state["suite"] = ime_tests.directed_suite(VLEN, full_vl_only=True)
    return _state["suite"]


def _elf_one():
    if "elf_one" not in _state:
        import nodes
        from chia.base.ChiaFunction import get
        name, asm, _ = _suite()[0]
        elf = get(nodes.build_ime_test.chia_remote(asm, name,
                                                   PREFLIGHT_WORK_DIR,
                                                   extension="ime"))
        if not elf:
            raise RuntimeError(f"{name} 組不起來（見第 4 項）")
        _state["elf_one"] = (name, elf)
    return _state["elf_one"]


def _baseline():
    if "baseline" not in _state:
        import nodes
        from chia.base.ChiaFunction import get
        from constants import BASELINE_CONFIG
        artifact = get(nodes.build_saturn.chia_remote(BASELINE_CONFIG, None,
                                                      extension="ime"))
        if not artifact.success:
            raise RuntimeError(f"{BASELINE_CONFIG} 建不起來（見第 6 項）")
        _state["baseline"] = artifact
    return _state["baseline"]


def _spike():
    if "spike" not in _state:
        import nodes
        from chia.base.ChiaFunction import get
        artifact = get(nodes.build_spike.chia_remote(extension="ime"))
        if not artifact.success:
            raise RuntimeError("libriscv 建不起來（見第 8 項）")
        _state["spike"] = artifact
    return _state["spike"]


# --------------------------------------------------------------------------
# 1-2: the environment itself
# --------------------------------------------------------------------------

@check(1, "titan 模組全部 import 得起來")
def c_imports() -> Result:
    names = ["constants", "ime_encodings", "rvv_ref", "ime_tests",
             "ime_stress", "helpers", "tools", "db_node", "nodes",
             "llm", "titan_loop"]
    failed = []
    for name in names:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    if failed:
        return bad("; ".join(failed),
                   "flat-module packaging puts examples/titan on sys.path via "
                   "RUNTIME_ENV['working_dir']; an ImportError here usually "
                   "means the job was submitted without that runtime env.")
    return ok(f"{len(names)} modules")


@check(2, "ray 叢集的資源表符合 cluster.yaml")
def c_resources() -> Result:
    import ray
    want = {"llm": "LLM agent", "chipyard": "chisel + spike build",
            "verilator_run": "模擬與 cosim", "riscv_build": "交叉編譯",
            "head_local": "工具與產生器", "database": "測試池"}
    have = ray.cluster_resources()
    missing = [f"{k} ({v})" for k, v in want.items() if k not in have]
    if missing:
        return bad("缺少 " + ", ".join(missing),
                   "對照 examples/titan/cluster.yaml 的 available_node_types。"
                   "build 節點常見的失敗是 SSH_AUTH_SOCK 為空導致 docker "
                   "run -v :/ssh-agent —— 先 eval $(ssh-agent -s) && ssh-add。")
    return ok(", ".join(f"{k}={have[k]:g}" for k in want))


# --------------------------------------------------------------------------
# 3-5: the test-generation chain, up to real ELFs
# --------------------------------------------------------------------------

@check(3, "產生 directed 測試程式（純 Python，跑在 head）")
def c_generate() -> Result:
    import ime_tests
    from constants import VLEN
    suite = _suite()
    if not suite:
        return bad(f"VLEN={VLEN} 產不出任何 round-one 幾何",
                   "檢查 rvv_ref.permissible_lambdas —— 這個 VLEN 可能整個在 "
                   "IME-legal domain 之外。")
    lines = sum(a.count("\n") for _, a, _ in suite)
    return ok(f"{len(suite)} 支程式，{lines:,} 行組語")


@check(4, "交叉編譯一支 —— 組譯器第一次看到我們產的程式")
def c_assemble_one() -> Result:
    import nodes
    from chia.base.ChiaFunction import get
    name, asm, geom = _suite()[0]
    started = time.time()
    elf = get(nodes.build_ime_test.chia_remote(asm, name, PREFLIGHT_WORK_DIR,
                                               extension="ime"))
    if not elf:
        return bad(f"{name} 組不起來（{geom.describe()}）",
                   "把 out/ 裡 dump 出來的 .S 拿到 chia-riscv-cross 容器裡手動 "
                   "組一次看訊息。最可能的兩件事：constants.IME_CFLAGS 沒帶 "
                   "_v（harness 的 baseline -march 連 v 都沒有），或 .insn "
                   "指令字的語法。")
    _state["elf_one"] = (name, elf)
    return ok(f"{name} → {len(elf):,} bytes，{time.time() - started:.0f}s")


@check(5, "交叉編譯整批")
def c_assemble_all() -> Result:
    import nodes
    from chia.base.ChiaFunction import get
    refs = {name: nodes.build_ime_test.chia_remote(
        asm, name, PREFLIGHT_WORK_DIR, extension="ime")
        for name, asm, _ in _suite()}
    built, failed = [], []
    for name, _, _ in _suite():
        elf = get(refs[name])
        (built if elf else failed).append(name)
        if elf:
            _state.setdefault("elfs", {})[name] = elf
    if failed:
        return bad(f"{len(failed)}/{len(refs)} 組不起來，例如 {failed[0]}",
                   "單支過但整批不過，通常是某個幾何才會踩到的東西 —— "
                   "分支距離、立即值範圍、或 .data 大小。")
    return ok(f"{len(built)} 支全過")


# --------------------------------------------------------------------------
# 6-7: the DUT
# --------------------------------------------------------------------------

@check(6, "baseline Saturn elaborate + verilate")
def c_build_baseline() -> Result:
    import nodes
    from chia.base.ChiaFunction import get
    from constants import BASELINE_CONFIG
    started = time.time()
    artifact = get(nodes.build_saturn.chia_remote(BASELINE_CONFIG, None,
                                                  extension="ime"))
    if not artifact.success:
        tail = (getattr(artifact, "stderr", "") or "")[-1500:]
        return bad(f"{BASELINE_CONFIG} 建不起來：{tail}",
                   "這是未改動的 Saturn，失敗代表環境問題而不是設計問題。"
                   "先確認 chipyard 的 env.sh 有 source、以及 config 名稱在這個 "
                   "checkout 裡存在。")
    _state["baseline"] = artifact
    return ok(f"{BASELINE_CONFIG}，{time.time() - started:.0f}s")


@check(7, "baseline 上跑一支 —— 必須失敗，而且要失敗得對")
def c_baseline_must_fail() -> Result:
    import helpers
    import nodes
    from chia.base.ChiaFunction import get
    name, elf = _elf_one()
    run = get(nodes.verilator_run_remote.chia_remote(
        _baseline(), elf, name, PREFLIGHT_WORK_DIR,
        extension="ime"))
    outcome = helpers.classify_run(run)
    if outcome.kind == "pass":
        return bad("未改動的 Saturn 竟然通過了 directed 測試",
                   "這代表測試沒有在測它宣稱的東西 —— IME 路徑大概根本沒執行到，"
                   "或比對從來沒觸發。在燒掉六十輪迭代之前必須查清楚。")
    if outcome.kind == "timeout":
        return bad("模擬逾時，沒有拿到判決",
                   "非法指令的 trap 應該讓程式很快結束。逾時通常代表 trap "
                   "handler 把機器帶進迴圈，或 htif 沒收到結束訊號。")
    return ok(f"如預期失敗：{outcome.kind} —— {outcome.detail[:90]}")


# --------------------------------------------------------------------------
# 8-10: Spike, the path nobody has walked
# --------------------------------------------------------------------------

@check(8, "建 Spike（未改動）—— 這條路沒有 example 走過")
def c_build_spike() -> Result:
    import nodes
    from chia.base.ChiaFunction import get
    started = time.time()
    artifact = get(nodes.build_spike.chia_remote(extension="ime"))
    if not artifact.success:
        tail = (getattr(artifact, "stderr", "") or "")[-1500:]
        return bad(f"libriscv 建不起來：{tail}",
                   "SpikeBuildNode 在整個 repo 裡只有單元測試呼叫過。檢查 "
                   "constants.SPIKE_SRC_REL 指到的 checkout 存在，且 $RISCV "
                   "在 chipyard 容器裡有設。")
    _state["spike"] = artifact
    return ok(f"libriscv {len(artifact.lib_content):,} bytes，"
              f"digest {artifact.digest[:12]}，{time.time() - started:.0f}s")


@check(9, "Spike 的 --varch 真的生效（VLEN 對不對）")
def c_spike_varch() -> Result:
    import nodes
    from chia.base.ChiaFunction import get
    from constants import VLEN
    probe = _vlenb_probe()
    elf = get(nodes.build_ime_test.chia_remote(probe, "vlenb_probe",
                                               PREFLIGHT_WORK_DIR,
                                               extension="ime"))
    if not elf:
        return bad("VLEN 探針組不起來",
                   "這支只用 csrr 與 printf，組不起來代表 harness 本身有問題。")
    res = get(nodes.spike_run.chia_remote(elf, "vlenb_probe",
                                          PREFLIGHT_WORK_DIR,
                                          extension="ime"))
    want = VLEN // 8
    marker = f"TITAN VLENB={want}"
    if marker not in (res.log or ""):
        return bad(f"預期 {marker}，實得：{(res.log or '')[-200:]}",
                   "Spike 用預設向量寬度在跑。所有 tile 幾何都是從 VLEN 導出的，"
                   "寬度不對會算出一個不一樣但完全合法的答案 —— 這是最惡劣的一種"
                   "錯。檢查 nodes.spike_run 的 --varch 有沒有傳出去。")
    return ok(f"vlenb={want}（VLEN={VLEN}）")


@check(10, "未改動的 Spike 上跑一支 —— 也必須失敗")
def c_spike_must_fail() -> Result:
    import helpers
    import nodes
    from chia.base.ChiaFunction import get
    name, elf = _elf_one()
    res = get(nodes.spike_run.chia_remote(elf, name, PREFLIGHT_WORK_DIR,
                                          extension="ime"))
    outcome = helpers.classify_run(res)
    if outcome.kind == "pass":
        return bad("upstream Spike 竟然通過了 IME 測試",
                   "upstream riscv-isa-sim 不認得 Zvvm。通過代表 .insn 指令字"
                   "被解成了別的既有指令 —— 回頭查 ime_encodings 的編碼。")
    return ok(f"如預期失敗：{outcome.kind}")


# --------------------------------------------------------------------------
# 11-12: the paths S3 depends on
# --------------------------------------------------------------------------

@check(11, "把 golden model link 進模擬器")
def c_golden_link() -> Result:
    import nodes
    from chia.base.ChiaFunction import get
    from constants import BASELINE_CONFIG
    started = time.time()
    artifact = get(nodes.build_saturn.chia_remote(
        BASELINE_CONFIG, _spike(), extension="ime"))
    if not artifact.success:
        tail = (getattr(artifact, "stderr", "") or "")[-1500:]
        return bad(f"帶 golden_model 的建置失敗：{tail}",
                   "這是計畫裡註明「沒人走過」的那條。ChiselBuildNode 需要 "
                   "clean_sim=True 才會為新 library 重新 link；若是 undefined "
                   "symbol，改用 static_golden_model=True 再試。")
    digest = getattr(artifact, "golden_model_digest", "")
    if digest and digest != _spike().digest:
        return bad(f"link 進去的 digest 不符：{digest[:12]} vs "
                   f"{_spike().digest[:12]}",
                   "模擬器綁到了 image 自帶的 libriscv，不是我們建的那個。")
    return ok(f"digest {digest[:12] or 'n/a'}，{time.time() - started:.0f}s")


@check(12, "S2 的 RVV 回歸套件已經 stage")
def c_regression_staged() -> Result:
    import db_node
    from chia.base.ChiaFunction import get
    import titan_loop
    suite = get(db_node.fetch_tests.chia_remote(titan_loop.RVV_REGRESSION_KEY))
    if not suite:
        return bad("DB_ROOT/tests/rvv/ 是空的",
                   "先跑 stage_rvv_tests.py。沒有它 S2 會直接報錯 —— 這是刻意的，"
                   "一個因為套件不存在而通過的閘門比沒有閘門更糟。")
    return ok(f"{len(suite)} 支已 stage")


# --------------------------------------------------------------------------

def _vlenb_probe() -> str:
    """A minimal RVV-only program that prints VLENB.

    No IME, no .insn -- if this does not run, the harness is broken and
    nothing about the matrix extension is being tested yet.
    """
    return """# preflight: report vlenb so we can tell whether --varch took effect
    .text
    .balign 4
    .globl main
main:
    addi  sp, sp, -16
    sd    ra, 8(sp)
    li    t0, 1536
    csrs  mstatus, t0        # enable vector state
    csrr  a1, vlenb
    la    a0, .Lfmt
    call  printf
    li    a0, 0
    ld    ra, 8(sp)
    addi  sp, sp, 16
    ret

    .data
.Lfmt:
    .asciz "TITAN VLENB=%d\\n"
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", type=int, default=1,
                        help="從第 N 項開始（前面的假設已經過）")
    parser.add_argument("--only", type=int, help="只跑第 N 項")
    parser.add_argument("--keep-going", action="store_true",
                        help="失敗後繼續（後面的檢查多半會連鎖失敗）")
    args = parser.parse_args()

    import ray
    from constants import RUNTIME_ENV
    ray.init(address="auto", runtime_env=RUNTIME_ENV)

    selected = [(n, t, f) for n, t, f in CHECKS
                if (args.only == n if args.only else n >= args.start)]
    results = []
    print(f"Titan preflight — {len(selected)} 項\n")
    for number, title, fn in selected:
        print(f"[{number:2d}] {title} ...", end=" ", flush=True)
        started = time.time()
        try:
            res = fn()
        except Exception:  # noqa: BLE001 - a crash is a failed check
            res = bad(traceback.format_exc(limit=3).strip().splitlines()[-1],
                      "檢查本身炸了 —— 多半是上一項留下的狀態沒建立，或 API "
                      "簽名不符。完整 traceback 在下方。")
            traceback.print_exc()
        res.seconds = time.time() - started
        results.append((number, title, res))
        print(("OK   " if res.ok else "FAIL ") + res.detail)
        if not res.ok:
            if res.remedy:
                print(f"     → {res.remedy}")
            if not args.keep_going:
                break

    passed = sum(1 for _, _, r in results if r.ok)
    print(f"\n{passed}/{len(results)} 通過"
          f"（共 {len(CHECKS)} 項，跑了 {len(results)} 項）")
    if passed == len(CHECKS):
        print("\n全部通過。可以跑 titan_loop.py 了 —— 第一個上場的是 Stage 0，"
              "寫 Spike 模型的那個 agent。")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
