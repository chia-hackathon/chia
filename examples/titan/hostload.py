"""主機負載閘門：送任務之前先看伺服器現在有多忙。

為什麼需要這個
--------------
`cluster.yaml` 的 CPU 配額是按「獨佔主機」寫的（所有節點加總正好等於主機的
96 執行緒），但這台機器是共用的 —— AETHER 叢集另有自己的一份宣告，還有約
40 位其他使用者。2026-09-23 曾觀測到 load average 155、可用記憶體剩 4 GB，
當時前幾名全是別人的工作（某人六隻掛了 22 天的 nvtop 就吃掉約 6 顆核心）。

兩個設計上的陷阱，這個模組刻意繞開：

1. **不能看總負載。** 我們自己的 verilator sim 也會推高 load average，所以
   「等負載降下來再送」會變成自我鎖死：負載只有在我們停下時才降。因此
   :func:`others_cpu_pct` 明確排除自己的 uid，只量別人佔了多少。

2. **不能無限等。** 那六隻 nvtop 已經掛了 22 天。任何「等到夠閒為止」的規則
   碰到長期佔用就會永遠等下去 —— 而永遠等下去跟當掉在外部看起來一模一樣，
   這個 session 已經為了分辨這兩者付出過代價。所以閘門**永遠會放行**，只是
   放行時的視窗大小隨餘裕縮放，並且把當下的理由講出來。

也就是說：這是節流閥，不是開關。它調整我們拿多少，不決定我們跑不跑。
"""
import os
import subprocess
import time

#: 主機的邏輯核心數。
TOTAL_CPUS = os.cpu_count() or 96

#: 我們願意佔用的上限（佔全機的比例）。0.35 × 96 ≈ 34 核，對應降載後的
#: verilator 2 節點 × 16 CPU。
SELF_BUDGET_FRAC = float(os.environ.get("TITAN_SELF_BUDGET_FRAC", "0.35"))

#: 低於這個可用記憶體（GB）就縮到最小視窗。Verilator 的 cosim 每支約 1-2 GB。
MIN_AVAIL_GB = float(os.environ.get("TITAN_MIN_AVAIL_GB", "16"))

#: 餘裕不足時仍然至少送這麼多 —— 保證一定有進展。
MIN_WINDOW = int(os.environ.get("TITAN_MIN_WINDOW", "2"))

#: 每支 cosim 估計吃掉的核心數（nodes.py 的 VERILATOR_THREADS）。
CPUS_PER_SIM = int(os.environ.get("TITAN_CPUS_PER_SIM", "8"))


def _self_uid():
    return os.getuid()


def others_cpu_pct():
    """別人（不含自己的 uid）合計佔用的 CPU 百分比；100 = 一顆核心。

    用 ``ps`` 的 ``%cpu``，那是該進程存活期間的平均值而非瞬時值，所以對剛
    啟動的尖峰反應偏慢、對長期佔用反應準確 —— 正好符合這裡的用途：我們要
    避開的是持續性的壅塞，不是某人跑了兩秒的腳本。
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "uid,pcpu", "--no-headers"],
            capture_output=True, text=True, timeout=20).stdout
    except (subprocess.SubprocessError, OSError):
        return None                      # 量不到就不要假裝量得到
    me, total = _self_uid(), 0.0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            uid, pcpu = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if uid != me:
            total += pcpu
    return total


def avail_gb():
    """/proc/meminfo 的 MemAvailable，單位 GB。"""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def snapshot():
    """回傳一份當下的負載快照，欄位量不到時為 ``None``。"""
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = None
    return {"load1": load1,
            "others_cpu_pct": others_cpu_pct(),
            "avail_gb": avail_gb(),
            "total_cpus": TOTAL_CPUS}


def headroom_cpus(snap=None):
    """現在還剩幾顆核心可以給我們用。

    量不到別人的用量時回傳 ``None`` —— 呼叫端應該退回設定值，而不是自己
    編一個數字出來。
    """
    snap = snap or snapshot()
    others = snap.get("others_cpu_pct")
    if others is None:
        return None
    budget = TOTAL_CPUS * SELF_BUDGET_FRAC
    free = TOTAL_CPUS - others / 100.0
    return max(0.0, min(budget, free))


def advised_window(configured, snap=None):
    """依當下餘裕給出建議的併發視窗，附一句人看得懂的理由。

    回傳 ``(window, reason)``。``configured`` 是設定上限（0 = 不限，此時以
    CPU 預算換算出的值為上限）。**視窗永遠 >= MIN_WINDOW**，所以這個函式
    不會讓呼叫端停擺。
    """
    snap = snap or snapshot()
    budget_window = max(MIN_WINDOW, int(TOTAL_CPUS * SELF_BUDGET_FRAC
                                        // CPUS_PER_SIM))
    ceiling = configured if configured and configured > 0 else budget_window

    head = headroom_cpus(snap)
    mem = snap.get("avail_gb")
    if head is None:
        return ceiling, "量不到別人的 CPU 用量，退回設定值"

    by_cpu = max(MIN_WINDOW, int(head // CPUS_PER_SIM))
    window = min(ceiling, by_cpu)
    why = (f"別人佔 {snap['others_cpu_pct'] / 100.0:.1f}/{TOTAL_CPUS} 核，"
           f"餘裕 {head:.1f} 核")

    if mem is not None and mem < MIN_AVAIL_GB:
        window = MIN_WINDOW
        why += f"；可用記憶體僅 {mem:.1f} GB < {MIN_AVAIL_GB:.0f} GB，縮到最小"
    elif mem is not None:
        why += f"，可用記憶體 {mem:.1f} GB"

    if window < ceiling:
        why += f" -> 視窗 {ceiling} 降為 {window}"
    else:
        why += f" -> 視窗 {window}"
    return window, why


def wait_for_headroom(need_cpus, max_wait_s=600, poll_s=30, log=print):
    """等到有 ``need_cpus`` 顆核心的餘裕，或等到逾時為止。

    **逾時後一定放行**，並且明說是逾時放行而不是等到了 —— 因為主機上有
    掛了數週的長期佔用，「等到夠閒」這個條件可能永遠不成立，而沉默地等下去
    在外部看起來跟當掉沒有分別。回傳 ``(ok, waited_s, snap)``，``ok`` 表示
    是真的等到了還是逾時放行。
    """
    t0 = time.time()
    while True:
        snap = snapshot()
        head = headroom_cpus(snap)
        if head is None or head >= need_cpus:
            return True, time.time() - t0, snap
        waited = time.time() - t0
        if waited >= max_wait_s:
            log(f"[hostload] 等了 {waited:.0f}s 仍只有 {head:.1f} 核餘裕"
                f"（需要 {need_cpus}），逾時放行 —— 主機上可能有長期佔用，"
                f"不再等待。")
            return False, waited, snap
        log(f"[hostload] 餘裕 {head:.1f} 核 < 需要的 {need_cpus}，"
            f"已等 {waited:.0f}s / {max_wait_s}s，{poll_s}s 後再看")
        time.sleep(poll_s)


if __name__ == "__main__":       # 手動看一眼現在的狀況
    snp = snapshot()
    print(f"total_cpus      {snp['total_cpus']}")
    print(f"load1           {snp['load1']}")
    print(f"others_cpu      {(snp['others_cpu_pct'] or 0) / 100.0:.1f} 核")
    print(f"avail_mem       {snp['avail_gb']:.1f} GB"
          if snp["avail_gb"] is not None else "avail_mem       ?")
    print(f"headroom        {headroom_cpus(snp):.1f} 核")
    for cfg in (0, 8, 16, 64):
        w, r = advised_window(cfg, snp)
        print(f"configured={cfg:<3} -> window={w:<3} ({r})")
