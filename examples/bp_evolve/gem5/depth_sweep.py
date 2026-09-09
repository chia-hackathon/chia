"""gem5 SE-mode O3 core with the prediction-to-execution distance as a knob.

Tier 2 exists to ask CBP-NG's fixed assumption as a question: what happens to
the ranking when the pipeline is not nine stages deep?  gem5's stock config
scripts do not expose that, because it is not usually a thing you sweep -- so
this script does, and nothing else.  It is deliberately small: every parameter
it does not set is a gem5 default, which is easier to defend than a config
tuned to make a point.

``--depth N`` is the number of cycles between a prediction being made and the
branch resolving in execute.  gem5's O3 model splits that across four
inter-stage delays, and this distributes N over them as evenly as four stages
allow.  Nothing in CBP-NG's model says *where* a pipeline is long -- only how
far a prediction is from its resolution -- so an even split is the assumption
that adds the least.

Usage (as invoked by Gem5Node.run_gem5):
    gem5.opt -d <outdir> depth_sweep.py --binary <path> --depth 9 \
             [--max-insts N] [--options "..."]
"""

import argparse

import m5
from m5.objects import (
    AddrRange, BadAddr, Cache, L2XBar, MemCtrl, Process, Root, SEWorkload,
    SrcClockDomain, SystemXBar, System, VoltageDomain, DDR3_1600_8x8,
)
from m5.objects import BranchPredictor, EvolvedBP, X86O3CPU


class L1Cache(Cache):
    assoc = 8
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 16
    tgts_per_mshr = 20


class L1ICache(L1Cache):
    size = "32kB"


class L1DCache(L1Cache):
    size = "32kB"


class L2Cache(Cache):
    size = "256kB"
    assoc = 8
    tag_latency = 12
    data_latency = 12
    response_latency = 12
    mshrs = 20
    tgts_per_mshr = 12


def split_depth(total):
    """Spread ``total`` cycles across fetch->decode->rename->IEW->execute.

    Each stage gets at least one cycle, because a zero-cycle stage in gem5's O3
    is not a shorter pipeline -- it is a differently-behaved one.  The minimum
    representable depth is therefore four, and a request below that is clamped
    rather than silently reinterpreted.
    """
    stages = 4
    total = max(stages, int(total))
    base, extra = divmod(total, stages)
    return [base + (1 if i < extra else 0) for i in range(stages)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--max-insts", type=int, default=0)
    ap.add_argument("--options", default="")
    args = ap.parse_args()

    f2d, d2r, r2i, i2e = split_depth(args.depth)

    system = System()
    system.clk_domain = SrcClockDomain(clock="3GHz",
                                       voltage_domain=VoltageDomain())
    system.mem_mode = "timing"
    system.mem_ranges = [AddrRange("2GB")]

    cpu = X86O3CPU()
    # The four knobs that are the whole point of this script.
    cpu.fetchToDecodeDelay = f2d
    cpu.decodeToRenameDelay = d2r
    cpu.renameToIEWDelay = r2i
    cpu.issueToExecuteDelay = i2e
    # gem5's inter-stage time buffers are sized by these two, and a stage delay
    # larger than the buffer trips an assertion inside IEW's constructor before
    # the simulation starts -- "Assertion `idx >= -past && idx <= future'
    # failed", which says nothing about pipeline depth.  The defaults (5) cap
    # the sweep at about twenty stages, so they are sized from the request.
    span = max(5, f2d, d2r, r2i, i2e) + 1
    cpu.forwardComSize = span
    cpu.backComSize = span
    # The predictor under test.  gem5 splits the BPU into a direction
    # predictor, a BTB, a RAS and an indirect predictor; only the first is the
    # evolved design, and the other three stay at their defaults so that a
    # depth-to-depth difference is about the design and not about the rest of
    # the front end.
    cpu.branchPred = BranchPredictor(conditionalBranchPred=EvolvedBP())
    system.cpu = cpu

    system.membus = SystemXBar()
    system.membus.badaddr_responder = BadAddr()
    system.membus.default = system.membus.badaddr_responder.pio

    cpu.icache = L1ICache()
    cpu.dcache = L1DCache()
    cpu.icache.cpu_side = cpu.icache_port
    cpu.dcache.cpu_side = cpu.dcache_port

    system.l2bus = L2XBar()
    cpu.icache.mem_side = system.l2bus.cpu_side_ports
    cpu.dcache.mem_side = system.l2bus.cpu_side_ports

    system.l2cache = L2Cache()
    system.l2cache.cpu_side = system.l2bus.mem_side_ports
    system.l2cache.mem_side = system.membus.cpu_side_ports

    cpu.createInterruptController()
    cpu.interrupts[0].pio = system.membus.mem_side_ports
    cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
    cpu.interrupts[0].int_responder = system.membus.mem_side_ports

    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR3_1600_8x8()
    system.mem_ctrl.dram.range = system.mem_ranges[0]
    system.mem_ctrl.port = system.membus.mem_side_ports
    system.system_port = system.membus.cpu_side_ports

    process = Process()
    process.cmd = [args.binary] + (args.options.split() if args.options else [])
    system.workload = SEWorkload.init_compatible(args.binary)
    cpu.workload = process
    cpu.createThreads()

    if args.max_insts > 0:
        cpu.max_insts_any_thread = args.max_insts

    root = Root(full_system=False, system=system)
    m5.instantiate()
    print(f"depth_sweep: depth={args.depth} "
          f"(fetch->decode {f2d}, decode->rename {d2r}, "
          f"rename->IEW {r2i}, issue->execute {i2e})")
    exit_event = m5.simulate()
    print(f"exiting @ tick {m5.curTick()} because {exit_event.getCause()}")


main()
