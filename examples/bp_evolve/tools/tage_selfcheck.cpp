// Run the ported predictor over a CBP-NG trace, without a simulator.
//
// Tier 1 answers "what does this predictor cost in a real out-of-order core",
// and answering it needs ChampSim.  But a much narrower question comes first:
// does the *port* predict the same branches the same way as the HARCOM
// original?  That question needs no core at all -- only the branch stream and
// the predictor -- and asking it here rather than through ChampSim turns a
// ten-minute build-and-simulate into about a second.
//
// The driver below is deliberately identical to ports/tage_champsim.h.in:
// predict on every branch, train only on conditional ones, advance the history
// on all of them.  So its conditional MPKI is the number Tier 1 reports, minus
// everything ChampSim contributes.  If the two disagree, the disagreement is
// in the adapter or the harness, not in the algorithm -- which is exactly the
// split that was impossible to make before.
//
// Warmup follows cbp.hpp: the predictor is driven throughout, and the counters
// are cleared once the warmup instruction count is passed, so the measurement
// window matches Tier 0's.
//
// Build:  see tools/port_selfcheck.py, which renders the core first.
// Run:    tage_selfcheck <trace.gz> [warmup_instructions] [sim_instructions]
//
// `sim_instructions` of 0 means "to the end of the trace", which is what
// cbp.hpp does when BPE_SIM_INSTRUCTIONS exceeds the trace length -- the
// default at the time of writing.  Pass a real limit only if Tier 0 was given
// one small enough to bite, or the two are measuring different windows.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "cbp_trace_reader.h"
#include "tage_core_rendered.h"

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <cbp-ng trace.gz> [warmup_instructions]\n",
                     argv[0]);
        return 2;
    }
    const uint64_t warmup = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 1000000;
    const uint64_t sim = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 0;

    cbp_reader reader(argv[1]);
    tage_core core;

    uint64_t seen = 0, instrs = 0, branches = 0, conditional = 0, mispredicts = 0;
    bool warmed = (warmup == 0);

    try {
        for (;;) {
            const decoded d = reader.next();
            seen++;
            if (warmed) instrs++;

            if (d.branch) {
                const tage_core::lookup L = core.predict(d.pc);
                if (d.inst_class == INST_CLASS::BR_COND) {
                    if (warmed) {
                        conditional++;
                        if (L.final_pred != d.taken) mispredicts++;
                    }
                    core.update(L, d.taken);
                }
                // Every branch ends a prediction block in the original, so
                // every branch advances the history -- conditional or not.
                core.advance(d.taken ? d.next_pc : (d.pc + 4));
                if (warmed) branches++;
            }

            if (!warmed && seen > warmup) {
                warmed = true;
                instrs = branches = conditional = mispredicts = 0;
            }
            if (sim != 0 && instrs >= sim) break;
        }
    } catch (const out_of_instructions&) {
    }

    if (instrs == 0) {
        std::fprintf(stderr, "trace ended during warmup; nothing measured\n");
        return 1;
    }

    std::printf("instructions,%llu\n", (unsigned long long)instrs);
    std::printf("branches,%llu\n", (unsigned long long)branches);
    std::printf("conditional,%llu\n", (unsigned long long)conditional);
    std::printf("mispredicts,%llu\n", (unsigned long long)mispredicts);
    std::printf("conditional_mpki,%.6f\n", 1000.0 * double(mispredicts) / double(instrs));
    return 0;
}
