// Convert a CBP-NG trace into a ChampSim trace.
//
// Tiers 0 and 1 are only comparable if they see the same branch stream.  They
// do not read the same file format, so one of them has to be translated, and
// translating the *trace* is far safer than translating the predictor: a trace
// converter is checkable against ground truth (the branch counts CBP-NG itself
// reports), whereas a mistranslated predictor is indistinguishable from the
// cost-model disagreement the whole loop exists to measure.
//
// The CBP-NG format carries more than cbp-ng's own trace_reader.hpp keeps.  It
// reads the effective address and access size of every load and store and then
// throws them away, because a branch-prediction harness has no use for them.
// ChampSim does: without them its caches see no memory traffic, every load
// hits, and the ROB occupancy at mispredict -- the one number Tier 1 exists to
// report -- is measured on a core that never stalls.  So this reader keeps
// them.
//
// Register mapping.  ChampSim infers the *kind* of a branch from which
// registers an instruction reads and writes (inc/instruction.h), so the
// mapping is not free: emit the wrong register signature and a return becomes
// a conditional branch, changing which predictor is even consulted.  CBP-NG
// names its registers 0-30 GPR, 31 SP, 32-63 SIMD, 64 flags, 65 zero;
// ChampSim reserves 6 (SP), 25 (flags), 26 (IP) and treats 0 as "no register".
// The map below is chosen to keep those three meaningful and every other
// register merely distinct.
//
// Build:  g++ -std=c++17 -O2 -o cbp2champsim cbp2champsim.cpp -lz
// Run:    cbp2champsim <in.gz> <out.champsimtrace.gz> [max_instructions]


#include "cbp_trace_reader.h"

// ChampSim's inc/trace_instruction.h, reproduced exactly: the on-disk layout
// is the ABI between the two programs.
constexpr std::size_t NUM_INSTR_DESTINATIONS = 2;
constexpr std::size_t NUM_INSTR_SOURCES = 4;

struct input_instr {
    unsigned long long ip;
    unsigned char is_branch;
    unsigned char branch_taken;
    unsigned char destination_registers[NUM_INSTR_DESTINATIONS];
    unsigned char source_registers[NUM_INSTR_SOURCES];
    unsigned long long destination_memory[NUM_INSTR_DESTINATIONS];
    unsigned long long source_memory[NUM_INSTR_SOURCES];
};

constexpr unsigned char CS_SP = 6;
constexpr unsigned char CS_FLAGS = 25;
constexpr unsigned char CS_IP = 26;

// CBP register name -> ChampSim register name.  0 means "drop it": the zero
// register carries no dependency, and emitting it as a real name would make
// every instruction that reads x31 falsely depend on every instruction that
// writes it.
static unsigned char map_reg(uint8_t r)
{
    if (r < 31) return static_cast<unsigned char>(64 + r);   // GPR 0-30 -> 64-94
    if (r == 31) return CS_SP;                                // stack pointer
    if (r < 64) return static_cast<unsigned char>(128 + (r - 32)); // SIMD -> 128-159
    if (r == 64) return CS_FLAGS;                             // condition flags
    return 0;                                                 // 65: zero register
}

// Push a register name into a fixed-width slot array, dropping duplicates and
// zeros.  Returns false when the array is full: ChampSim gives an instruction
// two destinations and four sources and no more, so a wide CBP instruction
// loses its tail.  That costs a little dependency fidelity and nothing at all
// in branch behaviour, which is what Tier 1 is measuring.
template <std::size_t N>
static bool push_reg(unsigned char (&slots)[N], unsigned char r)
{
    if (r == 0) return true;
    for (std::size_t i = 0; i < N; i++) {
        if (slots[i] == r) return true;
        if (slots[i] == 0) { slots[i] = r; return true; }
    }
    return false;
}

struct stats {
    uint64_t instrs = 0, branches = 0, conditional = 0, taken = 0;
    uint64_t loads = 0, stores = 0;
    uint64_t by_class[12] = {0};
};

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <cbp-ng trace.gz> <out.champsimtrace.gz> [max_instructions]\n",
            argv[0]);
        return 2;
    }
    const uint64_t limit = (argc > 3) ? std::strtoull(argv[3], nullptr, 10)
                                      : UINT64_MAX;

    cbp_reader in(argv[1]);
    gzFile out = gzopen(argv[2], "wb");
    if (!out) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 2; }

    stats st;
    try {
        while (st.instrs < limit) {
            decoded d = in.next();
            input_instr o{};
            o.ip = d.pc;
            o.is_branch = d.branch ? 1 : 0;
            o.branch_taken = d.taken ? 1 : 0;

            // Non-branch registers first; the branch signature is layered on
            // top so that it always wins a contested slot.
            if (!d.branch) {
                for (uint8_t r : d.in_regs) push_reg(o.source_registers, map_reg(r));
                for (uint8_t r : d.out_regs) push_reg(o.destination_registers, map_reg(r));
                if (d.has_mem) {
                    if (d.inst_class == INST_CLASS::LOAD) o.source_memory[0] = d.mem_addr;
                    else o.destination_memory[0] = d.mem_addr;
                }
            } else {
                // ChampSim's classifier is a chain of exact register-signature
                // tests, so each branch class gets the signature that lands on
                // the matching arm -- and nothing else, because one stray
                // "other" source register turns a direct jump into an indirect
                // one.  The conditional case is the exception worth keeping
                // real registers for: its flag dependency is what makes a
                // mispredict resolve late, which is the whole subject here.
                switch (d.inst_class) {
                case INST_CLASS::BR_COND: {
                    push_reg(o.source_registers, CS_IP);
                    bool any = false;
                    for (uint8_t r : d.in_regs) {
                        unsigned char m = map_reg(r);
                        if (m == CS_SP || m == CS_IP || m == 0) continue;
                        if (push_reg(o.source_registers, m)) any = any || true;
                    }
                    if (!any) push_reg(o.source_registers, CS_FLAGS);
                    push_reg(o.destination_registers, CS_IP);
                    break;
                }
                case INST_CLASS::BR_UNCOND_DIRECT:
                    push_reg(o.destination_registers, CS_IP);
                    break;
                case INST_CLASS::BR_UNCOND_INDIRECT: {
                    bool any = false;
                    for (uint8_t r : d.in_regs) {
                        unsigned char m = map_reg(r);
                        if (m == CS_SP || m == CS_IP || m == CS_FLAGS || m == 0) continue;
                        if (push_reg(o.source_registers, m)) any = true;
                    }
                    if (!any) push_reg(o.source_registers, 64);  // a plain GPR
                    push_reg(o.destination_registers, CS_IP);
                    break;
                }
                case INST_CLASS::BR_CALL_DIRECT:
                    push_reg(o.source_registers, CS_SP);
                    push_reg(o.source_registers, CS_IP);
                    push_reg(o.destination_registers, CS_SP);
                    push_reg(o.destination_registers, CS_IP);
                    break;
                case INST_CLASS::BR_CALL_INDIRECT: {
                    push_reg(o.source_registers, CS_SP);
                    push_reg(o.source_registers, CS_IP);
                    bool any = false;
                    for (uint8_t r : d.in_regs) {
                        unsigned char m = map_reg(r);
                        if (m == CS_SP || m == CS_IP || m == CS_FLAGS || m == 0) continue;
                        if (push_reg(o.source_registers, m)) { any = true; break; }
                    }
                    if (!any) push_reg(o.source_registers, 64);
                    push_reg(o.destination_registers, CS_SP);
                    push_reg(o.destination_registers, CS_IP);
                    break;
                }
                case INST_CLASS::BR_RETURN:
                    push_reg(o.source_registers, CS_SP);
                    push_reg(o.destination_registers, CS_SP);
                    push_reg(o.destination_registers, CS_IP);
                    break;
                default:
                    break;
                }
            }

            if (gzwrite(out, &o, sizeof(o)) != static_cast<int>(sizeof(o))) {
                std::fprintf(stderr, "write failed\n");
                gzclose(out);
                return 1;
            }

            st.instrs++;
            st.by_class[static_cast<unsigned>(d.inst_class)]++;
            if (d.branch) { st.branches++; if (d.taken) st.taken++; }
            if (d.inst_class == INST_CLASS::BR_COND) st.conditional++;
            if (d.inst_class == INST_CLASS::LOAD) st.loads++;
            if (d.inst_class == INST_CLASS::STORE) st.stores++;
        }
    } catch (const out_of_instructions&) {
        // The normal way a trace ends.
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error after %llu instructions: %s\n",
                     static_cast<unsigned long long>(st.instrs), e.what());
        gzclose(out);
        return 1;
    }

    gzclose(out);

    // These counts are the check.  cbp-ng prints its own instruction, branch
    // and conditional-branch totals for the same trace; if they do not match
    // these, the converter has desynchronised and nothing downstream of it
    // means anything.
    std::printf("instructions,%llu\n", (unsigned long long)st.instrs);
    std::printf("branches,%llu\n", (unsigned long long)st.branches);
    std::printf("conditional,%llu\n", (unsigned long long)st.conditional);
    std::printf("taken,%llu\n", (unsigned long long)st.taken);
    std::printf("loads,%llu\n", (unsigned long long)st.loads);
    std::printf("stores,%llu\n", (unsigned long long)st.stores);
    // Per-class histogram, because "loads,0" is a fact about the trace rather
    // than a bug in the converter and the two look identical from outside:
    // CBP-NG's branch-only traces record every instruction's PC but flatten
    // the non-branches to ALU, so ChampSim sees no memory stream at all.
    static const char* class_names[12] = {
        "ALU", "LOAD", "STORE", "BR_COND", "BR_UNCOND_DIRECT",
        "BR_UNCOND_INDIRECT", "FP", "ALU_SLOW", "UNDEF",
        "BR_CALL_DIRECT", "BR_CALL_INDIRECT", "BR_RETURN"};
    for (unsigned i = 0; i < 12; i++)
        if (st.by_class[i])
            std::printf("class_%s,%llu\n", class_names[i],
                        (unsigned long long)st.by_class[i]);
    return 0;
}
