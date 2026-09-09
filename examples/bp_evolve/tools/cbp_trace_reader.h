// Reader for the CBP-NG trace format.
//
// Split out of cbp2champsim.cpp so that the trace converter and the port
// self-check (tage_selfcheck.cpp) decode the stream with the same code.  Two
// readers would be two chances to desynchronise, and a desynchronised reader
// fails silently here -- the next PC is simply garbage, never an error -- so
// there is exactly one.

#ifndef CBP_TRACE_READER_H
#define CBP_TRACE_READER_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

enum class INST_CLASS : uint8_t {
    ALU = 0, LOAD = 1, STORE = 2,
    BR_COND = 3, BR_UNCOND_DIRECT = 4, BR_UNCOND_INDIRECT = 5,
    FP = 6, ALU_SLOW = 7, UNDEF = 8,
    BR_CALL_DIRECT = 9, BR_CALL_INDIRECT = 10, BR_RETURN = 11
};

struct decoded {
    uint64_t pc = 0;
    uint64_t next_pc = 0;
    INST_CLASS inst_class = INST_CLASS::ALU;
    bool branch = false;
    bool taken = false;
    bool has_mem = false;
    uint64_t mem_addr = 0;
    std::vector<uint8_t> in_regs;    // raw CBP names, all of them
    std::vector<uint8_t> out_regs;
};

struct out_of_instructions : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class cbp_reader
{
    gzFile fp;
    uint64_t n_read = 0;

public:
    explicit cbp_reader(const std::string& path)
    {
        fp = gzopen(path.c_str(), "rb");
        if (!fp) throw std::runtime_error("cannot open " + path);
        // The eight-byte magic marks the Ampere revision of the format.  A
        // trace without it starts at byte zero, so rewind rather than assume.
        const char magic[9] = "CBPNGAmp";
        for (unsigned i = 0; i < 8; i++) {
            char c;
            read(c);
            if (c != magic[i]) { gzseek(fp, 0, SEEK_SET); break; }
        }
    }
    ~cbp_reader() { if (fp) gzclose(fp); }

    uint64_t count() const { return n_read; }

    template <typename T> void read(T& obj)
    {
        long unsigned got = gzread(fp, static_cast<void*>(&obj), sizeof(T));
        if (got < sizeof(T)) {
            if (got == 0 && gzeof(fp)) throw out_of_instructions("eof");
            throw std::runtime_error("short read");
        }
    }
    void skip(size_t bytes)
    {
        if (gzseek(fp, static_cast<z_off_t>(bytes), SEEK_CUR) < 0)
            throw std::runtime_error("seek failed");
    }

    // Byte-for-byte the walk in cbp-ng's trace_reader.hpp, including the
    // base-update rule that decides how many value bytes follow.  Any
    // divergence here desynchronises the stream silently -- the next PC would
    // simply be garbage rather than an error -- so it is transcribed rather
    // than reimplemented.
    decoded next()
    {
        decoded d;
        read(d.pc);
        d.next_pc = d.pc + 4;
        read(d.inst_class);
        if (d.inst_class == INST_CLASS::UNDEF)
            throw std::runtime_error("undefined instruction class in trace");

        bool has_base_update = false;
        if (d.inst_class == INST_CLASS::LOAD || d.inst_class == INST_CLASS::STORE) {
            uint8_t access_size;
            read(d.mem_addr);
            read(access_size);
            d.has_mem = true;
            read(has_base_update);
            if (d.inst_class == INST_CLASS::STORE) skip(1);
        }

        const bool is_branch =
            d.inst_class == INST_CLASS::BR_COND ||
            d.inst_class == INST_CLASS::BR_UNCOND_DIRECT ||
            d.inst_class == INST_CLASS::BR_UNCOND_INDIRECT ||
            d.inst_class == INST_CLASS::BR_CALL_DIRECT ||
            d.inst_class == INST_CLASS::BR_CALL_INDIRECT ||
            d.inst_class == INST_CLASS::BR_RETURN;

        if (is_branch) {
            d.branch = true;
            read(d.taken);
            if (d.taken) read(d.next_pc);
        }

        uint8_t num_in = 0, num_out = 0;
        std::vector<uint8_t> int_in;
        read(num_in);
        for (int i = 0; i < num_in; i++) {
            uint8_t r; read(r);
            d.in_regs.push_back(r);
            if (r < 32) int_in.push_back(r);
        }
        read(num_out);
        for (int i = 0; i < num_out; i++) {
            uint8_t r; read(r);
            d.out_regs.push_back(r);
        }

        uint8_t base_update_reg = 0;
        if (has_base_update) {
            if (d.inst_class == INST_CLASS::STORE) {
                if (num_out != 1) has_base_update = false;
                else base_update_reg = d.out_regs[0];
            } else {
                if (num_out <= 1) has_base_update = false;
                else {
                    std::vector<uint8_t> int_out, overlap;
                    std::copy_if(d.out_regs.begin(), d.out_regs.end(),
                                 std::back_inserter(int_out),
                                 [](uint8_t r) { return r < 32; });
                    std::set_intersection(int_in.begin(), int_in.end(),
                                          int_out.begin(), int_out.end(),
                                          std::back_inserter(overlap));
                    if (overlap.size() == 1) base_update_reg = overlap[0];
                    else has_base_update = false;
                }
            }
        }

        for (int i = 0; i < num_out; i++) {
            skip(8);
            const bool matching_base_upd = has_base_update && base_update_reg == d.out_regs[i];
            const bool integer_register =
                (d.out_regs[i] < 32) || (d.out_regs[i] == 64) || (d.out_regs[i] == 65);
            if (!matching_base_upd && !integer_register) skip(8);
        }

        n_read++;
        return d;
    }
};

#endif // CBP_TRACE_READER_H
