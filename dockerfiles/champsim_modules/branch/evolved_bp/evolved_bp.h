#ifndef BRANCH_EVOLVED_BP_H
#define BRANCH_EVOLVED_BP_H

#include <array>
#include <cstdint>

#include "address.h"
#include "modules.h"
#include "msl/fwcounter.h"

// Header-only branch predictor module (mirrors evolved_pf's inline style so
// the generated environment needs no separate .cc).  Baseline: bimodal clone.
struct evolved_bp : champsim::modules::branch_predictor {
  using branch_predictor::branch_predictor;

  static constexpr std::size_t TABLE_SIZE = 16384;
  static constexpr std::size_t PRIME = 16381;
  static constexpr std::size_t BITS = 2;

  std::array<champsim::msl::fwcounter<BITS>, TABLE_SIZE> table;

  [[nodiscard]] static constexpr auto hash(champsim::address ip) { return ip.to<unsigned long>() % PRIME; }

  bool predict_branch(champsim::address ip)
  {
    auto value = table[hash(ip)];
    return value.value() > (value.maximum / 2);
  }

  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
  {
    table[hash(ip)] += taken ? 1 : -1;
  }
};

#endif
