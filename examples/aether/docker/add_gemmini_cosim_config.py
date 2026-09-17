#!/usr/bin/env python3
"""
Add a Rocket+Gemmini cosim config to chipyard.

generators/gemmini/chipyard/GemminiConfigs.scala already has GemminiRocketConfig
(gemmini.DefaultGemminiConfig == GemminiConfigs.defaultConfig).  defaultConfig is the
one whose parameters match the *static* gemmini_params.h that libgemmini.so was
compiled with (DIM 16, BANK_NUM 4, BANK_ROWS 4096, ACC_ROWS 1024, int8 elem_t /
int32 acc_t, float scale), so it is the only Gemmini config the spike golden model
can be compared against without rebuilding libgemmini.

The cosim variant just stacks WithCospike + WithTraceIO + WithDebugROB on it, the
same way saturn's MINV128D64RocketCosimConfig does for Rocket+Saturn.
"""
import sys

path = sys.argv[1]
src = open(path).read()

ANCHOR = """class GemminiShuttleConfig extends Config("""

NEW = """// Cosim config: Rocket + Gemmini, for lockstep co-simulation against spike's
// libgemmini golden model.  Run with +cospike-extension=gemmini so cospike appends
// "_xgemmini" to the ISA string it hands to spike.
//
// Uses DefaultGemminiConfig on purpose: libgemmini.so is built from a *static*
// gemmini_params.h (DIM 16, BANK_NUM 4, BANK_ROWS 4096, ACC_ROWS 1024, int8/int32),
// which is exactly GemminiConfigs.defaultConfig.  Any other Gemmini config would
// make the golden model structurally disagree with the RTL.
class GemminiRocketCosimConfig extends Config(
  new chipyard.harness.WithCospike ++
  new chipyard.config.WithTraceIO ++
  new gemmini.DefaultGemminiConfig ++
  new freechips.rocketchip.rocket.WithCease(false) ++
  new freechips.rocketchip.rocket.WithDebugROB ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class GemminiShuttleConfig extends Config("""

n = src.count(ANCHOR)
if n != 1:
    sys.exit("FATAL: expected exactly 1 anchor match in %s, found %d" % (path, n))

open(path, "w").write(src.replace(ANCHOR, NEW))
print("PATCHED %s" % path)
