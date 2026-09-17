#!/usr/bin/env python3
"""
Add the Saturn(RVV)+Gemmini(RoCC) co-existence configs to chipyard.

Until now no config in the tree instantiated both accelerators at once:
  * Saturn hangs off Shuttle through ShuttleCoreVectorParams (a dedicated vector
    port, NOT RoCC)  -- saturn.shuttle.WithShuttleVectorUnit
  * Gemmini hangs off the same tile through p(BuildRoCC)     -- gemmini.DefaultGemminiConfig
Shuttle's tile reads both (Tile.scala: `p(BuildRoCC).map(_(p))` plus the vector
unit), so the two are orthogonal and can simply be stacked.

Ordering matters: both fragments *rewrite* TilesLocated/BuildRoCC produced by
WithNShuttleCores, so they must sit to the LEFT of it in the Config chain.

Bus/beat width: Gemmini's default 16x16 int8 array wants a 128-bit dma_buswidth
and Saturn at DLEN=128 wants the same, so WithSystemBusWidth(128) +
WithShuttleTileBeatBytes(16) (already what GENV256D128ShuttleConfig uses) serves
both.

The cosim variant mirrors GENV256D128ShuttleCosimConfig (WithCospike +
WithTraceIO + WithShuttleDebugROB) so it can be run with +cospike-extension=gemmini.
Gemmini stays on DefaultGemminiConfig because libgemmini.so's static
gemmini_params.h == GemminiConfigs.defaultConfig.
"""
import sys

# The class the AETHER baseline design point elaborates as, and the anchor class
# that must already exist in GemminiConfigs.scala for the file to be the right one.
MARK = "class GENV256D128GemminiShuttleConfig"
ANCHOR = "class GemminiShuttleConfig extends Config("

# The baseline design point's Chisel source text.  loop/hwconfig.py renders this
# *same* text from its generic template for HwPoint() (its self-test asserts the
# two agree byte-for-byte), so this literal stays the single reference copy that
# the docker image build can use standalone, with no loop/ on PYTHONPATH.
BASELINE_CONFIG_TEXT = """

// ------------------------------------------------------------------
// AETHER: Saturn RVV 1.0 (vector port) + Gemmini (RoCC) on one Shuttle tile
// ------------------------------------------------------------------
class GENV256D128GemminiShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, saturn.common.VectorParams.genParams) ++
  new gemmini.DefaultGemminiConfig ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

// Same, plus cospike lockstep co-simulation (run Gemmini workloads with
// +cospike-extension=gemmini; plain RVV workloads need no extra plusarg).
class GENV256D128GemminiShuttleCosimConfig extends Config(
  new chipyard.harness.WithCospike ++
  new chipyard.config.WithTraceIO ++
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, saturn.common.VectorParams.genParams) ++
  new gemmini.DefaultGemminiConfig ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleDebugROB ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)
"""



def append_config(path, text, mark):
    """Idempotently append `text` to `path`; refuse if `mark` is already there."""
    src = open(path).read()
    if mark in src:
        return False
    if ANCHOR not in src:
        sys.exit("FATAL: anchor class GemminiShuttleConfig missing in %s" % path)
    open(path, "w").write(src.rstrip("\n") + "\n" + text)
    return True


def main(argv):
    path = argv[1]
    if not append_config(path, BASELINE_CONFIG_TEXT, MARK):
        sys.exit("FATAL: %s already patched" % path)
    print("PATCHED %s" % path)


if __name__ == "__main__":
    main(sys.argv)
