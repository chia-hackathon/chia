#!/usr/bin/env python3
"""Run ONE fixed kernel source on several simulator configs (hw design points).

No LLM, no loop: build sim -> compile the probe kernel -> run -> scrape PROBE lines.
Usage: python measure.py <kernel-name> <kernel.h path> <config> [<config> ...]
"""
import json, logging, os, re, sys, time
from pathlib import Path

ROOT = Path("/share1/saves/max410011/hackathon/aether")
sys.path.insert(0, str(ROOT / "loop"))
os.chdir(ROOT / "loop")

import ray
from constants import RUNTIME_ENV
import context, nodes

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("hwsweep")

kernel_name, src_path = sys.argv[1], sys.argv[2]
configs = sys.argv[3:]
src = Path(src_path).read_bytes()

ray.init(address="auto", runtime_env=RUNTIME_ENV)
out = ROOT / "out" / "hw-sweep"
out.mkdir(parents=True, exist_ok=True)

results = {}
for cfg in configs:
    t0 = time.time()
    context.select(kernel_name, cfg)
    log.info("=== %s / %s ===", cfg, kernel_name)
    try:
        sim = nodes.build_simulator()
        log.info("sim built in %.0fs", time.time() - t0)
        build = nodes.build_kernel(src)
        elf_name = context.K().elf_name
        if not build.success or elf_name not in build.files:
            raise RuntimeError("kernel build failed:\n" +
                               (build.stderr or build.stdout or "")[-3000:])
        run = nodes.run_kernel(sim, build.files[elf_name])
        logtext = f"{run.log or ''}\n{run.out or ''}"
        (out / f"simlog.{kernel_name}.{cfg}.txt").write_text(logtext)
        probes = dict(re.findall(r"PROBE (\w+)=(-?\d+)", logtext))
        cyc = re.findall(r"Cycles taken:\s*(\d+)", logtext)
        passed = "PASSED" in logtext
        results[cfg] = dict(ok=True, passed=passed,
                            cycles=int(cyc[0]) if cyc else None,
                            probes={k: int(v) for k, v in probes.items()},
                            secs=round(time.time() - t0, 1))
        log.info("%s -> %s", cfg, results[cfg])
    except Exception as exc:
        results[cfg] = dict(ok=False, error=str(exc)[-2000:],
                            secs=round(time.time() - t0, 1))
        log.exception("%s FAILED", cfg)
    (out / f"results.{kernel_name}.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
