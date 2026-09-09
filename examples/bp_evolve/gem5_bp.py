"""Install an evolved branch predictor into a gem5 checkout.

``Gem5Node`` can build gem5 and run it, but a branch predictor is not a runtime
argument to gem5 the way a ChampSim module is a build-time one: it is a
SimObject, so it has to exist in the source tree *and* be declared to the
Python object system *and* be listed in the SConscript before scons will
compile it.  Those three edits are what this module does, on the worker, before
:func:`Gem5Node.build_gem5` runs.

The edits are idempotent.  A sweep installs a predictor once per variant into
the same checkout, and appending the same SimObject declaration twice makes
scons fail with an error that has nothing to do with the predictor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from chia.base.ChiaFunction import ChiaFunction

BP_NAME = "EvolvedBP"
BP_STEM = "evolved_bp"

_SIMOBJECT_DECL = f'''

class {BP_NAME}(ConditionalPredictor):
    """Evolved predictor, installed by examples/bp_evolve. Do not hand-edit."""
    type = "{BP_NAME}"
    cxx_class = "gem5::branch_prediction::{BP_NAME}"
    cxx_header = "cpu/pred/{BP_STEM}.hh"
'''

# Match on the class name alone, not on its base or its quoting style. The
# first version of this checked for the full `class EvolvedBP(BranchPredictor):`
# line; when the base class changed to ConditionalPredictor the check stopped
# matching, the declaration was appended a second time, and gem5 died at
# startup inside pybind with `error_already_set` -- which says nothing about
# duplicate SimObjects. Idempotency checks have to be looser than the thing
# they are guarding.
_MARKER = f"class {BP_NAME}("


@dataclass
class InstallResult:
    success: bool
    message: str = ""
    installed_paths: tuple = ()


@ChiaFunction(resources={"gem5": 1.0})
def install_gem5_predictor(gem5_root: str, header_src: str, source_src: str,
                           config_src: str = "") -> InstallResult:
    """Write the predictor and register it with gem5's build system.

    Returns rather than raises on a missing checkout, because a Tier-2 failure
    should surface as a tier outcome the mapper can put in the agent's
    feedback, not as an exception that takes the sweep down.
    """
    pred_dir = os.path.join(gem5_root, "src", "cpu", "pred")
    if not os.path.isdir(pred_dir):
        return InstallResult(False, f"no gem5 predictor directory at {pred_dir}")

    hh = os.path.join(pred_dir, f"{BP_STEM}.hh")
    cc = os.path.join(pred_dir, f"{BP_STEM}.cc")
    with open(hh, "w") as f:
        f.write(header_src)
    with open(cc, "w") as f:
        f.write(source_src)

    # -- declare the SimObject ------------------------------------------------
    py_path = os.path.join(pred_dir, "BranchPredictor.py")
    try:
        with open(py_path) as f:
            py = f.read()
    except OSError as e:
        return InstallResult(False, f"cannot read {py_path}: {e}")
    if _MARKER not in py:
        with open(py_path, "a") as f:
            f.write(_SIMOBJECT_DECL)

    # -- add it to the build --------------------------------------------------
    # Two edits, and both are needed.  Source() compiles the translation unit;
    # naming the class in SimObject(sim_objects=[...]) is what makes scons
    # generate params/EvolvedBP.hh.  With only the first, the compile fails on
    # a missing params header -- which reads like a broken predictor and is
    # actually an unregistered one.
    scons_path = os.path.join(pred_dir, "SConscript")
    try:
        with open(scons_path) as f:
            scons = f.read()
    except OSError as e:
        return InstallResult(False, f"cannot read {scons_path}: {e}")

    changed = False
    src_line = f"Source('{BP_STEM}.cc')"
    if f"{BP_STEM}.cc" not in scons:
        scons += f"\n{src_line}\n"
        changed = True

    if f"'{BP_NAME}'" not in scons:
        marker = "SimObject('BranchPredictor.py'"
        at = scons.find(marker)
        if at < 0:
            return InstallResult(
                False,
                f"{scons_path} has no SimObject('BranchPredictor.py') block to "
                f"register {BP_NAME} in")
        list_at = scons.find("sim_objects=[", at)
        close_at = scons.find("]", list_at)
        if list_at < 0 or close_at < 0:
            return InstallResult(
                False, f"cannot find the sim_objects list in {scons_path}")
        scons = scons[:close_at] + f", '{BP_NAME}'" + scons[close_at:]
        changed = True

    if changed:
        with open(scons_path, "w") as f:
            f.write(scons)

    # -- the depth-sweep config script ---------------------------------------
    # It lives inside the checkout rather than being shipped per run: gem5 is
    # handed a path, and a path that only exists on the driver is the kind of
    # thing that works locally and fails on the first real worker.
    written = [hh, cc]
    if config_src:
        cfg_dir = os.path.join(gem5_root, "configs", "bp_evolve")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg = os.path.join(cfg_dir, "depth_sweep.py")
        with open(cfg, "w") as f:
            f.write(config_src)
        written.append(cfg)

    return InstallResult(True, f"installed {BP_NAME}", tuple(written))


def gem5_config_path(gem5_root: str) -> str:
    """Where :func:`install_gem5_predictor` puts the depth-sweep config."""
    return os.path.join(gem5_root, "configs", "bp_evolve", "depth_sweep.py")
