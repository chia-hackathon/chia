"""Titan: agentic design of the RISC-V Integrated Matrix Extension on Saturn.

Round one implements the integer non-widening subset of the Zvvm family --
`vmmacc.vv`, `vmtl.v`, `vmts.v` -- which is what an INT8 GEMM needs.

The modules are flat, not a package hierarchy: `constants.RUNTIME_ENV` ships
this directory as the Ray workers' working_dir, so every ChiaFunction here
deserializes on a worker by name with no pickle-by-value registration.  One
consequence worth remembering before adding a module: this directory lands on
sys.path on every worker, so a file named after a stdlib module would shadow
it.  That is why the encoding tables live in `ime_encodings.py`.
"""
