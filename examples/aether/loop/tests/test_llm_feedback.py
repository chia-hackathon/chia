"""Tests for loop.llm.format_feedback, in particular that passing runs now
carry a filtered tail of the benchmark's own stdout (see kernels.py's
HARNESS UPDATE note and loop.py's per-iteration simlog_NN.txt dump).

Run with the chia_env interpreter from the repo root:
    ~/.conda/envs/chia_env/bin/python -m pytest -q loop/tests/test_llm_feedback.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm  # noqa: E402
from nodes import Measurement  # noqa: E402


def _ok_measurement(cycles: int) -> Measurement:
    return Measurement(ok=True, kind="ok", cycles=cycles, instret=cycles * 2,
                       detail="ok")


def test_passing_feedback_includes_filtered_benchmark_stdout():
    sim_log = (
        "Verilator simulation starting...\n"
        "[HTIF] booting core 0\n"
        "cospike: comparing against spike model\n"
        "PROBE cold_bytes=4194304\n"
        "PROBE warm_tail=149021\n"
        "mcycle=132839\n"
    )
    fb = llm.format_feedback(_ok_measurement(132839), best=140000,
                             baseline=150000, sim_log=sim_log)

    assert "=== benchmark stdout (tail) ===" in fb
    assert "PROBE cold_bytes=4194304" in fb
    assert "PROBE warm_tail=149021" in fb
    # Framework noise should be filtered out of the appended tail section.
    tail_section = fb.split("=== benchmark stdout (tail) ===", 1)[1]
    assert "Verilator" not in tail_section
    assert "HTIF" not in tail_section
    assert "cospike" not in tail_section


def test_passing_feedback_omits_stdout_section_when_log_empty():
    fb = llm.format_feedback(_ok_measurement(132839), best=140000,
                             baseline=150000, sim_log="")
    assert "=== benchmark stdout (tail) ===" not in fb


def test_benchmark_stdout_tail_truncates_to_limit():
    huge = "\n".join(f"PROBE line {i}" for i in range(5000))
    out = llm._benchmark_stdout_tail(huge, limit=100)
    assert len(out.encode("utf-8")) <= 100 + len("...[truncated]...\n")
    assert out.startswith("...[truncated]...\n")


def test_failure_feedback_still_shows_full_sim_log_behavior():
    # incorrect/sim_failure branches already forwarded sim_log; make sure
    # that behavior is untouched by the new passing-run tail helper.
    m = Measurement(ok=False, kind="incorrect", cycles=1000, instret=None,
                    detail="self-check mismatch")
    fb = llm.format_feedback(m, best=None, baseline=150000,
                             sim_log="raw simulator dump here")
    assert "raw simulator dump here" in fb
    assert "=== benchmark stdout (tail) ===" not in fb
