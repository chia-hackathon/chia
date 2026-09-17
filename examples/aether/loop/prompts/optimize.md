# Task

${OBJECTIVE}

You are editing **one file**:

    ${WORK_DIR}/${KERNEL_REL}

That is the only file that reaches the compiler. Before every build the harness
reassembles the source tree from its own pristine copies of everything else and
drops your kernel in. Edits to the C harness, the dataset, `common/`, or the
linker script are discarded — not rejected with an error, just silently
ignored. Read them for context; don't spend turns changing them.

**This path is on a different machine than the one you're running on.** Your
built-in file tools (Read, Write, Edit, Glob, and a plain Bash call) only see
*this* machine's filesystem — `${WORK_DIR}` does not exist here and never will.
The only tool that reaches it is `kernel_bash` (an MCP tool): every read,
edit, and re-check of the kernel must go through `kernel_bash`'s
`run_command`, e.g. `cat ${WORK_DIR}/${KERNEL_REL}` to read it and a shell
heredoc / `sed`/`python3 -c ...` through the same tool to write it back. Do
not conclude the file is missing because your local Read/Bash can't find it —
that just means you used the wrong tool. To save you a round trip, the
current kernel source is inlined below.

# What happens after your turn

1. Your kernel is cross-compiled (`-march=rv64gcv_zfh_zvfh -mabi=lp64d -O2`).
2. The ELF runs on a **real RTL simulator** — Verilator on `${SIM_CONFIG}`,
   not a functional model. Cycle counts are the hardware's, not an estimate.
3. ${CHECK_DESC}

A kernel that is fast and wrong scores nothing. The self-check is the gate.

# The contract you must keep

${CONTRACT}

# The hardware you are targeting

${TARGET_NOTES}

${KERNEL_NOTES}

# Required output format for every turn

Before you touch the file with `kernel_bash`, write these three lines:

    HYPOTHESIS: <what you believe is costing cycles, and why — cite beats/cycles>
    CHANGE:     <the one concrete edit you will make>
    EXPECTED:   <predicted cycle count, and roughly how many cycles you expect to save>

Then make the edit, assemble-check it, and stop with a one-sentence summary.

Rules for the sequence of turns:

- **One isolated change per turn.** If you bundle three ideas and the number
  gets worse, you have learned nothing. You have very few turns; spend each on
  a single testable hypothesis.
- **Always start from the best-known kernel**, not from your last attempt. If
  the harness tells you your change regressed, revert to the best kernel
  (its source is given back to you) before applying the next idea.
- If your `EXPECTED` and the measured result disagree, say so explicitly next
  turn and update your model of the machine — that is more valuable than
  another blind edit.
- If you conclude the kernel is genuinely at the machine's limit, say that
  plainly and stop proposing changes rather than churning. An honest "this is
  the floor for this microarchitecture, here is the evidence" is an acceptable
  and useful outcome.

# Working method

Read the current kernel first and understand its existing strategy before
rewriting it. The assembler is available to you, so you can check that your
code assembles — but you cannot run it; only the harness can, and it will.
