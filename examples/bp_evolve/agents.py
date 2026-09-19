"""The LLM nodes: Design, Port, Repair.

Three places a model is consulted, and none of them decides anything.  Design
writes a HARCOM predictor; Port translates one into a ChampSim or gem5 module;
Repair fixes source the lint gate or the compiler rejected.  Whether the result
is legal is :mod:`harcom_lint`'s call, whether it is better is :mod:`vfs`'s, and
whether it enters the archive is :mod:`archive`'s.

Each returns a validated dict or raises :class:`AgentError`.  A malformed reply
is retried with the parse error appended, because a schema slip is a formatting
problem and says nothing about the design.

:func:`offline_design` is not an agent.  It is the control: a deterministic
mutation of the seed's template parameters, so the whole cascade can be
exercised -- lint, build, run, score, archive, promote -- without an LLM in the
loop.  If a sweep fails, running the same sweep with ``--arm offline`` is how
you tell a bad agent from a broken harness.
"""

from __future__ import annotations

import json
import random
import re
import time
from string import Template

from chia.base.ChiaFunction import get
from chia.base.llm_call import QueryResult

import constants as C


class AgentError(RuntimeError):
    """The node could not produce a reply matching its schema."""


def _load(name: str, **subs: str) -> str:
    """``${KEY}`` substitution, so prompt text can contain literal braces.

    C++ and JSON both use braces heavily and ``str.format`` would choke on
    every one of them.
    """
    text = (C.PROMPTS_DIR / name).read_text()
    for k, v in subs.items():
        text = text.replace("${" + k + "}", v)
    return text


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """Pull one JSON object out of a reply, fenced or bare."""
    for candidate in ([m.group(1) for m in _FENCE.finditer(text)] + [text]):
        candidate = candidate.strip()
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            obj = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise AgentError("no JSON object found in the reply")


# ---------------------------------------------------------------------------
# The LLM handle
# ---------------------------------------------------------------------------

def make_llm(logging_name: str = "bp_evolve_agent", *, resume_session: bool = True):
    """One instance per lineage, reused across its repair rounds.

    Deliberately not shared across lineages: a design agent that remembers the
    last variant's failures is doing hill-climbing in its context window rather
    than in the archive, and the archive is the thing being measured.

    ``resume_session=False`` gives a cold context on every call.  The bounds
    agent uses it: its prompt already carries the full history of what was
    asked for and what was refused, so a remembered session adds no
    information -- it only lets an earlier generation's framing persist into a
    later one, which is exactly the thing being measured.
    """
    if C.LLM_BACKEND == "vertex":
        from chia.models.vertex import VertexGeminiLLM
        return VertexGeminiLLM(
            model=C.GEMINI_MODEL,
            system_message=C.LLM_SYSTEM_MESSAGE,
            timeout_seconds=C.LLM_TIMEOUT_SECONDS,
            logging_name=logging_name,
            project=C.VERTEX_PROJECT,
            location=C.VERTEX_LOCATION,
            max_tokens=C.VERTEX_MAX_TOKENS,
        )
    if C.LLM_BACKEND != "claude":
        raise ValueError(
            f"unknown BPE_LLM_BACKEND {C.LLM_BACKEND!r}; expected 'claude' or 'vertex'")
    from chia.models.claude import ClaudeCodeLLM
    return ClaudeCodeLLM(
        model=C.LLM_MODEL,
        system_message=C.LLM_SYSTEM_MESSAGE,
        timeout_seconds=C.LLM_TIMEOUT_SECONDS,
        logging_name=logging_name,
        resume_session=resume_session,
        projects_cwd=C.CLAUDE_PROJECTS_DIR,
        extra_cli_args=list(C.LLM_EXTRA_CLI_ARGS),
        retry_on_timeout=False,
    )


def _ask(llm, prompt: str, tools: list | None = None) -> QueryResult:
    return get(
        llm.prompt.options(resources={"llm": C.LLM_RESOURCE}).chia_remote(
            llm, prompt, tools or [])
    )


def _ask_json(llm, prompt: str, validate, tools=None, retries: int = 2,
              budget_seconds: float | None = None) -> dict:
    """Ask, parse, validate; on failure re-ask with the reason appended."""
    last = ""
    deadline = time.monotonic() + budget_seconds if budget_seconds else None
    for attempt in range(retries + 1):
        if deadline is not None:
            left = deadline - time.monotonic()
            if left < 60:
                raise AgentError(f"design budget of {budget_seconds:.0f}s spent: {last}")
            # The llm travels to the worker with each call, so this is the
            # cap the CLI subprocess runs under.
            llm.timeout_seconds = int(left)
        text = prompt if attempt == 0 else (
            f"{prompt}\n\n---\nYour previous reply was rejected: {last}\n"
            f"Reply with ONLY the JSON object, matching the schema exactly.")
        res = _ask(llm, text, tools)
        if not res.success:
            last = f"the model call failed (rc={res.returncode})"
            continue
        try:
            obj = extract_json(res.result)
        except AgentError as e:
            last = str(e)
            continue
        ok, reason = validate(obj)
        if ok:
            return obj
        last = reason
    raise AgentError(f"no valid reply after {retries + 1} attempts: {last}")


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")


def _validate_source_reply(obj: dict, key: str = "source") -> tuple[bool, str]:
    """Every source-producing node returns the same three fields."""
    if not isinstance(obj.get(key), str) or len(obj[key].strip()) < 200:
        return False, f"'{key}' must be the complete source, not a fragment or a diff"
    name = obj.get("struct_name")
    if not isinstance(name, str) or not _IDENT.match(name or ""):
        return False, "'struct_name' must be a valid C++ identifier"
    if not re.search(rf"\bstruct\s+{re.escape(name)}\b", obj[key]):
        return False, f"'source' does not declare `struct {name}`"
    if not isinstance(obj.get("rationale"), str) or not obj["rationale"].strip():
        return False, "'rationale' must say what was changed and why"
    return True, ""


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------

def design(llm, *, parent_source: str, parent_summary: dict,
           archive_summary: str, feedback: str, generation: int,
           reference_source: str = "", winner_design: str = "",
           tools=None) -> dict:
    """Propose one new predictor by editing a parent's HARCOM source.

    The parent is drawn from the archive by the caller, not chosen here: which
    region of the space to explore is a search decision, and search decisions
    belong to the framework.
    """
    prompt = _load(
        "design.md",
        PARENT_SOURCE=parent_source,
        PARENT_SUMMARY=json.dumps(parent_summary, indent=2),
        ARCHIVE=archive_summary,
        FEEDBACK=feedback or "(this is the first variant from this parent)",
        GENERATION=str(generation),
        REFERENCE_SOURCE=reference_source or "// (reference not available)",
        WINNER_DESIGN=winner_design or "(no winning design to show)",
    )
    try:
        return _ask_json(llm, prompt, _validate_source_reply, tools,
                         budget_seconds=C.DESIGN_BUDGET_SECONDS)
    finally:
        llm.timeout_seconds = C.LLM_TIMEOUT_SECONDS   # repairs keep the default


def repair(llm, *, source: str, struct_name: str, diagnostics: str,
           round_no: int, tools=None) -> dict:
    """Fix source the gate or the compiler rejected.

    Handed the filtered diagnostics, not the raw template backtrace -- see
    ``cbp_ng._filter_cxx_diagnostics``.  A repair round that spends its context
    reading HARCOM's internals is a wasted one.
    """
    prompt = _load(
        "repair.md",
        SOURCE=source,
        STRUCT_NAME=struct_name,
        DIAGNOSTICS=diagnostics,
        ROUND=str(round_no),
        MAX_ROUNDS=str(C.MAX_REPAIR_ROUNDS),
    )
    return _ask_json(llm, prompt, _validate_source_reply, tools)


def port(llm, *, harcom_source: str, struct_name: str, target: str,
         tier0_mpki: float, tools=None) -> dict:
    """Translate a HARCOM predictor into a ChampSim or gem5 branch module.

    The MPKI it must reproduce is stated up front, because the port is checked
    against it (:func:`evaluator.check_port_fidelity`) and a translator that
    knows the acceptance test can aim at it.  That is not cheating: the test is
    "predict the same branches the same way", and there is no way to pass it
    except by doing so.
    """
    if target not in ("champsim", "gem5"):
        raise ValueError(f"target must be 'champsim' or 'gem5'; got {target!r}")
    prompt = _load(
        f"port_{target}.md",
        HARCOM_SOURCE=harcom_source,
        STRUCT_NAME=struct_name,
        TIER0_MPKI=f"{tier0_mpki:.4f}",
        TOLERANCE=f"{C.MPKI_AGREEMENT_TOLERANCE:.0%}",
    )
    return _ask_json(llm, prompt, _validate_source_reply, tools)


# ---------------------------------------------------------------------------
# The control arm: no LLM
# ---------------------------------------------------------------------------

# TAGE's template parameters, with the range each may take.  Ranges are not
# arbitrary: LOGG/LOGB span roughly a quarter to four times the shipped size,
# NUMG covers the usual 4-16 component count, and GHIST reaches far enough to
# make the longest history genuinely long.  Widening them further mostly buys
# designs that fail to compile.
_TAGE_PARAMS = {
    "LOGLB": (5, 7, 6),      # (low, high, default) -- log2 block length + 2
    "NUMG": (4, 12, 8),      # number of tagged components
    "LOGG": (9, 13, 11),     # log2 entries per component
    "LOGB": (10, 14, 12),    # log2 bimodal entries
    "TAGW": (8, 14, 11),     # tag width
    "GHIST": (40, 400, 100), # longest history length
    "LOGP1": (12, 16, 14),   # log2 first-level entries
    "GHIST1": (4, 10, 6),    # first-level history
}
_TAGE_ORDER = ("LOGLB", "NUMG", "LOGG", "LOGB", "TAGW", "GHIST", "LOGP1", "GHIST1")
# The table as shipped; `_TAGE_PARAMS` is the live one and the bounds arms move it.
DEFAULT_TAGE_PARAMS = dict(_TAGE_PARAMS)

# ---------------------------------------------------------------------------
# The search space itself, as an object of study
# ---------------------------------------------------------------------------
#
# `_TAGE_PARAMS` above is not a law of nature.  Every bound in it was a
# judgement call, and the ranges carry a comment justifying LOGG, LOGB, NUMG and
# GHIST -- but not LOGLB, whose ceiling of 7 was never argued for.  The audit
# arm exists to find out what an agent does when the bounds are handed to it
# rather than fixed: it may widen or narrow any of them, and what it reaches for
# first is the measurement.
#
# The guard below is not a taste filter and deliberately does not encode any
# opinion about which bounds are architecturally reasonable.  It rejects exactly
# one class of change: the one that breaks the renderer SILENTLY.
#
#   ports/tage_core.h.in:67   HTAGBITS    = TAGW - (LOGLB - 2)
#   ports/tage_core.h.in:68   BINDEX_BITS = LOGB - (LOGLB - 2)
#
# Both are `uint64_t`.  If LOGLB-2 ever reaches TAGW or LOGB the subtraction
# wraps, `bmask(bits >= 64)` returns ~0, and the predictor keeps running with
# its tag comparison disabled -- no crash, no build error, just a scored result
# that means nothing.  A bound set that permits that combination is rejected
# whatever its rationale, because the failure is invisible downstream and would
# enter the archive as a real design.

_BOUNDS_MIN_BITS = 1


def bounds_snapshot() -> dict:
    """The live bound table, as plain data."""
    return {k: list(v) for k, v in _TAGE_PARAMS.items()}


def validate_bounds(bounds: dict) -> tuple[bool, str]:
    """Structural check, plus the silent-underflow guard.

    Worst case over the whole box, not over any one design: the mutator may
    combine LOGLB at its ceiling with TAGW and LOGB at their floors, so that is
    the combination the guard has to hold for.
    """
    if set(bounds) != set(_TAGE_PARAMS):
        missing = set(_TAGE_PARAMS) ^ set(bounds)
        return False, f"bounds must name exactly the 8 parameters; differs by {sorted(missing)}"
    for k, v in bounds.items():
        if len(v) != 3 or not all(isinstance(x, int) for x in v):
            return False, f"{k}: expected three integers [low, high, default]"
        low, high, dflt = v
        if low < 1 or high < low:
            return False, f"{k}: need 1 <= low <= high, got low={low} high={high}"
        if not low <= dflt <= high:
            return False, f"{k}: default {dflt} outside [{low}, {high}]"

    lineinst_max = bounds["LOGLB"][1] - 2
    for dependent in ("TAGW", "LOGB"):
        slack = bounds[dependent][0] - lineinst_max
        if slack < _BOUNDS_MIN_BITS:
            return False, (
                f"LOGLB high={bounds['LOGLB'][1]} implies LOGLINEINST="
                f"{lineinst_max}, leaving {dependent} low={bounds[dependent][0]} "
                f"with {slack} bit(s). tage_core.h.in computes "
                f"{dependent} - LOGLINEINST as uint64_t; at {slack} it wraps and "
                f"the tag compare silently disables. Raise {dependent}'s low to "
                f"at least {lineinst_max + _BOUNDS_MIN_BITS}, or lower LOGLB's high.")
    return True, ""


def apply_bounds(bounds: dict) -> None:
    """Install a validated bound table.  Raises rather than half-applying."""
    ok, reason = validate_bounds(bounds)
    if not ok:
        raise AgentError(f"refusing to install bounds: {reason}")
    for k, v in bounds.items():
        _TAGE_PARAMS[k] = tuple(v)


def _validate_bounds_reply(obj: dict) -> tuple[bool, str]:
    changes = obj.get("changes")
    if not isinstance(changes, list):
        return False, "'changes' must be a list (use [] to leave the bounds alone)"
    if not isinstance(obj.get("rationale"), str) or not obj["rationale"].strip():
        return False, "'rationale' must say what is being widened or narrowed and why"
    for c in changes:
        if not isinstance(c, dict):
            return False, "each change must be an object"
        if c.get("param") not in _TAGE_PARAMS:
            return False, f"unknown parameter {c.get('param')!r}"
        for field in ("low", "high"):
            if not isinstance(c.get(field), int):
                return False, f"{c.get('param')}: '{field}' must be an integer"
    return True, ""


def tune_bounds(llm, *, archive_summary: str, history: str, generation: int,
                occupancy: str = "(unavailable)", tools=None) -> dict:
    """Ask the agent whether to move the search-space bounds.

    Returns the reply as given.  Applying it is the caller's job, because a
    rejected change is data -- what the agent reached for and why the harness
    would not allow it belongs in the record just as much as an accepted one.

    ``occupancy`` is where the evaluated designs actually landed inside each
    bound, elites and non-elites alike.  Without it the agent sees only the
    elites, and a parameter pinned against its ceiling is indistinguishable
    from one that converged -- an ambiguity measured at 3/6 either way across
    repeated cold-context calls on the same archive.
    """
    bounds = "\n".join(
        f"  {k:7s} low={v[0]:<5d} high={v[1]:<5d} default={v[2]}"
        for k, v in _TAGE_PARAMS.items())
    prompt = _load(
        "tune_bounds.md",
        BOUNDS=bounds,
        ARCHIVE=archive_summary,
        OCCUPANCY=occupancy or "(unavailable)",
        HISTORY=history or "(no bounds have been changed yet)",
        GENERATION=str(generation),
    )
    return _ask_json(llm, prompt, _validate_bounds_reply, tools)


# The baseline the bounds agent has to beat.  Without it an agent that widens
# LOGLB has only done the obvious thing -- "designs are piling up on the
# ceiling, raise it" -- and nothing says it did better than a threshold would.
# The rule sees exactly what the agent sees (the occupancy table, as numbers)
# and makes exactly that obvious move: a bound on which at least `threshold`
# of all evaluated designs sit is widened by one step, and nothing is ever
# narrowed.  Its reply goes through the same guard and the same record.
RULE_BOUNDS_THRESHOLD = 0.25


def rule_bounds(values: dict[str, list[int]], *,
                threshold: float = RULE_BOUNDS_THRESHOLD) -> dict:
    """Widen every bound that at least ``threshold`` of the designs sit on.

    ``values`` maps each parameter to the value every evaluated design took.
    Returns a reply shaped like :func:`tune_bounds`'s.
    """
    changes, skipped = [], []
    table = bounds_snapshot()
    for k in _TAGE_ORDER:
        v = values.get(k) or []
        if not v:
            continue
        low, high, _ = _TAGE_PARAMS[k]
        at_lo = sum(1 for x in v if x <= low) / len(v)
        at_hi = sum(1 for x in v if x >= high) / len(v)
        new_low = low - 1 if at_lo >= threshold and low > 1 else low
        new_high = high + 1 if at_hi >= threshold else high
        if (new_low, new_high) == (low, high):
            continue
        # The rule does not know to raise TAGW's floor along with LOGLB's
        # ceiling, so a move the guard would refuse is dropped here rather
        # than sinking the rule's other moves with it.
        trial = dict(table, **{k: [new_low, new_high,
                                   max(new_low, min(new_high, table[k][2]))]})
        if not validate_bounds(trial)[0]:
            skipped.append(k)
            continue
        table = trial
        changes.append({
            "param": k, "low": new_low, "high": new_high,
            "why": (f"{at_lo:.0%} of {len(v)} designs at low, "
                    f"{at_hi:.0%} at high; threshold {threshold:.0%}")})
    rationale = (f"rule: widen by one step every bound at least "
                 f"{threshold:.0%} of evaluated designs sit on")
    if skipped:
        rationale += f"; not moved, the guard would refuse it: {', '.join(skipped)}"
    return {"changes": changes, "rationale": rationale}


# Which of those survive the port.  LOGP1 and GHIST1 size the P1 gshare, and
# neither ChampSim nor gem5 has an interface that can hold a first-level
# predictor that a second level overrides a cycle later, so Tiers 1 and 2 see
# the TAGE alone.  `offline_design` still mutates them because they are real
# design parameters at Tier 0 -- they cost latency and energy there -- but a
# mutation confined to them is invisible downstream, and callers are told.
PORTED_PARAMS = ("LOGLB", "NUMG", "LOGG", "LOGB", "TAGW", "GHIST")
UNPORTED_PARAMS = ("LOGP1", "GHIST1")


def seed_template_args() -> str:
    """The seed predictor's own template arguments, in ``_TAGE_ORDER``.

    The starting point every offline mutation walks away from, and the
    parameter set ``ports/tage_core.h.in`` renders to reproduce the shipped
    TAGE -- which is what makes it the right input to the port self-check.
    """
    return ",".join(str(_TAGE_PARAMS[k][2]) for k in _TAGE_ORDER)


def offline_design(parent_source: str, struct_name: str, rng: random.Random,
                   *, parent_args: dict | None = None) -> dict:
    """Mutate the seed's template parameters. No model, no creativity.

    This is the harness control, and it is worth being explicit about what it
    can and cannot show.  It *can* show that the cascade runs: that a variant
    builds, scores, lands in a cell, gets promoted and comes back with a
    failure profile.  It *cannot* find a new prediction algorithm -- it only
    resizes the one it was given.  A sweep where the offline arm matches the
    LLM arm is not a sweep where the agent did well; it is one where the search
    space collapsed to table sizes.
    """
    args = dict(parent_args or {k: v[2] for k, v in _TAGE_PARAMS.items()})
    # Perturb two parameters, not one: single-parameter steps walk along an
    # axis of the archive grid and tend to land back in the parent's own cell.
    for key in rng.sample(_TAGE_ORDER, 2):
        low, high, _ = _TAGE_PARAMS[key]
        step = rng.choice([-2, -1, 1, 2]) if high - low > 4 else rng.choice([-1, 1])
        args[key] = max(low, min(high, args[key] + step))

    template_args = ",".join(str(args[k]) for k in _TAGE_ORDER)
    return {
        "source": parent_source,
        "struct_name": struct_name,
        "template_args": template_args,
        "args": args,
        "rationale": ("offline control: template parameters "
                      + ", ".join(f"{k}={args[k]}" for k in _TAGE_ORDER)),
    }


def offline_port(template_args: str, *, target: str = "champsim") -> dict:
    """Render the ChampSim or gem5 translation of the seed TAGE.

    The porting counterpart of :func:`offline_design`, and it exists for the
    same reason: so a failing Tier-1 or Tier-2 round can be diagnosed as a bad
    agent or a broken harness, rather than guessed at.

    It is also the only way to establish what the *interface* costs.  CBP-NG
    predicts a whole cache line per cycle and advances its global history once
    per prediction block; ChampSim and gem5 both ask once per branch and offer
    no block hook, so all three run the same predictor over histories that
    advance at different rates.  That gap is real and is not a translation
    error -- but the only way to tell the two apart is to have a port that is
    correct by construction, measure its gap, and hold every other port to it.
    See :func:`evaluator.check_port_fidelity`.

    Both targets are rendered from ``ports/tage_core.h.in``, which carries the
    algorithm, plus a thin per-simulator adapter.  One algorithm, two calling
    conventions: a cross-tier disagreement is then about the cost models, not
    about TAGE having been written twice.

    Unlike :func:`offline_design`, this cannot port an arbitrary design -- it
    renders one template, for the TAGE family the offline arm explores.  A
    design with a new prediction algorithm needs :func:`port`.
    """
    if target not in ("champsim", "gem5"):
        raise ValueError(f"target must be 'champsim' or 'gem5'; got {target!r}")

    params = (dict(zip(_TAGE_ORDER, (int(x) for x in template_args.split(","))))
              if template_args else {k: v[2] for k, v in _TAGE_PARAMS.items()})
    if set(_TAGE_ORDER) - set(params):
        raise AgentError(
            f"offline_port needs all of {_TAGE_ORDER}; got {sorted(params)}")

    # Template, not str.format: these files are C++ and full of braces.
    core = Template((C.PORTS_DIR / "tage_core.h.in").read_text()).substitute(
        {k: str(params[k]) for k in PORTED_PARAMS})

    # Say which parameters the port could not carry, rather than listing all
    # eight and implying it carried them.  LOGP1/GHIST1 size the P1 gshare,
    # which has no MPKI effect and no home in either simulator's interface --
    # see the header of ports/tage_core.h.in.  Two designs differing only in
    # those are one design to Tiers 1 and 2, and a rationale that did not say
    # so would make that look like a suspiciously reproducible result.
    rendered = ", ".join(f"{k}={params[k]}" for k in PORTED_PARAMS)
    dropped = ", ".join(f"{k}={params[k]}" for k in UNPORTED_PARAMS)
    if target == "champsim":
        source = Template(
            (C.PORTS_DIR / "tage_champsim.h.in").read_text()
        ).substitute(TAGE_CORE=core)
        return {"source": source, "struct_name": "evolved_bp",
                "unported": {k: params[k] for k in UNPORTED_PARAMS},
                "rationale": (f"offline port to ChampSim, rendered at {rendered}"
                              f"; not ported (P1 gshare): {dropped}")}

    # gem5 needs two files: the SimObject header and its implementation.
    header = Template(
        (C.PORTS_DIR / "tage_gem5.hh.in").read_text()).substitute(TAGE_CORE=core)
    impl = (C.PORTS_DIR / "tage_gem5.cc.in").read_text()
    return {"source": header, "impl": impl, "struct_name": "EvolvedBP",
            "unported": {k: params[k] for k in UNPORTED_PARAMS},
            "rationale": (f"offline port to gem5, rendered at {rendered}"
                          f"; not ported (P1 gshare): {dropped}")}


def rename_struct(source: str, old: str, new: str) -> str:
    """Rename a predictor struct so two variants can coexist in one checkout.

    Whole-word only.  ``tage`` appears inside ``tage_entry`` and in prose
    comments in the shipped header, and renaming those would produce source
    that no longer compiles for a reason nothing in the loop would explain.
    """
    if not _IDENT.match(new):
        raise ValueError(f"{new!r} is not a valid C++ identifier")
    return re.sub(rf"\b{re.escape(old)}\b", new, source)
