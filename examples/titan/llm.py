"""LLM dispatch: one agent, two prompts, and (from r5) a fresh session a turn.

The RTL agent used to run with ``resume_session=True``, memcpy-style, on the
theory that continuity is cheaper than re-orientation.  Measured, it was not:
the transcript being replayed was mostly pasted build logs, 145M cached tokens
over five calls.  So the session is fresh by default (``RESUME_SESSION``) and
the loop pays for orientation explicitly -- iteration number, diffstat, the
agent's own notes, the result, and a path to the logs -- which costs a few
kilobytes and, unlike a transcript, is something the agent can go and re-read.

The Stage M model agent keeps its session: it is short, cheap, and gets no
orientation block.
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
import time as _time
from typing import Optional, Sequence

from chia.base.ChiaFunction import get
from chia.models.claude import ClaudeCodeLLM, RateLimitError

from constants import (AGENT_LOG_DIR, CHIPYARD_PATH, CHIPYARD_SRC_PATH,
                       COSIM_CONFIG, DISALLOWED_TOOLS, SYNTH_CONFIG,
                       CLI_MCP_AUTO_BACKGROUND_MS, LLM_EXTRA_ARGS,
                       LLM_MODEL, LLM_TIMEOUT_SECONDS, MAX_TURNS, PROMPTS_DIR,
                       SATURN_SRC_REL, SPIKE_SRC_PATH)

LLM_RESOURCE = 1.0

#: Extra ``.options()`` for the LLM task, carrying one env var into the
#: ``claude`` subprocess (``ClaudeCodeLLM`` copies ``os.environ`` and offers
#: no per-call env hook, and the subprocess runs in *this* task's worker).
#:
#: ``CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`` is the CLI's threshold, default
#: 120000, for moving a slow MCP tool call into a background task and handing
#: control back to the model.  In ``-p`` mode that is a trapdoor: the model
#: has nothing left to do, ends its turn, and the session dies with the job
#: unfinished -- r8 iteration 3 was graded on an instrumented tree that way.
#: The tool interface no longer makes a call that long (see
#: ``tools.RunDirectedTool``); this is the second lock on the same door, and
#: unlike cluster.yaml it takes effect on the next run rather than the next
#: cluster restart.
LLM_TASK_OPTS = (
    {"runtime_env": {"env_vars": {
        "CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS": CLI_MCP_AUTO_BACKGROUND_MS}}}
    if CLI_MCP_AUTO_BACKGROUND_MS else {})

#: Whether a turn continues the previous turn's conversation.
#:
#: Off by default from r5 on, and the measurement is the argument: with it on,
#: five calls in r5 read 145M cached tokens -- every 100kB build log and every
#: 300-line simulator tail the loop had ever pasted, replayed on each turn, at
#: about $23 an iteration.  What that bought was continuity, and continuity is
#: now carried by three things the agent can re-read at will instead of being
#: re-sent: the knowledge notes, the working tree's own diff, and a directory
#: of logs inside the container.  Set TITAN_RESUME_SESSION=1 to go back.
RESUME_SESSION = os.environ.get("TITAN_RESUME_SESSION", "0") not in (
    "", "0", "false", "False", "no")

#: Rate-limit wait-and-retry.  A session usage limit is not an error the loop
#: can do anything about except outlive: r4 lost a converging S1 at iteration
#: 5 to one, and the whole run with it.  Waiting past the reset and
#: re-dispatching the same prompt costs an idle cluster for a few hours and
#: saves the run.
RATE_LIMIT_MAX_RETRIES = int(os.environ.get("TITAN_RATE_LIMIT_RETRIES", "6"))
#: Slack past the advertised reset.  The reset time is a server-side boundary
#: and clocks are not identical; retrying one second early buys another wait.
RATE_LIMIT_SLACK_S = int(os.environ.get("TITAN_RATE_LIMIT_SLACK_S", "90"))
#: What to wait when the exception carries no reset time at all.
RATE_LIMIT_BLIND_WAIT_S = int(os.environ.get("TITAN_RATE_LIMIT_BLIND_WAIT_S",
                                             "1800"))
#: Never sleep longer than this on one attempt, whatever the reset says -- a
#: malformed or far-future timestamp should not park the cluster for a week.
RATE_LIMIT_MAX_WAIT_S = int(os.environ.get("TITAN_RATE_LIMIT_MAX_WAIT_S",
                                           "43200"))

# SATURN_SRC_REL already starts with "generators/", so join it to
# CHIPYARD_PATH, not to CHIPYARD_SRC_PATH -- the latter produced a
# path with a "/../generators/" in the middle, which resolves fine
# but is shown verbatim to the agent in its prompt.
SATURN_SRC_PATH = os.path.join(CHIPYARD_PATH, SATURN_SRC_REL)


def _load_prompt(name: str, **subs: str) -> str:
    """Read a prompt and substitute ``${KEY}`` placeholders.

    ``${KEY}`` rather than ``str.format``'s ``{KEY}`` because these prompts
    quote Chisel and Scala, which is full of literal braces.
    """
    with open(os.path.join(PROMPTS_DIR, name), encoding="utf-8") as fh:
        text = fh.read()
    for key, value in subs.items():
        text = text.replace(f"${{{key}}}", value)
    return text


_SYSTEM = _load_prompt("system.md", AGENT_LOG_DIR=AGENT_LOG_DIR,
                       CHIPYARD_PATH=CHIPYARD_PATH)
_IMPLEMENT_TASK = _load_prompt("implement.md", COSIM_CONFIG=COSIM_CONFIG,
                               DIRECTED_CONFIG=SYNTH_CONFIG,
                               SATURN_SRC_PATH=SATURN_SRC_PATH,
                               AGENT_LOG_DIR=AGENT_LOG_DIR,
                               CHIPYARD_SRC_PATH=CHIPYARD_SRC_PATH)
_DEBUGGER_PREAMBLE = _load_prompt("debug.md",
                                  CHIPYARD_SRC_PATH=CHIPYARD_SRC_PATH,
                                  SATURN_SRC_PATH=SATURN_SRC_PATH,
                                  AGENT_LOG_DIR=AGENT_LOG_DIR,
                                  COSIM_CONFIG=COSIM_CONFIG)

# The Stage 2 model agent.  Separate prompts and, crucially, a separate
# session: it must never see the RTL, because its model is what judges the
# RTL in Stage 3.
_MODEL_SYSTEM = _load_prompt("spike_system.md")
_MODEL_TASK = _load_prompt("spike_task.md", SPIKE_SRC_PATH=SPIKE_SRC_PATH)
_MODEL_DEBUG = _load_prompt("spike_debug.md", SPIKE_SRC_PATH=SPIKE_SRC_PATH)


def make_llm(logging_name: str = "titan_generator",
             system_message: str = None,
             resume: bool = None):
    """Build one LLM with a persistent session.

    Backend is Claude; the cluster's ``llm`` node serves it.

    ``system_message`` selects which agent this is.  The default is the RTL
    implementer.  The Spike-model stage passes its own, and the two must
    never share a session: the whole value of the model as a Stage 3 judge
    comes from its author not having seen the RTL.
    """
    return ClaudeCodeLLM(
        model=LLM_MODEL,
        system_message=system_message or _SYSTEM,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
        logging_name=logging_name,
        resume_session=RESUME_SESSION if resume is None else resume,
        projects_cwd=None,
        extra_cli_args=LLM_EXTRA_ARGS,
        # No sub-agents.  In r5 the agent spawned its own via the built-in
        # Task tool: their token spend appeared in no accounting the loop
        # keeps, their edits landed in the tree with no record of which turn
        # made them, and none of them had the system prompt that explains
        # what may be edited and why the tests are not to be touched.
        disallowed_tools=DISALLOWED_TOOLS,
        # Zero means "do not pass the flag"; the CLI on this cluster has no
        # --max-turns.  See constants.MAX_TURNS.
        max_turns=MAX_TURNS or None,
    )


def make_model_llm(logging_name: str = "titan_spike_model"):
    """The Spike-model agent: a different system prompt and a fresh session.

    Keeps ``resume_session=True`` explicitly.  Stage M is not what made r5
    expensive -- it is short, its feedback is small, and it is *not* given the
    per-iteration orientation block that lets the RTL agent start cold, so
    dropping its session would take away continuity and put nothing back.
    """
    return make_llm(logging_name, system_message=_MODEL_SYSTEM, resume=True)


def implement_model(llm, tools: Sequence[object], note: str = ""):
    """First turn of Stage M.

    *note* is appended when the run started from a seeded model tree
    (``--model-seed``): the standing task is still "implement Zvvm in Spike",
    but a working model for the previous round's instructions is already
    applied, and the agent has to be told that -- and what it scores -- before
    it starts writing code that is already there.
    """
    task = f"{_MODEL_TASK}\n\n{note}" if note else _MODEL_TASK
    return _run(llm, task, tools)


def debug_model(llm, tools: Sequence[object], feedback: str):
    return _run(llm, f"{_MODEL_DEBUG}\n\n{feedback}", tools)


def _find_rate_limit(exc: BaseException) -> Optional[BaseException]:
    """The RateLimitError inside *exc*, if there is one anywhere in the chain.

    The error is raised on a worker, so what reaches the driver is a
    ``ray.exceptions.RayTaskError`` wrapping it.  Ray exposes the original as
    ``.cause`` (and as ``as_instanceof_cause()``), but the wrapper class is
    synthesised, so ``isinstance`` against it is not reliable -- hence the
    walk over ``.cause`` / ``__cause__`` / ``__context__`` with a class-name
    match as the last resort.  A false positive here costs one wait; a false
    negative costs the run.
    """
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, RateLimitError) or \
                type(cur).__name__ == "RateLimitError":
            return cur
        nxt = getattr(cur, "cause", None)
        if not isinstance(nxt, BaseException):
            nxt = cur.__cause__ or cur.__context__
        cur = nxt
    return None


def _rate_limit_wait_seconds(err: BaseException) -> tuple:
    """(seconds to sleep, reset time as text).

    ``chia.models.claude.RateLimitError`` carries ``reset_time`` -- the CLI's
    "resets 9:20am" line, already parsed into a tz-aware datetime by
    ``parse_rate_limit_reset`` -- so in the normal case the wait is exact
    rather than a guess.
    """
    reset = getattr(err, "reset_time", None)
    if reset is None:
        return RATE_LIMIT_BLIND_WAIT_S, "unknown"
    try:
        if reset.tzinfo is None:
            reset = reset.replace(tzinfo=_dt.timezone.utc)
        now = _dt.datetime.now(_dt.timezone.utc)
        wait = (reset - now).total_seconds() + RATE_LIMIT_SLACK_S
    except Exception:
        return RATE_LIMIT_BLIND_WAIT_S, str(reset)
    # A reset already in the past means the limit lifted while the error was
    # in flight; still pause for the slack so the retry is not instantaneous.
    wait = max(RATE_LIMIT_SLACK_S, min(wait, RATE_LIMIT_MAX_WAIT_S))
    return wait, reset.isoformat()


def _event(event: str, **kw) -> None:
    """Best-effort trace event.  Never let logging kill a live run."""
    try:
        from chia.trace.profiler import get_profiler
        get_profiler().log_event(event, **kw)
    except Exception:
        pass


def _run(llm, prompt: str, tools: Sequence[object]):
    """Dispatch one turn, waiting out session usage limits.

    Everything else propagates: a rate limit is the one failure mode where
    the right response is to do nothing for a while and try the identical
    request again.  Each attempt builds a fresh ``chia_remote`` call -- the
    ObjectRef from the failed one is spent.
    """
    tools = list(tools)
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return get(llm.prompt.options(resources={"llm": LLM_RESOURCE},
                                          **LLM_TASK_OPTS)
                       .chia_remote(llm, prompt, tools))
        except Exception as exc:                       # noqa: BLE001
            err = _find_rate_limit(exc)
            if err is None or attempt >= RATE_LIMIT_MAX_RETRIES:
                raise
            wait, reset = _rate_limit_wait_seconds(err)
            _event("rate_limit_wait", wait_seconds=wait, reset_time=reset,
                   attempt=attempt + 1, retries=RATE_LIMIT_MAX_RETRIES)
            print(f"[{_dt.datetime.now().isoformat(timespec='seconds')}] "
                  f"rate limit (attempt {attempt + 1}/"
                  f"{RATE_LIMIT_MAX_RETRIES}): resets {reset}, sleeping "
                  f"{wait:.0f}s before re-dispatching the same prompt",
                  file=sys.stderr, flush=True)
            _time.sleep(wait)
            print(f"[{_dt.datetime.now().isoformat(timespec='seconds')}] "
                  f"rate-limit wait over, retrying", file=sys.stderr,
                  flush=True)
    raise RuntimeError("unreachable")


def implement(llm, tools: Sequence[object], note: str = ""):
    """First turn: build the thing.

    *note* is appended when the run started from a resumed working tree
    (``--rtl-diff``): the standing task is unchanged, but the agent has to be
    told that a previous attempt is already applied, and what it scored.
    """
    task = f"{_IMPLEMENT_TASK}\n\n{note}" if note else _IMPLEMENT_TASK
    return _run(llm, task, tools)


def debug(llm, tools: Sequence[object], feedback: str):
    """Subsequent turns: the standing debug rules plus this run's evidence."""
    return _run(llm, f"{_DEBUGGER_PREAMBLE}\n\n{feedback}", tools)
