"""Llama-3.2-1B model spec + per-scenario operator inventory.

Pure analysis: no hardware, no simulation, no torch. Everything here is
arithmetic over a `ModelSpec`, so swapping models means adding one entry to
`MODELS` and nothing else.

Config provenance
-----------------
`llama-3.2-1B` was cross-checked byte-for-byte against the official
`meta-llama/Llama-3.2-1B` config.json found on this cluster at
    /share2/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/*/config.json
(and independently against the `unsloth/Llama-3.2-1B` mirror on HF). Fields
used: hidden_size 2048, intermediate_size 8192, num_hidden_layers 16,
num_attention_heads 32, num_key_value_heads 8, head_dim 64, vocab_size 128256,
tie_word_embeddings true, rms_norm_eps 1e-5, rope_theta 500000.

Kernel taxonomy (the `kernel` field on every Op)
------------------------------------------------
    int8_gemv    matrix-vector, one token   (decode weight-stationary path)
    int8_gemm    matrix-matrix, N tokens    (prefill path)
    rmsnorm      RMS normalisation
    rope         rotary position embedding
    attn_scores  Q @ K^T / sqrt(d)
    softmax      row softmax over the score matrix
    attn_pv      probs @ V
    silu_mul     SiLU(gate) * up
    add          residual add
    embedding    embedding-table row gather
    lm_head_gemv the output projection to vocab (own bucket: it dominates decode)

Sizes are in *elements*; byte counts use the dtype widths on `ModelSpec`, which
assume an int8-weight / int8-activation / int32-accumulator pipeline (what the
Gemmini 16x16 int8 array wants) with fp32 for the norm/softmax reductions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

Scenario = Literal["decode", "prefill"]


# --------------------------------------------------------------------------
# Model spec
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden: int              # hidden_size
    intermediate: int        # intermediate_size (MLP width)
    layers: int              # num_hidden_layers
    heads: int               # num_attention_heads
    kv_heads: int            # num_key_value_heads (GQA)
    head_dim: int
    vocab: int
    tied_embeddings: bool = True
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    verified: bool = False   # True == config.json was actually checked

    # dtype widths, in bytes, of the assumed quantised inference pipeline
    w_bytes: int = 1         # int8 weights
    act_bytes: int = 1       # int8 activations feeding the MAC arrays
    acc_bytes: int = 4       # int32/fp32 accumulators + norm/softmax working set
    kv_bytes: int = 1        # int8 KV cache
    emb_bytes: int = 1       # embedding table (tied with the LM head)

    # GQA: one K/V head's cache row is read once and reused by heads/kv_heads
    # query heads. True is the optimistic (and realistic, if you tile right)
    # assumption; flip to False for a pessimistic per-query-head re-read.
    gqa_share_kv_reads: bool = True

    # ---- derived ---------------------------------------------------------
    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def group_size(self) -> int:
        return self.heads // self.kv_heads

    def param_counts(self) -> dict[str, int]:
        h, i, v = self.hidden, self.intermediate, self.vocab
        per_layer = {
            "q_proj": h * self.q_dim,
            "k_proj": h * self.kv_dim,
            "v_proj": h * self.kv_dim,
            "o_proj": self.q_dim * h,
            "gate_proj": h * i,
            "up_proj": h * i,
            "down_proj": i * h,
            "norms": 2 * h,
        }
        layer_total = sum(per_layer.values())
        emb = v * h
        out = {f"layer.{k}": val for k, val in per_layer.items()}
        out["per_layer_total"] = layer_total
        out["all_layers"] = layer_total * self.layers
        out["final_norm"] = h
        out["embedding"] = emb
        out["lm_head"] = 0 if self.tied_embeddings else emb
        out["total"] = layer_total * self.layers + h + emb + out["lm_head"]
        out["non_embedding"] = out["total"] - emb
        out["norm_params"] = self.layers * 2 * h + h
        return out


MODELS: dict[str, ModelSpec] = {
    "llama-3.2-1B": ModelSpec(
        name="Llama-3.2-1B",
        hidden=2048,
        intermediate=8192,
        layers=16,
        heads=32,
        kv_heads=8,
        head_dim=64,
        vocab=128256,
        tied_embeddings=True,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        verified=True,
    ),
}

DEFAULT_MODEL = "llama-3.2-1B"


# --------------------------------------------------------------------------
# Operator inventory
# --------------------------------------------------------------------------
@dataclass
class Op:
    """One operator instance, aggregated over a whole scenario."""
    name: str
    kernel: str
    shape: str                 # human-readable, parameterised shape
    calls: int                 # invocations across the whole scenario
    macs: int                  # multiply-accumulates, scenario total
    bytes_read: int
    bytes_written: int
    elems: int = 0             # elements touched (the scaling knob for elementwise)
    stage: str = "layer"       # "prologue" | "layer" | "epilogue"
    note: str = ""

    @property
    def bytes_total(self) -> int:
        return self.bytes_read + self.bytes_written

    def scaled(self, factor: int) -> "Op":
        """Replicate this op `factor` times (used to fan a layer out over L)."""
        return replace(
            self,
            calls=self.calls * factor,
            macs=self.macs * factor,
            bytes_read=self.bytes_read * factor,
            bytes_written=self.bytes_written * factor,
            elems=self.elems * factor,
        )


def _gemv(spec: ModelSpec, name: str, m: int, k: int, *, note: str = "") -> Op:
    return Op(
        name=name,
        kernel="int8_gemv",
        shape=f"[{m}x{k}] @ [{k}] -> [{m}]",
        calls=1,
        macs=m * k,
        bytes_read=m * k * spec.w_bytes + k * spec.act_bytes,
        bytes_written=m * spec.acc_bytes,
        elems=m,
        note=note,
    )


def _gemm(spec: ModelSpec, name: str, m: int, k: int, n: int, *, note: str = "") -> Op:
    return Op(
        name=name,
        kernel="int8_gemm",
        shape=f"[{m}x{k}] @ [{k}x{n}] -> [{m}x{n}]",
        calls=1,
        macs=m * k * n,
        bytes_read=m * k * spec.w_bytes + n * k * spec.act_bytes,
        bytes_written=n * m * spec.acc_bytes,
        elems=n * m,
        note=note,
    )


def _elementwise(spec: ModelSpec, name: str, kernel: str, elems: int,
                 macs_per_elem: float, *, calls: int = 1, shape: str = "",
                 note: str = "") -> Op:
    return Op(
        name=name,
        kernel=kernel,
        shape=shape or f"[{elems}]",
        calls=calls,
        macs=int(elems * macs_per_elem),
        bytes_read=elems * spec.acc_bytes,
        bytes_written=elems * spec.acc_bytes,
        elems=elems,
        note=note,
    )


def decoder_layer_ops(spec: ModelSpec, scenario: Scenario, *, S: int = 512,
                      N: int = 1) -> list[Op]:
    """Ops for ONE decoder layer.

    decode : one new token, attending over `S` keys (S counts the current
             token, i.e. S = len(kv_cache) after the append).
    prefill: `N` tokens in one pass, causal, so token t attends to t+1 keys.
    """
    h, i, d = spec.hidden, spec.intermediate, spec.head_dim
    nH, nKV = spec.heads, spec.kv_heads
    qd, kvd = spec.q_dim, spec.kv_dim
    ops: list[Op] = []

    if scenario == "decode":
        proj = lambda nm, m: _gemv(spec, nm, m, h)
        proj_down = _gemv(spec, "mlp.down_proj", h, i)
        tokens = 1
    else:
        proj = lambda nm, m: _gemm(spec, nm, m, h, N)
        proj_down = _gemm(spec, "mlp.down_proj", h, i, N)
        tokens = N

    # --- attention block ---------------------------------------------------
    ops.append(_elementwise(spec, "input_layernorm", "rmsnorm", tokens * h, 2,
                            calls=tokens, shape=f"[{tokens}x{h}]",
                            note="sum-of-squares + rsqrt scale"))
    ops.append(proj("attn.q_proj", qd))
    ops.append(proj("attn.k_proj", kvd))
    ops.append(proj("attn.v_proj", kvd))
    ops.append(_elementwise(spec, "attn.rope_q", "rope", tokens * qd, 2,
                            calls=tokens, shape=f"[{tokens}x{nH}x{d}]"))
    ops.append(_elementwise(spec, "attn.rope_k", "rope", tokens * kvd, 2,
                            calls=tokens, shape=f"[{tokens}x{nKV}x{d}]"))

    kv_read_heads = nKV if spec.gqa_share_kv_reads else nH

    if scenario == "decode":
        score_macs = nH * S * d
        score_elems = nH * S
        ops.append(Op(
            name="attn.scores (QK^T/sqrt(d))", kernel="attn_scores",
            shape=f"per head [1x{d}] @ [{d}x{S}] -> [1x{S}], x{nH} heads",
            calls=nH, macs=score_macs,
            bytes_read=kv_read_heads * S * d * spec.kv_bytes + qd * spec.act_bytes,
            bytes_written=score_elems * spec.acc_bytes,
            elems=score_elems,
            note=f"K-cache read {'shared over GQA group' if spec.gqa_share_kv_reads else 'per query head'}",
        ))
        ops.append(_elementwise(spec, "attn.softmax", "softmax", score_elems, 1,
                                calls=nH, shape=f"[{nH}x{S}]"))
        ops.append(Op(
            name="attn.pv (probs@V)", kernel="attn_pv",
            shape=f"per head [1x{S}] @ [{S}x{d}] -> [1x{d}], x{nH} heads",
            calls=nH, macs=nH * S * d,
            bytes_read=kv_read_heads * S * d * spec.kv_bytes + score_elems * spec.act_bytes,
            bytes_written=qd * spec.acc_bytes,
            elems=qd,
        ))
    else:
        causal = N * (N + 1) // 2            # causal score entries per head
        score_macs = nH * causal * d
        score_elems = nH * causal
        ops.append(Op(
            name="attn.scores (QK^T/sqrt(d))", kernel="attn_scores",
            shape=f"per head [{N}x{d}] @ [{d}x{N}] causal, x{nH} heads",
            calls=nH, macs=score_macs,
            bytes_read=kv_read_heads * N * d * spec.kv_bytes + N * qd * spec.act_bytes,
            bytes_written=score_elems * spec.acc_bytes,
            elems=score_elems,
            note="lower-triangular only: N(N+1)/2 entries per head",
        ))
        ops.append(_elementwise(spec, "attn.softmax", "softmax", score_elems, 1,
                                calls=nH * N, shape=f"[{nH}x{N}x<=N] causal"))
        ops.append(Op(
            name="attn.pv (probs@V)", kernel="attn_pv",
            shape=f"per head [{N}x{N}] causal @ [{N}x{d}] -> [{N}x{d}], x{nH} heads",
            calls=nH, macs=nH * causal * d,
            bytes_read=kv_read_heads * N * d * spec.kv_bytes + score_elems * spec.act_bytes,
            bytes_written=N * qd * spec.acc_bytes,
            elems=N * qd,
        ))

    ops.append(_gemv(spec, "attn.o_proj", h, qd) if scenario == "decode"
               else _gemm(spec, "attn.o_proj", h, qd, N))
    ops.append(_elementwise(spec, "attn.residual_add", "add", tokens * h, 0,
                            calls=tokens, shape=f"[{tokens}x{h}]"))

    # --- MLP block ---------------------------------------------------------
    ops.append(_elementwise(spec, "post_attention_layernorm", "rmsnorm",
                            tokens * h, 2, calls=tokens, shape=f"[{tokens}x{h}]"))
    ops.append(proj("mlp.gate_proj", i))
    ops.append(proj("mlp.up_proj", i))
    ops.append(_elementwise(spec, "mlp.silu_mul", "silu_mul", tokens * i, 2,
                            calls=tokens, shape=f"[{tokens}x{i}]",
                            note="SiLU(gate)*up; sigmoid counted as ~1 MAC/elem"))
    ops.append(proj_down)
    ops.append(_elementwise(spec, "mlp.residual_add", "add", tokens * h, 0,
                            calls=tokens, shape=f"[{tokens}x{h}]"))

    for op in ops:
        op.stage = "layer"
    return ops


def model_ops(spec: ModelSpec, scenario: Scenario, *, S: int = 512, N: int = 1,
              lm_head_tokens: int | None = None) -> list[Op]:
    """Full-model inventory for the scenario, aggregated over all L layers.

    `lm_head_tokens` defaults to 1 for prefill (only the last position needs
    logits during inference) and to 1 for decode (one new token).
    """
    h, v = spec.hidden, spec.vocab
    tokens = 1 if scenario == "decode" else N
    if lm_head_tokens is None:
        lm_head_tokens = 1

    ops: list[Op] = []

    emb = Op(
        name="embed_tokens (lookup)", kernel="embedding",
        shape=f"gather {tokens} row(s) of [{v}x{h}]",
        calls=tokens, macs=0,
        bytes_read=tokens * h * spec.emb_bytes,
        bytes_written=tokens * h * spec.act_bytes,
        elems=tokens * h,
        stage="prologue",
        note="table is tied with the LM head weight",
    )
    ops.append(emb)

    layer = decoder_layer_ops(spec, scenario, S=S, N=N)
    ops.extend(op.scaled(spec.layers) for op in layer)

    final_norm = _elementwise(spec, "model.norm (final RMSNorm)", "rmsnorm",
                              tokens * h, 2, calls=tokens, shape=f"[{tokens}x{h}]")
    final_norm.stage = "epilogue"
    ops.append(final_norm)

    if scenario == "decode":
        head = _gemv(spec, "lm_head", v, h,
                     note="tied with embed_tokens; largest single decode op")
    else:
        head = _gemm(spec, "lm_head", v, h, lm_head_tokens,
                     note=f"inference computes logits for {lm_head_tokens} position(s) only")
    head.kernel = "lm_head_gemv" if scenario == "decode" else "lm_head_gemm"
    head.stage = "epilogue"
    ops.append(head)

    return ops


def totals(ops: Iterable[Op]) -> dict[str, int]:
    ops = list(ops)
    return {
        "macs": sum(o.macs for o in ops),
        "bytes_read": sum(o.bytes_read for o in ops),
        "bytes_written": sum(o.bytes_written for o in ops),
        "calls": sum(o.calls for o in ops),
    }


def tokens_in_scenario(scenario: Scenario, *, S: int = 512, N: int = 1) -> int:
    return 1 if scenario == "decode" else N


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def self_test(verbose: bool = True) -> None:
    spec = MODELS[DEFAULT_MODEL]
    p = spec.param_counts()

    def check(label: str, got, want, tol: float = 0.0):
        ok = got == want if tol == 0 else abs(got - want) <= tol * max(1, abs(want))
        if verbose:
            print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {got:,} (expected {want:,})")
        assert ok, f"{label}: {got} != {want}"

    if verbose:
        print("llama_model self-test")
        print(" parameter accounting")
    check("total params (tied)", p["total"], 1_235_814_400)
    check("embedding params", p["embedding"], 262_668_288)
    check("non-embedding params", p["non_embedding"], 973_146_112)
    assert 1.23e9 < p["total"] < 1.25e9, "total params should be ~1.24B"

    # Decode MACs: every GEMV weight is touched exactly once per token, so
    # GEMV MACs must equal total params minus the (non-GEMV) norm weights.
    ops = model_ops(spec, "decode", S=512)
    gemv_macs = sum(o.macs for o in ops if o.kernel in ("int8_gemv", "lm_head_gemv"))
    if verbose:
        print(" decode MAC identity")
    check("GEMV MACs == total params - norm params",
          gemv_macs, p["total"] - p["norm_params"])

    lm = next(o for o in ops if o.name == "lm_head")
    check("lm_head MACs == embedding params", lm.macs, p["embedding"])

    attn = sum(o.macs for o in ops if o.kernel in ("attn_scores", "attn_pv"))
    check("attention MACs @S=512", attn, 2 * spec.layers * spec.heads * 512 * spec.head_dim)

    tot = totals(ops)["macs"]
    ew = sum(o.macs for o in ops
             if o.kernel in ("rmsnorm", "rope", "softmax", "silu_mul", "add",
                             "embedding"))
    check("decode total MACs == GEMV + attention + elementwise",
          tot, gemv_macs + attn + ew)
    assert ew < 0.001 * tot, "elementwise MACs should be <0.1% of the total"
    if verbose:
        print(f"        (elementwise share of decode MACs: {100*ew/tot:.4f}%)")

    # Prefill: per-token projection MACs must equal the decode figure exactly.
    pre = model_ops(spec, "prefill", N=64)
    pre_gemm = sum(o.macs for o in pre if o.kernel == "int8_gemm")
    dec_gemv = sum(o.macs for o in ops
                   if o.kernel == "int8_gemv")  # excludes lm_head
    if verbose:
        print(" prefill scaling")
    check("prefill GEMM MACs == 64 x decode projection MACs", pre_gemm, 64 * dec_gemv)

    pre_attn = sum(o.macs for o in pre if o.kernel == "attn_scores")
    check("prefill causal score MACs",
          pre_attn, spec.layers * spec.heads * spec.head_dim * 64 * 65 // 2)

    # Layer count fan-out
    layer_ops = decoder_layer_ops(spec, "decode", S=512)
    per_layer = sum(o.macs for o in layer_ops)
    all_layers = sum(o.macs for o in ops if o.stage == "layer")
    check("layer fan-out x16", all_layers, per_layer * spec.layers)

    if verbose:
        print(" all llama_model self-tests passed")


if __name__ == "__main__":
    self_test()
