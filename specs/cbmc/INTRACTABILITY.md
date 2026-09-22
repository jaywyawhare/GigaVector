# Why 11 harnesses do not discharge

These are recorded with evidence rather than asserted, so nobody repeats the
attempts. Everything below was tried against CBMC 6.10.0 on this machine.

## 2026-08 re-trial: CBMC 6.11.0 + `--smt2 --fpa` (Z3 5.1.0), ESBMC 8.4.0

Every pending harness was retried on two alternative backends, 300 s each.

**The win: `grpc_frame` discharged.** CBMC 6.11.0's SMT2 route with native
floating-point theory (`--smt2 --fpa`, external Z3) proved all 2,980 checks in
5 s -- under 6.10 bit-blasting this harness never finished. The harness now
declares `CBMC-FLAGS: --smt2 --fpa` / `CBMC-SOLVER: z3`; `run_cbmc.py` skips it
cleanly when z3 is not on PATH, and `api/grpc.c` moved from `cbmc-pending` to
the proved `cbmc` tier.

**What still fails, and why:**

| Backend | Result |
|---|---|
| CBMC 6.11 `--cprover-smt2` | Still reaches `0 of N failed` then reports VERIFICATION ERROR -- the built-in backend remains incomplete; not countable as proof. |
| CBMC 6.11 `--smt2` (no `--fpa`) | Still crashes: `TODO floatbv_round_to_integral without FPA`. The `--fpa` flag is mandatory for float-heavy models. |
| CBMC 6.11 `--smt2 --fpa` on the rest | `distance` crashes with `flatten2bv of a non-constant FPA-encoded float is unsupported` (float bit-casts mixed into FPA encoding); `ann_index`, `bloom`, `bm25`, `compression`, `filter`, `geo`, `graph_db`, `id_bitmap`, `kdtree`, `parsers`, `ranking`, `wal_record` still time out at their existing bounds. |
| ESBMC 8.4 (all safety checks default, `-D__CPROVER__`, per-harness unwind) | No new proofs: every harness either times out or produces an artifact verdict. Its `distance`/`geo`/`ranking` "failures" are unsound NaN modelling of its own `nondet_float` intrinsic (counterexamples survive an explicit `x == x` NaN exclusion), and its `grpc_frame` bounds violation is contradicted by CBMC's 2,980-check proof. Not usable as evidence here. |

Net effect: 11 -> 10 pending harnesses. `bloom`, `geo` and `distance` remain
structurally blocked (float reasoning feeding symbolic indexing; missing libm;
FPA/bitcast mixes). A future CBMC/Z3 release that fixes `flatten2bv` or the
`--cprover-smt2` backend is the cheapest next lever -- re-run the trial before
touching any harness.

## Original findings (CBMC 6.10.0)

## What worked

**Scope reduction.** Shrinking N / buffer bounds / unwind limits while keeping
the properties identical turned four harnesses into proofs: `posting` (4,320
checks), `exact_search` (1,704), `graph_csr` (917), `quant` (1,013). A proof
over *all inputs* at N=2 is worth strictly more than a timeout at N=4, so this
is the first lever to reach for.

## What did not work

| Attempt | Result |
|---|---|
| `--smt2` (Z3 5.0.0, installed via `pip install z3-solver`) | CBMC crashes with an internal invariant violation and dumps core. This is a CBMC/Z3 integration defect, not something the harness can work around. |
| `--cprover-smt2` (CBMC's built-in SMT2 backend) | Reaches `0 of 624 failed` on `bloom` but then reports `VERIFICATION ERROR` rather than `SUCCESSFUL`. The backend is incomplete, so the result cannot be counted as a proof. |
| `--slice-formula`, `--full-slice` | No measurable effect on the float-heavy harnesses. |
| Allocator stubbing (`cbmc_alloc_stub.c`) | Linking `src/core/memory.c` drags the whole memory-accounting subsystem (pools, tracking tables, locks) into every model, and CBMC was visibly spending its budget inside `gv_memory_untrack_locked`. Replacing it with thin `malloc`/`free` stubs removes all of that — but `bloom`, `soa` and `id_bitmap` still time out, so the bookkeeping was not the dominant cost. The stub is kept because it is principled and reduces model size for free. |

## The dominant cost: exact IEEE-754 arithmetic

Four harnesses are blocked structurally, not by scope:

- **`distance`** — SIMD kernel vs scalar reference. Times out at DIM 8, at DIM 4
  (the smallest that still dispatches to SSE), and in a Manhattan-only variant
  with no `sqrt` at all.
- **`geo`** — CBMC ships no libm bodies (`asin`/`sin`/`cos`/`sqrt`) and offers
  no stub-override at this invocation level; range-contracted stubs are ignored.
- **`bloom`** — still times out at **one item of one byte**.
  `bloom_optimal_bits`/`bloom_optimal_hashes` size the filter with double
  `log`/`pow`, and `bloom_hash_i` then indexes with `% num_bits` on a value
  derived from that float arithmetic. Float reasoning feeding a symbolic modulo
  is the worst case for bit-blasting.
- **`ranking`** — MMR defaults to cosine, putting division and score
  normalisation on the critical path.

Bit-blasting floats to propositional logic is the root cause in every case. The
realistic fix is a solver with native floating-point theory, which is exactly
what the `--smt2` route would have provided had it not crashed.

## Honest accounting

The remaining 10 files sit in the `cbmc-pending` tier of `specs/coverage.yml`
and are **not** counted toward formal coverage. `make cbmc-smoke` still
compiles and runs each harness concretely, so they cannot silently rot, but a
concrete run is a test, not a proof, and the manifest does not pretend otherwise.
