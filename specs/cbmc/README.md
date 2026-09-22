# CBMC harnesses

Bounded model checking of the C itself -- Tier 2 of `docs/formal_verification_plan.md`.

These target pure, bounded, adversarially-fed code: bitmap algebra and byte
parsers. They upgrade "no crash on the corpus we tried" (libFuzzer) to "no
undefined behaviour for any input up to bound N".

## Running

CBMC is not required to build the tree. `make cbmc` skips cleanly when it is
absent, and `make cbmc-smoke` compiles every harness with the host compiler and
runs it concretely -- a cheap gate that keeps the harnesses from bit-rotting.

```sh
# full proof (install cbmc first: https://github.com/diffblue/cbmc)
make cbmc

# compile + concrete run, no cbmc needed
make cbmc-smoke
```

## Harnesses

| File | Target | Property |
|---|---|---|
| `harness_id_bitmap.c` | `src/core/id_bitmap.c` | AND/OR commutativity and definition, add/remove/contains agreement across container promotion, clone fidelity, serialize->deserialize identity |
| `harness_wal_record.c` | `wal_apply_record_buffer` | memory safety for an arbitrary byte string (same entry point as `tests/fuzz/fuzz_wal_apply.c`) |

Each harness compiles in two modes. Under `__CPROVER` it uses `nondet_*` inputs
and `__CPROVER_assume` bounds; otherwise it falls back to concrete values so it
still builds and runs as an ordinary test.
