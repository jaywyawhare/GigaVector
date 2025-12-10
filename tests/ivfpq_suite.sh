#!/usr/bin/env bash
set -euo pipefail

BIN_DIR="${BIN_DIR:-build/bench}"
RUN="${BIN_DIR}/benchmark_ivfpq"
RECALL="${BIN_DIR}/benchmark_ivfpq_recall"

if [[ ! -x "$RUN" || ! -x "$RECALL" ]]; then
  echo "Benchmarks not found. Run: make bench" >&2
  exit 1
fi

echo "== IVF-PQ throughput (default) =="
"$RUN"
echo
echo "== IVF-PQ throughput (cosine, nprobe=32, rerank=64) =="
"$RUN" 10000 200 256 8 8 32 64 1
echo
echo "== IVF-PQ recall (default) =="
"$RECALL" 20000 200 16 32 0
echo
echo "== IVF-PQ recall (cosine, nprobe=32, rerank=64) =="
"$RECALL" 20000 200 32 64 1

