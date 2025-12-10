#!/usr/bin/env bash
set -euo pipefail

SNAP="${1:-snapshots/database.bin}"
if [[ ! -f "$SNAP" ]]; then
  echo "Snapshot not found: $SNAP" >&2
  exit 1
fi

tmp="$(mktemp "${SNAP}.XXXX")"
cp "$SNAP" "$tmp"

# Flip a byte near the end of the file (before CRC) to trigger CRC mismatch.
size=$(stat -c%s "$tmp")
if (( size < 8 )); then
  echo "Snapshot too small to corrupt safely" >&2
  exit 1
fi
seek=$(( size - 8 ))
printf '\xFF' | dd of="$tmp" bs=1 seek="$seek" conv=notrunc status=none

echo "Corrupted snapshot copy at $tmp"
echo "Expected: load should fail CRC validation."

