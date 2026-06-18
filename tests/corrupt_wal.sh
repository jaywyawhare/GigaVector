#!/usr/bin/env bash
set -euo pipefail

WAL="${1:-snapshots/database.bin.wal}"
if [[ ! -f "$WAL" ]]; then
  echo "SKIP: WAL not found: $WAL" >&2
  exit 77
fi

tmp="$(mktemp "${WAL}.XXXX")"
cp "$WAL" "$tmp"

# Corrupt the last 4 bytes (likely CRC) to force validation failure.
if stat --version >/dev/null 2>&1; then
  size=$(stat -c%s "$tmp")
else
  size=$(stat -f%z "$tmp")
fi
# Ensure we have at least 4 bytes to flip (CRC or tail)
if (( size < 4 )); then
  # pad with zeroes to 8 bytes so we can corrupt the tail
  dd if=/dev/zero of="$tmp" bs=1 count=$((4 - size + 4)) conv=notrunc status=none
  if stat --version >/dev/null 2>&1; then
    size=$(stat -c%s "$tmp")
  else
    size=$(stat -f%z "$tmp")
  fi
fi
printf '\x00\x00\x00\x00' | dd of="$tmp" bs=1 seek=$(( size - 4 )) conv=notrunc status=none

echo "Corrupted WAL copy at $tmp"
echo "Expected: replay should fail with CRC or header mismatch."
