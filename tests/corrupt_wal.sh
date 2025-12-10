#!/usr/bin/env bash
set -euo pipefail

WAL="${1:-snapshots/database.bin.wal}"
if [[ ! -f "$WAL" ]]; then
  echo "WAL not found: $WAL" >&2
  exit 1
fi

tmp="$(mktemp "${WAL}.XXXX")"
cp "$WAL" "$tmp"

# Corrupt the last 4 bytes (likely CRC) to force validation failure.
printf '\x00\x00\x00\x00' | dd of="$tmp" bs=1 seek=$(( $(stat -c%s "$tmp") - 4 )) conv=notrunc status=none

echo "Corrupted WAL copy at $tmp"
echo "Expected: replay should fail with CRC or header mismatch."

