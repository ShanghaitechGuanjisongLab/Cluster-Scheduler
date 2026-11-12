#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE="$HERE/bin/hq.native"
LIBRARY="${HQ_LIBSTDCPP:-/public_bme2/Share200T/管吉松/envs/af3/lib}"
if [[ -f "$LIBRARY/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="$LIBRARY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "${TASKCTL_PYTHON:-python3}" "$HERE/GPUworker.py" "$NATIVE" "$@"