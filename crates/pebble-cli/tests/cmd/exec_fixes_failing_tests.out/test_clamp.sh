#!/usr/bin/env bash
set -euo pipefail
source ./clamp.sh

assert_clamp() {
    local label=$1 expected=$2 actual
    actual=$(clamp "$3" "$4" "$5")
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL: $label (expected $expected, got $actual)" >&2
        exit 1
    fi
}

assert_clamp 'below lower bound' 0 -2 0 10
assert_clamp 'inside bounds' 5 5 0 10
assert_clamp 'above upper bound' 10 12 0 10
assert_clamp 'at boundary' 0 0 0 10
echo 'PASS: all 4 clamp cases'
