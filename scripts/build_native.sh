#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc -shared -fPIC -O2 "$task_root/ACCESS2021_release/accessmath_lib.c" -o "$task_root/ACCESS2021_release/accessmath_lib.native.so" -lm
