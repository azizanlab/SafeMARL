#!/bin/bash
set -euo pipefail
algo=${1:?algo}; shift
if [[ ${1:-} =~ ^[0-9]+$ ]]; then seed=$1; shift; else seed=""; fi
here=$(cd "$(dirname "$0")" && pwd)
for task in HalfCheetah-v2-2x3 HalfCheetah-v2-3x2 Walker2d-v2-2x3 Walker2d-v2-3x2 Ant-v2-2x4 Ant-v2-4x2; do
  bash "$here/run.sh" "$algo" "$task" $seed "$@"
done
