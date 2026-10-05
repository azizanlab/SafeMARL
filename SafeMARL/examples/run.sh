#!/bin/bash
set -euo pipefail
algo=${1:?algo}; task=${2:?task}; shift 2
if [[ ${1:-} =~ ^[0-9]+$ ]]; then seed_args=(--seed "$1"); shift; else seed_args=(); fi
here=$(cd "$(dirname "$0")" && pwd)
config="$here/../tuned_configs/mujoco/$task/$algo/config.json"
[ -f "$config" ] || { echo "no tuned config at $config"; exit 1; }
cd "$here"
python train.py --load_config "$config" --exp_name default ${seed_args[@]+"${seed_args[@]}"} "$@"
