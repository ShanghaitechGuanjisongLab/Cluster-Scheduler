#!/usr/bin/env bash
set -uo pipefail

HERE="${TASKCTL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export HQ_SERVER_DIR="${TASKCTL_STATE:?缺少共享 TASKCTL_STATE}/server"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$GPU_IDS" ]]; then
    GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
fi
[[ -n "$GPU_IDS" ]] || { echo "未分配 GPU，拒绝启动扩容 worker" >&2; exit 2; }
TIME_ARGS=()
if [[ -n "${TASKCTL_WORKER_TIME_LIMIT:-}" ]]; then
    TIME_ARGS=(--time-limit "$TASKCTL_WORKER_TIME_LIMIT")
fi
exec "$HERE/bin/hq" worker start \
    --manager slurm --cpus "${SLURM_CPUS_PER_TASK:?}" \
    --resource "gpus=[$GPU_IDS]" --detect-resources=none \
    --idle-timeout "${TASKCTL_AUTO_IDLE_TIMEOUT:-5m}" \
    ${TIME_ARGS[@]+"${TIME_ARGS[@]}"} \
    --on-server-lost stop