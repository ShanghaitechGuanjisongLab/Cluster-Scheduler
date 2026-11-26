#!/usr/bin/env bash
set -uo pipefail

HERE="${TASKCTL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export HQ_SERVER_DIR="${TASKCTL_STATE:?缺少共享 TASKCTL_STATE}/server"
# 纯 worker：等主服务把 server 拉起来再连；冷启动（看门狗先交 bootstrap、主循环
# 作业还在排队）时 server 尚不存在，立刻退出会让扩容作业空转 churn。
if [[ "${TASKCTL_WORKER_WAIT_SERVER:-1}" != "0" ]]; then
    WAIT_MAX="${TASKCTL_WORKER_WAIT_SERVER_SEC:-86400}"
    for ((i=0; i<WAIT_MAX; i+=5)); do
        if compgen -G "$HQ_SERVER_DIR/*/access.json" >/dev/null; then break; fi
        sleep 5
    done
    compgen -G "$HQ_SERVER_DIR/*/access.json" >/dev/null || { echo "等待 HQ server 超时，退出" >&2; exit 2; }
fi
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