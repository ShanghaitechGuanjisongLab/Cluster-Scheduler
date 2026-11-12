#!/bin/bash
# 任务包装.sh — 任务执行包装器（由 任务控制.sh add 自动套用）
# ------------------------------------------------------------
# 职责：把 HyperQueue 分配到的 GPU 资源值翻译成 CUDA_VISIBLE_DEVICES，然后 exec 真正的任务命令。不请求 GPU 的任务原样透传。
#
# 原理：worker 侧声明 `--resource "gpus=[0,1,2,3]"`（非可分资源）后，任务进程会拿到环境变量 HQ_RESOURCE_VALUES_gpus（如 "3" 或 "1,2"），但 HQ 不会自动设置 CUDA_VISIBLE_DEVICES，需在此手工映射。
# ============================================================

if [ -n "${HQ_RESOURCE_VALUES_gpus:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${HQ_RESOURCE_VALUES_gpus}"
fi

HERE="${TASKCTL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec "${TASKCTL_PYTHON:-python3}" "$HERE/GPUworker.py" --task "$@"
