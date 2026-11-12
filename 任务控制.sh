#!/bin/bash
# ============================================================
# 任务控制.sh — 任务控制客户端（登录节点/任意节点上用，秒级即退）
# ------------------------------------------------------------
# 职责：向计算节点上常驻的 任务守护（HyperQueue server）提交/取消/查询       任务。不常驻任何进程，不受「登录节点进程脆弱」问题影响。
#
# 用法：
#   ./任务控制.sh add [--name N] [--gpus N] [--cpus N] [--priority P] [--cwd D] \                    [--] 命令 [参数……]
#                    # 添加一个任务；--gpus N 申请 N 块 GPU（默认 0）
#                    # --cpus N 申请 N 核 CPU（默认 1）
#                    # --gpu-memory 下限:上限 每张 GPU 的显存范围；未声明或为 0 时按 CPU 任务提交
#                    # --priority P 越大越先跑（默认 0，手动任务建议 10）
#   ./任务控制.sh add-json <任务.json>                    # 用 JSON 描述批量添加（字段见 README）
#   ./任务控制.sh cancel <任务ID|名字>     # 取消一个任务
#   ./任务控制.sh list [--all]            # 列出任务（默认含运行中与排队中）
#   ./任务控制.sh info <任务ID>            # 任务详情（含输出路径）
#   ./任务控制.sh logs <任务ID>            # 查看任务 stdout/stderr
#   ./任务控制.sh restart-service         # 重启受管服务（守护的 --run 进程），**不释放节点**
#   ./任务控制.sh service-status          # 受管服务的 pid、重启次数、命令
#   ./任务控制.sh stop                    # 请求 任务守护 优雅退出
#   ./任务控制.sh status                  # 任务守护 存活与资源占用概况
#   ./任务控制.sh autoscale start|stop|status|plan  # 配额感知的 GPU 弹性扩容
#
# 任务ID说明：一任务一 job，因此任务ID == HyperQueue 的 job ID，由 add 打印出来，也可用 list 查。
#
# cancel 与 restart-service 的区别：
#   cancel           取消**一个任务**（作业内的单个计算），守护与其余任务不动。
#   restart-service  重启**受管服务**（作业主体那个编排进程）。用于改了代码
#                    要换进程重新加载，又不想放弃已排到手的计算节点（重新                    排队可能等几小时）。已提交的任务由 worker 继续跑完，
#                    新进程按结果目录接管未完成部分，故无计算浪费。
# ============================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HQ="$HERE/bin/hq"
HQ_LIBSTDCPP="${HQ_LIBSTDCPP:-/public_bme2/Share200T/管吉松/envs/af3/lib}"
if [[ -f "$HQ_LIBSTDCPP/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="$HQ_LIBSTDCPP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
STATE_DIR="${TASKCTL_STATE:-$HERE/运行状态}"
export HQ_SERVER_DIR="$STATE_DIR/server"

die() { echo "❌ $*" >&2; exit 1; }

# hq 0.20 把 access.json 写在实例子目录下（运行状态/server/001/access.json）
access_ready() { compgen -G "$HQ_SERVER_DIR/*/access.json" >/dev/null; }

need_server() {
    access_ready || die "找不到 $HQ_SERVER_DIR/*/access.json
（任务守护 未启动？先在计算节点的作业里运行 任务守护.sh）"
}

cmd="${1:-}"; shift || true

case "$cmd" in
# ---------------- GPU 弹性扩容 ----------------
autoscale)
    operation="${1:-status}"
    case "$operation" in
        start|stop|status|plan|once) : ;;
        *) die "用法：$0 autoscale start|stop|status|plan|once" ;;
    esac
    need_server
    exec "${TASKCTL_PYTHON:-python3}" "$HERE/自动扩容.py" "$operation"
    ;;

# ---------------- 添加任务 ----------------
add)
    NAME="" GPUS=0 PRIO=0 CWD="" CPUS=1 GPU_MEMORY=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --name)     NAME="$2"; shift 2 ;;
            --gpus)     GPUS="$2"; shift 2 ;;
            --cpus)     CPUS="$2"; shift 2 ;;
            --gpu-memory) GPU_MEMORY="$2"; shift 2 ;;
            --priority) PRIO="$2"; shift 2 ;;
            --cwd)      CWD="$2"; shift 2 ;;
            --)         shift; break ;;
            -*)         die "未知选项 $1" ;;
            *)          break ;;
        esac
    done
    [ $# -gt 0 ] || die “用法：$0 add [--name N] [--gpus N] [--priority P] [--cwd D] -- 命令……”
    [[ "$GPUS" =~ ^[0-9]+$ ]] || die "GPU 数必须是非负整数"
    MEMORY_INFO=$("${TASKCTL_PYTHON:-python3}" -c 'import sys; sys.path.insert(0,sys.argv[1]); from GPUworker import 解析显存范围,原生显存范围; 下限,上限=解析显存范围(sys.argv[2]); print(原生显存范围(下限),原生显存范围(上限),原生显存范围((下限,上限),sys.argv[3]))' "$HERE" "$GPU_MEMORY" "$GPUS") || die "显存范围无效"
    read -r GPU_MEMORY_MIN GPU_MEMORY_MAX GPU_MEMORY_NATIVE <<< "$MEMORY_INFO"
    [[ "$GPU_MEMORY_MIN" == 0 || "$GPUS" -gt 0 ]] || die "--gpu-memory 必须与 --gpus 一起使用"
    [[ "$GPU_MEMORY_MIN" != 0 ]] || GPUS=0
    need_server

    ARGS=(submit --priority "$PRIO" --cpus "$CPUS")
    [ -n "$NAME" ] && ARGS+=(--name "$NAME")
    [ -n "$CWD" ]  && ARGS+=(--cwd "$CWD")
    if [ "$GPUS" -gt 0 ] 2>/dev/null; then
        ARGS+=(--resource "gpus=$GPUS")
        ARGS+=(--resource "gpumem=$GPU_MEMORY_NATIVE" --env "TASKCTL_GPU_MEMORY_MIN_GB=$GPU_MEMORY_MIN" --env "TASKCTL_GPU_MEMORY_MAX_GB=$GPU_MEMORY_MAX")
        # 包装器把 HQ_RESOURCE_VALUES_gpus 翻译成 CUDA_VISIBLE_DEVICES
        set -- bash "$HERE/任务包装.sh" "$@"
    fi
    # 输出落到 运行状态/logs/<jobid>/ 下，便于事后追查
    ARGS+=(--stdout "$STATE_DIR/logs/%{JOB_ID}/%{TASK_ID}.stdout"
           --stderr "$STATE_DIR/logs/%{JOB_ID}/%{TASK_ID}.stderr")
    ID=$("$HQ" --output-mode=quiet "${ARGS[@]}" -- "$@") || die "提交失败"
    echo "✅ 已添加任务 ID=$ID${NAME:+ name=$NAME}  （取消：$0 cancel $ID）"
    ;;

# ---------------- 批量添加（JSON）----------------
add-json)
    [ -n "${1:-}" ] && [ -f "$1" ] || die "用法：$0 add-json <任务.json>"
    need_server
    python3 - "$1" "$HERE" <<'PYEOF' || die "批量添加失败"
import json, subprocess, sys, os
path, here = sys.argv[1], sys.argv[2]
sys.path.insert(0, here)
from GPUworker import 解析显存范围, 原生显存范围
hq = os.path.join(here, "bin", "hq")
state = os.environ.get("TASKCTL_STATE", os.path.join(here, "运行状态"))
spec = json.load(open(path))
items = spec if isinstance(spec, list) else spec.get("tasks", [])
for it in items:
    args = [hq, "--output-mode=quiet", "submit",
            "--priority", str(it.get("priority", 0)),
            "--cpus", str(it.get("cpus", 1)),
            "--stdout", f"{state}/logs/%{{JOB_ID}}/%{{TASK_ID}}.stdout",
            "--stderr", f"{state}/logs/%{{JOB_ID}}/%{{TASK_ID}}.stderr"]
    if it.get("name"):     args += ["--name", it["name"]]
    if it.get("cwd"):      args += ["--cwd", it["cwd"]]
    argv = it["cmd"] if isinstance(it["cmd"], list) else it["cmd"].split()
    gpus = int(it.get("gpus", 0))
    下限, 上限 = 解析显存范围(it.get("gpu_memory", 0))
    if gpus < 0 or (下限 > 0 and gpus == 0):
        raise ValueError("GPU 数必须是非负整数，正数显存必须与 GPU 数一起声明")
    if gpus > 0 and 下限 > 0:
        args += ["--resource", f"gpus={gpus}",
                 "--resource", f"gpumem={原生显存范围((下限, 上限), gpus)}", "--env", f"TASKCTL_GPU_MEMORY_MIN_GB={下限}", "--env", f"TASKCTL_GPU_MEMORY_MAX_GB={上限}"]
        argv = ["bash", os.path.join(here, "任务包装.sh")] + argv
    args += ["--"] + argv
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"✅ {it.get('name','(匿名)')} -> 任务 ID={r.stdout.strip()}")
    else:
        print(f"❌ {it.get('name','(匿名)')} 提交失败：{r.stderr.strip()}")
PYEOF
    ;;

# ---------------- 取消 ----------------
cancel)
    [ -n "${1:-}" ] || die "用法：$0 cancel <任务ID|名字>"
    need_server
    ID="$1"
    if ! [[ "$ID" =~ ^[0-9]+$ ]]; then      # 按名字查 ID
        ID=$("$HQ" --output-mode=json job list --all 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
hit=[j['id'] for j in d if j.get('name')==sys.argv[1]]
print(hit[0] if hit else '')" "$1")
        [ -n "$ID" ] || die "找不到名为 $1 的任务"
    fi
    "$HQ" job cancel "$ID" && echo "✅ 任务 $ID 已取消"
    ;;

# ---------------- 列表 ----------------
list)
    need_server
    case "${1:-}" in
        # 机器可读输出：供上游编排进程在重启后**接管**队列里未完的任务，
        # 而不是把它们当成遗留重新提交（同一个预测算两遍）。字段含
        # id/name/task_stats{waiting,running,finished,failed,canceled,aborted}。
        --json) "$HQ" --output-mode=json job list --all ;;
        --all)  "$HQ" job list --all ;;
        *)      "$HQ" job list ;;
    esac
    ;;

# ---------------- 详情 ----------------
info)
    [ -n "${1:-}" ] || die "用法：$0 info <任务ID>"
    need_server
    "$HQ" job info "$1"
    ;;

# ---------------- 等待（阻塞至终态）----------------
wait)
    [ $# -gt 0 ] || die “用法：$0 wait <任务ID……>”
    need_server
    for jid in "$@"; do
        "$HQ" job wait "$jid" || exit $?
        "$HQ" --output-mode=json job info "$jid" | "${TASKCTL_PYTHON:-python3}" -c '
import json, sys
jobs = json.load(sys.stdin)
if not jobs:
    raise SystemExit("HQ 未返回待等待任务，拒绝视为成功")
for job in jobs:
    info = job["info"]
    stats = info.get("task_stats") or {}
    job_id = info["id"]
    if (any(stats.get(state) for state in ("failed", "canceled", "aborted", "waiting", "running"))
            or stats.get("finished", 0) != info["task_count"]):
        raise SystemExit(f"HQ 任务 {job_id} 未成功完成：{stats}")
' || exit $?
    done
    ;;

# ---------------- 日志 ----------------
logs)
    [ -n "${1:-}" ] || die "用法：$0 logs <任务ID>"
    need_server
    for f in "$STATE_DIR/logs/$1"/*.stdout "$STATE_DIR/logs/$1"/*.stderr; do
        [ -f "$f" ] || continue
        echo "───── $f ─────"
        tail -n 40 "$f"
    done
    ;;

# ---------------- 重启受管服务（不释放节点）----------------
restart-service)
    INFO="$STATE_DIR/service.info"
    [ -f "$INFO" ] || die "受管服务未登记（$INFO 不存在）
守护可能未用 --run 启动，或已退出。先用 ./任务控制.sh status 看一眼。"
    touch "$STATE_DIR/reload.requested"
    echo "✅ 已请求重启受管服务，守护 将在 ~5s 内发出 SIGTERM"
    echo "   ┈┈ 作业与节点分配保持不变；重启后 PID 会变，可查："
    echo "   ./任务控制.sh service-status"
    echo "   若服务未能在 --reload-timeout 内退出，守护会升级 SIGKILL"
    ;;

# ---------------- 受管服务状态 ----------------
service-status)
    INFO="$STATE_DIR/service.info"
    if [ ! -f "$INFO" ]; then
        echo "受管服务未登记（无 $INFO）"
        exit 0
    fi
    echo "── 受管服务 ──"
    cat "$INFO"
    # pid 只在它所属的节点上有意义；登录节点上 kill -0 可能撞到无关同名进程，故不探活。
    HOST=$(sed -n 's/^host=//p' "$INFO" | head -1)
    if [ "$HOST" = "$(hostname)" ]; then
        SPID=$(sed -n 's/^pid=//p' "$INFO" | head -1)
        if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then
            echo "状态：运行中（同节点探活）"
        else
            echo "状态：已退出（登记信息残留）"
        fi
    else
        echo "状态：运行在 $HOST 上；登录节点无法探活该 pid"
        echo "      改用 ./任务控制.sh status 看 HQ，或等 restart-service 后 pid 变化"
    fi
    ;;

# ---------------- 停止 任务守护 ----------------
stop)
    touch "$STATE_DIR/stop.requested"
    echo "✅ 已放置停止标志，任务守护 将在 ~5s 内优雅退出"
    ;;

# ---------------- 状态 ----------------
status)
    if access_ready; then
        echo "── server ──"
        "$HQ" server info 2>/dev/null | head -8
        echo "── worker ──"
        "$HQ" worker list 2>/dev/null
        echo "── 任务 ──"
        "$HQ" job list 2>/dev/null
    else
        echo "任务守护 未运行（无 $HQ_SERVER_DIR/access.json）"
    fi
    ;;

*)
    sed -n '2,28p' "$0"
    exit 2
    ;;
esac
