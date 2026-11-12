#!/bin/bash
# ============================================================
# 任务守护.sh — 单个 SLURM 分配内的任务守护（HyperQueue 封装）
# ------------------------------------------------------------
# 职责：在已分配的计算节点上常驻，对外提供「随时添加/取消多个并行       任务」与「重启受管服务而不释放节点分配」的能力。
#
# 组成：1 个 HyperQueue server（任务账本，对全网卡监听）
#       1 个 HyperQueue worker（执行任务，声明本分配的 GPU/CPU 资源）
#       0 或 1 个受管前台服务（--run 指定的长驻进程，如编排循环）
#       客户端（登录节点上的 任务控制.sh）经内网 TCP 连 server 提交任务，访问凭据放在共享盘 运行状态/server/ 下，两端都可见。
#
# 用法（通常充当作业脚本主体，让它成为前台进程）：
#     bash 任务守护.sh [--gpus 0,1,2,3] [--cpus 16] [--run "命令"] \                      [--reload-code 42] [--reload-timeout 25] [--max-restarts 10]
#
# 参数：
#   --gpus             GPU 卡号列表（逗号分隔）；默认取 SLURM 注入的
#                      CUDA_VISIBLE_DEVICES；传 none 表示纯 CPU worker
#   --cpus             CPU 核数；默认取 SLURM_CPUS_PER_TASK，兜底 nproc
#   --run "命令"       托管一个长驻前台服务；不带此参数时守护只提供 HQ
#   --reload-code N    受管服务以此码退出 → 自动重启（换代码但保留节点），默认 42
#   --reload-timeout T 重启请求后等服务优雅退出的秒数，超时升级 SIGKILL，默认 25
#   --max-restarts N   受管服务异常退出的自愈上限，默认 10（重载码退出不计数）
#   --adopt-worker     不新建 worker，而是**接管**已注册的那个（探活改走 hq worker list）。
#                      用于守护中途死亡后补看管：此时 server 与 worker 往往还活着，再起一个 worker 会让 HQ 以为多出资源而向同一张卡并发派任务
#
# 控制（三者都不重新分配/释放计算节点）：
#   ./任务控制.sh restart-service  重启受管服务；HQ 与已排队任务不受影响
#   ./任务控制.sh stop             守护优雅退出（受管服务 → worker → server）
#   kill <任务守护 的 pid>         SIGTERM 同样走优雅路径
#
# 为什么重启编排进程不浪费已跑的计算：计算跑在 HQ worker 进程里，受管服务只是提交者与等待者。杀掉它不会中断任何在跑的任务，新进程起来后按结果目录与台账接管未完成任务即可。
#
# 约定：状态与日志都在 <本目录>/运行状态/ 下，位于共享盘，登录节点可直接       查看；stop.requested 与 reload.requested 都会被消费（用后删除）。
# ============================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HQ="$HERE/bin/hq"
# CentOS 7 系统 libstdc++ 过旧；共享 AF3 环境提供 HQ 所需的 GLIBCXX/CXXABI。
HQ_LIBSTDCPP="${HQ_LIBSTDCPP:-/public_bme2/Share200T/管吉松/envs/af3/lib}"
if [[ -f "$HQ_LIBSTDCPP/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="$HQ_LIBSTDCPP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
STATE_DIR="${TASKCTL_STATE:-$HERE/运行状态}"
export HQ_SERVER_DIR="$STATE_DIR/server"
LOG_DIR="$STATE_DIR/logs"
STOP_FLAG="$STATE_DIR/stop.requested"
RELOAD_FLAG="$STATE_DIR/reload.requested"
SERVICE_INFO="$STATE_DIR/service.info"

# cleanup() 在 trap 安装后可能立即触发，故所有被它引用的变量都必须先有值（set -u）。
SERVER_OWNED=0
SERVER_PID=""
WORKER_PID=""
SVC_PID=""
SVC_RESTARTS=0
SVC_CRASHES=0
EXIT_RC=0                       # 传给 SLURM：任何意外退出都应为非零，以免误记 COMPLETED

mkdir -p "$HQ_SERVER_DIR" "$LOG_DIR"

# ---- 参数解析 ----
GPUS="${CUDA_VISIBLE_DEVICES:-}"
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
RUN_CMD=""
RELOAD_CODE=42
RELOAD_TIMEOUT=25
MAX_RESTARTS=10
ADOPT_WORKER=0
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --run) RUN_CMD="$2"; shift 2 ;;
        --reload-code) RELOAD_CODE="$2"; shift 2 ;;
        --reload-timeout) RELOAD_TIMEOUT="$2"; shift 2 ;;
        --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
        --adopt-worker) ADOPT_WORKER=1; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "未知参数：$1（--help 查看用法）" >&2; exit 2 ;;
    esac
done

command -v flock >/dev/null 2>&1 || { echo "缺少 flock，拒绝启动可能重复的 HQ 主服务" >&2; exit 2; }
exec 9>"$STATE_DIR/server.owner.lock" || exit 2
if ! flock -n 9; then
    echo "已有 HQ 主服务持有共享锁，本作业只加入 worker"
    READY=0
    for ((attempt=0; attempt<30; attempt++)); do
        if "$HQ" server info >/dev/null 2>&1; then READY=1; break; fi
        sleep 1
    done
    [[ "$READY" -eq 1 ]] || { echo "已有主服务未就绪，拒绝启动另一份编排" >&2; exit 2; }
    export TASKCTL_STATE="$STATE_DIR" TASKCTL_HOME="$HERE" SLURM_CPUS_PER_TASK="$CPUS"
    if [[ "$GPUS" == "none" ]]; then
        exec "$HQ" worker start --cpus "$CPUS" --detect-resources=none --on-server-lost=stop --idle-timeout=5m 9>&-
    fi
    export CUDA_VISIBLE_DEVICES="$GPUS"
    exec bash "$HERE/弹性worker.sh" 9>&-
fi
rm -f "$STOP_FLAG" "$RELOAD_FLAG" "$SERVICE_INFO"

# ---- 资源声明 ----
RES_ARGS=()
if [ "$GPUS" = "none" ]; then
    GPUS=""
elif [ -n "$GPUS" ]; then
    # 卡号需为整数列表；若 SLURM 给的是 UUID 等非整数，退化为 0..N-1
    if ! echo "$GPUS" | grep -qE '^[0-9]+(,[0-9]+)*$'; then
        n=$(echo "$GPUS" | tr ',' '\n' | wc -l)
        GPUS=$(seq -s, 0 $((n - 1)))
        echo "⚠️ CUDA_VISIBLE_DEVICES 非整数卡号，退化为 gpus=[$GPUS]"
    fi
    RES_ARGS=(--resource "gpus=[$GPUS]")
    # 资源探测旗标版本自适应：0.24 起 --no-detect-resources 被移除，更名为 --detect-resources=none（不探测，只认显式声明的资源）。
    if "$HQ" worker start --help 2>&1 | grep -q -- '--no-detect-resources'; then
        RES_ARGS+=(--no-detect-resources)          # HQ 0.20
    else
        RES_ARGS+=(--detect-resources=none)        # HQ 0.24+
    fi
fi

echo "==========================================="
echo "任务守护 启动：$(date)"
echo "  节点：$(hostname)"
echo "  GPU：${GPUS:-（无）}"
echo "  CPU:     $CPUS"
echo "  受管服务：${RUN_CMD:-（无）}"
echo "  状态目录：$STATE_DIR"
echo "==========================================="

# ---- 优雅退出 ----
stop_service() {
    [ -n "$SVC_PID" ] && kill -0 "$SVC_PID" 2>/dev/null || return 0
    kill -TERM "$SVC_PID" 2>/dev/null
    for _ in $(seq 1 "$RELOAD_TIMEOUT"); do
        kill -0 "$SVC_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$SVC_PID" 2>/dev/null; then
        echo "⚠️ 受管服务 ${RELOAD_TIMEOUT}s 内未退出，升级 SIGKILL"
        kill -KILL "$SVC_PID" 2>/dev/null
    fi
    wait "$SVC_PID" 2>/dev/null
    SVC_PID=""
}

cleanup() {
    trap - TERM INT EXIT
    echo "任务守护 退出中：$(date)"
    rm -f "$STOP_FLAG" "$RELOAD_FLAG" "$SERVICE_INFO"   # 消费请求标志与运行状态
    stop_service
    # 接管的 worker 不是本进程启的，不得代它收尸（否则会把仍在服务作业的 worker 杀掉）
    [ -n "$WORKER_PID" ] && kill "$WORKER_PID" 2>/dev/null
    if [ "$SERVER_OWNED" = 1 ] && [ -f "$HERE/自动扩容.py" ]; then
        "${TASKCTL_PYTHON:-python3}" "$HERE/自动扩容.py" cleanup >/dev/null 2>&1 || true
    fi
    # 只管自己启的 server：复用外部 server 时不得把它关掉（否则其他调用方会断联）。
    [ "$SERVER_OWNED" = 1 ] && "$HQ" server stop >/dev/null 2>&1
    wait 2>/dev/null
    echo "任务守护 已停止：$(date)"
    exit "$EXIT_RC"
}
trap cleanup TERM INT EXIT

# ---- 启动 server ----
# 先判是否已有活 server（复用则不接管它的生命周期），没有就清掉陈旧实例目录再启。
# 必要性：hq server 见到残留 access.json 会直接报 `File exists (os error 17)` 拒绝启动 → taskd 跟着退出 → 单任务能力永久丢失，只剩整包直跑（2026-10-20 实测踩到）；作业被 SLURM 掐断时 server 不会优雅退出，而看门狗每 5 天续交一次，故此清理是常态路径而非兜底。
SERVER_OWNED=0
if "$HQ" server info >/dev/null 2>&1; then
    echo "♻️  检测到已有 HQ server，直接复用（退出时不关它）"
else
    rm -rf "$HQ_SERVER_DIR"/* 2>/dev/null
    # 注意：hq 0.20 起把 access.json 写在实例子目录下（运行状态/server/001/access.json），故就绪判据用 glob 匹配任意实例，不能直接查 server 根目录。
    access_ready() { compgen -G "$HQ_SERVER_DIR/*/access.json" >/dev/null; }
    "$HQ" server start >>"$LOG_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    SERVER_OWNED=1
    for _ in $(seq 1 30); do                # 等 access.json 就绪（一般 <2s）
        access_ready && break
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "❌ server 启动失败，见 $LOG_DIR/server.log" >&2; exit 1; }
        sleep 0.5
    done
    access_ready || { echo "❌ server 未就绪" >&2; exit 1; }
fi
echo "✅ HQ server 就绪${SERVER_PID:+ (pid=$SERVER_PID)}"

# ---- 启动 worker ----
start_worker() {
    "$HQ" worker start --cpus "$CPUS" ${RES_ARGS[@]+"${RES_ARGS[@]}"} \
        >>"$LOG_DIR/worker.log" 2>&1 &
    WORKER_PID=$!
    echo "✅ HQ worker 已启动（pid=$WORKER_PID）"
}

# worker 探活：自启的按 pid；接管的没有 pid，只能查注册表里是否还有 RUNNING worker。
# 不能对空 WORKER_PID 用 kill -0（恒失败 → 会把健康集群误判为 worker 全死）。
worker_alive() {
    if [ -n "$WORKER_PID" ]; then
        kill -0 "$WORKER_PID" 2>/dev/null
    else
        "$HQ" worker list 2>/dev/null | grep -q RUNNING
    fi
}

if [ "$ADOPT_WORKER" = 1 ]; then
    if worker_alive; then
        echo "♻️  接管已有 worker（不新建，避免同卡并发）；失活时仍会自愈重启"
    else
        echo "⚠️  未见 RUNNING worker → 按资源声明新建一个"
        start_worker
    fi
else
    start_worker
fi

# ---- 受管前台服务（--run 指定时为作业主体，输出继承本脚本的 stdout/stderr）----
start_service() {
    [ -n "$RUN_CMD" ] || return 0
    bash -c "$RUN_CMD" &
    SVC_PID=$!
    echo "✅ 受管服务已启动（pid=$SVC_PID）"
    {
        echo "pid=$SVC_PID"
        echo "host=$(hostname)"
        echo "started_at=$(date '+%F %T')"
        echo "restarts=$SVC_RESTARTS"
        echo "cmd=$RUN_CMD"
    } > "$SERVICE_INFO"
}
start_service

if [[ "${TASKCTL_AUTOSCALE:-1}" == 1 && ! -f "$STATE_DIR/autoscale.disabled" ]]; then
    "${TASKCTL_PYTHON:-python3}" "$HERE/自动扩容.py" start || echo "⚠️ GPU 扩容监控未能启动"
fi

# 重启受管服务而不动守护自身 → SLURM 作业不退出 → 计算节点不释放。
request_reload() {
    rm -f "$RELOAD_FLAG"
    if [ -z "$SVC_PID" ] || ! kill -0 "$SVC_PID" 2>/dev/null; then
        echo "⚠️ 受管服务不在运行，忽略重启请求"
        return 0
    fi
    echo "🔁 收到重启请求 → 重启受管服务（作业与节点分配保持不变）"
    stop_service
    SVC_RESTARTS=$((SVC_RESTARTS + 1))
    start_service
}

# ---- 常驻监控循环 ----
WORKER_RESTARTS=0
while true; do
    sleep 5
    if [ -f "$STOP_FLAG" ]; then
        echo "收到停止标志 $STOP_FLAG"
        break
    fi
    [ -f "$RELOAD_FLAG" ] && request_reload
    # server 探活：自己启的按 pid 查；复用外部的没有 pid 可查，只能按命令探活，否则 kill -0 "" 会恒失败并被误判为 server 死亡。
    if [ "$SERVER_OWNED" = 1 ]; then
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "⚠️ server 意外退出"; EXIT_RC=1; break; }
    else
        "$HQ" server info >/dev/null 2>&1 || { echo "⚠️ 复用的 server 已失联"; EXIT_RC=1; break; }
    fi
    # 受管服务退出码三态：0 收工；重载码换代码重启；其余异常自愈（带上限）。
    if [ -n "$SVC_PID" ] && ! kill -0 "$SVC_PID" 2>/dev/null; then
        wait "$SVC_PID" 2>/dev/null
        svc_rc=$?
        SVC_PID=""
        if [ "$svc_rc" -eq 0 ]; then
            echo "✅ 受管服务正常结束 → 守护退出（作业随之结束）"
            break
        elif [ "$svc_rc" -eq "$RELOAD_CODE" ]; then
            echo "🔁 受管服务以重载码 $RELOAD_CODE 退出 → 重启以加载新代码"
            SVC_RESTARTS=$((SVC_RESTARTS + 1))
            start_service
        else
            SVC_CRASHES=$((SVC_CRASHES + 1))
            if [ "$SVC_CRASHES" -gt "$MAX_RESTARTS" ]; then
                echo "❌ 受管服务反复异常退出（最近码 $svc_rc）达上限 $MAX_RESTARTS" >&2
                EXIT_RC=1
                break
            fi
            echo "⚠️ 受管服务异常退出（码 $svc_rc），第 $SVC_CRASHES 次自愈重启"
            SVC_RESTARTS=$((SVC_RESTARTS + 1))
            start_service
        fi
    fi
    if ! worker_alive; then
        WORKER_RESTARTS=$((WORKER_RESTARTS + 1))
        if [ "$WORKER_RESTARTS" -gt 10 ]; then
            echo "❌ worker 反复崩溃达 10 次，放弃重启" >&2
            EXIT_RC=1
            break
        fi
        echo "⚠️ worker 意外退出，第 $WORKER_RESTARTS 次重启"
        start_worker
    fi
done

exit "$EXIT_RC"
