"""为 HQ worker 声明 Slurm 实际分配 GPU 的显存和算力。"""

import csv
from decimal import Decimal, InvalidOperation
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def 解析显存范围(显存):
    if isinstance(显存, str):
        范围 = 显存.split(":")
    elif isinstance(显存, (list, tuple)):
        if len(显存) != 2:
            raise ValueError("JSON 显存范围必须有且仅有上下限两个元素")
        范围 = list(显存)
    else:
        范围 = [显存]
    if len(范围) == 1:
        范围 *= 2
    if len(范围) != 2 or any(isinstance(数值, bool) for 数值 in 范围):
        raise ValueError("显存需求必须是单值或下限:上限，JSON 可使用两元素数组")
    try:
        下限, 上限 = (Decimal(str(数值)) for 数值 in 范围)
    except InvalidOperation as 错误:
        raise ValueError("显存范围必须是有限非负数") from 错误
    if not 下限.is_finite() or not 上限.is_finite() or not 0 <= 下限 <= 上限 or (下限 == 0 < 上限):
        raise ValueError("显存范围必须满足 0 < 下限 <= 上限，或同时为零表示 CPU 任务")
    if any(数值 * 10000 != (数值 * 10000).to_integral_value() for 数值 in (下限, 上限)):
        raise ValueError("显存范围最多支持四位小数")
    return float(下限), float(上限)


def 显存优先级(原优先级, 显存范围, 节点显存):
    下限, 上限 = 显存范围
    if 节点显存 < 下限:
        return None
    return 原优先级 - int(节点显存 < 上限)


def 原生显存范围(显存, 显卡数=1):
    下限, 上限 = 解析显存范围(显存)
    数值 = []
    for 边界 in (下限, 上限):
        文本 = format(Decimal(str(边界)) * int(显卡数), "f")
        数值.append(文本.rstrip("0").rstrip(".") if "." in 文本 else 文本)
    return 数值[0] if 下限 == 上限 else ":".join(数值)


def 显存运行环境(环境):
    结果 = dict(环境)
    下限 = 环境.get("TASKCTL_GPU_MEMORY_MIN_GB")
    上限 = 环境.get("TASKCTL_GPU_MEMORY_MAX_GB")
    if 下限 is None and 上限 is None:
        return 结果
    范围 = 解析显存范围((下限, 上限))
    if "TASKCTL_GPU_MEMORY_GB" not in 环境:
        raise ValueError("worker 未声明实际每卡显存，拒绝运行显存范围任务")
    显存 = 解析显存范围(环境["TASKCTL_GPU_MEMORY_GB"])[0]
    优先级 = 显存优先级(0, 范围, 显存)
    if 优先级 is None:
        raise ValueError(f"节点每卡显存 {显存:g}GiB 低于任务下限 {范围[0]:g}GiB，拒绝运行")
    结果["TASKCTL_GPU_MEMORY_COMPROMISED"] = "1" if 优先级 < 0 else "0"
    return 结果


def 检查旧worker任务边界(环境):
    if 环境.get("TASKCTL_NATIVE_TASK_BOUNDARY") == "1" or not all(环境.get(键) for 键 in ("SLURM_JOB_ID", "HQ_JOB_ID", "HQ_RESOURCE_VALUES_gpus")):
        return
    默认命令 = [环境.get("TASKCTL_PYTHON", sys.executable), str(Path(__file__).resolve().with_name("自动扩容.py")), "boundary"]
    命令 = json.loads(环境["TASKCTL_TASK_BOUNDARY_COMMAND"]) if 环境.get("TASKCTL_TASK_BOUNDARY_COMMAND") else 默认命令
    参数 = [*命令, "--已结束作业", 环境["HQ_JOB_ID"], "--任务尚未开始"]
    if 环境.get("HQ_WORKER_ID"):
        参数.extend(["--工作者", 环境["HQ_WORKER_ID"]])
    try:
        结果 = subprocess.run(参数, env=环境, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as 错误:
        print(f"任务启动边界检查异常，保留节点继续计算：{错误}", file=sys.stderr, flush=True)
        return
    if 结果.returncode == 0:
        return
    if 结果.returncode != 75:
        print("任务启动边界检查失败，保留节点继续计算：" + 结果.stderr.strip(), file=sys.stderr, flush=True)
        return
    try:
        决策 = json.loads(结果.stdout)
        工作者编号 = int(决策["工作者"])
    except (ValueError, KeyError, TypeError) as 错误:
        print(f"任务启动边界决策无效，保留节点继续计算：{错误}", file=sys.stderr, flush=True)
        return
    print("上一任务已结束，为等待大卡释放旧worker；本任务尚未开始，交回HQ等待", flush=True)
    subprocess.run([str(Path(__file__).resolve().parent / "bin" / "hq"), "worker", "stop", str(工作者编号)], env=环境, check=True, timeout=30)
    while True:
        signal.pause()


def prepare_submission(arguments):
    arguments = list(arguments)
    boundary = arguments.index("--") if "--" in arguments else len(arguments)
    options = arguments[:boundary]
    if "submit" not in options:
        return arguments
    memory = 0
    gpu_options = set()
    index = 0
    while index < boundary:
        value = options[index]
        if value == "--resource" and index + 1 < boundary:
            candidate, size = options[index + 1], 2
        elif value.startswith("--resource="):
            candidate, size = value.split("=", 1)[1], 1
        else:
            index += 1
            continue
        name, _, amount = candidate.partition("=")
        if name in ("gpus", "gpumem"):
            gpu_options.update(range(index, index + size))
        if name == "gpumem":
            memory = 解析显存范围(amount)[0]
        index += size
    if memory > 0:
        return arguments
    return [value for index, value in enumerate(arguments) if index not in gpu_options]


def prepare(arguments, environ, query_output):
    arguments = list(arguments)
    resource = None
    for index, value in enumerate(arguments):
        candidate = arguments[index + 1] if value == "--resource" else value.removeprefix("--resource=")
        if candidate.startswith("gpus=["):
            resource = candidate[6:-1].split(",")
            break
    if not resource:
        return arguments, dict(environ)
    rows = list(csv.reader(io.StringIO(query_output), skipinitialspace=True))
    selected = [row for row in rows if row[0] in resource or row[1] in resource]
    if len(rows) == len(resource):
        selected = rows
    if len(selected) != len(resource):
        raise ValueError("无法将 HQ GPU 卡号与实际可见显卡匹配，拒绝声明资源")
    memory = min(int(float(row[3])) // 1024 for row in selected)
    capability = min(float(row[4]) for row in selected)
    if memory < 1:
        raise ValueError("GPU 显存探测失败")
    if not any("gpumem=" in value for value in arguments):
        arguments += ["--resource", f"gpumem=sum({memory * len(resource)})"]
    environment = dict(environ, TASKCTL_GPU_MEMORY_GB=str(memory),
                       TASKCTL_GPU_COMPUTE_CAP=str(capability))
    environment["TASKCTL_TASK_BOUNDARY_COMMAND"] = json.dumps([environ.get("TASKCTL_PYTHON", sys.executable), str(Path(__file__).resolve().with_name("自动扩容.py")), "boundary"], ensure_ascii=False)
    environment["TASKCTL_NATIVE_TASK_BOUNDARY"] = "1"
    if "--cpus" in arguments:
        environment["TASKCTL_WORKER_CPUS"] = arguments[arguments.index("--cpus") + 1]
    return arguments, environment


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--task":
        if len(sys.argv) < 3:
            raise SystemExit("缺少待执行任务命令")
        环境 = 显存运行环境(os.environ)
        检查旧worker任务边界(环境)
        os.execvpe(sys.argv[2], sys.argv[2:], 环境)
    arguments = prepare_submission(sys.argv[1:])
    if not arguments:
        raise SystemExit("用法：python GPUworker.py <hq executable> worker start ...")
    if "worker" in arguments and "start" in arguments and any("gpus=[" in value for value in arguments):
        query = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits"], text=True, timeout=30)
        arguments, environment = prepare(arguments, os.environ, query)
    else:
        environment = dict(os.environ)
    os.execvpe(arguments[0], arguments, environment)


if __name__ == "__main__":
    main()