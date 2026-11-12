"""按 HQ 等待任务、Slurm 剩余配额和显存大小扩容，不复制计算任务。"""

import argparse
import csv
from dataclasses import dataclass
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

from GPUworker import 显存优先级

HERE = Path(__file__).resolve().parent
RESOURCE_SCALE = 10000


@dataclass(frozen=True)
class Request:
    job_id: int
    cpus: int
    gpus: int
    gpu_memory: float
    priority: int = 0
    显存上限: float = None


@dataclass(frozen=True)
class Offer:
    node: str
    partition: str
    gpu_type: str
    gpu_memory: int
    free_cpus: int
    free_gpus: int
    free_memory_mb: int
    budget_keys: tuple
    max_job_cpus: float = math.inf
    max_job_gpus: float = math.inf
    max_job_memory_mb: float = math.inf
    最长时限: str = "UNLIMITED"
    配置CPU: int = None
    配置GPU: int = None
    配置主机内存: int = None


def plan(requests, offers, budgets, max_workers=4, cpus_per_worker=8,
         memory_per_gpu_mb=65536):
    """按优先级和 FIFO 分配，放不下则回填后续任务；候选卡仍优先大显存。"""
    remaining = {key: dict(value) for key, value in budgets.items()}
    node_free = {}
    selected = []
    已分配 = set()
    候选 = [(编号, 请求, 节点) for 编号, 请求 in enumerate(requests) if 请求.gpus > 0 and 请求.gpu_memory > 0 for 节点 in offers if 节点.gpu_memory >= 请求.gpu_memory]
    for 编号, request, offer in sorted(候选, key=lambda 候选: (-显存优先级(候选[1].priority, (候选[1].gpu_memory, 候选[1].显存上限 or 候选[1].gpu_memory), 候选[2].gpu_memory), 候选[1].job_id, -候选[2].gpu_memory, 候选[2].partition, 候选[2].node, 候选[0])):
        if len(selected) >= max_workers:
            break
        if 编号 not in 已分配:
            available = node_free.setdefault(
                offer.node, {"cpu": offer.free_cpus, "gpu": offer.free_gpus,
                             "mem": offer.free_memory_mb})
            if offer.gpu_memory < request.gpu_memory or available["gpu"] < request.gpus:
                continue
            limits = [remaining[key] for key in offer.budget_keys]
            cpu_cap = min([available["cpu"], offer.max_job_cpus]
                          + [limit.get("cpu", math.inf) for limit in limits])
            cpus = min(max(cpus_per_worker, request.cpus), cpu_cap)
            resources = {"cpu": cpus, "gpu": request.gpus,
                         "mem": memory_per_gpu_mb * request.gpus,
                         "jobs": 1, "submit": 1, "node": 1}
            if (cpus < request.cpus or resources["mem"] > available["mem"]
                    or request.gpus > offer.max_job_gpus
                    or resources["mem"] > offer.max_job_memory_mb):
                continue
            if any(resources[key] > limit.get(key, math.inf)
                   for limit in limits for key in resources):
                continue
            有效优先级 = 显存优先级(request.priority, (request.gpu_memory, request.显存上限 or request.gpu_memory), offer.gpu_memory)
            selected.append({"job_id": request.job_id, "priority": 有效优先级,
                             "offer": offer, "cpus": int(cpus),
                             "gpus": request.gpus, "memory_mb": resources["mem"], "显存妥协": 有效优先级 != request.priority})
            已分配.add(编号)
            for key in ("cpu", "gpu", "mem"):
                available[key] -= resources[key]
            for limit in limits:
                for key in resources:
                    if key in limit:
                        limit[key] -= resources[key]
    return selected


def 逐档申请计划(请求列表, 节点列表, 额度, 已有档位=(), 每节点CPU=8, 每卡主机内存=65536):
    剩余请求 = [(编号, 请求) for 编号, 请求 in enumerate(请求列表) if 请求.gpus > 0 and 请求.gpu_memory > 0]
    if not 剩余请求:
        return []
    最大上限 = max(请求.显存上限 or 请求.gpu_memory for _, 请求 in 剩余请求)
    剩余额度 = {名称: dict(数值) for 名称, 数值 in 额度.items()}
    已申请 = set(已有档位)
    计划 = []
    for 分区 in sorted({节点.partition for 节点 in 节点列表}):
        分区节点 = [节点 for 节点 in 节点列表 if 节点.partition == 分区]
        档位列表 = sorted({节点.gpu_memory for 节点 in 分区节点})
        起始档位 = next((显存 for 显存 in 档位列表 if 显存 > 最大上限), 档位列表[-1])
        for 显存 in reversed([档位 for 档位 in 档位列表 if 档位 <= 起始档位]):
            if not 剩余请求:
                break
            if (分区, 显存) in 已申请:
                continue
            兼容请求 = [(编号, 请求) for 编号, 请求 in 剩余请求 if 请求.gpu_memory <= 显存]
            if not 兼容请求:
                break
            候选请求 = sorted(兼容请求, key=lambda 项: (-显存优先级(项[1].priority, (项[1].gpu_memory, 项[1].显存上限 or 项[1].gpu_memory), 显存), 项[1].job_id, 项[0]))
            分配成功 = False
            for 编号, 请求 in 候选请求:
                for 节点 in sorted((节点 for 节点 in 分区节点 if 节点.gpu_memory == 显存), key=lambda 节点: (节点.gpu_type, 节点.node)):
                    限额列表 = [剩余额度[键] for 键 in 节点.budget_keys]
                    CPU = min(max(每节点CPU, 请求.cpus), 节点.max_job_cpus, 节点.配置CPU if 节点.配置CPU is not None else math.inf)
                    主机内存 = math.ceil(每卡主机内存 * 请求.gpus)
                    if CPU < 请求.cpus or 请求.gpus > 节点.max_job_gpus or 主机内存 > 节点.max_job_memory_mb:
                        continue
                    if (节点.配置GPU is not None and 请求.gpus > 节点.配置GPU) or (节点.配置主机内存 is not None and 主机内存 > 节点.配置主机内存):
                        continue
                    if any(限额.get("submit", math.inf) < 1 for 限额 in 限额列表):
                        continue
                    有效优先级 = 显存优先级(请求.priority, (请求.gpu_memory, 请求.显存上限 or 请求.gpu_memory), 显存)
                    计划.append({"job_id": 请求.job_id, "priority": 有效优先级, "offer": 节点, "cpus": int(CPU), "gpus": int(请求.gpus), "memory_mb": 主机内存, "显存妥协": 有效优先级 != 请求.priority, "通用显卡": True, "排除节点": sorted({候选.node for 候选 in 分区节点 if 候选.gpu_memory != 显存})})
                    for 限额 in 限额列表:
                        if "submit" in 限额:
                            限额["submit"] -= 1
                    剩余请求 = [项 for 项 in 剩余请求 if 项[0] != 编号]
                    已申请.add((分区, 显存))
                    分配成功 = True
                    break
                if 分配成功:
                    break
    return 计划


def memory_mb(value):
    match = re.fullmatch(r"([0-9.]+)([KMGTPE]?)(?:[cn])?", value or "", re.I)
    if not match:
        raise ValueError(f"无法解析内存值：{value!r}")
    return int(float(match.group(1)) * {"": 1, "K": 1 / 1024, "M": 1,
               "G": 1024, "T": 1024 ** 2, "P": 1024 ** 3,
               "E": 1024 ** 4}[match.group(2).upper()])


def tres(value):
    result = {}
    for entry in (value or "").split(","):
        if "=" not in entry:
            continue
        name, amount = entry.split("=", 1)
        if name == "mem":
            result["mem"] = memory_mb(amount)
        elif name in ("cpu", "node", "gres/gpu"):
            result["gpu" if name == "gres/gpu" else name] = int(amount)
    return result


def gpu_memory(gpu_type):
    match = re.search(r"(192|141|128|96|80|64|48|40|32|24|16|12|8)GB", gpu_type, re.I)
    return int(match.group(1)) if match else None


def slurm_fields(line):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", line))


def resource_amount(entry):
    amount = next(iter(entry.get("request", {}).values()), 0)
    if isinstance(amount, list) and len(amount) == 2:
        amount = amount[0]
    if not isinstance(amount, (int, float)):
        raise ValueError(f"不支持的 HQ 资源请求：{entry}")
    return amount / RESOURCE_SCALE


def job_requests(detail):
    requests = []
    for submit in detail.get("submits", []):
        task = submit.get("array")
        if not task:
            continue
        variants = task.get("resources") or []
        if len(variants) != 1 or variants[0].get("n_nodes", 0):
            continue
        values = {entry["resource"]: resource_amount(entry)
                  for entry in variants[0].get("resources", [])}
        gpus = values.get("gpus", 0)
        memory_total = values.get("gpumem", 0)
        if gpus <= 0 or memory_total <= 0:
            continue
        memory = memory_total / gpus
        显存请求 = next(资源.get("request", {}) for 资源 in variants[0].get("resources", []) if 资源["resource"] == "gpumem")
        显存值 = next(iter(显存请求.values()))
        上限 = 显存值[1] / RESOURCE_SCALE / gpus if isinstance(显存值, list) else None
        waiting = sum(task.get("state") == "waiting" for task in detail.get("tasks", []))
        requests.extend([Request(detail["info"]["id"], values.get("cpus", 1),
                     gpus, memory, int(task.get("priority", 0)), 显存上限=上限)] * waiting)
    return requests


def worker_capacity(worker):
    values = {}
    for item in worker.get("configuration", {}).get("resources", {}).get("resources", []):
        if item["kind"] == "range":
            amount = item["end"] - item["start"] + 1
        elif item["kind"] == "list":
            amount = len(item["values"])
        elif item["kind"] == "sum":
            amount = item["size"] / RESOURCE_SCALE
        else:
            continue
        values[item["name"]] = amount
    return values


def consume_capacity(requests, capacities):
    remaining = sorted(requests, key=lambda item: (-item.priority, item.job_id))
    for capacity in capacities:
        单卡显存 = capacity.get("单卡显存", capacity.get("gpumem", 0) / max(1, capacity.get("gpus", 0)))
        for request in sorted(remaining, key=lambda 请求: (-显存优先级(请求.priority, (请求.gpu_memory, 请求.显存上限 or 请求.gpu_memory), 单卡显存) if 单卡显存 >= 请求.gpu_memory else math.inf, 请求.job_id)):
            if (capacity.get("gpus", 0) >= request.gpus
                    and capacity.get("cpus", 0) >= request.cpus
                    and capacity.get("gpumem", 0) >= request.gpu_memory * request.gpus):
                remaining.remove(request)
                capacity["gpus"] -= request.gpus
                capacity["cpus"] -= request.cpus
                capacity["gpumem"] -= request.gpu_memory * request.gpus
    return remaining


def _limit(value):
    return int(value) if value and value.isdigit() else math.inf


def 已占用额度(作业, 资源):
    if 资源 == "submit":
        return 1
    if 作业["state"] == "PENDING":
        return 0
    return 1 if 资源 == "jobs" else 作业.get(资源, 0)


def 可申请主机内存(配置, 已分配, 系统预留):
    return max(0, 配置.get("mem", 0) - 已分配.get("mem", 0) - int(系统预留 or 0))


def 小卡任务边界让位(请求列表, 当前作业, 大卡作业列表, 剩余额度, 节点列表):
    GPU请求 = [请求 for 请求 in 请求列表 if 请求.gpus > 0 and 请求.gpu_memory > 0]
    当前显存 = 当前作业.get("gpu_memory") or 0
    if not GPU请求 or 当前作业.get("state") != "RUNNING" or 当前显存 <= 0:
        return None
    最高请求 = min(GPU请求, key=lambda 请求: (-请求.priority, 请求.job_id))
    充足显存 = 最高请求.显存上限 or 最高请求.gpu_memory
    if 当前显存 >= 充足显存:
        return None
    for 大卡作业 in 大卡作业列表:
        if 大卡作业.get("state") != "PENDING" or 大卡作业.get("account") != 当前作业.get("account") or 大卡作业.get("partition") != 当前作业.get("partition"):
            continue
        if (大卡作业.get("gpu_memory") or 0) < 充足显存 or 大卡作业.get("gpu", 0) < 最高请求.gpus or 大卡作业.get("cpu", 0) < 最高请求.cpus:
            continue
        for 节点 in 节点列表:
            if 节点.partition != 大卡作业["partition"] or 节点.gpu_memory != 大卡作业["gpu_memory"]:
                continue
            分区额度键 = 当前作业.get("分区额度键")
            if not 分区额度键 or 分区额度键 not in 节点.budget_keys or 剩余额度.get(分区额度键, {}).get("jobs", math.inf) != 0:
                continue
            if any((1 if 资源 == "jobs" else 大卡作业.get(资源, 0)) > 限额 + 已占用额度(当前作业, 资源) for 键 in 节点.budget_keys for 资源, 限额 in 剩余额度[键].items() if 资源 != "submit"):
                continue
            return {"释放作业": str(当前作业["id"]), "等待大卡作业": str(大卡作业["id"]), "分区": 当前作业["partition"], "当前显存": 当前显存, "大卡显存": 大卡作业["gpu_memory"], "最高优先级任务": 最高请求.job_id, "优先级": 最高请求.priority}
    return None


class Controller:
    def __init__(self, state=None, 启动模式=False):
        self.state = Path(state or os.environ.get("TASKCTL_STATE", HERE / "运行状态")).resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        self.启动模式 = 启动模式
        配置路径 = self.state / "逐档启动.json"
        self.启动配置 = json.loads(配置路径.read_text()) if 配置路径.exists() else {}
        access = sorted((self.state / "server").glob("*/access.json"))
        if not access and not 启动模式:
            raise RuntimeError("没有可用 HQ server，先启动任务守护")
        digest = hashlib.sha256(access[-1].read_bytes() if access else str(self.state).encode()).hexdigest()[:10]
        self.name = f"hq-bootstrap-{hashlib.sha256(str(self.state).encode()).hexdigest()[:10]}" if 启动模式 else f"hq-elastic-{digest}"
        self.username = getpass.getuser()
        self.account = os.environ.get("TASKCTL_AUTO_ACCOUNT", os.environ.get("SLURM_JOB_ACCOUNT", self.username))
        self.env = dict(os.environ, TASKCTL_STATE=str(self.state),
                        TASKCTL_HOME=str(HERE), HQ_SERVER_DIR=str(self.state / "server"))
        if 启动模式:
            self.env.update(self.启动配置.get("环境", {}))
        self.account = self.env.get("TASKCTL_AUTO_ACCOUNT", self.env.get("SLURM_JOB_ACCOUNT", self.username))
        library = os.environ.get("HQ_LIBSTDCPP", "/public_bme2/Share200T/管吉松/envs/af3/lib")
        self.env["LD_LIBRARY_PATH"] = library + ":" + self.env.get("LD_LIBRARY_PATH", "")
        self.status_file = self.state / "autoscale.json"
        self.cpus = int(self.env.get("TASKCTL_AUTO_CPUS", "8"))
        self.memory_mb = memory_mb(self.env.get("TASKCTL_AUTO_MEMORY", "64G"))
        if min(self.cpus, self.memory_mb) < 1:
            raise ValueError("CPU和内存必须为正数")

    def run(self, arguments, env=None):
        result = subprocess.run(arguments, env=env or self.env, text=True,
                                capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f"{arguments[0]} 失败：{result.stderr.strip()[:500]}")
        return result.stdout.strip()

    def hq(self, *arguments):
        return json.loads(self.run([str(HERE / "bin" / "hq"), "--output-mode=json", *arguments]))

    def slurm_jobs(self):
        output = self.run(["squeue", "-h", "-u", self.username, "-o",
                           "%i|%a|%P|%q|%T|%C|%b|%D|%j|%V|%R|%m"])
        jobs = []
        for row in csv.reader(output.splitlines(), delimiter="|"):
            if len(row) != 12:
                raise ValueError("Slurm 作业输出格式异常，停止扩容")
            job_id, account, partition, qos, state, cpus, gres, nodes, name, submit, reason, memory = row
            counts = re.findall(r"(?:^|,)gpu(?::[^:,]+)?:(\d+)", gres)
            gpu_type = re.search(r"gpu:([^:,]+):\d+", gres)
            显存 = gpu_memory(gpu_type.group(1)) if gpu_type else None
            if counts and 显存 is None:
                明细 = slurm_fields(self.run(["scontrol", "-o", "show", "job", job_id]))
                分配节点 = 明细.get("NodeList", "")
                if 分配节点 and 分配节点 not in ("(null)", "None"):
                    节点明细 = self.run(["scontrol", "-o", "show", "node", 分配节点])
                    显存列表 = [gpu_memory(型号) for 型号 in re.findall(r"gpu:([^:,()]+):\d+", 节点明细)]
                    if 显存列表 and all(显存列表):
                        显存 = min(显存列表)
                if 显存 is None and 明细.get("ExcNodeList") not in (None, "(null)", "None", ""):
                    排除节点 = {行.strip() for 行 in self.run(["scontrol", "show", "hostnames", 明细["ExcNodeList"]]).splitlines()}
                else:
                    排除节点 = set()
            else:
                排除节点 = set()
            jobs.append({"id": job_id, "account": account, "partition": partition,
                         "qos": qos, "state": state, "cpu": int(cpus),
                         "gpu": sum(map(int, counts)) * int(nodes), "node": int(nodes),
                         "mem": memory_mb(memory) * (int(cpus) if memory.endswith("c") else int(nodes)),
                         "name": name, "submit": submit, "reason": reason,
                         "gpu_memory": 显存, "排除节点": sorted(排除节点), "通用显卡": gpu_type is None})
        return jobs

    def snapshot(self):
        partitions = {fields["PartitionName"]: fields for line in
                      self.run(["scontrol", "-o", "show", "partition"]).splitlines()
                      if (fields := slurm_fields(line)).get("PartitionName")}
        allowed = {}
        requested = set(filter(None, self.env.get("TASKCTL_AUTO_PARTITIONS", "").split(",")))
        groups = {__import__("grp").getgrgid(group).gr_name for group in os.getgroups()}
        for name, fields in partitions.items():
            if fields.get("State") != "UP" or (requested and name not in requested):
                continue
            accounts = fields.get("AllowAccounts", "ALL").split(",")
            denied = fields.get("DenyAccounts", "").split(",")
            allowed_groups = fields.get("AllowGroups", "ALL").split(",")
            if self.account in denied or ("ALL" not in accounts and self.account not in accounts):
                continue
            if "ALL" not in allowed_groups and not groups.intersection(allowed_groups):
                continue
            allowed[name] = fields
        jobs = self.slurm_jobs()
        user_qos = self.env.get("TASKCTL_AUTO_QOS", self.env.get("SLURM_JOB_QOS", ""))
        if not user_qos:
            user_qos = next((job["qos"] for job in jobs if job["account"] == self.account), "")
        names = {fields.get("QoS", "") for fields in allowed.values()} | {job["qos"] for job in jobs} | {user_qos}
        names.discard("")
        names.discard("N/A")
        qos_rows = self.run(["sacctmgr", "-Pn", "show", "qos", "where", "name=" + ",".join(sorted(names)),
            "format=Name,MaxTRESPerUser,MaxJobsPerUser,MaxSubmitJobsPerUser,MaxTRES"])
        qos = {}
        budgets = {}
        for row in csv.reader(qos_rows.splitlines(), delimiter="|"):
            if len(row) < 5:
                raise ValueError("无法读取 QOS 限额，停止扩容")
            name, max_tres, max_jobs, max_submit, per_job = row[:5]
            qos[name] = tres(per_job)
            budgets[f"qos:{name}"] = dict(tres(max_tres), jobs=_limit(max_jobs), submit=_limit(max_submit))
        if not qos or any(f"qos:{name}" not in budgets for name in names):
            raise ValueError("QOS 限额不完整，停止扩容")
        association_rows = self.run(["sacctmgr", "-Pn", "show", "assoc", "where", f"account={self.account}",
            "format=Account,User,GrpTRES,GrpJobs,MaxJobs,MaxSubmit"])
        association_keys = []
        for row in csv.reader(association_rows.splitlines(), delimiter="|"):
            if len(row) < 6 or row[0] != self.account or row[1] not in ("", self.username):
                continue
            key = f"assoc:{row[0]}:{row[1]}"
            association_keys.append(key)
            budgets[key] = dict(tres(row[2]), jobs=min(_limit(row[3]), _limit(row[4])), submit=_limit(row[5]))
        if not association_keys:
            raise ValueError("账号关联不可读，停止扩容")
        for job in jobs:
            keys = {f"qos:{job['qos']}", f"qos:{partitions.get(job['partition'], {}).get('QoS', '')}"}
            if job["account"] == self.account:
                keys.update(association_keys)
            for key in keys:
                if key not in budgets:
                    continue
                for resource, limit in budgets[key].items():
                    amount = 已占用额度(job, resource)
                    budgets[key][resource] = max(0, limit - amount)
        offers = []
        for line in self.run(["scontrol", "-o", "show", "node"]).splitlines():
            fields = slurm_fields(line)
            state = fields.get("State", "")
            if not any(state.startswith(item) for item in ("IDLE", "MIXED", "ALLOCATED")) or any(
                    flag in state for flag in ("DRAIN", "DOWN", "FAIL", "RESERVED", "NO_RESPOND")):
                continue
            typed = re.findall(r"(?:^|,)gpu:([^:,()]+):(\d+)", fields.get("Gres", ""))
            if len(typed) != 1 or not (memory := gpu_memory(typed[0][0])):
                continue
            configured, allocated = tres(fields.get("CfgTRES")), tres(fields.get("AllocTRES"))
            free_gpu = configured.get("gpu", 0) - allocated.get("gpu", 0)
            for partition in fields.get("Partitions", "").split(","):
                if partition not in allowed:
                    continue
                part = allowed[partition]
                allowed_qos = part.get("AllowQos", "ALL").split(",")
                if user_qos and "ALL" not in allowed_qos and user_qos not in allowed_qos:
                    continue
                keys = tuple(dict.fromkeys(association_keys + [f"qos:{name}" for name in
                             (part.get("QoS"), user_qos) if name and name != "N/A"]))
                if any(key not in budgets for key in keys):
                    continue
                per_job = [qos.get(part.get("QoS"), {}), qos.get(user_qos, {})]
                offers.append(Offer(fields["NodeName"], partition, typed[0][0], memory,
                    configured.get("cpu", 0) - allocated.get("cpu", 0), free_gpu,
                    可申请主机内存(configured, allocated, fields.get("MemSpecLimit", 0)), keys,
                    min(item.get("cpu", math.inf) for item in per_job),
                    min(item.get("gpu", math.inf) for item in per_job),
                    min(item.get("mem", math.inf) for item in per_job), 最长时限=part.get("MaxTime", "UNLIMITED"), 配置CPU=configured.get("cpu", 0), 配置GPU=configured.get("gpu", 0), 配置主机内存=可申请主机内存(configured, {}, fields.get("MemSpecLimit", 0))))
        for 作业 in jobs:
            作业["分区额度键"] = f"qos:{partitions.get(作业['partition'], {}).get('QoS', '')}"
            档位 = {节点.gpu_memory for 节点 in offers if 节点.partition == 作业["partition"] and 节点.node not in 作业.get("排除节点", set())}
            if 作业["gpu"] > 0 and 作业["gpu_memory"] is None and len(档位) == 1:
                作业["gpu_memory"] = next(iter(档位))
        return jobs, offers, budgets, user_qos

    def demand(self):
        jobs = self.hq("job", "list", "--all")
        active = [job["id"] for job in jobs if (job.get("task_stats") or {}).get("waiting")
                  or (job.get("task_stats") or {}).get("running")]
        details = self.hq("job", "info", ",".join(map(str, active))) if active else []
        requests = []
        used = {}
        for detail in details:
            requests.extend(job_requests(detail))
            for task in detail.get("tasks", []):
                if task.get("state") != "running":
                    continue
                usage = used.setdefault(task.get("worker"), {"cpus": 0, "gpus": 0, "gpumem": 0})
                arrays = [submit.get("array") for submit in detail.get("submits", []) if submit.get("array")]
                if not arrays:
                    continue
                variants = arrays[0].get("resources") or []
                if len(variants) != 1:
                    continue
                for entry in variants[0].get("resources", []):
                    name = entry["resource"]
                    usage[name] = usage.get(name, 0) + resource_amount(entry)
        workers = [worker for worker in self.hq("worker", "list") if worker.get("ended") is None]
        capacities = []
        for worker in workers:
            capacity = worker_capacity(worker)
            capacity["单卡显存"] = capacity.get("gpumem", 0) / max(1, capacity.get("gpus", 0))
            for resource, amount in used.get(worker["id"], {}).items():
                capacity[resource] = max(0, capacity.get(resource, 0) - amount)
            capacities.append(capacity)
        return consume_capacity(requests, capacities), workers, details

    def submit(self, allocation, qos):
        offer = allocation["offer"]
        logs = self.state / "elastic"
        logs.mkdir(exist_ok=True)
        environment = {key: value for key, value in self.env.items() if not key.startswith(
            ("SLURM_", "SLURMD_", "HQ_RESOURCE_", "TASKCTL_GPU_"))
            and key not in ("CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
        environment["TASKCTL_AUTO_IDLE_TIMEOUT"] = os.environ.get("TASKCTL_AUTO_IDLE_TIMEOUT", "5m")
        时限秒 = parse_duration(offer.最长时限)
        if math.isfinite(时限秒):
            if 时限秒 <= 0:
                raise ValueError(f"分区 {offer.partition} 的最大时限无效：{offer.最长时限}")
            environment["TASKCTL_WORKER_TIME_LIMIT"] = f"{max(1, int(时限秒) - 120)}s"
            时限请求 = offer.最长时限
        else:
            environment.pop("TASKCTL_WORKER_TIME_LIMIT", None)
            时限请求 = "0"
        arguments = ["sbatch", "--parsable", f"--job-name={self.name}",
            f"--account={self.account}", f"--partition={offer.partition}",
            "--nodes=1", "--ntasks=1", f"--cpus-per-task={allocation['cpus']}",
            f"--mem={math.ceil(allocation['memory_mb'])}M", f"--gres=gpu:{int(allocation['gpus'])}" if allocation.get("通用显卡") else f"--gres=gpu:{offer.gpu_type}:{int(allocation['gpus'])}",
            f"--time={时限请求}", "--export=ALL", f"--chdir={HERE}",
            f"--output={logs}/%j.out", f"--error={logs}/%j.err"]
        if qos:
            arguments.append(f"--qos={qos}")
        if allocation.get("排除节点"):
            arguments.append("--exclude=" + ",".join(allocation["排除节点"]))
        脚本 = self.启动配置.get("作业脚本") if getattr(self, "启动模式", False) else str(HERE / "弹性worker.sh")
        if not 脚本 or not Path(脚本).is_file():
            raise ValueError("缺少可用的逐档启动作业脚本")
        job_id = self.run(arguments + [脚本], environment).split(";", 1)[0]
        if not job_id.isdigit():
            raise ValueError("sbatch 未返回合法 job ID")
        print(f"扩容提交 {job_id}：{offer.partition}/{offer.node} "
              f"{offer.gpu_type} ×{allocation['gpus']} CPU={allocation['cpus']} "
              f"HQ={allocation['job_id']} priority={allocation['priority']} 最长时限={时限请求}", flush=True)
        return job_id

    def 任务边界检查(self, 工作者编号, 已结束作业号, 任务尚未开始=False):
        请求列表, 工作者列表, 任务详情 = self.demand()
        if 工作者编号 is None:
            编号列表 = {任务["worker"] for 详情 in 任务详情 if 详情["info"]["id"] == 已结束作业号 for 任务 in 详情.get("tasks", []) if 任务.get("state") == "running" and 任务.get("worker") is not None}
            if len(编号列表) != 1:
                raise ValueError("无法唯一确定任务所在worker，拒绝释放")
            工作者编号 = 编号列表.pop()
        if 任务尚未开始:
            当前详情 = next((详情 for 详情 in 任务详情 if 详情["info"]["id"] == 已结束作业号), None)
            if 当前详情 is None:
                raise ValueError("启动边界任务不存在，拒绝释放")
            待运行详情 = dict(当前详情, tasks=[dict(任务, state="waiting") for 任务 in 当前详情.get("tasks", []) if 任务.get("worker") == 工作者编号 and 任务.get("state") == "running"])
            请求列表.extend(job_requests(待运行详情))
        工作者 = next((条目 for 条目 in 工作者列表 if 条目["id"] == 工作者编号), None)
        if 工作者 is None:
            raise ValueError("任务边界的 worker 已不存在，拒绝释放")
        分配号 = str((工作者.get("allocation") or {}).get("id", ""))
        作业列表, 节点列表, 额度, _ = self.snapshot()
        当前作业 = next((作业 for 作业 in 作业列表 if str(作业["id"]) == 分配号 and 作业["name"] == self.name), None)
        if 当前作业 is None:
            return None
        if any(任务.get("worker") == 工作者编号 and 任务.get("state") == "running" and 详情["info"]["id"] != 已结束作业号 for 详情 in 任务详情 for 任务 in 详情.get("tasks", [])):
            return None
        保留作业号 = {str(编号) for 编号 in self.启动配置.get("保留作业", [])}
        上游作业名 = set(filter(None, self.env.get("TASKCTL_AUTO_JOB_NAMES", "").split(",")))
        大卡作业 = [作业 for 作业 in 作业列表 if 作业["name"] == self.name or 作业["name"] in 上游作业名 or str(作业["id"]) in 保留作业号]
        决策 = 小卡任务边界让位(请求列表, 当前作业, 大卡作业, 额度, 节点列表)
        if 决策:
            决策["工作者"] = 工作者编号
            记录目录 = self.state / "任务边界"
            记录目录.mkdir(exist_ok=True)
            路径 = 记录目录 / f"{分配号}.json"
            临时路径 = 路径.with_suffix(f".tmp.{os.getpid()}")
            临时路径.write_text(json.dumps({**决策, "已完成HQ作业": 已结束作业号, "时间": time.strftime("%F %T")}, ensure_ascii=False, indent=2), encoding="utf-8")
            临时路径.replace(路径)
        return 决策

    def upgrade_idle_workers(self, workers, details):
        if not (HERE / "bin" / "hq.native").exists():
            return False
        for worker in workers:
            capacity = worker_capacity(worker)
            if not capacity.get("gpus") or capacity.get("gpumem"):
                continue
            allocation = worker.get("allocation") or {}
            if str(allocation.get("id")) != self.env.get("SLURM_JOB_ID"):
                continue
            busy = any(task.get("worker") == worker["id"] and task.get("state") == "running"
                       for detail in details if detail.get("info", {}).get("name") != "taskctl-autoscale"
                       for task in detail.get("tasks", []))
            if busy:
                continue
            print(f"worker {worker['id']} 无用户计算，交由守护重建并补齐显存声明", flush=True)
            self.run([str(HERE / "bin" / "hq"), "worker", "stop", str(worker["id"])])
            return True
        return False

    def tick(self, apply=False):
        if apply:
            with open(self.state / "逐档申请.lock", "a") as 锁:
                fcntl.flock(锁, fcntl.LOCK_EX)
                return self.申请周期(apply=True)
        return self.申请周期()

    def 申请周期(self, apply=False):
        if getattr(self, "启动模式", False):
            if not self.启动配置.get("作业脚本"):
                raise ValueError("尚无逐档启动配置，请由上游提供真实等待任务和作业入口")
            requests, workers, details = [Request(**请求) for 请求 in self.启动配置.get("请求", [])], [], []
        else:
            requests, workers, details = self.demand()
        if apply and requests and self.upgrade_idle_workers(workers, details):
            return {"upgrading_idle_worker": True}
        jobs, offers, budgets, qos = self.snapshot()
        owned = [job for job in jobs if job["name"] == self.name]
        registered = {str((worker.get("allocation") or {}).get("id")) for worker in workers}
        保留作业 = {str(编号) for 编号 in getattr(self, "启动配置", {}).get("保留作业", [])}
        上游作业名 = set(filter(None, (self.env.get("TASKCTL_AUTO_JOB_NAMES", "") if hasattr(self, "env") else "").split(",")))
        保留作业.update(str(作业["id"]) for 作业 in jobs if 作业["name"] in 上游作业名)
        保留作业.update(registered)
        保留作业.add(self.env.get("SLURM_JOB_ID", "") if hasattr(self, "env") else "")
        已有档位 = {(作业["partition"], 作业["gpu_memory"]) for 作业 in jobs if 作业["account"] == getattr(self, "account", 作业["account"]) and (作业["name"].startswith(("hq-elastic-", "hq-bootstrap-")) or str(作业["id"]) in 保留作业) and (作业.get("state", "PENDING") == "PENDING" or (作业.get("state") == "RUNNING" and str(作业["id"]) not in registered))}
        allocations = 逐档申请计划(requests, offers, budgets, 已有档位=已有档位, 每节点CPU=self.cpus, 每卡主机内存=self.memory_mb)
        升级计划 = []
        GPU请求 = [请求 for 请求 in requests if 请求.gpus > 0 and 请求.gpu_memory > 0]
        剩余需求 = list(requests)
        if GPU请求:
            规划额度 = {名称: dict(数值, submit=math.inf) for 名称, 数值 in budgets.items()}
            for 分区 in sorted({节点.partition for 节点 in offers}):
                未覆盖GPU请求 = [请求 for 请求 in 剩余需求 if 请求.gpus > 0 and 请求.gpu_memory > 0]
                if not 未覆盖GPU请求:
                    break
                最高请求 = min(未覆盖GPU请求, key=lambda 请求: (-请求.priority, 请求.job_id))
                候选计划 = 逐档申请计划([最高请求], [节点 for 节点 in offers if 节点.partition == 分区], 规划额度, 每节点CPU=self.cpus, 每卡主机内存=self.memory_mb)
                if not 候选计划:
                    continue
                大卡计划 = 候选计划[0]
                节点 = 大卡计划["offer"]
                if (分区, 节点.gpu_memory) in 已有档位 or any(分配["offer"].partition == 分区 and 分配["offer"].gpu_memory == 节点.gpu_memory for 分配 in allocations):
                    continue
                if all(budgets[键].get("submit", math.inf) >= 1 for 键 in 节点.budget_keys):
                    continue
                可升级 = [作业 for 作业 in owned if 作业["state"] == "PENDING" and 作业.get("通用显卡") and 作业["partition"] == 分区 and 0 < (作业.get("gpu_memory") or 0) < 节点.gpu_memory and 作业["gpu"] >= 大卡计划["gpus"] and 作业["cpu"] >= 最高请求.cpus and 作业["mem"] >= 大卡计划["memory_mb"]]
                if 可升级:
                    作业 = min(可升级, key=lambda 作业: (作业["gpu_memory"], int(作业["id"])))
                    升级计划.append({"job_id": 作业["id"], "partition": 分区, "原显存": 作业["gpu_memory"], "gpu_memory": 节点.gpu_memory, "排除节点": 大卡计划["排除节点"], "需求任务": 最高请求.job_id})
                    剩余需求.remove(最高请求)
        if 升级计划:
            allocations = 逐档申请计划(剩余需求, offers, budgets, 已有档位=已有档位, 每节点CPU=self.cpus, 每卡主机内存=self.memory_mb)
        cancelled = []
        if apply:
            for 升级 in 升级计划:
                self.run(["scontrol", "update", f"JobId={升级['job_id']}", "ExcNodeList=" + ",".join(升级["排除节点"])])
                print(f"升级等待分配 {升级['job_id']}：{升级['partition']} {升级['原显存']}GiB → {升级['gpu_memory']}GiB，保留作业号，HQ需求={升级['需求任务']}", flush=True)
            for job in owned:
                if job["state"] != "PENDING":
                    continue
                if not requests:
                    self.run(["scancel", job["id"]])
                    cancelled.append(job["id"])
            for allocation in allocations:
                self.submit(allocation, qos)
        status = {"time": time.strftime("%F %T"), "allocation_name": self.name,
                  "waiting_gpu_tasks": len(requests), "owned_allocations": owned,
                  "quota_headroom": {key: {name: (value if math.isfinite(value) else None)
                                           for name, value in limit.items()} for key, limit in budgets.items()},
                  "plan": [{**{key: value for key, value in item.items() if key != "offer"},
                            "node": item["offer"].node, "partition": item["offer"].partition,
                            "gpu_memory": item["offer"].gpu_memory} for item in allocations],
                  "upgraded_pending": 升级计划,
                  "cancelled_pending": cancelled}
        if apply:
            temporary = self.status_file.with_suffix(f".tmp.{os.getpid()}")
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2))
            temporary.replace(self.status_file)
        return status

    def cleanup(self, all_jobs=False):
        for job in self.slurm_jobs():
            if job["name"] == self.name and (all_jobs or job["state"] == "PENDING"):
                self.run(["scancel", job["id"]])

    def start(self):
        self.state.joinpath("autoscale.disabled").unlink(missing_ok=True)
        for job in self.hq("job", "list", "--all"):
            stats = job.get("task_stats") or {}
            if job.get("name") == "taskctl-autoscale" and (stats.get("waiting") or stats.get("running")):
                return job["id"]
        arguments = [str(HERE / "bin" / "hq"), "--output-mode=quiet", "submit",
            "--name=taskctl-autoscale", "--cpus=1", "--priority=100", "--crash-limit=unlimited",
            f"--stdout={self.state}/logs/%{{JOB_ID}}/%{{TASK_ID}}.stdout",
            f"--stderr={self.state}/logs/%{{JOB_ID}}/%{{TASK_ID}}.stderr",
            f"--env=TASKCTL_STATE={self.state}", f"--env=TASKCTL_AUTO_ACCOUNT={self.account}"]
        for name, value in self.env.items():
            if name.startswith("TASKCTL_AUTO_"):
                arguments.append(f"--env={name}={value}")
        return self.run(arguments + ["--", sys_python(), str(HERE / "自动扩容.py"), "watch"])

    def watch(self):
        with open(self.state / "autoscale.lock", "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            stopped = False

            def stop(signum, frame):
                nonlocal stopped
                stopped = True

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            failures = 0
            try:
                while not stopped and not (self.state / "autoscale.disabled").exists():
                    try:
                        self.tick(apply=True)
                        failures = 0
                    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
                        failures += 1
                        print(f"扩容检查失败，未继续申请：{error}", flush=True)
                        if failures >= 5:
                            break
                    time.sleep(30)
            finally:
                self.cleanup()


def parse_duration(value):
    if value in ("UNLIMITED", "INFINITE"):
        return math.inf
    days, _, clock = value.rpartition("-")
    fields = list(map(int, clock.split(":")))
    if len(fields) != 3:
        raise ValueError(f"无法解析时限 {value}")
    return (int(days or 0) * 24 + fields[0]) * 3600 + fields[1] * 60 + fields[2]


def sys_python():
    import sys
    return os.environ.get("TASKCTL_PYTHON", sys.executable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "once", "start", "stop", "status", "watch", "cleanup", "bootstrap", "boundary"))
    parser.add_argument("--仅计划", action="store_true", help="启动模式只查看逐档计划，不提交 Slurm 作业")
    parser.add_argument("--工作者", type=int)
    parser.add_argument("--已结束作业", type=int)
    parser.add_argument("--任务尚未开始", action="store_true")
    args = parser.parse_args()
    controller = Controller(启动模式=args.command == "bootstrap")
    if args.command == "boundary":
        if args.已结束作业 is None:
            parser.error("任务边界检查必须指定 --已结束作业")
        决策 = controller.任务边界检查(args.工作者, args.已结束作业, 任务尚未开始=args.任务尚未开始)
        if 决策:
            print(json.dumps(决策, ensure_ascii=False), flush=True)
        return 75 if 决策 else 0
    elif args.command in ("plan", "once", "bootstrap"):
        print(json.dumps(controller.tick(apply=args.command in ("once", "bootstrap") and not args.仅计划), ensure_ascii=False, indent=2))
    elif args.command == "start":
        print(f"扩容监控 HQ ID={controller.start()}")
    elif args.command == "watch":
        controller.watch()
    elif args.command == "status":
        print(controller.status_file.read_text() if controller.status_file.exists() else "尚无扩容检查记录")
    elif args.command == "stop":
        (controller.state / "autoscale.disabled").touch()
        controller.cleanup()
        print("已停用自动扩容；只取消尚未启动的扩容作业，运行中的计算不打断")
    else:
        controller.cleanup(all_jobs=True)


if __name__ == "__main__":
    raise SystemExit(main())