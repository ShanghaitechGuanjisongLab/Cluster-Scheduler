"""HQ GPU 资源声明和弹性扩容的离线测试，不提交 Slurm 作业。"""

import json
import importlib.util
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("taskctl_autoscale", ROOT / "自动扩容.py")
AUTOSCALE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUTOSCALE
SPEC.loader.exec_module(AUTOSCALE)
GPU_SPEC = importlib.util.spec_from_file_location("gpu_worker", ROOT / "GPUworker.py")
GPU_WORKER = importlib.util.module_from_spec(GPU_SPEC)
GPU_SPEC.loader.exec_module(GPU_WORKER)
QUERY_SPEC = importlib.util.spec_from_file_location("queue_query", ROOT / "查看队列.py")
QUEUE_QUERY = importlib.util.module_from_spec(QUERY_SPEC)
QUERY_SPEC.loader.exec_module(QUEUE_QUERY)


class 队列查询测试(unittest.TestCase):
    def test_只查询等待任务且按已提交优先级和FIFO排序(self):
        def 作业(编号, 优先级, 资源, 等待=1):
            return {"info": {"id": 编号, "name": f"任务{编号}", "task_stats": {"waiting": 等待}}, "submits": [{"array": {"priority": 优先级, "resources": [{"resources": 资源}]}}]}

        范围资源 = [{"resource": "gpus", "request": {"Compact": 20000}}, {"resource": "gpumem", "request": {"显存范围": [650000, 1300000]}}]
        固定资源 = [{"resource": "gpus", "request": {"Compact": 10000}}, {"resource": "gpumem", "request": {"Compact": 320000}}]
        详情 = [作业(3, 21, 范围资源), 作业(2, 21, 固定资源), 作业(1, -2, [{"resource": "cpus", "request": {"Compact": 10000}}]), 作业(4, 100, 范围资源, 等待=0)]
        调用 = []

        def 查询(*参数):
            调用.append(参数)
            if 参数[1] == "list":
                return [条目["info"] for 条目 in 详情]
            return 详情

        结果 = QUEUE_QUERY.读取等待任务(查询)
        self.assertEqual([条目["作业号"] for 条目 in 结果], [2, 3, 1])
        self.assertEqual(调用[-1], ("job", "info", "3,2,1"))
        self.assertEqual((结果[1]["每卡显存下限GiB"], 结果[1]["每卡显存上限GiB"]), (32.5, 65))
        self.assertEqual(结果[1]["妥协优先级"], 20)
        self.assertIsNone(结果[0]["妥协优先级"])
        self.assertEqual(结果[2]["GPU数"], 0)
        文本 = QUEUE_QUERY.格式化队列(结果)
        self.assertIn("32.5:65", 文本)
        self.assertIn("CPU", 文本)
        self.assertIn("共 3 个等待作业", 文本)

    def test_空队列不查询详情且错误不被当作空队列(self):
        调用 = []

        def 空查询(*参数):
            调用.append(参数)
            return []

        self.assertEqual(QUEUE_QUERY.读取等待任务(空查询), [])
        self.assertEqual(len(调用), 1)
        self.assertIn("没有等待任务", QUEUE_QUERY.格式化队列([]))

        def 失败查询(*参数):
            raise RuntimeError("服务离线")

        with self.assertRaisesRegex(RuntimeError, "服务离线"):
            QUEUE_QUERY.读取等待任务(失败查询)


class GpuResourceTests(unittest.TestCase):
    def test_旧worker边界查询异常不误停任务(self):
        环境 = dict(os.environ, SLURM_JOB_ID="1", HQ_JOB_ID="2", HQ_RESOURCE_VALUES_gpus="0", TASKCTL_NATIVE_TASK_BOUNDARY="0")
        for 错误 in (subprocess.TimeoutExpired("boundary", 120), OSError("服务离线")):
            with self.subTest(错误=错误), patch.object(GPU_WORKER.subprocess, "run", side_effect=错误) as 查询:
                GPU_WORKER.检查旧worker任务边界(环境)
                self.assertEqual(查询.call_count, 1)
        with patch.object(GPU_WORKER.subprocess, "run", return_value=subprocess.CompletedProcess([], 75, stdout="invalid", stderr="")) as 查询:
            GPU_WORKER.检查旧worker任务边界(环境)
            self.assertEqual(查询.call_count, 1)

    def test_任务包装按真实每卡显存标记妥协并拒绝不足(self):
        基础环境 = dict(os.environ, TASKCTL_HOME=str(ROOT), TASKCTL_PYTHON=sys.executable, TASKCTL_GPU_MEMORY_MIN_GB="22", TASKCTL_GPU_MEMORY_MAX_GB="44", HQ_RESOURCE_VALUES_gpus="1")
        for 显存, 标记 in ((22, "1"), (32, "1"), (44, "0"), (80, "0")):
            with self.subTest(显存=显存):
                环境 = dict(基础环境, TASKCTL_GPU_MEMORY_GB=str(显存))
                self.assertEqual(GPU_WORKER.显存运行环境(环境)["TASKCTL_GPU_MEMORY_COMPROMISED"], 标记)
                结果 = subprocess.run(["bash", str(ROOT / "任务包装.sh"), sys.executable, "-c", 'import os; print(os.environ["TASKCTL_GPU_MEMORY_COMPROMISED"], os.environ["CUDA_VISIBLE_DEVICES"])'], env=环境, capture_output=True, text=True, timeout=5)
                self.assertEqual(结果.returncode, 0, 结果.stderr)
                self.assertEqual(结果.stdout.strip(), 标记 + " 1")
        with self.assertRaisesRegex(ValueError, "低于任务下限"):
            GPU_WORKER.显存运行环境(dict(基础环境, TASKCTL_GPU_MEMORY_GB="21"))
        基础环境.pop("TASKCTL_GPU_MEMORY_GB", None)
        with self.assertRaisesRegex(ValueError, "未声明实际每卡显存"):
            GPU_WORKER.显存运行环境(基础环境)

    def test_显存范围解析与边界优先级(self):
        for 输入 in ("22:44", [22, 44], (22, 44)):
            self.assertEqual(GPU_WORKER.解析显存范围(输入), (22, 44))
        self.assertEqual(GPU_WORKER.解析显存范围("32.5:65"), (32.5, 65))
        self.assertEqual(GPU_WORKER.原生显存范围("32.5:65", 2), "65:130")
        self.assertEqual(GPU_WORKER.原生显存范围(80), "80")
        self.assertEqual(GPU_WORKER.解析显存范围(80), (80, 80))
        self.assertEqual(GPU_WORKER.解析显存范围(0), (0, 0))
        for 显存, 期望 in ((21, None), (22, 19), (32, 19), (44, 20), (80, 20)):
            self.assertEqual(GPU_WORKER.显存优先级(20, (22, 44), 显存), 期望)
        self.assertEqual(GPU_WORKER.显存优先级(-(2 ** 31), (22, 44), 32), -(2 ** 31) - 1)

    def test_拒绝无效显存范围(self):
        for 输入 in ("44:22", "0:44", "-1:44", "nan", "inf", "22:44:80", [22], [22, 44, 80], True, "1.00001:2"):
            with self.subTest(输入=输入), self.assertRaises(ValueError):
                GPU_WORKER.解析显存范围(输入)

    def test_弹性worker不再默认限制四小时(self):
        with tempfile.TemporaryDirectory() as 目录:
            根目录 = Path(目录)
            (根目录 / "bin").mkdir()
            执行器 = 根目录 / "bin" / "hq"
            执行器.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
            执行器.chmod(0o755)
            for 时限 in (None, "431880s"):
                with self.subTest(时限=时限):
                    环境 = dict(os.environ, TASKCTL_HOME=str(根目录), TASKCTL_STATE=str(根目录 / "运行状态"), CUDA_VISIBLE_DEVICES="0", SLURM_CPUS_PER_TASK="8")
                    环境.pop("TASKCTL_WORKER_TIME_LIMIT", None)
                    if 时限:
                        环境["TASKCTL_WORKER_TIME_LIMIT"] = 时限
                    结果 = subprocess.run(["bash", str(ROOT / "弹性worker.sh")], env=环境, capture_output=True, text=True, timeout=5)
                    self.assertEqual(结果.returncode, 0, 结果.stderr)
                    参数 = json.loads(结果.stdout)
                    if 时限:
                        self.assertEqual(参数[参数.index("--time-limit") + 1], 时限)
                    else:
                        self.assertNotIn("--time-limit", 参数)

    def test_第二份作业只加入worker并保留主服务状态(self):
        with tempfile.TemporaryDirectory() as 目录:
            根目录 = Path(目录)
            (根目录 / "bin").mkdir()
            状态目录 = 根目录 / "运行状态"
            状态目录.mkdir()
            for 文件名 in ("service.info", "stop.requested", "reload.requested"):
                (状态目录 / 文件名).write_text("existing-owner", encoding="utf-8")
            shutil.copyfile(ROOT / "任务守护.sh", 根目录 / "任务守护.sh")
            (根目录 / "弹性worker.sh").write_text('#!/bin/bash\nprintf "WORKER_ONLY:%s:%s\\n" "$TASKCTL_STATE" "$SLURM_CPUS_PER_TASK"\n', encoding="utf-8")
            执行器 = 根目录 / "bin" / "hq"
            执行器.write_text(f"#!{sys.executable}\nimport sys\nraise SystemExit(0 if sys.argv[1:3] == ['server', 'info'] else 7)\n", encoding="utf-8")
            执行器.chmod(0o755)
            with open(状态目录 / "server.owner.lock", "a") as 锁:
                fcntl.flock(锁, fcntl.LOCK_EX | fcntl.LOCK_NB)
                结果 = subprocess.run(["bash", str(根目录 / "任务守护.sh"), "--gpus", "0", "--cpus", "4", "--run", ":"], env=dict(os.environ, TASKCTL_STATE=str(状态目录), TASKCTL_AUTOSCALE="0"), capture_output=True, text=True, timeout=12)
            self.assertEqual(结果.returncode, 0, 结果.stdout + 结果.stderr)
            self.assertIn(f"WORKER_ONLY:{状态目录}:4", 结果.stdout)
            for 文件名 in ("service.info", "stop.requested", "reload.requested"):
                self.assertEqual((状态目录 / 文件名).read_text(), "existing-owner")

    def test_submission_requires_positive_gpu_memory(self):
        prefix = ["hq", "--output-mode=quiet", "submit"]
        command = ["--", "echo", "gpumem=80", "gpus=9"]
        for resources in (["--resource", "gpus=1"],
                          ["--resource=gpus=2"],
                          ["--resource", "gpus=1", "--resource", "gpumem=0"],
                          ["--resource=gpus=2", "--resource=gpumem=0"]):
            with self.subTest(resources=resources):
                arguments = prefix + ["--resource", "custom=2"] + resources + command
                self.assertEqual(GPU_WORKER.prepare_submission(arguments),
                                 prefix + ["--resource", "custom=2"] + command)
        for resources in (["--resource", "gpus=1", "--resource", "gpumem=32"],
                  ["--resource=gpus=2", "--resource=gpumem=80"]):
            arguments = prefix + resources + command
            self.assertEqual(GPU_WORKER.prepare_submission(arguments), arguments)

    def test_worker_declares_only_allocated_gpu_memory(self):
        output = "0, GPU-small, TeslaV100, 32768, 7.0\n1, GPU-large, A100, 81920, 8.0\n"
        command, environment = GPU_WORKER.prepare(
            ["hq", "worker", "start", "--cpus", "8", "--resource", "gpus=[1]"], {}, output)
        self.assertIn("gpumem=sum(80)", command)
        self.assertEqual(environment["TASKCTL_GPU_COMPUTE_CAP"], "8.0")
        self.assertEqual(json.loads(environment["TASKCTL_TASK_BOUNDARY_COMMAND"])[-1], "boundary")
        command, environment = GPU_WORKER.prepare(
            ["hq", "worker", "start", "--resource", "gpus=[0]"], {}, output)
        self.assertIn("gpumem=sum(32)", command)
        self.assertEqual(environment["TASKCTL_GPU_COMPUTE_CAP"], "7.0")

    def test_unknown_gpu_mapping_fails_closed(self):
        with self.assertRaises(ValueError):
            GPU_WORKER.prepare(["hq", "worker", "start", "--resource", "gpus=[7]"], {},
                               "0, one, V100, 32768, 7.0\n1, two, A100, 81920, 8.0\n")

    def test_clients_require_positive_gpu_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            server = root / "运行状态" / "server" / "001"
            server.mkdir(parents=True)
            (server / "access.json").write_text("{}")
            shutil.copyfile(ROOT / "任务控制.sh", root / "任务控制.sh")
            shutil.copyfile(ROOT / "GPUworker.py", root / "GPUworker.py")
            recorded = root / "args.json"
            executable = root / "bin" / "hq"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['RECORDED_ARGS'], 'w') as stream:\n"
                "    json.dump(sys.argv[1:], stream)\n"
                "print(17)\n")
            executable.chmod(0o755)
            for mode in ("add", "add-json"):
                for memory in (None, 0, 80, "22:44", [32.5, 65]):
                    with self.subTest(mode=mode, memory=memory):
                        if mode == "add":
                            command = ["bash", str(root / "任务控制.sh"), "add",
                                       "--name", "test", "--gpus", "1", "--cpus", "6"]
                            if memory is not None:
                                command += ["--gpu-memory", ":".join(map(str, memory)) if isinstance(memory, list) else str(memory)]
                            command += ["--", "echo", "test"]
                        else:
                            task = {"name": "test", "gpus": 1, "cpus": 6,
                                    "cmd": ["echo", "test"]}
                            if memory is not None:
                                task["gpu_memory"] = memory
                            path = root / "tasks.json"
                            path.write_text(json.dumps({"tasks": [task]}))
                            command = ["bash", str(root / "任务控制.sh"), "add-json", str(path)]
                        result = subprocess.run(
                            command, env=dict(os.environ, RECORDED_ARGS=str(recorded)),
                            capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        arguments = json.loads(recorded.read_text())
                        self.assertEqual("gpus=1" in arguments, bool(memory))
                        if memory:
                            self.assertIn("gpumem=" + GPU_WORKER.原生显存范围(memory), arguments)
                        self.assertEqual(str(root / "任务包装.sh") in arguments, bool(memory))


    def test_wait_checks_task_state_even_when_native_wait_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            server = root / "运行状态" / "server" / "001"
            server.mkdir(parents=True)
            (server / "access.json").write_text("{}")
            shutil.copyfile(ROOT / "任务控制.sh", root / "任务控制.sh")
            executable = root / "bin" / "hq"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if 'info' in sys.argv:\n"
                "    print(json.dumps([{'info': {'id': 17, 'task_count': 1,\n"
                "        'task_stats': json.loads(os.environ['NATIVE_STATS'])}}]))\n")
            executable.chmod(0o755)
            for state in ("finished", "failed", "canceled", "aborted", "waiting", "running", "unknown"):
                with self.subTest(state=state):
                    result = subprocess.run(
                        ["bash", str(root / "任务控制.sh"), "wait", "17"],
                        env=dict(os.environ, NATIVE_STATS=json.dumps({state: 1})),
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode == 0, state == "finished", result.stderr)
                    if state != "finished":
                        self.assertIn("未成功完成", result.stderr)


class PlanningTests(unittest.TestCase):
    def test_边界检查仅标记可释放worker不取消Slurm且保护主服务(self):
        with tempfile.TemporaryDirectory() as 目录:
            控制器 = AUTOSCALE.Controller.__new__(AUTOSCALE.Controller)
            控制器.state = Path(目录)
            控制器.name = "hq-elastic-test"
            控制器.启动配置, 控制器.env = {}, {}
            请求 = [AUTOSCALE.Request(9, 6, 1, 22, priority=1199, 显存上限=44)]
            当前 = {"id": "small", "name": 控制器.name, "account": "account", "partition": "gpu", "state": "RUNNING", "gpu_memory": 40, "gpu": 1, "cpu": 8, "mem": 32768, "分区额度键": "gpu"}
            大卡 = dict(当前, id="large", state="PENDING", gpu_memory=80)
            详情 = [{"info": {"id": 8}, "tasks": [{"worker": 2, "state": "running"}]}]
            控制器.demand = lambda: (请求, [{"id": 2, "allocation": {"id": "small"}}], 详情)
            控制器.snapshot = lambda: ([当前, 大卡], self.offers, {"gpu": {"jobs": 0, "gpu": 0, "cpu": 0}, "public": {}}, "")
            控制器.run = lambda *参数: self.fail("边界检查不应自行取消Slurm或停止worker")
            self.assertEqual(控制器.任务边界检查(2, 8)["等待大卡作业"], "large")
            self.assertTrue((控制器.state / "任务边界" / "small.json").exists())
            详情.append({"info": {"id": 10}, "tasks": [{"worker": 2, "state": "running"}]})
            self.assertIsNone(控制器.任务边界检查(2, 8))
            详情.pop()
            当前["name"] = "pepopt-owner"
            self.assertIsNone(控制器.任务边界检查(2, 8))

    def test_小卡仅在同分区名额阻塞大卡时任务边界让位(self):
        请求 = [AUTOSCALE.Request(9, 6, 1, 22, priority=1199, 显存上限=44), AUTOSCALE.Request(1, 6, 1, 5, priority=0, 显存上限=10)]
        当前 = {"id": "small", "account": "account", "partition": "gpu", "state": "RUNNING", "gpu_memory": 40, "gpu": 1, "cpu": 8, "mem": 32768, "分区额度键": "gpu"}
        大卡 = dict(当前, id="large", state="PENDING", gpu_memory=80)
        额度 = {"gpu": {"jobs": 0, "gpu": 0, "cpu": 0, "submit": 0}, "public": {}}
        决策 = AUTOSCALE.小卡任务边界让位(请求, 当前, [大卡], 额度, self.offers)
        self.assertEqual((决策["释放作业"], 决策["等待大卡作业"], 决策["最高优先级任务"]), ("small", "large", 9))
        for 当前值, 大卡列表, 请求值, 额度值 in ((当前, [], 请求, 额度), (当前, [dict(大卡, partition="public")], 请求, 额度), (当前, [dict(大卡, account="other")], 请求, 额度), (当前, [dict(大卡, state="RUNNING")], 请求, 额度), (当前, [大卡], [], 额度), (当前, [大卡], 请求[1:], 额度), (dict(当前, gpu_memory=80), [大卡], 请求, 额度), (当前, [大卡], 请求, {"gpu": {"jobs": 1}}), (当前, [dict(大卡, cpu=16)], 请求, 额度)):
            with self.subTest(当前=当前值, 大卡=大卡列表, 请求=请求值, 额度=额度值):
                self.assertIsNone(AUTOSCALE.小卡任务边界让位(请求值, 当前值, 大卡列表, 额度值, self.offers))

    def test_名额已满时只原地升级自己的等待小卡不动运行或外部作业(self):
        with tempfile.TemporaryDirectory() as 目录:
            控制器 = AUTOSCALE.Controller.__new__(AUTOSCALE.Controller)
            控制器.state = Path(目录)
            控制器.status_file = 控制器.state / "autoscale.json"
            控制器.name, 控制器.account = "hq-elastic-test", "test-account"
            控制器.env = {"TASKCTL_AUTO_JOB_NAMES": "pepopt-loop"}
            控制器.cpus, 控制器.memory_mb = 8, 32768
            控制器.demand = lambda: ([AUTOSCALE.Request(9, 6, 1, 22, priority=1199, 显存上限=44)], [{"allocation": {"id": "1"}}], [])
            运行 = {"id": "1", "name": "pepopt-loop", "account": 控制器.account, "partition": "gpu", "gpu_memory": 80, "state": "RUNNING", "gpu": 1, "cpu": 8, "mem": 32768, "通用显卡": True}
            小卡 = dict(运行, id="2", name=控制器.name, gpu_memory=40, state="PENDING")
            外部作业 = dict(小卡, id="3", name="cky-other")
            作业列表 = [运行, 小卡, 外部作业]
            控制器.snapshot = lambda: (作业列表, self.offers, {"gpu": {"submit": 0}, "public": {"submit": 1}}, "")
            控制器.upgrade_idle_workers = lambda *参数: False
            命令 = []
            控制器.run = lambda 参数: 命令.append(参数)
            结果 = 控制器.tick(apply=True)
            self.assertEqual(命令, [["scontrol", "update", "JobId=2", "ExcNodeList=medium"]])
            self.assertEqual(结果["upgraded_pending"][0]["gpu_memory"], 80)
            self.assertEqual(结果["plan"], [])
            self.assertEqual(结果["cancelled_pending"], [])
            控制器.snapshot = lambda: (作业列表, self.offers, {"gpu": {"submit": 0}, "public": {"submit": 0}}, "")
            作业列表[1] = dict(小卡, gpu_memory=80)
            命令.clear()
            self.assertEqual(控制器.tick(apply=True)["upgraded_pending"], [])
            self.assertEqual(命令, [])
            作业列表[1] = dict(小卡, state="RUNNING")
            self.assertEqual(控制器.tick(apply=True)["upgraded_pending"], [])
            self.assertEqual(命令, [])

    def test_运行中大卡不阻止为剩余等待任务申请同档大卡(self):
        控制器 = AUTOSCALE.Controller.__new__(AUTOSCALE.Controller)
        控制器.name, 控制器.account = "hq-elastic-test", "test-account"
        控制器.env = {"TASKCTL_AUTO_JOB_NAMES": "pepopt-loop"}
        控制器.cpus, 控制器.memory_mb = 8, 32768
        控制器.demand = lambda: ([AUTOSCALE.Request(3, 6, 1, 22, priority=1199, 显存上限=44)], [{"allocation": {"id": "running-large"}}], [])
        运行作业 = {"id": "running-large", "name": "pepopt-loop", "account": 控制器.account, "partition": "gpu", "gpu_memory": 80, "state": "RUNNING"}
        作业列表 = [运行作业]
        控制器.snapshot = lambda: (作业列表, self.offers, {"gpu": {"submit": 1, "jobs": 0}, "public": {"submit": 0}}, "")
        结果 = 控制器.tick()
        self.assertEqual([(项["partition"], 项["gpu_memory"]) for 项 in 结果["plan"]], [("gpu", 80)])
        作业列表.append(dict(运行作业, id="pending-large", name=控制器.name, state="PENDING"))
        结果 = 控制器.tick()
        self.assertEqual([(项["partition"], 项["gpu_memory"]) for 项 in 结果["plan"]], [("gpu", 40)])
        作业列表.pop()
        控制器.demand = lambda: ([], [{"allocation": {"id": "running-large"}}], [])
        self.assertEqual(控制器.tick()["plan"], [])

    def test_通用GPU排队档位信息可写入JSON状态(self):
        控制器 = AUTOSCALE.Controller.__new__(AUTOSCALE.Controller)
        控制器.username = "test-user"

        def 命令(参数):
            if 参数[0] == "squeue":
                return "100|account|gpu|qos|PENDING|8|gpu:1|1|hq-bootstrap-test|2026-11-11T00:00:00|Priority|32G"
            if 参数[-1] == "100":
                return "JobId=100 NodeList=(null) ExcNodeList=large"
            return "large"

        控制器.run = 命令
        作业 = 控制器.slurm_jobs()
        self.assertEqual(作业[0]["排除节点"], ["large"])
        self.assertIn("large", json.dumps(作业))

    def test_已有大卡等待不会因运行名额或等待时长而取消(self):
        with tempfile.TemporaryDirectory() as 目录:
            控制器 = AUTOSCALE.Controller.__new__(AUTOSCALE.Controller)
            控制器.state = Path(目录)
            控制器.name, 控制器.account = "hq-elastic-test", "test-account"
            控制器.env = {}
            控制器.cpus, 控制器.memory_mb = 8, 32768
            控制器.status_file = 控制器.state / "autoscale.json"
            控制器.demand = lambda: ([AUTOSCALE.Request(编号, 6, 1, 22, 显存上限=44) for 编号 in range(8)], [], [])
            旧作业 = {"id": "1", "name": 控制器.name, "account": 控制器.account, "partition": "gpu", "gpu_memory": 80, "state": "PENDING", "submit": "2000-01-01T00:00:00"}
            控制器.snapshot = lambda: ([旧作业], self.offers, {"gpu": {"submit": 3, "jobs": 0}, "public": {"submit": 2, "jobs": 0}}, "")
            控制器.upgrade_idle_workers = lambda *参数: False
            命令 = []
            控制器.run = lambda 参数: 命令.append(参数)
            控制器.submit = lambda 分配, qos: 命令.append((分配["offer"].partition, 分配["offer"].gpu_memory))
            结果 = 控制器.tick(apply=True)
            self.assertEqual(命令, [("gpu", 40), ("public", 32)])
            self.assertEqual(结果["cancelled_pending"], [])

    def test_无HQ启动读取显式需求且不重复保留的现有档位(self):
        with tempfile.TemporaryDirectory() as 目录:
            状态 = Path(目录)
            配置 = {"作业脚本": str(ROOT / "任务守护.sh"), "请求": [{"job_id": 编号, "cpus": 6, "gpus": 1, "gpu_memory": 22, "显存上限": 44} for 编号 in range(8)], "保留作业": ["old-large", "old-small"], "环境": {"TASKCTL_AUTO_CPUS": "8", "TASKCTL_AUTO_MEMORY": "32G"}}
            (状态 / "逐档启动.json").write_text(json.dumps(配置), encoding="utf-8")
            控制器 = AUTOSCALE.Controller(state=状态, 启动模式=True)
            作业 = [{"id": "old-large", "name": "pepopt-loop", "account": 控制器.account, "partition": "gpu", "gpu_memory": 80}, {"id": "old-small", "name": "pepopt-small", "account": 控制器.account, "partition": "public", "gpu_memory": 32}]
            控制器.snapshot = lambda: (作业, self.offers, {"gpu": {"submit": 3}, "public": {"submit": 1}}, "")
            结果 = 控制器.tick()
            self.assertEqual([(项["partition"], 项["gpu_memory"]) for 项 in 结果["plan"]], [("gpu", 40)])
            self.assertEqual(控制器.memory_mb, 32768)

    def test_逐档申请从刚好高于最大上限的档位向下且每档一次(self):
        节点 = [AUTOSCALE.Offer(str(显存), "gpu", f"GPU{显存}GB", 显存, 0, 0, 0, ("gpu",)) for 显存 in (16, 24, 32, 40, 80, 128)]
        请求 = [AUTOSCALE.Request(编号, 6, 1, 22, 显存上限=44) for 编号 in range(8)]
        计划 = AUTOSCALE.逐档申请计划(请求, 节点, {"gpu": {"submit": 10, "jobs": 0, "gpu": 0, "cpu": 0}})
        self.assertEqual([项["offer"].gpu_memory for 项 in 计划], [80, 40, 32, 24])
        self.assertEqual(AUTOSCALE.逐档申请计划([], 节点, {}), [])
        self.assertEqual(AUTOSCALE.逐档申请计划(请求, 节点, {"gpu": {"submit": 0}}), [])

    def test_逐档申请在名额任务或已有档位处停止(self):
        请求 = [AUTOSCALE.Request(编号, 6, 1, 22, 显存上限=40) for 编号 in range(8)]
        配额 = {"gpu": {"submit": 2}, "public": {"submit": 2}}
        计划 = AUTOSCALE.逐档申请计划(请求, self.offers, 配额, 已有档位=(("gpu", 80),))
        self.assertEqual([(项["offer"].partition, 项["offer"].gpu_memory) for 项 in 计划], [("gpu", 40), ("public", 32)])
        self.assertEqual(配额["gpu"]["submit"], 2)
        self.assertEqual(len(AUTOSCALE.逐档申请计划(请求[:1], self.offers, 配额)), 1)
        self.assertEqual(len(AUTOSCALE.逐档申请计划(请求, self.offers, {"gpu": {"submit": 1}, "public": {"submit": 0}})), 1)

    def test_同一显存档位不重复申请不同型号(self):
        节点 = list(self.offers) + [AUTOSCALE.Offer("small-other", "public", "TeslaV100S-32GB", 32, 0, 0, 0, ("public",))]
        计划 = AUTOSCALE.逐档申请计划([AUTOSCALE.Request(编号, 6, 1, 22, 显存上限=44) for 编号 in range(8)], 节点, {"gpu": {"submit": 4}, "public": {"submit": 2}})
        self.assertEqual([(项["offer"].partition, 项["offer"].gpu_memory) for 项 in 计划], [("gpu", 80), ("gpu", 40), ("public", 32)])
        self.assertTrue(all(项["通用显卡"] for 项 in 计划))
        self.assertEqual(计划[-1]["排除节点"], [])
        self.assertEqual(计划[0]["排除节点"], ["medium"])

    def test_扩容和已有容量均按节点上的有效优先级排序(self):
        请求 = [AUTOSCALE.Request(1, 6, 1, 22, priority=20, 显存上限=44), AUTOSCALE.Request(2, 6, 1, 5, priority=20, 显存上限=10)]
        配额 = {"gpu": {"jobs": 0}, "public": {"gpu": 1, "cpu": 8, "jobs": 1}}
        分配 = AUTOSCALE.plan(请求, self.offers, 配额)
        self.assertEqual([条目["job_id"] for 条目 in 分配], [2])
        剩余 = AUTOSCALE.consume_capacity(请求, [{"gpus": 1, "cpus": 8, "gpumem": 32}])
        self.assertEqual([条目.job_id for 条目 in 剩余], [1])
        请求.pop()
        分配 = AUTOSCALE.plan(请求, self.offers, 配额)
        self.assertEqual(分配[0]["priority"], 19)
        self.assertTrue(分配[0]["显存妥协"])

    def test_扩容申请每个分区的最大时限(self):
        with tempfile.TemporaryDirectory() as 目录:
            控制器 = AUTOSCALE.Controller.__new__(AUTOSCALE.Controller)
            控制器.state = Path(目录)
            控制器.name = "test-elastic"
            控制器.account = "test-account"
            控制器.env = {"TASKCTL_AUTO_HOURS": "4", "TASKCTL_WORKER_TIME_LIMIT": "14280s"}
            提交记录 = []

            def 录制提交(参数, 环境):
                提交记录.append((参数, 环境.copy()))
                return "4312412"

            控制器.run = 录制提交
            for 时限, 请求, 工作者时限 in (("5-00:00:00", "5-00:00:00", "431880s"), ("12:00:00", "12:00:00", "43080s"), ("UNLIMITED", "0", None)):
                with self.subTest(时限=时限):
                    候选 = AUTOSCALE.Offer("node", "partition", "A10080GB", 80, 8, 1, 32768, (), 最长时限=时限)
                    分配 = {"offer": 候选, "cpus": 8, "gpus": 1.0, "memory_mb": 32768.0, "job_id": 17, "priority": 20}
                    self.assertEqual(控制器.submit(分配, ""), "4312412")
                    参数, 环境 = 提交记录[-1]
                    self.assertIn(f"--time={请求}", 参数)
                    self.assertIn("--mem=32768M", 参数)
                    self.assertTrue(any(值.endswith(":1") for 值 in 参数 if 值.startswith("--gres=")))
                    self.assertFalse(any(值.startswith("--nodelist") for 值 in 参数))
                    self.assertEqual(环境.get("TASKCTL_WORKER_TIME_LIMIT"), 工作者时限)

    def test_排队作业仅占提交槽不占运行资源(self):
        作业 = {"state": "PENDING", "cpu": 8, "gpu": 1, "mem": 32768, "node": 1}
        self.assertEqual({资源: AUTOSCALE.已占用额度(作业, 资源) for 资源 in ("cpu", "gpu", "mem", "node", "jobs", "submit")}, {"cpu": 0, "gpu": 0, "mem": 0, "node": 0, "jobs": 0, "submit": 1})
        作业["state"] = "RUNNING"
        self.assertEqual({资源: AUTOSCALE.已占用额度(作业, 资源) for 资源 in ("cpu", "gpu", "mem", "node", "jobs", "submit")}, {"cpu": 8, "gpu": 1, "mem": 32768, "node": 1, "jobs": 1, "submit": 1})

    def test_主机内存扣除系统预留(self):
        self.assertEqual(AUTOSCALE.可申请主机内存({"mem": 515470}, {"mem": 471040}, 10240), 34190)
        self.assertEqual(AUTOSCALE.可申请主机内存({"mem": 515470}, {"mem": 510000}, "10240"), 0)

    def setUp(self):
        self.offers = [
            AUTOSCALE.Offer("small", "public", "TeslaV100-32GB", 32, 16, 2, 200000, ("public",)),
            AUTOSCALE.Offer("medium", "gpu", "NVIDIAA100-PCIE-40GB", 40, 16, 2, 200000, ("gpu",)),
            AUTOSCALE.Offer("large", "gpu", "NVIDIAA10080GBPCIe", 80, 16, 2, 200000, ("gpu",)),
        ]

    def test_prefers_largest_and_shares_partition_quota(self):
        requests = [AUTOSCALE.Request(1, 6, 1, 32)] * 4
        budgets = {"gpu": {"gpu": 1, "cpu": 8, "jobs": 1},
                   "public": {"gpu": 1, "cpu": 8, "jobs": 1}}
        result = AUTOSCALE.plan(requests, self.offers, budgets)
        self.assertEqual([item["offer"].node for item in result], ["large", "small"])
        self.assertEqual(budgets["gpu"]["gpu"], 1)

    def test_other_job_exhausting_gpu_partition_does_not_block_public(self):
        budgets = {"gpu": {"gpu": 0, "cpu": 0, "jobs": 0},
                   "public": {"gpu": 1, "cpu": 8, "jobs": 1}}
        result = AUTOSCALE.plan([AUTOSCALE.Request(1, 6, 1, 32)], self.offers, budgets)
        self.assertEqual(result[0]["offer"].partition, "public")
        self.assertEqual(AUTOSCALE.plan([AUTOSCALE.Request(2, 6, 1, 80)],
                                       self.offers, budgets), [])

    def test_no_demand_or_quota_does_not_allocate(self):
        self.assertEqual(AUTOSCALE.plan([], self.offers, {}), [])
        self.assertEqual(AUTOSCALE.plan([AUTOSCALE.Request(1, 6, 1, 80)],
            self.offers, {"gpu": {"submit": 0}, "public": {"jobs": 0}}), [])
        self.assertEqual(AUTOSCALE.plan([AUTOSCALE.Request(1, 6, 1, 0)],
                                       self.offers, {"gpu": {}, "public": {}}), [])

    def test_cpu_and_host_memory_are_required(self):
        budgets = {"gpu": {"cpu": 4}, "public": {"jobs": 0}}
        self.assertEqual(AUTOSCALE.plan([AUTOSCALE.Request(1, 6, 1, 32)],
                                       self.offers, budgets), [])
        self.assertEqual(AUTOSCALE.plan([AUTOSCALE.Request(1, 6, 1, 80)],
            [AUTOSCALE.Offer("lowram", "gpu", "A10080GB", 80, 8, 1, 1024, ("gpu",))],
            {"gpu": {}}), [])

    def test_priority_precedes_memory_size_and_fifo_breaks_ties(self):
        budgets = {"gpu": {"gpu": 1, "cpu": 8, "jobs": 1}, "public": {"jobs": 0}}
        requests = [AUTOSCALE.Request(1, 6, 1, 80, priority=0),
                    AUTOSCALE.Request(3, 6, 1, 13, priority=25)]
        selected = AUTOSCALE.plan(requests, self.offers, budgets)
        self.assertEqual(selected[0]["job_id"], 3)
        self.assertEqual(selected[0]["priority"], 25)
        requests = [AUTOSCALE.Request(3, 6, 1, 80, priority=25),
                    AUTOSCALE.Request(1, 6, 1, 13, priority=25)]
        self.assertEqual(AUTOSCALE.plan(requests, self.offers, budgets)[0]["job_id"], 1)

    def test_small_node_backfills_when_high_priority_task_cannot_fit(self):
        requests = [AUTOSCALE.Request(1, 6, 1, 65, priority=100),
                    AUTOSCALE.Request(2, 6, 1, 13, priority=0)]
        budgets = {"gpu": {"gpu": 0, "jobs": 0},
                   "public": {"gpu": 1, "cpu": 8, "jobs": 1}}
        selected = AUTOSCALE.plan(requests, self.offers, budgets)
        self.assertEqual([(item["job_id"], item["offer"].node) for item in selected],
                         [(2, "small")])
        budgets = {"gpu": {"gpu": 1, "cpu": 8, "jobs": 1},
                   "public": {"gpu": 1, "cpu": 8, "jobs": 1}}
        selected = AUTOSCALE.plan(requests, self.offers, budgets)
        self.assertEqual([(item["job_id"], item["offer"].node) for item in selected],
                         [(1, "large"), (2, "small")])

    def test_existing_capacity_is_reserved_in_priority_order(self):
        requests = [AUTOSCALE.Request(1, 6, 1, 65, priority=0),
                    AUTOSCALE.Request(2, 6, 1, 13, priority=30)]
        remaining = AUTOSCALE.consume_capacity(
            requests, [{"gpus": 1, "cpus": 8, "gpumem": 80}])
        self.assertEqual([request.job_id for request in remaining], [1])

    def test_autoscale_accepts_workers_without_allocation(self):
        controller = AUTOSCALE.Controller.__new__(AUTOSCALE.Controller)
        controller.name = "test-elastic"
        controller.max_workers, controller.cpus, controller.memory_mb = 4, 8, 65536
        controller.demand = lambda: ([], [{"allocation": None}], [])
        controller.snapshot = lambda: ([], [], {}, "")
        self.assertEqual(controller.tick()["plan"], [])

    def test_detects_gpu_sizes_and_requires_explicit_memory(self):
        self.assertEqual(AUTOSCALE.gpu_memory("NVIDIAA10080GBPCIe"), 80)
        self.assertEqual(AUTOSCALE.gpu_memory("TeslaV100S-PCIE-32GB"), 32)
        detail = {"info": {"id": 17}, "tasks": [{"state": "waiting"}],
                  "submits": [{"array": {"resources": [{"n_nodes": 0,
                      "resources": [{"resource": "gpus", "request": {"Compact": 10000}},
                                    {"resource": "cpus", "request": {"Compact": 60000}}]}]}}]}
        self.assertEqual(AUTOSCALE.job_requests(detail), [])
        resources = detail["submits"][0]["array"]["resources"][0]["resources"]
        resources.append({"resource": "gpumem", "request": {"Compact": 0}})
        self.assertEqual(AUTOSCALE.job_requests(detail), [])
        resources[-1]["request"]["Compact"] = 320000
        self.assertEqual(AUTOSCALE.job_requests(detail), [AUTOSCALE.Request(17, 6, 1, 32)])
        detail["submits"][0]["array"]["priority"] = 27
        self.assertEqual(AUTOSCALE.job_requests(detail),
                 [AUTOSCALE.Request(17, 6, 1, 32, priority=27)])
        resources[-1]["request"] = {"显存范围": [325000, 650000]}
        self.assertEqual(AUTOSCALE.job_requests(detail), [AUTOSCALE.Request(17, 6, 1, 32.5, priority=27, 显存上限=65)])


if __name__ == "__main__":
    unittest.main()