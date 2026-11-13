"""Opt-in native scheduling checks; only no-op tasks and virtual GPU resources."""

import json
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(os.environ.get("HQ_TEST_BINARY"), "HQ_TEST_BINARY is required")
class NativeFifoTests(unittest.TestCase):
    def test_AF3通过真实HQ完成事件等待且正常关闭订阅(self):
        二进制 = str(Path(os.environ["HQ_TEST_BINARY"]).resolve())
        项目 = Path(__file__).resolve().parents[2] / "Optimize-peptide-sequences"
        with tempfile.TemporaryDirectory(prefix="hq-event-check-") as 目录:
            环境 = dict(os.environ, TASKCTL_STATE=目录, HQ_SERVER_DIR=str(Path(目录) / "server"), LD_LIBRARY_PATH="/public_bme2/Share200T/管吉松/envs/af3/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
            服务 = subprocess.Popen([二进制, "server", "start", "--host=127.0.0.1"], env=环境, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            工作者 = None

            def 查询(*参数):
                结果 = subprocess.run([二进制, *参数], env=环境, capture_output=True, text=True, timeout=20)
                self.assertEqual(结果.returncode, 0, 结果.stdout + 结果.stderr)
                return 结果.stdout

            try:
                for 行 in 服务.stdout:
                    if "Journal path" in 行:
                        break
                else:
                    self.fail("隔离事件测试服务未启动")
                编号 = int(查询("--output-mode=quiet", "submit", "--cpus=1", "--stdout=" + 目录 + "/task.out", "--stderr=" + 目录 + "/task.err", "--", "/usr/bin/true"))
                规范 = importlib.util.spec_from_file_location("af3_event_integration", 项目 / "af3适配" / "af3任务.py")
                适配器 = importlib.util.module_from_spec(规范)
                with patch.dict(os.environ, 环境):
                    规范.loader.exec_module(适配器)
                    适配器.HQ程序 = 二进制
                    适配器.queue_jobs = lambda: json.loads(查询("--output-mode=json", "job", "list", "--all"))
                    状态记录 = []

                    def 回调(作业):
                        nonlocal 工作者
                        状态记录.append(作业[0]["task_stats"].copy())
                        if 工作者 is None:
                            工作者 = subprocess.Popen([二进制, "worker", "start", "--cpus=1", "--detect-resources=none", "--idle-timeout=5s", "--time-limit=15s"], env=环境, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    with patch.object(适配器, "time", Mock(wraps=适配器.time)) as 时钟:
                        适配器.wait([编号], on_poll=回调)
                        时钟.sleep.assert_not_called()
                    self.assertGreaterEqual(len(状态记录), 2)
                    self.assertEqual(状态记录[-1].get("finished"), 1)
            finally:
                subprocess.run([二进制, "server", "stop"], env=环境, capture_output=True, timeout=10)
                for 进程 in (工作者, 服务):
                    if 进程 is not None:
                        try:
                            进程.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            进程.kill()
                            进程.wait(timeout=10)
                服务.stdout.close()

    def test_范围CLI_JSON与实际包装器端到端联动(self):
        原目录 = Path(__file__).resolve().parents[1]
        二进制 = str(Path(os.environ["HQ_TEST_BINARY"]).resolve())
        with tempfile.TemporaryDirectory(prefix="hq-range-cli-") as 目录:
            根目录 = Path(目录)
            (根目录 / "bin").mkdir()
            (根目录 / "bin" / "hq").symlink_to(二进制)
            for 文件名 in ("任务控制.sh", "任务包装.sh", "GPUworker.py"):
                shutil.copyfile(原目录 / 文件名, 根目录 / 文件名)
            环境 = dict(os.environ, TASKCTL_STATE=str(根目录 / "状态"), HQ_SERVER_DIR=str(根目录 / "状态" / "server"), TASKCTL_HOME=str(根目录), TASKCTL_PYTHON=sys.executable, TASKCTL_GPU_MEMORY_GB="32", LD_LIBRARY_PATH="/public_bme2/Share200T/管吉松/envs/af3/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
            服务 = subprocess.Popen([二进制, "server", "start", "--host=127.0.0.1"], env=环境, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            工作者 = None
            try:
                for 行 in 服务.stdout:
                    if "Journal path" in 行:
                        break
                else:
                    self.fail("隔离 HQ 服务未启动")
                任务命令 = [sys.executable, "-c", 'import os; print(os.environ["TASKCTL_GPU_MEMORY_COMPROMISED"], os.environ["CUDA_VISIBLE_DEVICES"])']
                CLI = subprocess.run(["bash", str(根目录 / "任务控制.sh"), "add", "--gpus", "1", "--gpu-memory", "22:44", "--", *任务命令], env=环境, capture_output=True, text=True, timeout=15)
                self.assertEqual(CLI.returncode, 0, CLI.stdout + CLI.stderr)
                配置 = 根目录 / "任务.json"
                配置.write_text(json.dumps({"tasks": [{"gpus": 1, "gpu_memory": [22, 44], "cmd": 任务命令}]}), encoding="utf-8")
                JSON = subprocess.run(["bash", str(根目录 / "任务控制.sh"), "add-json", str(配置)], env=环境, capture_output=True, text=True, timeout=15)
                self.assertEqual(JSON.returncode, 0, JSON.stdout + JSON.stderr)
                任务编号 = re.findall(r"ID=(\d+)", CLI.stdout + JSON.stdout)
                self.assertEqual(len(任务编号), 2)
                工作者 = subprocess.Popen([二进制, "worker", "start", "--cpus=1", "--resource=gpus=[0]", "--resource=gpumem=sum(32)", "--detect-resources=none", "--idle-timeout=30s", "--time-limit=1m"], env=环境, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                等待 = subprocess.run(["bash", str(根目录 / "任务控制.sh"), "wait", *任务编号], env=环境, capture_output=True, text=True, timeout=30)
                self.assertEqual(等待.returncode, 0, 等待.stdout + 等待.stderr)
                for 编号 in 任务编号:
                    self.assertEqual((根目录 / "状态" / "logs" / 编号 / "0.stdout").read_text().strip(), "1 0")
            finally:
                subprocess.run([二进制, "server", "stop"], env=环境, capture_output=True, timeout=10)
                for 进程 in (工作者, 服务):
                    if 进程 is not None:
                        try:
                            进程.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            进程.kill()
                            进程.wait(timeout=10)
                服务.stdout.close()

    def run_queue(self, tasks, memory, 显卡数=1, 边界返回码=None, 旧worker保护=False):
        binary = str(Path(os.environ["HQ_TEST_BINARY"]).resolve())
        with tempfile.TemporaryDirectory(prefix="hq-fifo-check-") as directory:
            environment = dict(os.environ, HQ_SERVER_DIR=str(Path(directory) / "server"))
            environment["LD_LIBRARY_PATH"] = (
                "/public_bme2/Share200T/管吉松/envs/af3/lib:"
                + environment.get("LD_LIBRARY_PATH", ""))
            server = subprocess.Popen(
                [binary, "server", "start", "--host=127.0.0.1"], env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            worker = None

            def command(*arguments):
                result = subprocess.run([binary, *arguments], env=environment,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            try:
                for line in server.stdout:
                    if "Journal path" in line:
                        break
                else:
                    self.fail("Native test server did not start")
                submitted = []
                for name, priority, requested in tasks:
                    执行命令 = [sys.executable, str(Path(__file__).resolve().parents[1] / "GPUworker.py"), "--task", "/usr/bin/true"] if 旧worker保护 else ["/usr/bin/true"]
                    job_id = int(command(
                        "--output-mode=quiet", "submit", "--name=" + name,
                        "--priority=" + str(priority), "--cpus=1", "--resource=gpus=1",
                        "--resource=gpumem=" + str(requested),
                        "--stdout=" + directory + "/%{JOB_ID}.out",
                        "--stderr=" + directory + "/%{JOB_ID}.err", "--", *执行命令))
                    submitted.append((job_id, requested))
                if 边界返回码 is not None:
                    environment["TASKCTL_TASK_BOUNDARY_COMMAND"] = json.dumps([sys.executable, "-c", f"raise SystemExit({边界返回码})"])
                if 旧worker保护:
                    environment.update(SLURM_JOB_ID="test-boundary", TASKCTL_NATIVE_TASK_BOUNDARY="0", TASKCTL_GPU_MEMORY_GB=str(memory))
                    environment["TASKCTL_TASK_BOUNDARY_COMMAND"] = json.dumps([sys.executable, "-c", 'import json,sys; 作业号=int(sys.argv[sys.argv.index("--已结束作业")+1]); print(json.dumps({"工作者":1})); raise SystemExit(75 if 作业号>1 else 0)'])
                worker = subprocess.Popen(
                    [binary, "worker", "start", "--cpus=1", "--resource=gpus=[" + ",".join(map(str, range(显卡数))) + "]",
                     "--resource=gpumem=sum(" + str(memory * 显卡数) + ")", "--detect-resources=none",
                     "--idle-timeout=30s", "--time-limit=1m"], env=environment,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                runnable = [str(job_id) for job_id, requested in submitted if float(str(requested).split(":")[0]) <= memory]
                if 边界返回码 == 75 or 旧worker保护:
                    runnable = runnable[:1]
                command("job", "wait", ",".join(runnable))
                if 边界返回码 == 75 or 旧worker保护:
                    self.assertEqual(worker.wait(timeout=10), 0)
                details = json.loads(command("--output-mode=json", "job", "info",
                                             ",".join(str(job_id) for job_id, _ in submitted)))
                finished = [detail for detail in details
                            if detail["info"]["task_stats"].get("finished")]
                finished.sort(key=lambda detail: detail["tasks"][0]["started_at"])
                waiting = [detail["info"]["name"] for detail in details
                           if detail["info"]["task_stats"].get("waiting")]
                return [detail["info"]["name"] for detail in finished], waiting
            finally:
                subprocess.run([binary, "server", "stop"], env=environment,
                               capture_output=True, timeout=10)
                for process in (worker, server):
                    if process is not None:
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                server.stdout.close()

    def test_priority_then_fifo_across_resource_sizes(self):
        order, waiting = self.run_queue([
            ("low-first", 0, 13), ("equal-first-small", 10, 13),
            ("equal-later-large", 10, 65), ("high-later", 40, 13),
            ("oversized-highest", 100, 96),
        ], 80)
        self.assertEqual(order, ["high-later", "equal-first-small", "equal-later-large", "low-first"])
        self.assertEqual(waiting, ["oversized-highest"])

    def test_任务边界退出先记成功且不执行下一任务(self):
        顺序, 等待 = self.run_queue([("first-finished", 10, "22:44"), ("second-waiting", 10, "22:44")], 40, 边界返回码=75)
        self.assertEqual(顺序, ["first-finished"])
        self.assertEqual(等待, ["second-waiting"])

    def test_任务边界保留或检查失败均继续运行(self):
        for 返回码 in (0, 2):
            with self.subTest(返回码=返回码):
                顺序, 等待 = self.run_queue([("first", 10, "22:44"), ("second", 10, "22:44")], 40, 边界返回码=返回码)
                self.assertEqual(顺序, ["first", "second"])
                self.assertFalse(等待)

    def test_旧worker在下一任务计算前释放并将该任务交回等待(self):
        顺序, 等待 = self.run_queue([("first-finished", 10, "22:44"), ("second-not-started", 10, "22:44")], 40, 旧worker保护=True)
        self.assertEqual(顺序, ["first-finished"])
        self.assertEqual(等待, ["second-not-started"])

    def test_backfill_skips_oversized_high_priority_task(self):
        order, waiting = self.run_queue([
            ("large-high", 100, 65), ("small-low", 0, 13),
        ], 32)
        self.assertEqual(order, ["small-low"])
        self.assertEqual(waiting, ["large-high"])

    def test_full_signed_priority_range_and_fifo(self):
        order, waiting = self.run_queue([
            ("min-old-small", -(2 ** 31), 13),
            ("min-new-large", -(2 ** 31), 65),
            ("max-old-small", 2 ** 31 - 1, 13),
            ("max-new-large", 2 ** 31 - 1, 65),
        ], 80)
        self.assertEqual(order, ["max-old-small", "max-new-large", "min-old-small", "min-new-large"])
        self.assertFalse(waiting)

    def test_显存妥协只在不足上限的节点降一级(self):
        任务 = [("range-old", 20, "22:44"), ("sufficient-new", 20, "5:10")]
        for 显存, 期望 in ((32, ["sufficient-new", "range-old"]), (44, ["range-old", "sufficient-new"]), (80, ["range-old", "sufficient-new"])):
            with self.subTest(显存=显存):
                顺序, 等待 = self.run_queue(任务, 显存)
                self.assertEqual(顺序, 期望)
                self.assertFalse(等待)

    def test_妥协降级后仍按用户优先级和FIFO(self):
        for 原优先级, 正常优先级, 期望 in ((20, 19, ["range-old", "normal-new"]), (100, 20, ["range-old", "normal-new"]), (-(2 ** 31), -(2 ** 31), ["normal-new", "range-old"]), (2 ** 31 - 1, 2 ** 31 - 1, ["normal-new", "range-old"])):
            with self.subTest(原优先级=原优先级, 正常优先级=正常优先级):
                顺序, 等待 = self.run_queue([("range-old", 原优先级, "22:44"), ("normal-new", 正常优先级, "5:10")], 32)
                self.assertEqual(顺序, 期望)
                self.assertFalse(等待)

    def test_显存下限不足时回填且保留小数(self):
        顺序, 等待 = self.run_queue([("fractional-high", 100, "32.5:65"), ("normal-low", 0, "5:10")], 32)
        self.assertEqual(顺序, ["normal-low"])
        self.assertEqual(等待, ["fractional-high"])

    def test_多卡节点按单卡容量判断显存妥协(self):
        顺序, 等待 = self.run_queue([("range-old", 20, "22:44"), ("normal-new", 20, "5:10")], 32, 显卡数=2)
        self.assertEqual(顺序, ["normal-new", "range-old"])
        self.assertFalse(等待)


if __name__ == "__main__":
    unittest.main()