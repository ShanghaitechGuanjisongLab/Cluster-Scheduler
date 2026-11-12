"""只读列出 HQ 等待任务的显存范围、提交优先级和妥协时优先级。"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import unicodedata

from GPUworker import 显存优先级, 解析显存范围
from 自动扩容 import RESOURCE_SCALE, resource_amount

本目录 = Path(__file__).resolve().parent


def 查询HQ(*参数, 状态目录=None):
    环境 = dict(os.environ)
    if 状态目录 is not None:
        环境["HQ_SERVER_DIR"] = str(Path(状态目录) / "server")
    else:
        环境.setdefault("HQ_SERVER_DIR", str(Path(环境.get("TASKCTL_STATE", 本目录 / "运行状态")) / "server"))
    环境["LD_LIBRARY_PATH"] = 环境.get("HQ_LIBSTDCPP", "/public_bme2/Share200T/管吉松/envs/af3/lib") + ":" + 环境.get("LD_LIBRARY_PATH", "")
    结果 = subprocess.run([str(本目录 / "bin" / "hq"), "--output-mode=json", *参数], env=环境, capture_output=True, text=True, timeout=60)
    if 结果.returncode:
        raise RuntimeError("HQ 查询失败：" + (结果.stderr.strip() or 结果.stdout.strip()))
    数据 = json.loads(结果.stdout)
    if not isinstance(数据, list):
        raise ValueError("HQ 返回的列表格式无效")
    return 数据


def 读取等待任务(查询=查询HQ):
    作业列表 = 查询("job", "list", "--all")
    作业编号 = [作业["id"] for 作业 in 作业列表 if (作业.get("task_stats") or {}).get("waiting", 0) > 0]
    if not 作业编号:
        return []
    作业详情 = 查询("job", "info", ",".join(map(str, 作业编号)))
    if set(作业编号) - {作业["info"]["id"] for 作业 in 作业详情}:
        raise RuntimeError("HQ 未返回全部等待作业详情，请重新查询")
    记录列表 = []
    for 作业 in 作业详情:
        信息 = 作业["info"]
        等待数 = (信息.get("task_stats") or {}).get("waiting", 0)
        if not 等待数:
            continue
        提交列表 = 作业.get("submits", [])
        if len(提交列表) != 1 or not 提交列表[0].get("array"):
            raise ValueError(f"作业 {信息['id']} 不是中间层的单提交作业，无法准确归属资源和优先级")
        数组 = 提交列表[0]["array"]
        优先级 = int(数组.get("priority", 0))
        for 方案编号, 方案 in enumerate(数组.get("resources", []), 1):
            资源 = {条目["resource"]: 条目 for 条目 in 方案.get("resources", [])}
            显卡数 = resource_amount(资源["gpus"]) if "gpus" in 资源 else 0
            显存值 = next(iter(资源.get("gpumem", {}).get("request", {}).values()), 0)
            总下限, 总上限 = 显存值 if isinstance(显存值, list) else (显存值, 显存值)
            下限, 上限 = 解析显存范围((总下限 / RESOURCE_SCALE / 显卡数, 总上限 / RESOURCE_SCALE / 显卡数)) if 显卡数 else (0, 0)
            妥协优先级 = 显存优先级(优先级, (下限, 上限), 下限) if 0 < 下限 < 上限 else None
            记录列表.append({"作业号": 信息["id"], "任务名": 信息["name"], "等待任务数": 等待数, "资源方案": 方案编号, "GPU数": 显卡数 if 下限 else 0, "每卡显存下限GiB": 下限, "每卡显存上限GiB": 上限, "优先级": 优先级, "妥协优先级": 妥协优先级})
        if not 数组.get("resources"):
            raise ValueError(f"作业 {信息['id']} 没有资源方案")
    return sorted(记录列表, key=lambda 记录: (-记录["优先级"], 记录["作业号"], 记录["资源方案"]))


def 格式化队列(记录列表):
    if not 记录列表:
        return "HQ 当前没有等待任务。"
    表格 = [["HQ ID", "任务名", "等待数", "方案", "GPU数", "显存范围(GiB/卡)", "优先级", "妥协优先级"]]
    for 记录 in 记录列表:
        显存 = f"{记录['每卡显存下限GiB']:g}:{记录['每卡显存上限GiB']:g}" if 记录["GPU数"] else "CPU"
        妥协 = str(记录["妥协优先级"]) if 记录["妥协优先级"] is not None else "不适用"
        表格.append([str(记录["作业号"]), 记录["任务名"], str(记录["等待任务数"]), str(记录["资源方案"]), f"{记录['GPU数']:g}", 显存, str(记录["优先级"]), 妥协])

    def 显示宽度(文本):
        return sum(2 if unicodedata.east_asian_width(字符) in ("W", "F") else 1 for 字符 in 文本)

    列宽 = [max(显示宽度(行[列]) for 行 in 表格) for 列 in range(len(表格[0]))]
    行列表 = ["  ".join(文本 + " " * (列宽[列] - 显示宽度(文本)) for 列, 文本 in enumerate(行)).rstrip() for 行 in 表格]
    作业数量 = len({记录["作业号"] for 记录 in 记录列表})
    行列表.append(f"共 {作业数量} 个等待作业；按已提交优先级降序、作业号升序排列。")
    return "\n".join(行列表)


def 主程序():
    参数解析 = argparse.ArgumentParser(description=__doc__)
    参数解析.add_argument("--状态目录", type=Path, help="共享状态目录，默认遵循 TASKCTL_STATE 或 HQ_SERVER_DIR")
    参数解析.add_argument("--json", dest="机器可读", action="store_true", help="输出 JSON")
    参数 = 参数解析.parse_args()
    try:
        记录列表 = 读取等待任务(lambda *查询参数: 查询HQ(*查询参数, 状态目录=参数.状态目录))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as 错误:
        print(f"查询失败：{错误}", file=sys.stderr)
        return 1
    if 参数.机器可读:
        print(json.dumps(记录列表, ensure_ascii=False, indent=2))
    else:
        print("查询时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(格式化队列(记录列表))
    return 0


if __name__ == "__main__":
    raise SystemExit(主程序())