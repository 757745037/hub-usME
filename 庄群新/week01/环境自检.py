# -*- coding: utf-8 -*-
"""第 1 周作业：Python 开发环境自检。

运行方式：
    python 环境自检.py

脚本会依次检查 Python 版本、运行目录、编码环境，并输出一份自检清单。
"""

import platform
import sys


def check_python_version() -> tuple[bool, str]:
    """检查 Python 版本是否不低于 3.8。"""
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 8)
    info = f"Python {platform.python_version()} ({platform.system()} {platform.release()})"
    return ok, info


def check_encoding() -> tuple[bool, str]:
    """检查标准输出编码，避免中文乱码。"""
    encoding = sys.stdout.encoding or "unknown"
    ok = encoding.lower().replace("-", "") in {"utf8", "cp65001"}
    return ok, f"标准输出编码：{encoding}"


def use_utf8_output() -> None:
    """Windows 控制台默认可能是 GBK，这里强制使用 UTF-8，避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def check_workdir() -> tuple[bool, str]:
    """检查当前运行目录。"""
    import os

    cwd = os.getcwd()
    return True, f"当前工作目录：{cwd}"


def main() -> int:
    use_utf8_output()
    print("=" * 46)
    print("          Python 开发环境自检")
    print("=" * 46)

    checks = [
        ("解释器版本", check_python_version),
        ("输出编码", check_encoding),
        ("工作目录", check_workdir),
    ]

    failed = 0
    for index, (title, func) in enumerate(checks, start=1):
        try:
            ok, detail = func()
        except Exception as exc:  # noqa: BLE001 - 自检脚本需要兜底
            ok, detail = False, f"检查异常：{exc!r}"

        mark = "通过" if ok else "未通过"
        if not ok:
            failed += 1
        print(f"[{index}] {title:<8} {mark:<4} {detail}")

    print("-" * 46)
    print("基础语法小练习：")

    name = "庄群新"
    week = 1
    print(f"  你好，{name}！这是第 {week} 周的作业。")

    numbers = [1, 2, 3, 4, 5]
    print(f"  列表 {numbers} 的和为 {sum(numbers)}，平均值为 {sum(numbers) / len(numbers):.2f}")

    print("-" * 46)
    if failed:
        print(f"自检结束：{failed} 项未通过，请检查环境配置。")
        return 1

    print("自检结束：全部通过，环境可以正常开始学习。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
