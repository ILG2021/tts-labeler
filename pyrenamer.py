#!/usr/bin/env python3
"""
rename_files.py

重命名文件，去除文件名中的空白符号和引号（包括中文引号），替换成下划线。

用法（命令行）:
    # 重命名单个或多个文件
    python rename_files.py file1.txt "file 2.txt"

    # 使用通配符（Linux/macOS 由 shell 展开；Windows 由脚本内部展开，效果相同）
    python rename_files.py *.txt
    python rename_files.py "docs/*.md"      # Windows 需加引号防止 shell 报错
    python rename_files.py "**/*.txt"       # 递归匹配所有子目录（跨平台）

    # 重命名目录下所有文件
    python rename_files.py -d /path/to/dir

    # 按后缀名过滤（可多个）
    python rename_files.py -d /path/to/dir -e .txt .md

    # 递归处理子目录
    python rename_files.py -d /path/to/dir -r

    # 试运行（不实际重命名）
    python rename_files.py -d /path/to/dir --dry-run
"""

import re
import glob as _glob
import argparse
from pathlib import Path


# 需要替换为下划线的字符：空白符 + 各类引号（英文、中文）
_REPLACE_PATTERN = re.compile(
    r'[\s'           # 所有空白符（空格、制表符、换行等）
    r'\'"'           # 英文单引号、双引号
    r'\u2018\u2019'  # 中文单引号 ''
    r'\u201c\u201d'  # 中文双引号 ""
    r'\u300c\u300d'  # 日文引号 「」
    r'\u300e\u300f'  # 日文引号 『』
    r']+'            # 合并连续匹配为单个下划线
)


def clean_filename(name: str) -> str:
    """
    清洗文件名（不含扩展名部分），将空白和引号替换为下划线。
    首尾的下划线会被去除。
    """
    cleaned = _REPLACE_PATTERN.sub('_', name)
    cleaned = cleaned.strip('_')
    return cleaned if cleaned else '_'


def rename_file(filepath: Path, dry_run: bool = False) -> tuple[Path, Path] | None:
    """
    重命名单个文件。

    参数:
        filepath: 文件路径
        dry_run:  若为 True，只打印计划，不实际执行

    返回:
        (原路径, 新路径) 元组；若文件名无需变更则返回 None
    """
    stem = filepath.stem
    suffix = filepath.suffix
    new_stem = clean_filename(stem)
    new_name = new_stem + suffix

    if new_name == filepath.name:
        return None  # 无需改动

    new_path = filepath.parent / new_name

    if not dry_run:
        # 目标文件已存在时自动加序号避免覆盖
        if new_path.exists():
            counter = 1
            while new_path.exists():
                new_path = filepath.parent / f"{new_stem}_{counter}{suffix}"
                counter += 1

        filepath.rename(new_path)

    return filepath, new_path


def _expand_path(p: str | Path) -> list[Path]:
    """
    展开单个路径或通配符模式，返回匹配到的文件列表。

    在 Linux/macOS 上，shell 通常会在传入 argv 前自动展开通配符；
    在 Windows 上，cmd.exe 和 PowerShell 均不展开通配符，需由此函数负责。
    为保持跨平台一致性，无论什么平台都在此统一展开。
    """
    p_str = str(p)
    if any(c in p_str for c in ('*', '?', '[')):
        # recursive=True 支持 ** 跨目录匹配
        matched = _glob.glob(p_str, recursive=True)
        if not matched:
            print(f"[警告] 通配符未匹配到任何文件: {p_str}")
            return []
        return [Path(m) for m in sorted(matched) if Path(m).is_file()]
    else:
        fp = Path(p_str)
        if not fp.is_file():
            print(f"[警告] 跳过（不是文件或不存在）: {fp}")
            return []
        return [fp]


def collect_files(
    paths: list[str | Path] | None = None,
    directory: str | Path | None = None,
    extensions: list[str] | None = None,
    recursive: bool = False,
) -> list[Path]:
    """
    收集待处理的文件列表。

    参数:
        paths:      明确指定的文件路径或通配符模式列表
        directory:  扫描整个目录
        extensions: 后缀名过滤列表，如 ['.txt', '.md']；None 表示不过滤
        recursive:  是否递归子目录（仅对 directory 有效）

    返回:
        Path 对象列表
    """
    files: list[Path] = []

    # 处理明确指定的文件（含通配符展开，Windows 上 shell 不展开时由此负责）
    if paths:
        for p in paths:
            files.extend(_expand_path(p))

    # 扫描目录
    if directory:
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"不是有效目录: {d}")
        globber = d.rglob('*') if recursive else d.glob('*')
        for fp in sorted(globber):
            if fp.is_file():
                files.append(fp)

    # 后缀名过滤
    if extensions:
        exts = {e if e.startswith('.') else f'.{e}' for e in extensions}
        files = [f for f in files if f.suffix.lower() in exts]

    return files


def batch_rename(
    paths: list[str | Path] | None = None,
    directory: str | Path | None = None,
    extensions: list[str] | None = None,
    recursive: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> list[tuple[Path, Path]]:
    """
    批量重命名文件（模块调用入口）。

    参数:
        paths:      明确指定的文件路径或通配符模式列表
        directory:  扫描整个目录
        extensions: 后缀名过滤列表
        recursive:  是否递归子目录
        dry_run:    试运行，不实际重命名
        verbose:    是否打印操作日志

    返回:
        实际发生重命名的 [(原路径, 新路径), ...] 列表
    """
    files = collect_files(paths=paths, directory=directory,
                          extensions=extensions, recursive=recursive)

    if not files:
        if verbose:
            print("未找到符合条件的文件。")
        return []

    results: list[tuple[Path, Path]] = []
    skipped = 0

    for fp in files:
        result = rename_file(fp, dry_run=dry_run)
        if result is None:
            skipped += 1
            continue
        old, new = result
        results.append((old, new))
        if verbose:
            prefix = "[DRY-RUN] " if dry_run else ""
            print(f"{prefix}{old.name}  ->  {new.name}  (在 {old.parent})")

    if verbose:
        action = "将重命名" if dry_run else "已重命名"
        print(f"\n共扫描 {len(files)} 个文件，{action} {len(results)} 个，跳过 {skipped} 个。")

    return results


# ── 命令行入口 ────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='rename_files',
        description='重命名文件：将文件名中的空白符和引号替换为下划线。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'files', nargs='*', metavar='FILE',
        help='要重命名的文件，支持通配符（如 *.txt、"docs/**/*.md"）',
    )
    parser.add_argument(
        '-d', '--directory', metavar='DIR',
        help='扫描指定目录下的所有文件',
    )
    parser.add_argument(
        '-e', '--extensions', nargs='+', metavar='EXT',
        help='只处理指定后缀名的文件，如 .txt .md',
    )
    parser.add_argument(
        '-r', '--recursive', action='store_true',
        help='递归处理子目录（仅对 --directory 有效）',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='试运行：只显示将要进行的重命名，不实际执行',
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='静默模式，不打印操作日志',
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.files and not args.directory:
        parser.print_help()
        return

    batch_rename(
        paths=args.files or None,
        directory=args.directory,
        extensions=args.extensions,
        recursive=args.recursive,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
