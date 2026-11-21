"""任务包发布前自检（preflight lint）。

只做**静态**检查：目录结构、配置可解析、答案泄漏、清单哈希一致性。
不执行包内任何代码，不联网。

检查项来自反复返修时踩过的真实问题类别（题包版本号一路 v4 / r5 / repaired
才过质检，说明这些错误很常见）。规则编号保持稳定，方便在 issue 里引用。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

try:  # Python 3.11+ 自带；3.10 需要 tomli
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

Severity = Literal["high", "medium", "low"]

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# tests/ 下至少要有一个入口脚本，否则 Verifier 无从执行
ENTRY_CANDIDATES = ("test.sh", "run_tests.sh", "run.sh", "verify.py", "run_checks.py")

# environment/ 里出现这些名字，通常意味着标准答案被挂进了选手可见目录
ANSWER_NAME = re.compile(
    r"(gold|answer|label|ground[_-]?truth|private|hidden|secret|expected)", re.IGNORECASE
)

# 任务说明里不该出现指向私有资产的路径
PRIVATE_REF = re.compile(
    r"(/opt/private|private[_-]?gold|tests/private|hidden[_-]?test|solution/private)",
    re.IGNORECASE,
)

SCORE_KEYWORD = re.compile(r"(评分|得分|score|reward|指标|metric)", re.IGNORECASE)

MAX_FILE_MB = 50

__all__ = ["Finding", "Severity", "format_report", "lint_task"]


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: Severity
    path: str
    message: str

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _pick_dir(root: Path, *names: str) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _manifest_entries(data: object) -> list[tuple[str, str | None]]:
    """把几种常见 manifest 写法统一成 [(path, sha256|None)]。"""

    def from_items(items: object) -> list[tuple[str, str | None]]:
        out: list[tuple[str, str | None]] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    path = item.get("path") or item.get("file")
                    if isinstance(path, str):
                        sha = item.get("sha256") or item.get("sha")
                        out.append((path, sha if isinstance(sha, str) else None))
        elif isinstance(items, dict):
            for path, sha in items.items():
                if isinstance(path, str):
                    out.append((path, sha if isinstance(sha, str) else None))
        return out

    if isinstance(data, list):
        return from_items(data)
    if isinstance(data, dict):
        if "files" in data:
            return from_items(data["files"])
        # 顶层直接是 {path: sha}
        if all(isinstance(k, str) and "/" in k for k in data):
            return from_items(data)
    return []


def lint_task(root: str | Path) -> list[Finding]:
    """对一个任务包目录做发布前自检，返回按严重度排序的问题列表。"""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"不是一个目录：{root}")

    findings: list[Finding] = []

    def add(rule: str, severity: Severity, path: str, message: str) -> None:
        findings.append(Finding(rule=rule, severity=severity, path=path, message=message))

    # R001 任务说明
    instruction = root / "instruction.md"
    if not instruction.is_file():
        add("R001", "high", "instruction.md", "缺少 instruction.md，选手看不到任务目标")
    elif not instruction.read_text(encoding="utf-8", errors="replace").strip():
        add("R001", "high", "instruction.md", "instruction.md 为空")

    # R002 / R003 运行配置
    task_toml = root / "task.toml"
    if not task_toml.is_file():
        add("R002", "high", "task.toml", "缺少 task.toml，资源与超时无从配置")
    elif tomllib is None:  # pragma: no cover
        add("R002", "medium", "task.toml", "当前 Python 无 tomllib，跳过 task.toml 解析（需 3.11+）")
    else:
        try:
            raw = task_toml.read_bytes()
            config = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, Exception) as exc:  # tomllib.TOMLDecodeError
            add("R002", "high", "task.toml", f"task.toml 无法解析：{exc}")
            config = {}
        if isinstance(config, dict) and config:
            timeouts = config.get("timeouts") or config.get("timeout") or config.get("limits")
            if timeouts is None:
                add("R003", "medium", "task.toml", "未声明超时，长任务可能被平台默认值截断")
            elif isinstance(timeouts, dict):
                for key, value in timeouts.items():
                    if isinstance(value, (int, float)) and value <= 0:
                        add("R003", "medium", "task.toml", f"超时项 {key} 不是正数：{value}")

    # R004 / R006 environment
    env = _pick_dir(root, "environment", "env")
    if env is None:
        add("R004", "high", "environment/", "缺少 environment/，任务没有可运行的工作区")
    else:
        for path in _files(env):
            name = path.name
            if ANSWER_NAME.search(name):
                add(
                    "R006",
                    "high",
                    _rel(path, root),
                    "选手可见目录里出现疑似答案文件：一旦泄漏，任务失去区分度且返修代价极高",
                )

    # R005 tests
    tests = _pick_dir(root, "tests")
    if tests is None:
        add("R005", "high", "tests/", "缺少 tests/，没有固定的评价与验收入口")
    elif not _files(tests):
        add("R005", "high", "tests/", "tests/ 为空")
    elif not any((tests / name).is_file() for name in ENTRY_CANDIDATES):
        add(
            "R005",
            "high",
            "tests/",
            f"tests/ 下没有入口脚本（期望其中之一：{', '.join(ENTRY_CANDIDATES)}）",
        )

    # R007 environment 与 solution 内容完全相同 → oracle 泄漏
    solution = _pick_dir(root, "solution")
    if solution is None:
        add("R013", "low", "solution/", "没有 solution/，缺少 oracle 参考实现")
    elif env is not None:
        env_hashes = {}
        for path in _files(env):
            try:
                env_hashes.setdefault(_sha256(path), []).append(path)
            except OSError as exc:
                add("R007", "low", _rel(path, root), f"读取失败，跳过哈希对比：{exc}")
        for path in _files(solution):
            try:
                digest = _sha256(path)
            except OSError as exc:
                add("R007", "low", _rel(path, root), f"读取失败，跳过哈希对比：{exc}")
                continue
            if digest in env_hashes:
                for twin in env_hashes[digest]:
                    add(
                        "R007",
                        "high",
                        _rel(path, root),
                        f"与 {_rel(twin, root)} 内容完全相同：参考实现被复制进了选手工作区",
                    )

    # R008 manifest 一致性
    for candidate in (root / "manifest.json", root / "artifacts" / "manifest.json"):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            add("R008", "high", _rel(candidate, root), f"manifest 不是合法 JSON：{exc}")
            continue
        entries = _manifest_entries(data)
        if not entries:
            add("R008", "medium", _rel(candidate, root), "manifest 里没有可识别的文件清单")
            continue
        for rel_path, want in entries:
            target = root / rel_path
            if not target.is_file():
                add("R008", "high", rel_path, "manifest 声明的文件不存在")
                continue
            if want:
                want_clean = want.split(":")[-1].strip().lower()
                try:
                    got = _sha256(target)
                except OSError as exc:
                    add("R008", "low", rel_path, f"读取失败：{exc}")
                    continue
                if got != want_clean:
                    add(
                        "R008",
                        "high",
                        rel_path,
                        f"哈希与 manifest 不符：manifest={want_clean[:12]}… actual={got[:12]}…",
                    )

    # R009 说明里引用私有路径
    if instruction.is_file():
        text = instruction.read_text(encoding="utf-8", errors="replace")
        for match in sorted({m.group(0) for m in PRIVATE_REF.finditer(text)}):
            add("R009", "high", "instruction.md", f"任务说明引用了私有资产路径：{match}")
        if not SCORE_KEYWORD.search(text):
            add("R012", "low", "instruction.md", "任务说明里没有出现评分/指标相关表述")

    # R010 大文件
    for path in _files(root):
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
        except OSError:
            continue
        if size_mb > MAX_FILE_MB:
            add(
                "R010",
                "low",
                _rel(path, root),
                f"单文件 {size_mb:.1f} MB，超过 {MAX_FILE_MB} MB：考虑外置或用数据清单引用",
            )

    # R011 tests 与 environment 存在同内容文件（可能是无意的资源重复）
    if tests is not None and env is not None:
        test_hashes: dict[str, Path] = {}
        for path in _files(tests):
            try:
                test_hashes.setdefault(_sha256(path), path)
            except OSError:
                continue
        for path in _files(env):
            try:
                digest = _sha256(path)
            except OSError:
                continue
            if digest in test_hashes:
                add(
                    "R011",
                    "low",
                    _rel(path, root),
                    f"与 {_rel(test_hashes[digest], root)} 内容相同：确认这是有意共享的资源",
                )

    findings.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.rule, f.path))
    return findings


def _width(text: str) -> int:
    """终端显示宽度：东亚全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def format_report(findings: Iterable[Finding]) -> str:
    """把问题列表渲染成对齐的文本表。"""
    items = list(findings)
    if not items:
        return "PASS 未发现问题"
    headers = ("严重度", "规则", "位置", "说明")
    rows = [(f.severity, f.rule, f.path, f.message) for f in items]
    widths = [
        max(_width(h), max(_width(row[i]) for row in rows))
        for i, h in enumerate(headers[:3])
    ]
    lines = [" ".join(_pad(headers[i], widths[i]) for i in range(3)) + " " + headers[3]]
    lines.append("-" * 110)
    for row in rows:
        lines.append(
            " ".join(_pad(row[i], widths[i]) for i in range(3)) + " " + row[3]
        )
    high = sum(1 for f in items if f.severity == "high")
    lines.append("-" * 110)
    lines.append(f"合计 {len(items)} 项，其中 high {high} 项")
    return "\n".join(lines)
