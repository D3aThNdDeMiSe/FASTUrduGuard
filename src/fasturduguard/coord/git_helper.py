"""Tiny git wrapper used as the inter-node 'wire'.

Either GitPython (preferred) or plain `git` subprocess (fallback). All ops are
best-effort: a failure to push is logged but never crashes training.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from fasturduguard.utils import repo_root

log = logging.getLogger("fug.coord.git")


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    cwd = cwd or repo_root()
    try:
        out = subprocess.run(
            cmd, cwd=str(cwd),
            capture_output=True, text=True, timeout=120,
        )
        return out.returncode, (out.stdout + out.stderr).strip()
    except Exception as e:
        return 1, str(e)


def pull() -> bool:
    rc, msg = _run(["git", "pull", "--rebase", "--autostash"])
    log.info("git pull: rc=%d msg=%s", rc, msg.splitlines()[-1] if msg else "")
    return rc == 0


def commit_paths(paths: list[Path], message: str) -> bool:
    rel = []
    root = repo_root()
    for p in paths:
        try:
            rel.append(str(p.resolve().relative_to(root)))
        except Exception:
            rel.append(str(p))
    rc, _ = _run(["git", "add", "-f", *rel])
    if rc != 0:
        return False
    rc, msg = _run(["git", "commit", "-m", message])
    if "nothing to commit" in msg.lower():
        return True
    return rc == 0


def push() -> bool:
    rc, msg = _run(["git", "push"])
    log.info("git push: rc=%d msg=%s", rc, msg.splitlines()[-1] if msg else "")
    return rc == 0


def publish(paths: list[Path], message: str) -> bool:
    """add+commit+push helper. Best-effort; safe to call when no remote configured."""
    if not commit_paths(paths, message):
        return False
    return push()
