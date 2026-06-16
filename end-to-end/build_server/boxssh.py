"""Tiny non-interactive SSH/SFTP helper for the Threadripper build box.

The build box (192.168.10.5) uses password auth for root; the Bash tool
can't drive an interactive password prompt, so we use paramiko (present in
.venv-qairt). Credentials come from env to keep them out of argv/history:

    BOX_HOST (default 192.168.10.5) BOX_USER (root) BOX_PASS

Usage:
    .venv-qairt/Scripts/python.exe end-to-end/build_server/boxssh.py run "ls /workspace"
    ... put <local> <remote>
    ... get <remote> <local>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def connect() -> paramiko.SSHClient:
    host = os.environ.get("BOX_HOST", "192.168.10.5")
    user = os.environ.get("BOX_USER", "root")
    pw = os.environ.get("BOX_PASS")
    if not pw:
        sys.exit("set BOX_PASS")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=pw, timeout=20,
              look_for_keys=False, allow_agent=False)
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: float | None = None) -> int:
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
    for line in stdout:
        sys.stdout.write(line)
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if err.strip():
        sys.stderr.write(err)
    return rc


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    op = sys.argv[1]
    c = connect()
    try:
        if op == "run":
            return run(c, sys.argv[2])
        if op == "put":
            sftp = c.open_sftp()
            sftp.put(sys.argv[2], sys.argv[3])
            print(f"put {sys.argv[2]} -> {sys.argv[3]}")
            return 0
        if op == "putx":
            # base64 over exec stdin — works on FUSE mounts where SFTP
            # create fails (unRAID /mnt/vm_8tb).
            import base64
            data = Path(sys.argv[2]).read_bytes()
            b64 = base64.b64encode(data).decode()
            stdin, stdout, stderr = c.exec_command(f"base64 -d > {sys.argv[3]}")
            stdin.write(b64)
            stdin.channel.shutdown_write()
            rc = stdout.channel.recv_exit_status()
            err = stderr.read().decode("utf-8", "replace")
            if err.strip():
                sys.stderr.write(err)
            print(f"putx {sys.argv[2]} -> {sys.argv[3]} ({len(data)} B, rc={rc})")
            return rc
        if op == "get":
            sftp = c.open_sftp()
            sftp.get(sys.argv[2], sys.argv[3])
            print(f"get {sys.argv[2]} -> {sys.argv[3]}")
            return 0
        sys.exit(f"unknown op {op}")
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(main())
