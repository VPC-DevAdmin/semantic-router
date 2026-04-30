"""Manage the `vllm-sr` router as a subprocess.

The router is a Go binary that exposes an apiserver (default :8080) and an
Envoy-fronted OpenAI-compatible endpoint (default :8801). The lifecycle:

  1. Spawn `vllm-sr serve [serve_args...]` with stdout/stderr → log file.
  2. Poll `GET http://apiserver/ready` until it returns 200 (router fully up).
  3. Yield to the caller.
  4. On exit, send SIGTERM. If the process has not exited within
     `stop_timeout_s`, send SIGKILL.

Tests mock the spawn + poll seams; this module never starts a real router in CI.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import IO

import httpx

from .config import RouterProcessConfig


class RouterStartupError(RuntimeError):
    pass


class RouterProcess:
    """Async context manager owning a `vllm-sr` subprocess.

    With `external=True`, we don't spawn anything — we only health-check the
    already-running router. Useful when you have a long-running dev stack.
    """

    def __init__(self, cfg: RouterProcessConfig) -> None:
        self.cfg = cfg
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_handle: IO[bytes] | None = None

    @property
    def apiserver_base(self) -> str:
        return f"http://{self.cfg.apiserver_host}:{self.cfg.apiserver_port}"

    @property
    def frontend_base(self) -> str:
        return f"http://{self.cfg.frontend_host}:{self.cfg.frontend_port}"

    async def __aenter__(self) -> RouterProcess:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def start(self) -> None:
        if not self.cfg.external:
            self._spawn()
        try:
            await self._wait_until_ready()
        except Exception:
            await self.stop()
            raise

    def _spawn(self) -> None:
        binary = shutil.which(self.cfg.binary) or self.cfg.binary
        if not Path(binary).is_file() and shutil.which(self.cfg.binary) is None:
            raise RouterStartupError(
                f"router binary {self.cfg.binary!r} not found on PATH; install with "
                "`curl -fsSL https://vllm-semantic-router.com/install.sh | bash` "
                "or set `binary:` in config/router.yaml"
            )
        cmd = [binary, "serve", *self.cfg.serve_args]
        env = os.environ.copy()
        env.update(self.cfg.env)

        log_handle: IO[bytes] | None = None
        if self.cfg.log_path:
            Path(self.cfg.log_path).parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(self.cfg.log_path, "wb")
        self._log_handle = log_handle

        self._proc = subprocess.Popen(
            cmd,
            stdout=log_handle if log_handle else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_handle else subprocess.DEVNULL,
            env=env,
        )

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.cfg.ready_timeout_s
        url = f"{self.apiserver_base}/ready"
        last_err: str | None = None
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                # Surface a clear failure if the subprocess died during startup.
                if self._proc is not None and self._proc.poll() is not None:
                    raise RouterStartupError(
                        f"router process exited during startup with "
                        f"code {self._proc.returncode}; "
                        f"see {self.cfg.log_path or '(no log path configured)'}"
                    )
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return
                    last_err = f"HTTP {resp.status_code}"
                except httpx.HTTPError as e:
                    last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(0.5)
        raise RouterStartupError(
            f"router not ready at {url} within {self.cfg.ready_timeout_s}s "
            f"(last: {last_err})"
        )

    async def stop(self) -> None:
        proc = self._proc
        self._proc = None
        try:
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                # Wait without blocking the event loop.
                deadline = time.monotonic() + self.cfg.stop_timeout_s
                while time.monotonic() < deadline and proc.poll() is None:
                    await asyncio.sleep(0.2)
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
        finally:
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                finally:
                    self._log_handle = None
