"""Manage the `vllm-sr` router lifecycle.

`vllm-sr serve` is a *launcher*: it brings up a stack of Docker containers
(router + envoy + dashboard + simulator + datastores + observability) and
EXITS CLEANLY with code 0. The actual router service then lives in those
containers, managed by the host `vllm-sr` CLI. `vllm-sr stop` tears the
stack down.

Lifecycle the harness implements:

  1. Run `vllm-sr serve [serve_args...]` with stdout/stderr → log file.
  2. Wait for the launcher subprocess to exit. Exit 0 = launch succeeded;
     non-zero = launch failed (read the log to diagnose).
  3. Poll `GET http://apiserver/ready` against the now-background router
     until it returns 200, with a clear timeout error pointing at the
     dashboard if the router is in setup mode.
  4. Yield to the caller.
  5. On exit, by default LEAVE THE STACK RUNNING (re-runs are fast and the
     user controls the long-lived lifecycle via `vllm-sr stop` themselves).
     If `stop_on_exit: true`, run `vllm-sr stop` to tear down.

Tests mock the spawn + poll seams; this module never starts a real router in CI.
"""
from __future__ import annotations

import asyncio
import contextlib
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
        """Wait for the launcher to exit cleanly, then poll /ready.

        The `vllm-sr serve` launcher runs synchronously and exits 0 once the
        Docker stack is up. A non-zero exit is a real failure. After the
        launcher exits, we poll the router's apiserver /ready endpoint until
        it returns 200, which signals the router is configured and routing.
        """
        # Phase 1: wait for the launcher subprocess (if we spawned one) to exit.
        if self._proc is not None:
            launcher_deadline = time.monotonic() + self.cfg.ready_timeout_s
            while time.monotonic() < launcher_deadline:
                rc = self._proc.poll()
                if rc is None:
                    await asyncio.sleep(0.5)
                    continue
                if rc != 0:
                    raise RouterStartupError(
                        f"`vllm-sr serve` exited with code {rc}; "
                        f"see {self.cfg.log_path or '(no log path configured)'}"
                    )
                break  # exit 0 = launch succeeded; move on to readiness probe

        # Phase 2: poll /ready on the apiserver.
        deadline = time.monotonic() + self.cfg.ready_timeout_s
        url = f"{self.apiserver_base}/ready"
        last_err: str | None = None
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
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
            f"(last: {last_err}).\n"
            f"If `vllm-sr status` shows 'Router: Status unknown', the router is "
            f"likely in setup mode with no models configured. Open the dashboard "
            f"at http://{self.cfg.apiserver_host}:8700 to configure models, then "
            f"retry. See logs with `vllm-sr logs router`."
        )

    async def stop(self) -> None:
        """Tear down the router stack if `stop_on_exit` is set.

        Default behavior is to LEAVE THE STACK RUNNING — `vllm-sr serve` is
        slow (image pulls, container starts) and benchmark runs are short.
        The user controls the long-lived lifecycle with `vllm-sr stop`.

        If the launcher subprocess is still alive somehow (timed out before
        finishing its handoff), we SIGTERM it as a safety net.
        """
        proc = self._proc
        self._proc = None
        try:
            # Safety net: kill a still-running launcher (shouldn't happen normally).
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                deadline = time.monotonic() + self.cfg.stop_timeout_s
                while time.monotonic() < deadline and proc.poll() is None:
                    await asyncio.sleep(0.2)
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)

            # Tear down the background stack only if requested.
            if not self.cfg.external and self.cfg.stop_on_exit:
                binary = shutil.which(self.cfg.binary) or self.cfg.binary
                # Best-effort teardown: swallow any failure (already exiting).
                with contextlib.suppress(Exception):
                    subprocess.run(
                        [binary, "stop"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=self.cfg.stop_timeout_s,
                        check=False,
                    )
        finally:
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                finally:
                    self._log_handle = None
