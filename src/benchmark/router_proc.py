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
import hashlib
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


# Hash of the router-config.yaml the running stack was launched with.
# Written here by `ensure_router_stack` after a successful start/restart;
# consulted on the next call to decide whether the stack is still running
# the right config. Lives at repo root, gitignored.
ROUTER_CONFIG_HASH_FILE = Path(".router-config-hash")


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


# ─────────────────────────────────────────────────────────────────────────
# Hash-based "ensure stack is up with the right config" helper
# ─────────────────────────────────────────────────────────────────────────
# The router stack is a long-running docker collection that loads its
# config at startup. `make route` rebuilds `config/router-config.yaml`
# before each pass; without an explicit restart, the running stack keeps
# serving the old config and the new exemplars/backends silently never
# take effect.
#
# `ensure_router_stack` solves that with a hash-and-restart-if-different
# check: cheap when the config didn't change (just verifies the apiserver
# is healthy), slower when it did (one `vllm-sr stop && vllm-sr serve`
# cycle). The hash of the active config is persisted in
# .router-config-hash so the check survives across CLI invocations.


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_stored_hash(hash_file: Path) -> str | None:
    if not hash_file.exists():
        return None
    return hash_file.read_text().strip() or None


def _write_stored_hash(hash_file: Path, h: str) -> None:
    hash_file.write_text(h + "\n")


async def _is_apiserver_healthy(cfg: RouterProcessConfig, timeout_s: float = 2.0) -> bool:
    url = f"http://{cfg.apiserver_host}:{cfg.apiserver_port}/ready"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(url)
            return r.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


def _vllm_sr_stop(cfg: RouterProcessConfig) -> None:
    """Tolerant teardown — succeeds whether the stack is up or already gone."""
    binary = shutil.which(cfg.binary) or cfg.binary
    with contextlib.suppress(Exception):
        subprocess.run(
            [binary, "stop"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=cfg.stop_timeout_s, check=False,
        )


async def ensure_router_stack(
    cfg: RouterProcessConfig,
    config_path: Path,
    *,
    hash_file: Path | None = None,
    is_healthy: callable[[RouterProcessConfig], object] | None = None,
    stop: callable[[RouterProcessConfig], None] | None = None,
    spawn_serve: callable[[RouterProcessConfig], object] | None = None,
) -> str:
    """Ensure the router stack is running with the config at `config_path`.

    Returns one of:
      'unchanged' — stack is up and was already serving this config; we did nothing.
      'started'   — stack was down; we ran `vllm-sr serve --config <path>`.
      'restarted' — stack was up with a different config; we stopped + restarted.

    Fast path: when the SHA256 of `config_path` matches what we recorded
    on the last successful start AND `/ready` returns 200, we skip the
    docker-restart cycle entirely (sub-second).
    """
    hash_file = hash_file or ROUTER_CONFIG_HASH_FILE
    health_probe = is_healthy or _is_apiserver_healthy
    do_stop = stop or _vllm_sr_stop

    async def _default_spawn(c: RouterProcessConfig) -> None:
        # Reuse the existing launcher path — spawns `vllm-sr serve [serve_args]`,
        # waits for it to exit 0, then polls /ready until healthy.
        proc = RouterProcess(c)
        await proc.start()
        # Leave the stack running; nothing to clean up (launcher already exited).

    do_spawn = spawn_serve or _default_spawn

    current = _hash_file(config_path)
    stored = _read_stored_hash(hash_file)
    healthy = await health_probe(cfg)

    if stored == current and healthy:
        return "unchanged"

    action = "restarted" if healthy else "started"
    if healthy:
        do_stop(cfg)
    await do_spawn(cfg)
    _write_stored_hash(hash_file, current)
    return action
