"""RouterProcess lifecycle tests.

We don't spawn a real `vllm-sr` here — these tests mock subprocess.Popen and
httpx so the lifecycle logic (spawn → poll /ready → teardown) is exercised
deterministically.
"""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from benchmark.config import RouterProcessConfig
from benchmark.router_proc import RouterProcess, RouterStartupError


class _FakeProc:
    """Minimal subprocess.Popen stand-in."""

    def __init__(self, exit_after: int | None = None, returncode: int = 0) -> None:
        self._calls = 0
        self._exit_after = exit_after
        self.returncode: int | None = None
        self._returncode_after_exit = returncode
        self.signals_received: list[int] = []

    def poll(self):
        self._calls += 1
        if self._exit_after is not None and self._calls >= self._exit_after:
            self.returncode = self._returncode_after_exit
            return self.returncode
        return None

    def send_signal(self, sig: int) -> None:
        self.signals_received.append(sig)
        # Pretend it exits cleanly on SIGTERM.
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0


def _patch_httpx(monkeypatch, transport: httpx.MockTransport) -> None:
    real_init = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


@pytest.mark.asyncio
async def test_external_mode_does_not_spawn(monkeypatch) -> None:
    """external=True: no subprocess, just health-check."""
    cfg = RouterProcessConfig(external=True, ready_timeout_s=2)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/ready")
        return httpx.Response(200)

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))

    with patch("benchmark.router_proc.subprocess.Popen") as popen:
        async with RouterProcess(cfg):
            pass
        popen.assert_not_called()


@pytest.mark.asyncio
async def test_spawn_and_ready(monkeypatch) -> None:
    cfg = RouterProcessConfig(ready_timeout_s=2, stop_timeout_s=1)
    fake_proc = _FakeProc()

    # First call to /ready returns 503, second returns 200.
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(503)
        return httpx.Response(200)

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))

    with patch("benchmark.router_proc.shutil.which", return_value="/usr/bin/vllm-sr"), \
         patch("benchmark.router_proc.subprocess.Popen", return_value=fake_proc) as popen:
        async with RouterProcess(cfg):
            popen.assert_called_once()
            args, kwargs = popen.call_args
            assert args[0][0] == "/usr/bin/vllm-sr"
            assert args[0][1] == "serve"

    # Teardown sent SIGTERM.
    import signal as _signal
    assert _signal.SIGTERM in fake_proc.signals_received


@pytest.mark.asyncio
async def test_missing_binary_raises(monkeypatch) -> None:
    cfg = RouterProcessConfig(binary="definitely-not-installed-xyz", ready_timeout_s=1)

    with patch("benchmark.router_proc.shutil.which", return_value=None):
        with pytest.raises(RouterStartupError, match="not found on PATH"):
            async with RouterProcess(cfg):
                pass


@pytest.mark.asyncio
async def test_subprocess_dies_during_startup(monkeypatch) -> None:
    cfg = RouterProcessConfig(ready_timeout_s=2, stop_timeout_s=1)
    # poll() returns 1 immediately, simulating crash.
    fake_proc = _FakeProc(exit_after=1, returncode=1)

    def handler(request: httpx.Request) -> httpx.Response:
        # /ready will never come up; whether we hit it or not, the proc-died
        # check should fire first.
        return httpx.Response(503)

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))

    with patch("benchmark.router_proc.shutil.which", return_value="/usr/bin/vllm-sr"), \
         patch("benchmark.router_proc.subprocess.Popen", return_value=fake_proc):
        with pytest.raises(RouterStartupError, match="exited during startup"):
            async with RouterProcess(cfg):
                pass


@pytest.mark.asyncio
async def test_ready_timeout(monkeypatch) -> None:
    cfg = RouterProcessConfig(ready_timeout_s=1, stop_timeout_s=1)
    fake_proc = _FakeProc()  # never exits

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))

    with patch("benchmark.router_proc.shutil.which", return_value="/usr/bin/vllm-sr"), \
         patch("benchmark.router_proc.subprocess.Popen", return_value=fake_proc):
        with pytest.raises(RouterStartupError, match="not ready"):
            async with RouterProcess(cfg):
                pass
