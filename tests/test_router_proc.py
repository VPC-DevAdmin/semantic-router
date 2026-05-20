"""RouterProcess lifecycle tests.

We don't spawn a real `vllm-sr` here — these tests mock subprocess.Popen and
httpx so the lifecycle logic (run launcher → wait for exit → poll /ready →
teardown) is exercised deterministically.

vllm-sr serve is a launcher: it runs synchronously, brings up a Docker stack,
and exits 0. The actual router runs in those background containers. The
harness must NOT treat that clean exit as a crash.
"""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from benchmark.config import RouterProcessConfig
from benchmark.router_proc import (
    RouterProcess,
    RouterStartupError,
    ensure_router_stack,
)


class _FakeProc:
    """Minimal subprocess.Popen stand-in.

    `exit_after`: number of poll() calls before the process appears to exit.
    None means "never exits" (simulates a hung launcher).
    `returncode`: the exit code reported once it exits.
    """

    def __init__(self, exit_after: int | None = 1, returncode: int = 0) -> None:
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
async def test_launcher_exits_zero_then_ready_polled(monkeypatch) -> None:
    """The expected happy path: `vllm-sr serve` exits 0, then /ready returns 200."""
    cfg = RouterProcessConfig(ready_timeout_s=5, stop_timeout_s=1)
    fake_proc = _FakeProc(exit_after=1, returncode=0)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/ready")
        return httpx.Response(200)

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))

    with patch("benchmark.router_proc.shutil.which", return_value="/usr/bin/vllm-sr"), \
         patch("benchmark.router_proc.subprocess.Popen", return_value=fake_proc) as popen:
        async with RouterProcess(cfg):
            popen.assert_called_once()
            args, kwargs = popen.call_args
            assert args[0][0] == "/usr/bin/vllm-sr"
            assert args[0][1] == "serve"


@pytest.mark.asyncio
async def test_launcher_nonzero_exit_raises(monkeypatch) -> None:
    """A real failure: launcher exits with code != 0."""
    cfg = RouterProcessConfig(ready_timeout_s=2, stop_timeout_s=1)
    fake_proc = _FakeProc(exit_after=1, returncode=1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))

    with patch("benchmark.router_proc.shutil.which", return_value="/usr/bin/vllm-sr"), \
         patch("benchmark.router_proc.subprocess.Popen", return_value=fake_proc):
        with pytest.raises(RouterStartupError, match=r"exited with code 1"):
            async with RouterProcess(cfg):
                pass


@pytest.mark.asyncio
async def test_missing_binary_raises(monkeypatch) -> None:
    cfg = RouterProcessConfig(binary="definitely-not-installed-xyz", ready_timeout_s=1)
    with patch("benchmark.router_proc.shutil.which", return_value=None):
        with pytest.raises(RouterStartupError, match="not found on PATH"):
            async with RouterProcess(cfg):
                pass


@pytest.mark.asyncio
async def test_ready_timeout_after_launcher_exits(monkeypatch) -> None:
    """Launcher exits 0 but /ready never returns 200 → timeout error."""
    cfg = RouterProcessConfig(ready_timeout_s=1, stop_timeout_s=1)
    fake_proc = _FakeProc(exit_after=1, returncode=0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))

    with patch("benchmark.router_proc.shutil.which", return_value="/usr/bin/vllm-sr"), \
         patch("benchmark.router_proc.subprocess.Popen", return_value=fake_proc):
        with pytest.raises(RouterStartupError, match="not ready"):
            async with RouterProcess(cfg):
                pass


@pytest.mark.asyncio
async def test_stop_on_exit_invokes_vllm_sr_stop(monkeypatch) -> None:
    """When stop_on_exit=True, `vllm-sr stop` is run to tear down the stack."""
    cfg = RouterProcessConfig(ready_timeout_s=2, stop_timeout_s=1, stop_on_exit=True)
    fake_proc = _FakeProc(exit_after=1, returncode=0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))

    with patch("benchmark.router_proc.shutil.which", return_value="/usr/bin/vllm-sr"), \
         patch("benchmark.router_proc.subprocess.Popen", return_value=fake_proc), \
         patch("benchmark.router_proc.subprocess.run") as run_mock:
        async with RouterProcess(cfg):
            pass
        run_mock.assert_called_once()
        args, kwargs = run_mock.call_args
        assert args[0] == ["/usr/bin/vllm-sr", "stop"]


@pytest.mark.asyncio
async def test_stop_default_leaves_stack_running(monkeypatch) -> None:
    """Default (stop_on_exit=False) does NOT invoke `vllm-sr stop`."""
    cfg = RouterProcessConfig(ready_timeout_s=2, stop_timeout_s=1)
    fake_proc = _FakeProc(exit_after=1, returncode=0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))

    with patch("benchmark.router_proc.shutil.which", return_value="/usr/bin/vllm-sr"), \
         patch("benchmark.router_proc.subprocess.Popen", return_value=fake_proc), \
         patch("benchmark.router_proc.subprocess.run") as run_mock:
        async with RouterProcess(cfg):
            pass
        run_mock.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# ensure_router_stack — hash-based skip-or-restart
# ─────────────────────────────────────────────────────────────────────────

def _config(tmp_path) -> tuple[RouterProcessConfig, type]:
    """Builds a RouterProcessConfig + a fresh path to the (fake) router
    config file in tmp_path."""
    cfg = RouterProcessConfig(
        binary="vllm-sr",
        ready_timeout_s=5,
        apiserver_host="127.0.0.1", apiserver_port=8080,
        frontend_host="127.0.0.1", frontend_port=8899,
    )
    return cfg


@pytest.mark.asyncio
async def test_ensure_unchanged_when_hash_matches_and_healthy(tmp_path) -> None:
    """Fast path: hash file matches current config AND apiserver healthy →
    no restart, no spawn. Returns 'unchanged'."""
    cfg = _config(tmp_path)
    config_path = tmp_path / "router-config.yaml"
    config_path.write_text("identical: content")
    hash_file = tmp_path / ".router-config-hash"
    # Pre-seed the hash file with the right hash.
    import hashlib
    hash_file.write_text(hashlib.sha256(config_path.read_bytes()).hexdigest())

    calls: dict[str, int] = {"stop": 0, "spawn": 0}

    async def healthy(_c): return True
    def stop(_c): calls["stop"] += 1
    async def spawn(_c): calls["spawn"] += 1

    action = await ensure_router_stack(
        cfg, config_path, hash_file=hash_file,
        is_healthy=healthy, stop=stop, spawn_serve=spawn,
    )
    assert action == "unchanged"
    assert calls == {"stop": 0, "spawn": 0}


@pytest.mark.asyncio
async def test_ensure_restarts_when_hash_changed(tmp_path) -> None:
    """Hash mismatch + stack up → stop + spawn → 'restarted'. Hash file
    is updated to the new value."""
    cfg = _config(tmp_path)
    config_path = tmp_path / "router-config.yaml"
    config_path.write_text("new content")
    hash_file = tmp_path / ".router-config-hash"
    hash_file.write_text("0" * 64)  # stale hash

    calls: dict[str, int] = {"stop": 0, "spawn": 0}

    async def healthy(_c): return True
    def stop(_c): calls["stop"] += 1
    async def spawn(_c): calls["spawn"] += 1

    action = await ensure_router_stack(
        cfg, config_path, hash_file=hash_file,
        is_healthy=healthy, stop=stop, spawn_serve=spawn,
    )
    assert action == "restarted"
    assert calls == {"stop": 1, "spawn": 1}
    # Stored hash reflects the new content.
    import hashlib
    assert hash_file.read_text().strip() == \
        hashlib.sha256(config_path.read_bytes()).hexdigest()


@pytest.mark.asyncio
async def test_ensure_starts_when_stack_is_down(tmp_path) -> None:
    """Health check fails (stack not running) → spawn, no prior stop.
    Returns 'started'. Works whether or not a hash file exists."""
    cfg = _config(tmp_path)
    config_path = tmp_path / "router-config.yaml"
    config_path.write_text("anything")
    hash_file = tmp_path / ".router-config-hash"
    # No hash file → first-ever start.

    calls: dict[str, int] = {"stop": 0, "spawn": 0}

    async def healthy(_c): return False
    def stop(_c): calls["stop"] += 1
    async def spawn(_c): calls["spawn"] += 1

    action = await ensure_router_stack(
        cfg, config_path, hash_file=hash_file,
        is_healthy=healthy, stop=stop, spawn_serve=spawn,
    )
    assert action == "started"
    # No stop (stack wasn't running); spawn happened.
    assert calls == {"stop": 0, "spawn": 1}
    assert hash_file.exists()


@pytest.mark.asyncio
async def test_ensure_restarts_when_hash_match_but_stack_down(tmp_path) -> None:
    """Edge case: hash file says we last started with this config, but
    the user `docker stop`'d the containers manually. We must detect that
    via the health check and restart the stack, not trust the hash file."""
    cfg = _config(tmp_path)
    config_path = tmp_path / "router-config.yaml"
    config_path.write_text("the right content")
    hash_file = tmp_path / ".router-config-hash"
    import hashlib
    hash_file.write_text(hashlib.sha256(config_path.read_bytes()).hexdigest())

    calls: dict[str, int] = {"stop": 0, "spawn": 0}

    async def healthy(_c): return False  # user killed the containers
    def stop(_c): calls["stop"] += 1
    async def spawn(_c): calls["spawn"] += 1

    action = await ensure_router_stack(
        cfg, config_path, hash_file=hash_file,
        is_healthy=healthy, stop=stop, spawn_serve=spawn,
    )
    assert action == "started"
    # No stop needed (already down); spawn happened.
    assert calls == {"stop": 0, "spawn": 1}
