"""`make start_LLM` is a tier-agnostic dispatcher: walk every env slot
whose URL hits localhost, look it up in `config/local_models.yaml`,
match the per-CPU-vendor block, exec the verbatim argv with
{port}/{served_name}/{container_name} filled in.

These tests inject a fake recipe library and a fake `run` function so
no docker / no actual subprocess is invoked. The real `_run` is exercised
indirectly by the live `make start_LLM` workflow on the dev server.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmark import start_llm
from benchmark.config import ModelsConfig, TierConfig, TierModel


def _tier(level: int, models: list[TierModel]) -> TierConfig:
    return TierConfig(
        name=f"tier{level}", level=level, specializations=["general"],
        router_alias=f"tier{level}", models=models,
    )


@pytest.fixture
def library_file(tmp_path: Path) -> Path:
    """A minimal local-models library with one recipe per CPU vendor."""
    p = tmp_path / "local_models.yaml"
    p.write_text(yaml.safe_dump({
        "Qwen3-1.7B": {
            "description": "Qwen 3 1.7B small dense.",
            "container_name": "vllm-{served_name}",
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "amd": {
                "start": [
                    "docker", "run", "-d", "--name", "{container_name}",
                    "-p", "{port}:8000",
                    "amd-image",
                    "--served-model-name", "{served_name}",
                ],
                "stop": ["docker", "stop", "{container_name}"],
            },
            "intel": {
                "start": [
                    "docker", "run", "-d", "--name", "{container_name}",
                    "-p", "{port}:8000",
                    "intel-image",
                    "--served-model-name", "{served_name}",
                ],
                "stop": ["docker", "stop", "{container_name}"],
            },
        },
        "Qwen3-30B-A3B": {
            "description": "Qwen 3 30B MoE.",
            "container_name": "vllm-{served_name}",
            "amd": {
                "start": [
                    "docker", "run", "-d", "--name", "{container_name}",
                    "--network", "host",
                    "amd-moe-image",
                    "--port", "{port}",
                    "--served-model-name", "{served_name}",
                ],
                "stop": ["docker", "stop", "{container_name}"],
            },
        },
    }))
    return p


def _capture_runs(monkeypatch) -> list[list[str]]:
    """Record every command the launcher would issue."""
    runs: list[list[str]] = []
    monkeypatch.setattr(start_llm, "_run", lambda cmd, *a, **kw: runs.append(cmd))
    return runs


def _patch_models(monkeypatch, models: ModelsConfig) -> None:
    """Stub out load_models so the dispatcher gets exactly our tiers."""
    monkeypatch.setattr(start_llm, "load_models", lambda *_a, **_kw: models)


def test_dispatches_only_localhost_slots(monkeypatch, library_file) -> None:
    """Vendor slots (non-localhost) are ignored. Localhost slots match
    a recipe, and the per-vendor argv runs with placeholders filled in."""
    runs = _capture_runs(monkeypatch)
    models = ModelsConfig(tiers=[
        _tier(1, [
            TierModel(
                slot=1, url="http://localhost:8001/v1",
                served_model_name="Qwen3-1.7B",
            ),
            TierModel(  # vendor slot — must be skipped
                slot=2, url="https://api.openai.com/v1",
                served_model_name="gpt-4o-mini",
            ),
        ]),
    ])
    _patch_models(monkeypatch, models)

    start_llm.start_local_engines(library_path=library_file, vendor="amd")

    # Two runs: stop (idempotent cleanup) + start. Vendor slot ignored.
    assert len(runs) == 2
    stop_cmd, start_cmd = runs
    assert stop_cmd == ["docker", "stop", "vllm-Qwen3-1.7B"]
    assert "--port" not in start_cmd  # this recipe has -p (host:cont), not --port
    assert "vllm-Qwen3-1.7B" in start_cmd
    assert "8001:8000" in start_cmd  # {port} from URL
    assert "Qwen3-1.7B" in start_cmd  # {served_name}


def test_vendor_picks_matching_block(monkeypatch, library_file) -> None:
    """AMD on AMD picks the amd block; Intel picks the intel block."""
    runs = _capture_runs(monkeypatch)
    _patch_models(monkeypatch, ModelsConfig(tiers=[
        _tier(1, [TierModel(
            slot=1, url="http://localhost:8001/v1",
            served_model_name="Qwen3-1.7B",
        )]),
    ]))
    start_llm.start_local_engines(library_path=library_file, vendor="intel")
    # The start command should contain the Intel image.
    assert any("intel-image" in cmd for cmd in runs)
    assert not any("amd-image" in cmd for cmd in runs)


def test_unknown_model_raises_with_known_list(monkeypatch, library_file) -> None:
    """An env slot referencing a model the library doesn't know about
    must fail loud — and the error should list the known recipes so the
    operator can either add a recipe or fix the .env."""
    _capture_runs(monkeypatch)
    _patch_models(monkeypatch, ModelsConfig(tiers=[
        _tier(1, [TierModel(
            slot=1, url="http://localhost:8001/v1",
            served_model_name="NotInTheLibrary",
        )]),
    ]))
    with pytest.raises(SystemExit, match="no recipe for 'NotInTheLibrary'"):
        start_llm.start_local_engines(library_path=library_file, vendor="amd")


def test_vendor_block_missing_raises(monkeypatch, library_file) -> None:
    """Qwen3-30B-A3B only has an `amd:` block in the fixture library.
    Running on Intel must fail explicitly rather than silently skip."""
    _capture_runs(monkeypatch)
    _patch_models(monkeypatch, ModelsConfig(tiers=[
        _tier(2, [TierModel(
            slot=1, url="http://localhost:8002/v1",
            served_model_name="Qwen3-30B-A3B",
        )]),
    ]))
    with pytest.raises(ValueError, match="no 'intel' launcher block"):
        start_llm.start_local_engines(library_path=library_file, vendor="intel")


def test_idempotent_pre_stop_tolerates_nonexistent_container(
    monkeypatch, library_file
) -> None:
    """The pre-stop before `docker run` must tolerate failure (the
    container may not exist yet — `docker stop` returns non-zero in
    that case, which we treat as 'good, nothing to clean up'). The
    actual `docker run` start command, in contrast, must be check=True
    so a real launch failure surfaces."""
    captured: list[tuple[list[str], bool]] = []

    def fake(cmd, *a, **kw):
        captured.append((cmd, kw.get("check", True)))

    monkeypatch.setattr(start_llm, "_run", fake)
    _patch_models(monkeypatch, ModelsConfig(tiers=[
        _tier(1, [TierModel(
            slot=1, url="http://localhost:8001/v1",
            served_model_name="Qwen3-1.7B",
        )]),
    ]))
    start_llm.start_local_engines(library_path=library_file, vendor="amd")

    # Two calls: stop then start. Stop must be check=False; start check=True.
    assert len(captured) == 2
    stop_cmd, stop_check = captured[0]
    start_cmd, start_check = captured[1]
    assert stop_cmd[:2] == ["docker", "stop"]
    assert stop_check is False, "pre-stop must tolerate nonexistent container"
    assert start_cmd[:2] == ["docker", "run"]
    assert start_check is True, "the actual launch should fail loud"


def test_stop_local_engines_tolerates_already_stopped(
    monkeypatch, library_file
) -> None:
    """`make stop_LLM` against an already-stopped container is a no-op
    success, not an error."""
    captured: list[tuple[list[str], bool]] = []

    def fake(cmd, *a, **kw):
        captured.append((cmd, kw.get("check", True)))

    monkeypatch.setattr(start_llm, "_run", fake)
    _patch_models(monkeypatch, ModelsConfig(tiers=[
        _tier(1, [TierModel(
            slot=1, url="http://localhost:8001/v1",
            served_model_name="Qwen3-1.7B",
        )]),
    ]))
    start_llm.stop_local_engines(library_path=library_file, vendor="amd")

    assert len(captured) == 1
    cmd, check = captured[0]
    assert cmd[:2] == ["docker", "stop"]
    assert check is False, "stop_local_engines must tolerate nonexistent containers"


def test_stop_runs_recipe_stop_commands(monkeypatch, library_file) -> None:
    runs = _capture_runs(monkeypatch)
    _patch_models(monkeypatch, ModelsConfig(tiers=[
        _tier(1, [TierModel(
            slot=1, url="http://localhost:8001/v1",
            served_model_name="Qwen3-1.7B",
        )]),
    ]))
    start_llm.stop_local_engines(library_path=library_file, vendor="amd")
    assert runs == [["docker", "stop", "vllm-Qwen3-1.7B"]]


def test_dedupes_same_model_on_same_port(monkeypatch, library_file) -> None:
    """If the same (served_name, port) appears in two tiers, launch once."""
    runs = _capture_runs(monkeypatch)
    common = TierModel(
        slot=1, url="http://localhost:8001/v1",
        served_model_name="Qwen3-1.7B",
    )
    _patch_models(monkeypatch, ModelsConfig(tiers=[
        _tier(1, [common]),
        _tier(2, [common]),
    ]))
    start_llm.start_local_engines(library_path=library_file, vendor="amd")
    # One stop + one start, not two stops + two starts.
    assert len(runs) == 2


def test_dispatcher_skips_when_no_local_slots(monkeypatch, library_file) -> None:
    """A config with only vendor URLs runs nothing — clean exit, no error."""
    runs = _capture_runs(monkeypatch)
    _patch_models(monkeypatch, ModelsConfig(tiers=[
        _tier(3, [TierModel(
            slot=1, url="https://api.openai.com/v1",
            served_model_name="gpt-5-mini",
        )]),
    ]))
    start_llm.start_local_engines(library_path=library_file, vendor="amd")
    assert runs == []


def test_recipe_argv_uses_unknown_placeholder_raises(
    monkeypatch, tmp_path: Path
) -> None:
    """A typo in a placeholder name (e.g. {portt}) must be caught with a
    clear error rather than emit a partially-rendered argv."""
    bad = tmp_path / "local_models.yaml"
    bad.write_text(yaml.safe_dump({
        "Qwen3-1.7B": {
            "amd": {
                "start": ["docker", "run", "-p", "{portt}:8000", "img"],
                "stop": ["docker", "stop", "x"],
            },
        },
    }))
    _capture_runs(monkeypatch)
    _patch_models(monkeypatch, ModelsConfig(tiers=[
        _tier(1, [TierModel(
            slot=1, url="http://localhost:8001/v1",
            served_model_name="Qwen3-1.7B",
        )]),
    ]))
    with pytest.raises(ValueError, match="unknown placeholder"):
        start_llm.start_local_engines(library_path=bad, vendor="amd")


# ---- helper: detect_cpu_vendor (parsing /proc/cpuinfo) ----

def test_detect_cpu_vendor_amd(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "cpuinfo"
    fake.write_text("processor\t: 0\nvendor_id\t: AuthenticAMD\n")
    monkeypatch.setattr(start_llm, "Path", lambda p: fake if p == "/proc/cpuinfo" else Path(p))
    assert start_llm.detect_cpu_vendor() == "amd"


def test_detect_cpu_vendor_intel(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "cpuinfo"
    fake.write_text("processor\t: 0\nvendor_id\t: GenuineIntel\n")
    monkeypatch.setattr(start_llm, "Path", lambda p: fake if p == "/proc/cpuinfo" else Path(p))
    assert start_llm.detect_cpu_vendor() == "intel"
