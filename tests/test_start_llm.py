"""`make start_LLM` reads the served model name from the resolved slot-1
model — meaning `.env`'s `TIER{N}_1_MODEL` is the single source for the
per-tier model name across both the docker launcher and the request
body. The tier YAML's `served_model_name` is a silent fallback.
"""
from __future__ import annotations

from pathlib import Path

from benchmark import start_llm
from benchmark.config import (
    BackendSpec,
    TierConfig,
    TierEndpoint,
    TierModel,
)


def _tier(level: int, *, served: str, model_dir: Path,
          slot_models: list[TierModel] | None = None,
          backend_kind: str = "docker_vllm_zendnn_single") -> TierConfig:
    t = TierConfig(
        name=f"tier{level}", level=level, specializations=["general"],
        router_alias=f"tier{level}", served_model_name=served,
        endpoint=TierEndpoint(url=f"http://localhost:800{level}/v1"),
        backend=BackendSpec(
            kind=backend_kind,
            image="vllm/test:latest",
            model_host_path=str(model_dir),
            host_port=8000 + level,
            cpus="0-31",
            mems="0",
            max_model_len=8192,
            max_num_seqs=8,
            dtype="bfloat16",
            block_size=32,
            kv_gb=24,
        ),
    )
    if slot_models is not None:
        t.models = slot_models
    return t


def _capture_cmd(monkeypatch):
    """Replace `_run` + `_stop_one` so the launcher records the docker
    command it WOULD execute without actually invoking docker."""
    cmds: list[list[str]] = []
    monkeypatch.setattr(start_llm, "_run", lambda cmd, *a, **kw: cmds.append(cmd))
    monkeypatch.setattr(start_llm, "_stop_one", lambda name, **kw: None)
    return cmds


def test_zendnn_single_uses_env_model_name_via_resolved_slot1(
    tmp_path: Path, monkeypatch
) -> None:
    """When .env populates slot 1 with a different name than the YAML,
    the docker launch must use the env name as `--served-model-name`."""
    cmds = _capture_cmd(monkeypatch)

    # YAML default = "tier1"; env-provided slot 1 = "Qwen3-1.7B" → env wins.
    tier = _tier(1, served="tier1", model_dir=tmp_path, slot_models=[
        TierModel(
            slot=1, url="http://localhost:8001/v1",
            served_model_name="Qwen3-1.7B",
        ),
    ])
    start_llm._start_docker_vllm_zendnn_single(tier)

    assert cmds, "no docker command issued"
    cmd = cmds[-1]
    assert "--served-model-name" in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "Qwen3-1.7B"


def test_zendnn_single_falls_back_to_yaml_when_no_env(
    tmp_path: Path, monkeypatch
) -> None:
    """No env slots → resolved_models() synthesizes from YAML and the
    launcher uses the YAML's served_model_name verbatim."""
    cmds = _capture_cmd(monkeypatch)

    tier = _tier(1, served="tier1", model_dir=tmp_path)  # no env slots
    start_llm._start_docker_vllm_zendnn_single(tier)

    cmd = cmds[-1]
    assert cmd[cmd.index("--served-model-name") + 1] == "tier1"


def test_dual_socket_uses_env_model_name(tmp_path: Path, monkeypatch) -> None:
    """Same behaviour for the dual-socket launcher (used by T2)."""
    cmds = _capture_cmd(monkeypatch)

    tier = _tier(
        2, served="tier2", model_dir=tmp_path,
        backend_kind="docker_vllm_dual_socket",
        slot_models=[
            TierModel(
                slot=1, url="http://localhost:8002/v1",
                served_model_name="Qwen3-30B-A3B",
            ),
        ],
    )
    # The dual_socket launcher reads `backend.replicas` for ports/cpus.
    tier.backend.replicas = [
        {"name": "r0", "cpus": "0-31", "mems": "0", "port": 8002},
    ]
    tier.backend.kv_gb_per_replica = 40
    start_llm._start_docker_vllm_dual_socket(tier)

    # Each replica gets its own docker command; all must use the env name.
    served = [
        cmd[cmd.index("--served-model-name") + 1]
        for cmd in cmds if "--served-model-name" in cmd
    ]
    assert served and all(s == "Qwen3-30B-A3B" for s in served)
