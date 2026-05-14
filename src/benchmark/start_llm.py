"""Launch / stop local-CPU tier backends from per-tier YAML files.

`make start_LLM` invokes `start_local_tiers()`; `make stop_LLM` invokes
`stop_local_tiers()`. Each function iterates `config/tiers/*.yaml`, dispatches
on `backend.kind`, and runs the appropriate docker commands.

Supported kinds (extend as new runners are wired):
  - `docker_vllm_dual_socket`    : two NUMA-pinned vLLM replicas (e.g. T2 Qwen).
  - `docker_vllm_zendnn_single`  : one AMD ZenDNN/Zentorch vLLM replica
                                   on a single NUMA socket (e.g. T1 Qwen3-1.7B).
  - `remote`                     : someone else manages this endpoint; no-op.
  - `placeholder`                : not yet provisioned; no-op (with note).

Container names follow `vllm-{tier_name}-{replica_name}` so stop is a
straightforward filter. Stable names mean `start_LLM` is idempotent —
re-running just kills old containers and starts fresh.

Errors abort with a clear message rather than half-bring-up a stack.
"""
from __future__ import annotations

import contextlib
import shlex
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from .config import TierConfig, load_models

console = Console()

DEFAULT_TIERS_DIR = Path("config/tiers")


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, streaming output. Echoes the command for transparency."""
    console.print(f"[dim]$ {' '.join(shlex.quote(c) for c in cmd)}[/]")
    return subprocess.run(cmd, check=check)


def _container_name(tier_name: str, replica: str) -> str:
    return f"vllm-{tier_name}-{replica}"


def _running_container_names(prefix: str = "vllm-") -> list[str]:
    """List running container names matching the prefix."""
    out = subprocess.check_output(
        ["docker", "ps", "--format", "{{.Names}}", "--filter", f"name={prefix}"],
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def _stop_one(name: str, *, timeout: int = 30) -> None:
    # Suppress "no such container" — that's the success case for stop_LLM.
    with contextlib.suppress(subprocess.CalledProcessError):
        subprocess.run(
            ["docker", "stop", "-t", str(timeout), name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ---- backend dispatchers ----

def _start_docker_vllm_dual_socket(tier: TierConfig) -> None:
    """Launch two NUMA-pinned vLLM CPU containers for one tier."""
    # BackendSpec has extra='allow', so model_dump exposes kind-specific fields.
    extra = tier.backend.model_dump()
    image = extra.get("image")
    model_host_path = extra.get("model_host_path")
    hf_host_path = extra.get("hf_host_path")
    max_model_len = extra.get("max_model_len", 8192)
    block_size = extra.get("block_size", 32)
    dtype = extra.get("dtype", "bfloat16")
    kv_gb = extra.get("kv_gb_per_replica", 120)
    replicas = extra.get("replicas") or []

    if not image or not model_host_path:
        console.print(
            f"[red]error[/]: tier {tier.name} backend missing required fields "
            f"(image, model_host_path)"
        )
        sys.exit(2)

    model_dir = Path(model_host_path)
    if not model_dir.is_dir():
        console.print(
            f"[red]error[/]: tier {tier.name} model dir not found: {model_dir}\n"
            f"  download with:\n"
            f"    huggingface-cli download <repo> --local-dir {model_dir}"
        )
        sys.exit(2)

    if not replicas:
        console.print(
            f"[red]error[/]: tier {tier.name} has docker_vllm_dual_socket "
            f"but no `backend.replicas` list"
        )
        sys.exit(2)

    # Stop any previously-running replicas for this tier (idempotent).
    for r in replicas:
        _stop_one(_container_name(tier.name, r["name"]))

    for r in replicas:
        name = _container_name(tier.name, r["name"])
        cpus = str(r["cpus"])
        mems = str(r["mems"])
        port = int(r["port"])
        _run(
            [
                "docker", "run", "-d", "--rm",
                "--name", name,
                "--network", "host",
                "--shm-size", "4g",
                "--security-opt", "seccomp=unconfined",
                "--cap-add", "SYS_NICE",
                "--cpuset-cpus", cpus,
                "--cpuset-mems", mems,
                "-v", f"{model_host_path}:/models/served:ro",
                *(["-v", f"{hf_host_path}:/root/.cache/huggingface"] if hf_host_path else []),
                "-e", f"VLLM_CPU_KVCACHE_SPACE={kv_gb}",
                "-e", f"VLLM_CPU_OMP_THREADS_BIND={cpus}",
                "-e", "OMP_NUM_THREADS=32",
                image,
                "--model", "/models/served",
                "--dtype", str(dtype),
                "--max-model-len", str(max_model_len),
                "--trust-remote-code",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--served-model-name", tier.served_model_name,
                "--block-size", str(block_size),
            ]
        )
    console.print(
        f"[green]tier {tier.name}[/]: launched {len(replicas)} replica(s); "
        f"vllm cold load ~90-120s. Readiness probe: "
        f"curl -sf http://127.0.0.1:{replicas[0]['port']}/v1/models"
    )


def _stop_docker_vllm_dual_socket(tier: TierConfig) -> None:
    extra = tier.backend.model_dump()
    for r in extra.get("replicas") or []:
        name = _container_name(tier.name, r["name"])
        console.print(f"[dim]stopping {name}[/]")
        _stop_one(name)


def _cpuset_count(cpus_str: str) -> int:
    """Count CPUs in a docker cpuset string like '0-31' or '0,2,4-7'."""
    n = 0
    for part in cpus_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            n += b - a + 1
        elif part:
            n += 1
    return max(n, 1)


def _start_docker_vllm_zendnn_single(tier: TierConfig) -> None:
    """Launch one AMD ZenDNN/Zentorch vLLM CPU container on a single NUMA socket.

    Differs from dual_socket in three ways:
      • Single container, single replica.
      • AMD-specific ZenDNN tunings (ZENDNNL_MATMUL_ALGO, TORCHINDUCTOR_*).
      • Bridge networking with explicit `-p host:container` port mapping,
        rather than `--network host`. This lets multiple ZenDNN-style
        tiers coexist without colliding on container-internal port 8000.
    """
    extra = tier.backend.model_dump()
    image = extra.get("image")
    model_host_path = extra.get("model_host_path")
    cpus = str(extra.get("cpus", "0-31"))
    mems = str(extra.get("mems", "0"))
    host_port = int(extra.get("host_port", 8001))
    max_model_len = int(extra.get("max_model_len", 16384))
    max_num_seqs = int(extra.get("max_num_seqs", 8))
    dtype = str(extra.get("dtype", "bfloat16"))
    enable_prefix_caching = bool(extra.get("enable_prefix_caching", True))
    reasoning_parser = extra.get("reasoning_parser")  # optional, may be None
    kv_gb = int(extra.get("kv_gb", 64))

    if not image or not model_host_path:
        console.print(
            f"[red]error[/]: tier {tier.name} backend missing required fields "
            f"(image, model_host_path)"
        )
        sys.exit(2)

    model_dir = Path(model_host_path)
    if not model_dir.is_dir():
        console.print(
            f"[red]error[/]: tier {tier.name} model dir not found: {model_dir}\n"
            f"  download with:\n"
            f"    hf download <repo> --local-dir {model_dir}"
        )
        sys.exit(2)

    omp_threads = _cpuset_count(cpus)

    # Stop any previously-running replica (idempotent).
    name = _container_name(tier.name, "r0")
    _stop_one(name)

    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", name,
        "--cpuset-cpus", cpus,
        "--cpuset-mems", mems,
        "--shm-size", "4g",
        "--security-opt", "seccomp=unconfined",
        "--cap-add", "SYS_NICE",
        "-p", f"{host_port}:8000",
        "-v", f"{model_host_path}:/models/served:ro",
        # ZenDNN / Zentorch tunings — verbatim from the validated docker run.
        "-e", f"VLLM_CPU_KVCACHE_SPACE={kv_gb}",
        "-e", f"VLLM_CPU_OMP_THREADS_BIND={cpus}",
        "-e", f"OMP_NUM_THREADS={omp_threads}",
        "-e", "OMP_PROC_BIND=close",
        "-e", "OMP_PLACES=cores",
        "-e", "TORCHINDUCTOR_FREEZING=1",
        "-e", "VLLM_USE_AOT_COMPILE=0",
        "-e", "TORCHINDUCTOR_AUTOGRAD_CACHE=0",
        "-e", "ZENDNNL_MATMUL_ALGO=1",
        image,
        "vllm", "serve", "/models/served",
        "--served-model-name", tier.served_model_name,
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
    ]
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", str(reasoning_parser)])
    cmd.extend(["--host", "0.0.0.0", "--port", "8000"])

    _run(cmd)
    console.print(
        f"[green]tier {tier.name}[/]: launched ZenDNN single-socket replica on "
        f"host port {host_port}; vllm cold load ~30-90s. Readiness probe: "
        f"curl -sf http://127.0.0.1:{host_port}/v1/models"
    )


def _stop_docker_vllm_zendnn_single(tier: TierConfig) -> None:
    name = _container_name(tier.name, "r0")
    console.print(f"[dim]stopping {name}[/]")
    _stop_one(name)


# ---- entry points ----

_STARTERS: dict[str, callable] = {
    "docker_vllm_dual_socket": _start_docker_vllm_dual_socket,
    "docker_vllm_zendnn_single": _start_docker_vllm_zendnn_single,
}
_STOPPERS: dict[str, callable] = {
    "docker_vllm_dual_socket": _stop_docker_vllm_dual_socket,
    "docker_vllm_zendnn_single": _stop_docker_vllm_zendnn_single,
}


def start_local_tiers(tiers_dir: Path = DEFAULT_TIERS_DIR) -> None:
    """Start every tier whose backend.kind has a registered starter."""
    models = load_models(tiers_dir)
    for tier in sorted(models.tiers, key=lambda t: t.level):
        kind = tier.backend.kind
        starter = _STARTERS.get(kind)
        if starter is None:
            console.print(f"[dim]tier {tier.name}: backend.kind={kind!r} — nothing to start[/]")
            continue
        starter(tier)


def stop_local_tiers(tiers_dir: Path = DEFAULT_TIERS_DIR) -> None:
    """Stop every tier whose backend.kind has a registered stopper."""
    models = load_models(tiers_dir)
    for tier in sorted(models.tiers, key=lambda t: t.level):
        kind = tier.backend.kind
        stopper = _STOPPERS.get(kind)
        if stopper is None:
            continue
        stopper(tier)
    # Belt-and-suspenders: stop anything else matching our prefix that
    # wasn't covered by a per-tier stopper (e.g. stale containers from a
    # previous YAML config).
    leftover = [n for n in _running_container_names("vllm-")
                if n.startswith("vllm-tier")]
    for name in leftover:
        console.print(f"[dim]stopping leftover {name}[/]")
        _stop_one(name)
