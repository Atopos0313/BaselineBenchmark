# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare the local T-Head backend with the vendored official FLA forward."""

from __future__ import annotations

import math
import os
import statistics
import sys
from pathlib import Path

import pytest
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = next((
    path for path in (
        Path(os.environ["FLAGATTN_ROOT"]).resolve() if os.environ.get("FLAGATTN_ROOT") else None,
        *Path(__file__).resolve().parents,
        BENCHMARK_ROOT.parent / "FlagAttention-Log_linear",
        BENCHMARK_ROOT.parent / "FlagAttention",
    )
    if path is not None and (path / "src" / "flag_attn").is_dir()
), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Set FLAGATTN_ROOT to a FlagAttention checkout containing src/flag_attn")
FLA_ROOT = (
    BENCHMARK_ROOT / "backends" / "_thead"
    / "ops" / "FLA" / "log_linear_attn" / "flash-linear-attention-for-sail"
)
if not (FLA_ROOT / "fla" / "ops" / "log_linear_attn" / "chunk.py").is_file():
    raise FileNotFoundError(f"Official T-Head FLA source is missing: {FLA_ROOT}")
for source_dir in (PROJECT_ROOT, PROJECT_ROOT / "src", FLA_ROOT):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from benchmark.recording import benchmark_metric, record_benchmark_result  # noqa: E402
from fla.ops.log_linear_attn import chunk_log_linear_attn as official_fla  # noqa: E402
from flag_attn import log_linear_attn as project_backend  # noqa: E402
from flag_attn.testing import backend as test_backend  # noqa: E402

pytestmark = pytest.mark.log_linear_attn

BENCHMARK_SHAPES = [
    (1, 8192, 96, 128),
    (2, 16384, 16, 128),
    (4, 2048, 16, 128),
    (4, 4096, 64, 128),
    (8, 2048, 32, 256),
    (8, 1024, 8, 64),
]
BENCHMARK_ROUNDS = 7
BENCHMARK_WARMUP_MS = 1000
BENCHMARK_REP_MS = 100


def _make_inputs(batch, sequence, heads, dim):
    levels = math.ceil(math.log2(sequence)) + 1
    q = torch.randn(batch, sequence, 1, dim, device=test_backend.device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, heads, dim, device=test_backend.device, dtype=torch.bfloat16)
    g = -0.05 * torch.rand(batch, sequence, heads, device=test_backend.device, dtype=torch.float32)
    level_scales = torch.sigmoid(
        torch.randn(batch, sequence, heads, levels, device=test_backend.device, dtype=torch.float32)
    ).to(torch.bfloat16)
    return q, k, v, g, level_scales


@torch.inference_mode()
def run_benchmark(
    *, shapes=None, rounds=BENCHMARK_ROUNDS,
    warmup=BENCHMARK_WARMUP_MS, rep=BENCHMARK_REP_MS, record_property=None,
):
    if not test_backend.supports_operator("log_linear_attn") or not test_backend.has_specialization("log_linear_attn"):
        pytest.skip("The selected device has no FlagAttention Log-Linear backend")
    if rounds < 1 or warmup < 0 or rep <= 0:
        raise ValueError("rounds and rep must be positive; warmup must be nonnegative")

    metrics = []
    print(f"Device={test_backend.get_device_name()} dtype=BF16", flush=True)
    print(f"rounds={rounds}, warmup={warmup} ms, rep={rep} ms", flush=True)
    print("| Shape | Official FLA (ms) | Project backend (ms) | Speedup (FLA/project) | RMS relative error |", flush=True)
    print("| --- | ---: | ---: | ---: | ---: |", flush=True)
    for index, (batch, sequence, heads, dim) in enumerate(BENCHMARK_SHAPES if shapes is None else shapes):
        torch.manual_seed(0)
        inputs = _make_inputs(batch, sequence, heads, dim)

        def run_fla():
            return official_fla(*inputs, scale=1.0, output_final_state=False)[0]

        def run_project():
            return project_backend(*inputs)

        runners = {"fla": run_fla, "project": run_project}
        outputs = {name: run() for name, run in runners.items()}
        test_backend.device_fn.synchronize()
        for name, output in outputs.items():
            if not torch.isfinite(output).all():
                raise AssertionError(f"{name} produced NaN/Inf for {(batch, sequence, heads, dim)}")
        reference, actual = outputs["fla"].float(), outputs["project"].float()
        error = ((reference - actual).square().mean().sqrt()
                 / (reference.square().mean().sqrt() + 1e-8)).item()

        times = {}
        order = ("fla", "project") if index % 2 == 0 else ("project", "fla")
        for name in order:
            samples = [test_backend.do_bench(runners[name], warmup=warmup, rep=rep, return_mode="median")
                       for _ in range(rounds)]
            times[name] = statistics.median(samples)
        speedup = times["fla"] / times["project"]
        shape = f"B{batch}_T{sequence}_H{heads}_D{dim}"
        print(f"| {shape} | {times['fla']:.6f} | {times['project']:.6f} | {speedup:.3f}x | {error:.6f} |", flush=True)
        metrics.append(benchmark_metric(
            shape_detail={"B": batch, "T": sequence, "H": heads, "D": dim},
            latency_base=times["fla"], latency=times["project"], speedup=speedup,
            accuracy={"rms_relative": error}, measurement_scope="full_forward",
            candidate_source="FlagAttention _thead vs vendored official FLA",
        ))

    record_benchmark_result(
        record_property, op_name="log_linear_attn", dtype="bf16", result=metrics,
        mode="api", baseline="official_thead_fla", phase="forward",
        rounds=rounds, warmup_ms=warmup, rep_ms=rep,
    )
    return metrics


def test_log_linear_attn_benchmark(record_property):
    run_benchmark(record_property=record_property)


if __name__ == "__main__":
    run_benchmark()
