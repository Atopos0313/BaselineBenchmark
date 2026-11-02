# T-Head Wall Attention benchmark

The reference implementation is the t-head/flash-linear-attention-for-sail
Git submodule at ops/FLA/wall_attn/flash-linear-attention-for-sail, pinned to
ea54b7d30c9015bec8c0837f4b5c3d627d1f745f. The benchmark imports it
directly as the FLA baseline; it is not the candidate implementation.

Clone this branch with its pinned reference source:

    git clone --branch thead-wallattention --recurse-submodules https://github.com/Atopos0313/BaselineBenchmark.git

For an existing checkout, run git submodule update --init --recursive.

The candidate flag_attn package and benchmark.recording helper must be
available from a FlagAttention source checkout. From the BaselineBenchmark root:

    FLAGATTN_ROOT=/path/to/FlagAttention
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$FLAGATTN_ROOT/src:$FLAGATTN_ROOT" python -m pytest -q -s backends/_thead/benchmarks/test_parallel_wall_attn_benchmark.py

The script verifies the imported FLA source path and compares all measured
outputs with the candidate before reporting per-shape latency.
