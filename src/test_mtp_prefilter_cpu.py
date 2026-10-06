#!/usr/bin/env python3
"""CPU check for patch 18 (MTP name prefilter) against a real checkpoint; no GPU needed.

    docker run --rm -v "$PWD/src:/t" -w /t -v /path/to/huggingface_hub:/hub:ro \\
      --entrypoint python3 qwen38-flash-dgx:v0.30 test_mtp_prefilter_cpu.py /hub/models--.../snapshots/<rev>

Runs vLLM's safetensors iterator over every file of the snapshot inside the drafter's prefilter
and checks that it yields exactly the names the drafter's load_weights would keep from the
unfiltered stream (the checkpoint index names that _remap_mtp_weight_name maps), and that the
filter is off again afterwards.
"""
import glob
import json
import os
import sys
import time

import vllm.model_executor.model_loader.weight_utils as wu
from vllm.models.qwen4_exp.nvidia.mtp import _remap_mtp_weight_name

snap = sys.argv[1]
names = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
want = {n for n in names if _remap_mtp_weight_name(n) is not None}
files = sorted(glob.glob(os.path.join(snap, "*.safetensors")))

t = time.perf_counter()
with wu.qwen38_keep_only(lambda n: _remap_mtp_weight_name(n) is not None, "MTP (test)"):
    got = {n for n, _ in wu.safetensors_weights_iterator(files, use_tqdm_on_load=False)}
dt = time.perf_counter() - t
assert got == want, (sorted(want - got)[:5], sorted(got - want)[:5])
assert wu._qwen38_keep_name is None, "filter still active after the block"
assert wu._qwen38_keep_stats == [len(want), len(names) - len(want)], wu._qwen38_keep_stats
print(f"patch 18 OK: {len(got)} of {len(names)} tensors kept (the drafter's exact set), "
      f"{len(names) - len(got)} skipped before reading; {dt:.1f} s for {len(files)} files")
