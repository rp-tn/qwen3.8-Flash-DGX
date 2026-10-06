#!/usr/bin/env python3
"""MTP drafter: skip non-MTP tensors before they are read (VLLM_MTP_NAME_PREFILTER=1, default on).

The drafter is loaded as a second model from the same checkpoint. Its loader walks all 299,845
tensors of the 11 files and the drafter's load_weights keeps the ~3,100 whose name
_remap_mtp_weight_name maps, discarding the rest after they were created; with patch 15 that
means after they were read (~70 GiB). vLLM's safetensors iterator already asks
weight_utils.should_skip_weight(name, ...) before reading each tensor; this wraps it with an
optional keep-filter, and the drafter's load_weights sets it to its own name function while it
loads. The kept names are exactly the ones the drafter would have kept (same function, same names:
the primary checkpoint source has an empty prefix; with secondary sources the filter is not used).
src/test_mtp_prefilter_cpu.py checks the kept set against the checkpoint index."""
import ast
import os
import re
import sys

SP = sys.argv[1]
WU = f"{SP}/vllm/model_executor/model_loader/weight_utils.py"
MTP = f"{SP}/vllm/models/qwen4_exp/nvidia/mtp.py"
assert os.path.exists(MTP), f"mtp.py not found: {MTP}"

s = open(WU).read()
assert "from vllm.model_executor.model_loader.ep_weight_filter import (\n    should_skip_weight,\n)" in s
assert s.count("should_skip_weight(name, local_expert_ids)") >= 2
s += '''

# --- qwen38-flash-dgx: optional keep-filter in front of should_skip_weight (patch 18) ---
import contextlib as _qwen38_pf_contextlib

_qwen38_keep_name = None
_qwen38_keep_stats = [0, 0]  # kept, skipped
_qwen38_should_skip_weight = should_skip_weight


def should_skip_weight(weight_name, local_expert_ids):  # noqa: F811
    keep = _qwen38_keep_name
    if keep is not None:
        if not keep(weight_name):
            _qwen38_keep_stats[1] += 1
            return True
        _qwen38_keep_stats[0] += 1
    return _qwen38_should_skip_weight(weight_name, local_expert_ids)


@_qwen38_pf_contextlib.contextmanager
def qwen38_keep_only(keep, what: str):
    global _qwen38_keep_name
    prev, _qwen38_keep_name = _qwen38_keep_name, keep
    _qwen38_keep_stats[:] = [0, 0]
    try:
        yield
    finally:
        _qwen38_keep_name = prev
        logger.info("qwen38 %s name prefilter: %d tensors kept, %d skipped before reading "
                    "(VLLM_MTP_NAME_PREFILTER=0 disables)", what, *_qwen38_keep_stats)
'''
ast.parse(s)
open(WU, "w").write(s)

m = open(MTP).read()
# v0.30 passes mapper=mapper; it only drops names after the remap, so the kept set is unchanged.
ret = re.compile(r"^        return (loader\.load_weights\(remap_weight_names\(\)(?:, mapper=mapper)?\))\n", re.M)
found = ret.findall(m)
assert len(found) == 1, f"MTP load_weights return: expected 1, found {len(found)}"
call = found[0]
m = ret.sub(lambda _: (
    "        if _qwen38_os_pf.environ.get(\"VLLM_MTP_NAME_PREFILTER\", \"1\") == \"0\" or getattr(\n"
    "            self, \"secondary_weights\", ()\n"
    "        ):\n"
    f"            return {call}\n"
    "        from vllm.model_executor.model_loader import weight_utils as _q38_wu\n"
    "\n"
    "        with _q38_wu.qwen38_keep_only(lambda n: _remap_mtp_weight_name(n) is not None, \"MTP\"):\n"
    f"            return {call}\n"), m)
anchor = "def _remap_mtp_weight_name(name: str) -> str | None:\n"
assert m.count(anchor) == 1, "_remap_mtp_weight_name not found"
m = m.replace(anchor, "import os as _qwen38_os_pf  # qwen38-flash-dgx patch 18\n\n\n" + anchor)
ast.parse(m)
open(MTP, "w").write(m)
print("weight_utils.py + mtp.py: MTP name prefilter applied OK")
