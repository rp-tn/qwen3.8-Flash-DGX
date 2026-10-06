#!/usr/bin/env python3
"""Read small checkpoint tensors with pread instead of mmap views (VLLM_LOAD_PREAD=1, default on).

vLLM's safetensors_weights_iterator yields mmap views; every consumer then page-faults the file
pages in on its first touch (patch 14's clone, the linear layers' copy_). In the loader's own
order, pread of each tensor into ordinary memory runs at ~10.9 GiB/s against ~2.2 GiB/s for the
mmap view + clone (cold client cache, docs/HOW-IT-WORKS.md). Tensors above 64 MiB stay
mmap views, and the PLE table (ngram_embedding.shard_*) is never read here: vllm_ple_mmap serves it
from its own memmap and drops what the loader hands it. The storage of every pread tensor is
tagged, so patch 14 skips its now redundant clone. Needs patch 14 applied first."""
import ast
import sys

SP = sys.argv[1]
WU = f"{SP}/vllm/model_executor/model_loader/weight_utils.py"
RE = f"{SP}/vllm/model_executor/layers/fused_moe/routed_experts.py"

s = open(WU).read()
old = ('        else:\n'
       '            with safe_open(st_file, framework="pt") as f:\n'
       '                for name in f.keys():  # noqa: SIM118\n'
       '                    if should_skip_weight(name, local_expert_ids):\n'
       '                        continue\n'
       '                    param = f.get_tensor(name)\n'
       '                    yield name, param\n')
assert s.count(old) == 1, f"safetensors_weights_iterator default branch: expected 1, found {s.count(old)}"
s = s.replace(old, (
       '        else:\n'
       '            with safe_open(st_file, framework="pt") as f, _qwen38_preader(st_file) as _q38_read:\n'
       '                for name in f.keys():  # noqa: SIM118\n'
       '                    if should_skip_weight(name, local_expert_ids):\n'
       '                        continue\n'
       '                    param = _q38_read(name)\n'
       '                    if param is None:\n'
       '                        param = f.get_tensor(name)\n'
       '                    yield name, param\n'))
s += '''

# --- qwen38-flash-dgx: pread small tensors instead of mmap views (VLLM_LOAD_PREAD=1) ---
import contextlib as _qwen38_contextlib
import json as _qwen38_json
import os as _qwen38_os
import struct as _qwen38_struct

_QWEN38_LOAD_PREAD = _qwen38_os.environ.get("VLLM_LOAD_PREAD", "1") != "0"
_QWEN38_PREAD_MAX = 64 << 20
_QWEN38_PREAD_SKIP = ("ngram_embedding.shard_",)
_QWEN38_ST_DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16,
    "I32": torch.int32, "I64": torch.int64, "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


def _qwen38_pread_exact(fd: int, n: int, offset: int, out=None):
    buf = memoryview(out) if out is not None else memoryview(bytearray(n))
    got = 0
    while got < n:
        k = _qwen38_os.preadv(fd, [buf[got:]], offset + got)
        if k <= 0:
            raise OSError(f"short read at offset {offset + got}")
        got += k
    return buf


@_qwen38_contextlib.contextmanager
def _qwen38_preader(path: str):
    if not _QWEN38_LOAD_PREAD:
        yield lambda name: None
        return
    fd = _qwen38_os.open(path, _qwen38_os.O_RDONLY)
    stats = [0, 0, 0]  # tensors read, bytes read, left as mmap views
    try:
        n = _qwen38_struct.unpack("<Q", bytes(_qwen38_pread_exact(fd, 8, 0)))[0]
        header = _qwen38_json.loads(bytes(_qwen38_pread_exact(fd, n, 8)))
        base = 8 + n

        def read(name: str):
            meta = header.get(name)
            dtype = _QWEN38_ST_DTYPES.get(meta["dtype"]) if isinstance(meta, dict) else None
            if dtype is None or any(p in name for p in _QWEN38_PREAD_SKIP):
                stats[2] += 1
                return None
            start, end = meta["data_offsets"]
            nbytes = end - start
            numel = 1
            for d in meta["shape"]:
                numel *= d
            if nbytes > _QWEN38_PREAD_MAX or nbytes != numel * dtype.itemsize:
                stats[2] += 1
                return None
            buf = torch.empty(nbytes, dtype=torch.uint8)
            if nbytes:
                _qwen38_pread_exact(fd, nbytes, base + start, buf.numpy())
            buf.untyped_storage()._qwen38_anon = True
            stats[0] += 1
            stats[1] += nbytes
            return buf.view(dtype).reshape(meta["shape"])

        logger.info_once("qwen38 pread loader: tensors <= %d MiB read with pread (VLLM_LOAD_PREAD=0 disables)",
                         _QWEN38_PREAD_MAX >> 20)
        yield read
    finally:
        _qwen38_os.close(fd)
        if stats[2]:
            logger.info("qwen38 pread loader: %s: %d tensors (%.2f GiB) read, %d left as mmap views",
                        _qwen38_os.path.basename(path), stats[0], stats[1] / 2**30, stats[2])
'''
ast.parse(s)
open(WU, "w").write(s)

r = open(RE).read()
assert "def _qwen38_h2d_src(" in r, "patch 14 (patch_moe_load_clone.py) must be applied first"
r += '''

# --- qwen38-flash-dgx: patch 15 already read these into ordinary memory, skip patch 14's clone ---
_qwen38_h2d_src_clone = _qwen38_h2d_src


def _qwen38_h2d_src(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:  # noqa: F811
    if getattr(src.untyped_storage(), "_qwen38_anon", False):
        return src
    return _qwen38_h2d_src_clone(dst, src)
'''
ast.parse(r)
open(RE, "w").write(r)
print("weight_utils.py: pread loader applied OK; routed_experts.py: clone skip for pread tensors OK")
