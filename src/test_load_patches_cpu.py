#!/usr/bin/env python3
"""CPU checks for patches 15 (pread loader) and 17 (chunked embedding copy); no GPU needed.

    docker run --rm -v "$PWD/src:/t" -w /t -v /path/to/snapshot:/snap:ro \\
      --entrypoint python3 qwen38-flash-dgx:v0.30 test_load_patches_cpu.py /snap/model-00010-of-00010.safetensors ...

Patch 15: for each file, the default iterator with VLLM_LOAD_PREAD on must yield the same names,
dtypes, shapes and bytes as with it off; tensors <= 64 MiB come back tagged (read into ordinary
memory), larger ones and every PLE shard come back as untagged mmap views.
Patch 17: the chunked copy reproduces a plain copy for row counts that do not divide the chunk.
"""
import os
import sys

import torch

import vllm.model_executor.layers.vocab_parallel_embedding as vpe
import vllm.model_executor.model_loader.weight_utils as wu

for path in sys.argv[1:]:
    wu._QWEN38_LOAD_PREAD = False
    ref = dict(wu.safetensors_weights_iterator([path], use_tqdm_on_load=False))
    wu._QWEN38_LOAD_PREAD = True
    n = tagged = big = ple = 0
    for name, t in wu.safetensors_weights_iterator([path], use_tqdm_on_load=False):
        r = ref.pop(name)
        is_tagged = getattr(t.untyped_storage(), "_qwen38_anon", False)
        nbytes = t.numel() * t.element_size()
        if "ngram_embedding.shard_" in name:
            assert not is_tagged, f"PLE shard was read: {name}"
            ple += 1
        elif nbytes > wu._QWEN38_PREAD_MAX:
            assert not is_tagged, f"large tensor was read: {name}"
            big += 1
        else:
            assert is_tagged, f"small tensor not read with pread: {name}"
            assert t.dtype == r.dtype and t.shape == r.shape, (name, t.dtype, r.dtype, t.shape, r.shape)
            assert torch.equal(t.view(torch.uint8) if t.dim() else t.reshape(1).view(torch.uint8),
                               r.view(torch.uint8) if r.dim() else r.reshape(1).view(torch.uint8)), name
            tagged += 1
        n += 1
    assert not ref, f"names missing with pread on: {list(ref)[:3]}"
    print(f"patch 15 OK {os.path.basename(path)}: {n} tensors, {tagged} read + identical, "
          f"{big} large views, {ple} PLE shard views")

vpe._QWEN38_CHUNK_BYTES = 1000
for rows, cols in ((1, 7), (37, 11), (250, 40), (4096, 3)):
    src = torch.randn(rows, cols).to(torch.bfloat16)
    dst = torch.zeros_like(src)
    vpe._qwen38_copy_in_chunks(dst, src)
    assert torch.equal(dst, src), (rows, cols)
print("patch 17 OK: chunked copy == plain copy")
