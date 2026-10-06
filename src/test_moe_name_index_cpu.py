#!/usr/bin/env python3
"""CPU unit test for patch_moe_name_index.py: the indexed candidates must be exactly the mapping
entries the original loop in RoutedExperts.load_weights visits, in the same order (no GPU needed).

    docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx:v0.30 test_moe_name_index_cpu.py
"""
import os
import time

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

RE = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py"
src = open(RE).read()
ns = {"os": os}
exec(src[src.index("# qwen38-moe-name-index-begin"):src.index("# qwen38-moe-name-index-end")], ns)  # noqa: S102
index_of, candidates = ns["_qwen38_expert_index"], ns["_qwen38_expert_candidates"]


def original(mapping, qual_name, is_fused):
    out, matched = [], False
    for entry in mapping:
        if entry[1] not in qual_name:
            if matched and is_fused:
                break
            continue
        matched = True
        out.append(entry)
    return out


def names(prefix, n_experts, projs):
    for e in list(range(n_experts)) + [n_experts + 3]:
        for p in projs:
            for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale", "weight_scale_inv", "bias"):
                yield f"{prefix}.experts.{e}.{p}.{suffix}", False
    for p in ("gate_up_proj", "down_proj", "w13", "w2"):
        yield f"{prefix}.experts.{p}", True
        yield f"{prefix}.experts.{p}.weight", True
        yield f"{prefix}.experts.3.{p}.weight", False
    yield f"{prefix}.gate.weight", False
    yield f"{prefix}.shared_expert.gate_proj.weight", False
    yield f"{prefix}.experts.experts.1.gate_proj.weight", False
    yield f"{prefix}.experts.1.gate_proj.base_layer.weight", False


configs = [
    dict(ckpt=("gate_proj", "down_proj", "up_proj"), n=512, redundant=0, lora=""),
    dict(ckpt=("gate_proj", "down_proj", "up_proj"), n=64, redundant=8, lora=""),
    dict(ckpt=("w1", "w2", "w3"), n=16, redundant=0, lora=""),
    dict(ckpt=("gate_proj", "down_proj", "up_proj"), n=16, redundant=0, lora="base_layer."),
    dict(ckpt=("gate", "down", "up"), n=8, redundant=0, lora=""),  # no fused entries
]
checked = 0
for c in configs:
    g, d, u = c["ckpt"]
    mapping = RoutedExperts.build_expert_params_mapping(
        g, d, u, num_experts=c["n"], num_redundant_experts=c["redundant"],
        lora_base_layer_prefix=c["lora"], include_fused=True)
    index = index_of(mapping)
    assert index is not None, "every entry should start with 'experts.'"
    for q, fused in names("language_model.model.layers.7.mlp", c["n"], (g, d, u)):
        for is_fused in (fused, not fused):
            want, got = original(mapping, q, is_fused), list(candidates(index, mapping, q, is_fused))
            assert got == want, (c, q, is_fused, got[:4], want[:4])
            checked += 1

odd = [("p", "experts.0.gate_proj.", 0, "w1"), ("p", "mlp.experts.1.up_proj.", 1, "w3")]
assert index_of(odd) is None and candidates(None, odd, "x", False) is odd, "must fall back to the full scan"

mapping = RoutedExperts.build_expert_params_mapping("gate_proj", "down_proj", "up_proj", num_experts=512, include_fused=True)
index = index_of(mapping)
qs = [f"language_model.model.layers.7.mlp.experts.{e}.{p}.weight" for e in range(512) for p in ("gate_proj", "up_proj", "down_proj")]
t = time.perf_counter(); [original(mapping, q, False) for q in qs]; t_orig = (time.perf_counter() - t) / len(qs)
t = time.perf_counter(); [candidates(index, mapping, q, False) for q in qs]; t_new = (time.perf_counter() - t) / len(qs)
print(f"MoE name index OK: {checked} name/config cases identical to the original loop; "
      f"{t_orig * 1e6:.1f} -> {t_new * 1e6:.1f} us per tensor ({len(mapping)} mapping entries)")
