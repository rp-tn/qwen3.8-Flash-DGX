#!/usr/bin/env python3
"""fp8_e4m3 main KV cache for the Qwen3.8-Flash-Next QSA (patch 7).

After @Nanetnounou's original for the preview image (issue #6, vllm#54426), on
``vllm/models/qwen4_exp/nvidia``:

* vLLM already quantizes on the WRITE side: ``do_kv_cache_update`` goes through the generic
  cache-update path with the layer's ``_k_scale``/``_v_scale``, and ``get_kv_cache_spec`` passes
  ``kv_quant_mode``. The gaps are the READ side (the split-K Triton kernel loads the cache as
  bf16) and the guards that refuse anything but bf16.
* The kernel dequantizes with vLLM's own ``_cast_kv_tile`` (mode 1 = fp8 per-tensor), the same
  helper unified attention uses. It is compiled out when the cache is bf16 (``KV_QUANT_MODE`` is
  a constexpr), so ``--kv-cache-dtype auto`` behaves exactly like upstream.
* The READ uses the layer's real scales (the preview version read with a fixed 1.0, which only
  matched because the writes used 1.0 too).
* What v0.30 already covers and this patch leaves alone: the QSA *indexer* caches. The raw-key
  ring is fixed to bf16, and the compressed key cache has its own dtype knob
  (``attention_config.indexer_kv_dtype``, native fp8 since vllm#54890); they are untouched.
* sm_121 caps shared memory at 101376 bytes and ``_cast_kv_tile`` materializes an fp32 tile, so
  BLOCK_N is halved under quantization, at run time and in the warmup.

Usage: python3 patch_qsa_fp8_kv.py <site-packages>
"""
import ast
import sys

SP = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: patch_qsa_fp8_kv.py <site-packages>")
BASE = f"{SP}/vllm/models/qwen4_exp/nvidia"
OPS = f"{BASE}/ops/qsa.py"
OWNER = f"{BASE}/qsa.py"


def sub(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    assert n == 1, f"anchor '{what}' found {n} times (expected 1): upstream moved"
    return text.replace(old, new)


# ------------------------------------------------------------------------ ops/qsa.py
ops = open(OPS).read()

ops = sub(ops,
    "from vllm.triton_utils import HAS_TRITON, tl, triton\n",
    "from vllm.triton_utils import HAS_TRITON, tl, triton\n"
    "from vllm.v1.attention.ops.triton_unified_attention import _cast_kv_tile\n",
    "import of vLLM's dequant helper")

# Kernel signature: two scale pointers and the constexpr quant mode.
ops = sub(ops,
    "    num_requests,\n    TOPK: tl.constexpr,\n    PAGE_SIZE: tl.constexpr,\n    PAGE_TABLE_WIDTH: tl.constexpr,",
    "    num_requests,\n"
    "    k_scale_ptr,\n"
    "    v_scale_ptr,\n"
    "    KV_QUANT_MODE: tl.constexpr,\n"
    "    TOPK: tl.constexpr,\n    PAGE_SIZE: tl.constexpr,\n    PAGE_TABLE_WIDTH: tl.constexpr,",
    "split-K kernel signature")

# Dequantize between the loads and the first dot.
ops = sub(ops,
    "        scores = tl.dot(query, keys)\n"
    "        # Scaling scores avoids re-quantizing a scaled query to BF16.\n",
    "        keys = _cast_kv_tile(keys, query, k_scale_ptr, KV_QUANT_MODE)\n"
    "        values = _cast_kv_tile(values, query, v_scale_ptr, KV_QUANT_MODE)\n"
    "        scores = tl.dot(query, keys)\n"
    "        # Scaling scores avoids re-quantizing a scaled query to BF16.\n",
    "split-K kernel dequantization")

# Public wrapper: optional scales (None -> 1.0, a numerical no-op for unmodified callers).
ops = sub(ops,
    "    out: torch.Tensor | None = None,\n"
    "    *,\n"
    "    output_gate: torch.Tensor,\n"
    ") -> torch.Tensor:\n"
    '    """Run sparse GQA directly over paged BF16 K/V caches.',
    "    out: torch.Tensor | None = None,\n"
    "    *,\n"
    "    output_gate: torch.Tensor,\n"
    "    k_scale: torch.Tensor | None = None,\n"
    "    v_scale: torch.Tensor | None = None,\n"
    ") -> torch.Tensor:\n"
    '    """Run sparse GQA directly over paged BF16 or fp8_e4m3 K/V caches.',
    "wrapper signature")

# Dtype guard: bf16 query over a bf16 or fp8_e4m3 cache.
ops = sub(ops,
    "    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16\n",
    "    assert q.dtype == torch.bfloat16\n"
    "    assert k_cache.dtype == v_cache.dtype\n"
    "    # 1 = FP8_PER_TENSOR in KVQuantMode; the per-token-head modes are not wired here.\n"
    "    _kv_mode = 1 if k_cache.dtype == torch.float8_e4m3fn else 0\n"
    "    assert k_cache.dtype == torch.bfloat16 or _kv_mode, (\n"
    '        f"QSA: KV cache is {k_cache.dtype}, expected bf16 or fp8_e4m3"\n'
    "    )\n",
    "wrapper dtype guard")

# Block size under quantization (shared-memory cap on sm_121).
ops = sub(ops,
    "    block_n, partial_warps, num_tiles, num_splits = _select_config(\n"
    "        q.shape[0], k_cache.shape[2], use_prefill_config, selection_width\n"
    "    )\n",
    "    block_n, partial_warps, num_tiles, num_splits = _fp8_block_fit(\n"
    "        _select_config(q.shape[0], k_cache.shape[2], use_prefill_config, selection_width),\n"
    "        _kv_mode,\n"
    "        selection_width,\n"
    "    )\n"
    "    _one = torch.ones((), dtype=torch.float32, device=q.device)\n"
    "    _k_scale_t = k_scale if k_scale is not None else _one\n"
    "    _v_scale_t = v_scale if v_scale is not None else _one\n",
    "runtime config + scales")

# Launch: pass scales and mode (runtime launch: its last positional is block_table.shape[0]).
ops = sub(ops,
    "        k_cache.shape[0],\n"
    "        block_table.shape[0],\n"
    "        TOPK=selection_width,\n",
    "        k_cache.shape[0],\n"
    "        block_table.shape[0],\n"
    "        _k_scale_t,\n"
    "        _v_scale_t,\n"
    "        KV_QUANT_MODE=_kv_mode,\n"
    "        TOPK=selection_width,\n",
    "runtime launch")

# Warmup: the cache arrives as uint8 raw bytes when quantized; compile the fp8 specialization.
ops = sub(ops,
    "    head_dim = kv_cache.shape[-1] // 2\n"
    "    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)\n",
    "    if kv_cache.dtype == torch.uint8:\n"
    "        kv_cache = kv_cache.view(torch.float8_e4m3fn)\n"
    "    _kv_mode = 1 if kv_cache.dtype == torch.float8_e4m3fn else 0\n"
    "    head_dim = kv_cache.shape[-1] // 2\n"
    "    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)\n",
    "warmup cache view")

ops = sub(ops,
    "    profiles = {\n"
    "        _select_config(num_rows, num_kv_heads, use_prefill_config, selection_width)\n",
    "    profiles = {\n"
    "        _fp8_block_fit(\n"
    "            _select_config(num_rows, num_kv_heads, use_prefill_config, selection_width),\n"
    "            _kv_mode,\n"
    "            selection_width,\n"
    "        )\n",
    "warmup profiles")

ops = sub(ops,
    "            num_requests,\n"
    "            TOPK=selection_width,\n",
    "            num_requests,\n"
    "            TritonWarmupTensor(torch.float32, shape=()),\n"
    "            TritonWarmupTensor(torch.float32, shape=()),\n"
    "            KV_QUANT_MODE=_kv_mode,\n"
    "            TOPK=selection_width,\n",
    "warmup launch")

ops = sub(ops,
    "def qsa_sparse_paged_attention(\n",
    "def _fp8_block_fit(config, kv_mode: int, num_columns: int):\n"
    '    """Halve BLOCK_N under a quantized cache: _cast_kv_tile materializes an fp32 tile\n'
    "    (twice the bf16 shared memory for K and V) and sm_121 caps at 101376 bytes.\"\"\"\n"
    "    block_n, warps, num_tiles, num_splits = config\n"
    "    if kv_mode == 0:\n"
    "        return config\n"
    "    block_n = max(16, block_n // 2)\n"
    "    num_tiles = triton.cdiv(num_columns, block_n)\n"
    "    return block_n, warps, num_tiles, min(num_splits, num_tiles)\n"
    "\n\n"
    "def qsa_sparse_paged_attention(\n",
    "block-fit helper")

ast.parse(ops)
open(OPS, "w").write(ops)
print("ops/qsa.py: fp8_e4m3 read path (split-K kernel + warmup) added OK")

# --------------------------------------------------------------------------- qsa.py
owner = open(OWNER).read()
FP8 = '("auto", "bfloat16", "fp8", "fp8_e4m3")'

owner = sub(owner,
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]\n',
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
    '        "auto",\n        "bfloat16",\n        "fp8",\n        "fp8_e4m3",\n    ]\n'
    "\n"
    "    @classmethod\n"
    "    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:\n"
    "        # The parent asks whether FlashAttention's kernels read the dtype; QSA reads the\n"
    "        # cache with its own Triton kernel.\n"
    "        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes\n",
    "backend supported dtypes")

# Impl init: the parent FlashAttentionImpl refuses fp8 on this device because ITS kernels
# cannot read it; QSA does not use them. Neutralize the dtype for the parent init, restore it.
owner = sub(owner,
    "    def __init__(self, *args, **kwargs) -> None:\n"
    "        super().__init__(*args, **kwargs)\n"
    "        if not is_flash_attn_varlen_func_available():",
    "    def __init__(self, *args, **kwargs) -> None:\n"
    '        _fp8 = ("fp8", "fp8_e4m3")\n'
    "        _real_kv_dtype = None\n"
    '        if kwargs.get("kv_cache_dtype") in _fp8:\n'
    '            _real_kv_dtype = kwargs["kv_cache_dtype"]\n'
    '            kwargs["kv_cache_dtype"] = "auto"\n'
    "        elif len(args) > 6 and args[6] in _fp8:\n"
    "            _real_kv_dtype = args[6]\n"
    '            args = args[:6] + ("auto",) + args[7:]\n'
    "        super().__init__(*args, **kwargs)\n"
    "        if _real_kv_dtype is not None:\n"
    "            self.kv_cache_dtype = _real_kv_dtype\n"
    "        if not is_flash_attn_varlen_func_available():",
    "impl init (parent guard)")

owner = sub(owner,
    '        if self.kv_cache_dtype not in ("auto", "bfloat16"):\n'
    '            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")\n'
    "        self.supports_quant_query_input = False\n",
    f"        if self.kv_cache_dtype not in {FP8}:\n"
    "            raise NotImplementedError(\n"
    '                f"Qwen4Exp QSA: {self.kv_cache_dtype} is not supported (bf16 and fp8_e4m3 are)"\n'
    "            )\n"
    "        self.supports_quant_query_input = False\n",
    "impl dtype guard")

# forward_qsa: reinterpret the uint8 storage as fp8 (bit view, no copy; what triton_attn.py
# does), and pass the layer's real scales to the kernel.
owner = sub(owner,
    "        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)\n"
    "        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:\n"
    '            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")\n',
    "        if kv_cache.dtype == torch.uint8:\n"
    "            kv_cache = kv_cache.view(torch.float8_e4m3fn)\n"
    "        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)\n"
    "        if query.dtype != torch.bfloat16:\n"
    '            raise NotImplementedError("Qwen4Exp QSA requires a BF16 query")\n'
    "        if key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):\n"
    '            raise NotImplementedError(f"Qwen4Exp QSA: cache dtype {key_cache.dtype} is not supported")\n'
    "        _quantized = key_cache.dtype == torch.float8_e4m3fn\n",
    "forward_qsa cache view + guard")

owner = sub(owner,
    "            output[:num_tokens],\n"
    "            output_gate=output_gate[:num_tokens],\n"
    "        )\n"
    "        return output\n",
    "            output[:num_tokens],\n"
    "            output_gate=output_gate[:num_tokens],\n"
    '            k_scale=getattr(layer, "_k_scale", None) if _quantized else None,\n'
    '            v_scale=getattr(layer, "_v_scale", None) if _quantized else None,\n'
    "        )\n"
    "        return output\n",
    "forward_qsa scales")

owner = sub(owner,
    '        if cache_config.cache_dtype not in ("auto", "bfloat16"):\n'
    '            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")\n',
    f"        if cache_config.cache_dtype not in {FP8}:\n"
    "            raise NotImplementedError(\n"
    '                f"Qwen4Exp QSA: cache_dtype {cache_config.cache_dtype} is not supported "\n'
    '                "(bf16 and fp8_e4m3 are)"\n'
    "            )\n",
    "owner cache_config guard")

owner = sub(owner,
    "        if self.kv_cache_torch_dtype != torch.bfloat16:\n"
    '            raise NotImplementedError("Qwen4Exp QSA requires BF16 cache storage")\n',
    "        # vLLM allocates a quantized cache as uint8 raw bytes, reinterpreted in forward_qsa.\n"
    "        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8):\n"
    "            raise NotImplementedError(\n"
    '                f"Qwen4Exp QSA: storage dtype {self.kv_cache_torch_dtype} is not supported"\n'
    "            )\n",
    "owner storage guard")

# Left in place on purpose: `quant_config.kv_cache_scheme is not None` targets a KV scheme
# declared in the checkpoint (compressed-tensors), not --kv-cache-dtype.

ast.parse(owner)
open(OWNER, "w").write(owner)
print("qsa.py: fp8_e4m3 main KV declared supported, guards widened, real scales passed OK")
