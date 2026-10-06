# How it works

## The memory problem

Qwen3.8-Flash-Next is a sparse MoE with an unusual extra component: a **51B-parameter
n-gram embedding table** (the paper calls it PLE / "Engram"). The NVFP4
checkpoint breaks down roughly as:

| Component | Format | Size |
|---|---|---|
| Routed experts (48 layers × 512 experts, 10 active) | NVFP4 | ~63 GiB |
| Attention / GDN / QSA / shared experts / gate / lm_head / MTP | bf16 | ~15 GiB |
| **N-gram (PLE) table** — 16 heads × 20M rows × 160 dims | FP8 e4m3 + 1 scale | **~48 GiB** |
| **Total** | | **~126 GiB** |

A DGX Spark has **128 GB unified memory**, of which ~10 GiB is OS/driver/Docker. So
126 GiB of weights leaves essentially nothing for the KV cache — you cannot serve.

vLLM ships an offload path (`VLLM_PLE_CPU_OFFLOAD=1`) that moves the table to pinned
**host** RAM. On a discrete-GPU server that frees VRAM. On a Spark, host and device
are the **same physical pool**, so it frees nothing. That is why, until now, the only
thing that ran Flash-Next on a Spark was a llama.cpp GGUF — which mmaps its weights by
default, but has no sparse-attention kernel and so has poor prefill and no MTP.

## The lever

The table is a **lookup**, not compute. Per token the model reads exactly
**16 rows × 160 bytes = 2.5 KB**, at hashed (random) addresses. Even a 20k-token
prefill is ~320k row reads ≈ 1.3 GB — under a second on NVMe — and natural language
and code hit a very concentrated set of n-grams, so the hot rows stay in the page
cache after the first pass.

So the table does not need to be resident. This repo `mmap`s the checkpoint's
`model-plefp8-*.safetensors` shards and gathers rows on demand. That is exactly what
llama.cpp does with its GGUF — we just bring it to the vLLM path, which keeps the real
QSA/GDN kernels and MTP.

Result: **~76 GiB resident** (78 GiB of non-table weights minus a little), leaving
~20–22 GiB for KV at `GPU_MEM=0.85` — a 720–790k-token pool, i.e. ~3× concurrency at
the native 262k or a single 500k request with YaRN.

## The patch (`src/vllm_ple_mmap.py`)

Enabled by `VLLM_PLE_MMAP=1`; a complete no-op otherwise. It patches exactly one
class, `Qwen4ExpNGramEmbedding` (`vllm/models/qwen4_exp/nvidia/ngram_embedding.py`), in
three small ways:

1. **`__init__`** — run the stock constructor with the resident embedding classes
   (`Qwen4ExpPLEDeviceEmbedding`, and `Qwen4ExpPLEPinnedHostEmbedding` for engram CPU
   offload) swapped for a tiny placeholder. No large parameter is ever allocated. The
   placeholder's lookup gathers rows from `np.memmap` views of the shards (dedup + sort for
   locality, a thread pool so page faults overlap) and returns an fp8 tensor on the GPU; it
   also answers what the layer asks of its embedding (`dequantize`, `supports_prefetch`, a
   dummy `weight` for the init log line).

2. **`load_weights`** — drop the 128 shard tensors on the floor (they're served from
   disk) and keep only the global FP8 `weight_scale`, which the placeholder's `dequantize`
   applies. Then open the memmaps.

3. **`forward`** — keep the stock n-gram hashing (a Triton kernel, `compute_ngram_ids`) and
   route the lookup through a custom op, `vllm::ple_mmap_lookup_ids`. This is the crucial
   bit for GB10 (below).

Everything else — the n-gram hashing, the short-conv, the dequant, the sparse
attention — is stock vLLM.

## Three GB10 bugs this works around

Bringing the model up on a real Spark with real weights surfaced three issues. All are
handled by the patch + the flags in `scripts/serve.sh`:

1. **The gather cannot live inside a CUDA graph.** It is CPU work plus a pageable
   host→device copy (`Cannot copy between CPU and CUDA tensors during CUDA graph capture`, or
   `cudaErrorStreamCaptureUnsupported` for the synchronize). vLLM v0.30 captures this model
   with *breakable* piecewise CUDA graphs, so the lookup ends a graph segment itself and runs
   eagerly between two segments — details in [vLLM v0.30 specifics](#vllm-v030-specifics).
   Never `FULL*` capture. `--enforce-eager` also avoids it but is slower.

2. **`KeyError` on the layer registry during capture.** The custom op looks the layer
   up by name; registering it in the forward pass fails because a graph replay does not
   re-run that Python line. Fix: register in `__init__`.

3. **Prefix caching crashed** (`CUBLAS_STATUS_INTERNAL_ERROR` in a GDN `in_proj` GEMM, later
   `illegal memory access` in the Mamba state copy) on the cached-block path. Unrelated to
   this patch but required to run: the root cause is a vLLM block-size bug, fixed in this
   image — see [Prefix caching](#prefix-caching-the-root-cause-and-the-fix) below. The old
   advice (`--no-enable-prefix-caching`) is no longer needed.

## Long context: what works and what does not

Measured on the GX10 with the mmap patch, `GPU_MEM=0.85`, MTP=2 unless noted.

| Config | Result |
|---|---|
| 262144 native, MTP | KV pool ~720–790k tokens, ~3× concurrency at full length. Baseline prod. |
| **YaRN, CTX 500000, MTP** | **Works.** Pool ~724k tokens. Needle-in-a-haystack found at 276k and 414k tokens; decode 25–28 tok/s typical (36 on predictable text, ~94% draft acceptance there); no OOM. **This is the validated ceiling.** |
| YaRN, CTX 800000, MTP, `GPU_MEM=0.875` | Boots (pool 928k) and answers, but a 300k-token prefill got **SIGTERM from earlyoom** at 1.96% free memory: the prefill's activation peak plus the draft do not fit in ~5 GiB of headroom. |
| `--kv-cache-dtype fp8_e4m3` | Refused by the stock model (`QSA requires a BF16 main KV cache`); enabled by @Nanetnounou's patch in this image — see [fp8 KV cache](#fp8-kv-cache-on-the-qsa-path-opt-in). In bf16 a single 1M request needs 26.3 GiB of KV (28 KB/token, of which the attention K/V is the only part fp8 halves). |

Two YaRN-specific traps, both handled by `scripts/serve.sh`:

- YaRN is applied with Qwen's published `--hf-overrides` (rope `yarn`, factor 4,
  `original_max_position_embeddings` 262144) and needs `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.
- **YaRN + MTP fails to boot** with `--mamba-block-size can only be set with
  --enable-prefix-caching`. Cause: dict `hf_overrides` are not propagated to the draft
  model (`SpeculativeConfig.compose_draft_hf_overrides` only forwards callables), so the
  draft keeps `max_model_len=262144` while sharing the `cache_config`, whose
  `mamba_block_size` was auto-set to the target's `max_model_len`. Fix: put
  `"max_model_len": <CTX>` inside `--speculative-config`, which overrides the draft's
  length (`_maybe_override_draft_max_model_len`).

## Correctness

`src/test_ple_mmap_cpu.py` builds synthetic FP8 shards (with the real safetensors
layout and non-trivial data offsets) and checks the mmap gather bit-for-bit against a
reference `table[ids]`, including dedup, multi-shard spans, the fp8 view path used by
the placeholder, and out-of-range → `IndexError`. It needs only numpy+torch (no GPU):

```bash
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_ple_mmap_cpu.py
```

End-to-end, the served model is coherent ("The capital of France is Paris."), which is
the real test that the FP8 rows are being gathered and dequantized correctly — a wrong
gather turns the n-gram contribution to noise and the model degrades immediately.

## Performance notes

- **Prefill** ~2,400–2,660 tok/s (ctx 32k, single request). This is the axis that
  matters most versus llama.cpp (~540 tok/s), because Flash-Next's QSA prefill kernels
  only exist in vLLM/SGLang.
- **Decode** ~17 tok/s without speculation; with `MTP=2` **25–28 tok/s** on free-form
  prose (~63% draft acceptance) and up to ~36 tok/s on predictable text (~94%). The gather does one host↔device sync per decode step, which is pure
  latency at batch 1; MTP amortizes it. Removing that sync (staging ids through a
  pinned buffer, or a small resident hot-row cache) is the obvious next optimization.
- **First request into a cold region** of the table pays some NVMe I/O; it smooths out
  as the page cache warms. This makes single-shot prefill measurements cache-state
  dependent: the same prompt can run 2–3× slower on the first pass than once the rows it
  touches are resident, and the lower `GPU_MEM` is, the more RAM the page cache keeps for
  the 48 GiB table. Benchmark prefill on a second pass (or after `PREWARM=1`, which
  streams the whole table once at boot, ~10 s) and say which one you are quoting.

## Contributed GB10 fixes and the faster gather

Three changes from [@Saren-Arterius](https://github.com/Saren-Arterius)'s fork
([qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)),
merged in and measured A/B on the GX10 (same flags, YaRN 500k, MTP=2, greedy, real
prompts — no `ignore_eos`, which pushes this model into a degenerate post-EOS regime
and makes decode numbers meaningless):

| | before | after |
|---|---|---|
| Decode, 400-token answers (median of 6) | 21.8 tok/s | **26.2 tok/s** (+20%; 25–29 on a warm server) |
| Prefill 8k (warm) | 2,289 tok/s | **~2,570 tok/s** (+12%) |
| Prefill 32k | 2,316 tok/s | 2,418 tok/s (+4%) |
| Needle at 92k tokens | found, 47.1 s | found, 44.7 s |

1. **FLA shared-memory gate.** sm_121 reports 99 KiB of shared memory per block; the
   flash-linear-attention gate (`ops/utils.py`, `DEFAULT = 102400`) asks for 100 KiB, so
   all 36 GDN layers silently ran the small-tile kernels. Lowering the constant to
   101376 lets the GB10 take the big-tile path. This is the same fix the
   Qwen3.5-122B Spark recipe carried as `patch_fla_shmem.py`.
2. **`chunk_delta_h` `num_warps=2` pin** — [fla#953](https://github.com/fla-org/flash-linear-attention/issues/953),
   a `tl.dot` race on Blackwell with `num_warps=4`. A correctness fix; no speed effect expected.
3. **PLE gather hot path** in `src/vllm_ple_mmap.py`: dedup row ids on CPU (`np.unique`),
   gather only unique rows, stage them through a persistent pinned buffer with an async
   H2D copy, expand on the GPU via the inverse index; decode-sized gathers (≤
   `VLLM_PLE_MMAP_FAST_ROWS` unique rows, module default 512) can skip the thread pool. Also
   bf16/f16 tables, `VLLM_PLE_MMAP_DIR`, and a periodic `PLE mmap stats` line — which shows
   where the remaining decode cost is: on that inline path ~6.5 ms of the ~9.5 ms per lookup
   is the disk gather itself (the page cache holds only part of the 48 GiB table at
   `GPU_MEM=0.80`). Those misses are why `serve.sh` now sets the threshold to 0 (`FAST_ROWS`):
   on the pool their page faults overlap instead of queueing on one thread, measured at +8%
   decode at 1 stream and +17% aggregate at 4 streams with the same rows gathered.

## Independent reproduction and the native offload path

[@jschmied](https://github.com/jschmied) reproduced this recipe on a DGX Spark
([issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1)). They also
ran vLLM's native `VLLM_PLE_CPU_OFFLOAD=1` path and documented what it needs on the
NVFP4 checkpoint (the `Fp8Config` gate in `_get_ple_embedding_quant_method`, and
`CAP_SYS_PTRACE` because `yama.ptrace_scope=1` blocks the sibling-process
`pidfd_getfd` used for the CUDA-IPC handoff), plus concurrency traces showing
aggregate throughput of ~267 tok/s at 48 streams with page-fault cost per token
*falling* with batch size. Full notes:
<https://github.com/jschmied/qwen38-flash-next-gb10>.

## Upstream references

- vLLM recipe: <https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next>
- vLLM PR (Flash-Next support): <https://github.com/vllm-project/vllm/pull/53896>
- vLLM v0.29.0 release (first official build with the model, as `qwen4_exp`): <https://github.com/vllm-project/vllm/releases/tag/v0.29.0> — port notes in [HISTORY.md](HISTORY.md#the-vllm-v0290-port-dockerfilev029)
- vLLM v0.30.0 release (the base image): <https://github.com/vllm-project/vllm/releases/tag/v0.30.0> — see [vLLM v0.30 specifics](#vllm-v030-specifics)
- NVFP4 checkpoints: <https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4> (the default since 2026-09-14) and <https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4>
- SGLang day-0 write-up (PLE offload mechanics): <https://www.lmsys.org/blog/2026-08-26-qwen-flash-next>

## Prefix caching: the root cause and the fix

With `--enable-prefix-caching` vLLM puts this hybrid model's Mamba-style layers (the 36
GDN layers and the PLE short-conv) in cache mode `align`: their recurrent state is
captured at every `mamba_block_size` boundary (1600 tokens here — vLLM also raises the
attention block to 1600 so both page sizes agree) and restored when a later request
hits a cached prefix.

What we saw on GB10, in order:

1. Stock image: `CUDA illegal memory access` on the first batch of cached requests.
2. With [vllm#50729](https://github.com/vllm-project/vllm/pull/50729) (a genuine fix
   for an overlapping-copy race) — same crash.
3. With @Saren-Arterius's bounds guard on top: no crash, but 40–80 "out-of-range block
   id" skips per ~100 requests, and greedy outputs that **changed on cache hits** (an
   answer became an empty completion, or the reverse).
4. Ruling things out one at a time (all with the deterministic top-k, so any difference
   is real): `--mamba-ssm-cache-dtype float32`, `MTP=0`, `--no-async-scheduling`,
   `--mamba-cache-mode all` — every variant produced the *same* wrong outputs. So it was
   structural, not numerical.
5. Instrumenting the two state-restore sites (checksums of the GDN layer-0 state and the
   PLE conv state, with the block index and `has_initial_state`) gave the answer in one
   probe. Cold request, 5150 tokens: state after 5136 tokens = `4.443284220e3`. Cache hit
   at 3200 tokens: `has_initial_state=True`, restored state checksum **`0.000`**, from a
   block slot that had never been written.

Steps 2–3 describe the state *before* the fix below. The bounds guard stayed in the image as a
safety net (a skipped copy plus a counter instead of a dead CUDA context); since the block_size
fix its counter has been 0 on every run, tournaments included, and a non-zero count would now
mean a new bug, not this one.

The bug: `EngineCore._initialize_kv_caches` (`vllm/v1/engine/core.py`) sets
`cache_config.block_size = min(group.block_size for every KV group)`. For this model
one of the groups is the QSA raw-key ring (`CircularBufferSpec`, `qsa_cache.py`), whose
block is its ring capacity `compress_ratio × cdiv(compress_ratio + num_spec, compress_ratio)`
= 8 tokens with MTP=2, 4 without. (This group type does not exist on upstream vLLM `main`,
where the same `min()` is harmless because every group is 1600.) `cache_config.mamba_block_size` stays 1600, but two consumers used
`cache_config.block_size` *as* the Mamba block size:

- `v1/worker/gpu/model_states/mamba_hybrid.py`, `add_request`: the running state slot
  is seeded from `(num_computed_tokens - 1) // block_size` → `3199 // 8 = 399` instead
  of `1`. Column 399 is past the end of the request's Mamba block-table row, the
  persistent table is zero-filled there, block id 0 is the null block — in range, so the
  guard never fires — and its all-zero page is copied in as the "restored" state.
- `v1/core/sched/scheduler.py`, `_mamba_block_aligned_split`: prefill chunks were
  aligned to 8-token boundaries instead of 1600, so states were almost never captured
  at a real boundary and cold requests rarely cached anything. This is directly visible
  in the traces: a 5,150-token cold prompt stopped its chunk at 5,136 with MTP=2
  (`5150 - 5150 % 8 - 8`, the Eagle back-off) and at 5,148 with MTP=0 (`5150 - 5150 % 4`)
  — exactly the ring block sizes, never 4,800.

The fix (`src/patch_mamba_block_size.py`) is two lines: use
`cache_config.mamba_block_size` in the first and the scheduler's own `self.block_size`
(the LCM of all group block sizes, 1600) in the second. Validation on the same probe:
hit restores `4.149867426e3` = exactly the state the cold run wrote at 3200; first-token
top-5 logprobs identical to the 4th decimal between cold and two hits on 4k/8k/15k/31k
prompts; 32/32 greedy completions identical cold vs hit; guard counter 0 through 100
concurrent multi-turn requests; tournament 45/51 with caching vs 44–45 without.

Note that in single-process mode (`UniProcExecutor`, the default on one GPU) the
scheduler and the worker share the same `vllm_config` object, so both halves of the bug
apply; with the multiprocess executor only the scheduler half would. We will upstream
this.

## Exact top-k for the sparse attention

QSA scores every query against compressed key blocks and keeps the top `k` (512–2048
tokens' worth). vLLM does that with `torch.ops._C.persistent_topk`, a histogram-based
approximate select. On GB10 the faster `cooperative_topk` is disabled
(`not is_device_capability_family(120)`), so `persistent_topk` runs for prefill *and*
decode.

@k3dani ([issue #3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3),
[vllm#51782](https://github.com/vllm-project/vllm/issues/51782)) showed that the kernel
drops legitimate candidates when more than 16384 logits share a coarse histogram bin —
which trained indexer logits do — and that its output differs between launches. We
confirmed both: with the stock kernel, 3 identical greedy runs of the same prompt gave
3 different outputs on 2 of 4 prompts (2.6k–128k tokens), and first-token top-5 sets
that didn't even overlap.

`src/patch_qsa_exact_topk.py` adds `VLLM_QSA_EXACT_TOPK=1`: mask the columns the
scoring kernel never wrote (≥ `visible_blocks[row]`, they are `torch.empty`) to `-inf`
in place, then `torch.topk` over the row. Results: 4/4 prompts stable, first-token
logprobs identical to the 4th decimal across runs, tournament score unchanged (44/51 →
44/51 on NVFP4; 45/51 with prefix caching). Cost on the GX10: decode unchanged, prefill
−8% at 8k, −20–40% at 32k+ (the top-k runs over the full visible width per chunk).
Masking the uninitialized columns and keeping the stock kernel (`VLLM_QSA_EXACT_TOPK=fill`)
does **not** restore determinism, so the kernel itself is the problem, not the garbage.

### The kernel-side fix (patch 8, `VLLM_QSA_DET_TOPK=1`)

vLLM's `persistent_topk` hands out output slots with `atomicAdd` (thread-arrival order) and
takes exact-key ties at the last radix round first-come; when more elements share the
threshold key than fit its candidate buffers, the selected *set* changes too. Since the
sparse attention sums the selected keys in output order, either forks the hidden state.
[@jschmied](https://github.com/jschmied)'s rewrite ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122))
makes every single-CTA row go through a radix select that rescans the row per key byte (no
candidate buffers, exact pivot) followed by an index-ordered block scan, and gives the
multi-CTA path a deterministic emission (per-CTA counts + prefix over CTAs, ties ranked by
index). Micro-benchmark cost is 1.3–4× per call, which at model level is noise.

We build it as a standalone extension (`_C_det.so`) with the image's `nvcc` at `docker build`
time, from his repo at a pinned commit, and route the QSA indexer's `_topk` to
`torch.ops._C_det.persistent_topk` when `VLLM_QSA_DET_TOPK=1`. Measured on the GX10 (hybrid,
MTP=2, prefix caching, same box, same bench script and prompts; the exact and stock columns are the earlier runs from the sections above):

| | stock kernel | exact `torch.topk` | **deterministic kernel** |
|---|---|---|---|
| Deterministic (4 prompts × 3, first-token logprobs) | no | yes (0.000) | **yes (0.000)** |
| Decode | 32.4 tok/s | 30.8 | **32.5** |
| Prefill 8k | ~1,650 | 1,476 | **2,436** |
| Prefill 32k | 2,105 | 1,794 | **2,904** |
| Needle 92k | 45 s | 69 s | **48 s** |

With the 2026-09-07 pin (PR #10: signed-zero canonicalisation, deterministic low-shared-memory
path, launcher shared-memory fix, faster kernel) the same bench gives decode 31.8 tok/s, prefill
2,488 / 2,996 tok/s, needle 46.3 s, still 4/4 deterministic; his suite is now 210 cases and the
previous pin fails one of them with a hard launch error at 64 rows × 40k+ columns.

His standalone `test_det.py` (177 cases: bit-identical across calls, equal to an exact
reference, adversarial tie populations around every buffer size the original kernels used)
passes 177/177 on the GX10, and the stock op fails to reproduce itself on the same inputs.

## The MTP drafter's vocabulary, and `MADV_RANDOM` on the table

Two changes taken as ideas from MiaAI-Lab's recipe and reimplemented here.

**Reduced draft vocabulary.** `llm_base_proposer._maybe_share_lm_head` gives the MTP draft the
target's `lm_head`, so each draft step is a (B × 2560) · (2560 × 248,320) bf16 GEMV: 1.27 GiB
read per drafted token, twice per step at MTP=2, on a decode step that is bandwidth-bound.
`src/patch_mtp_draft_vocab.py` wraps `Qwen4ExpMTP.compute_logits`: on first call it
slices the shared head to the ids in `VLLM_MTP_DRAFT_VOCAB` (a private 65,536 × 2560 copy,
320 MiB), then each call computes the reduced logits and scatters them into a full-width tensor
filled with −∞, so `argmax`, the rejection sampler and `VocabMapping` see the usual shape. The
target's head is never touched. Correctness argument: with greedy drafting the rejection sampler
accepts a drafted token with probability p_target(token) and otherwise resamples from the target
with that token removed, which reproduces the target distribution exactly for *any* draft; a
smaller vocabulary only changes which token is drafted. The tournament confirmed it (45/51, the
best run), and acceptance moved 75% → 68%.

The id set: tokens of a local corpus by frequency, then the lowest ids (Qwen's BPE vocabulary is
in merge order, a frequency proxy), plus all special/added tokens and the 256 byte fallbacks —
`tools/build_draft_vocab.py`. On a French document 6.6% of tokens sit above the 65,536 cut, which
is the acceptance loss you should expect on prose in a language the corpus does not cover; a
98,304-id set recovers part of it for 160 MiB more per step (not separable from 65,536 in our
6-run bench).

**`MADV_RANDOM`.** Row lookups are 160-byte reads at hashed addresses. Without the advice the
kernel's mmap readahead pulls a window of pages around every faulting row and fills the page
cache with neighbours that are never used. `np.memmap` exposes the underlying `mmap`, and one
`madvise(MADV_RANDOM)` per shard (`VmFlags: rr` in `/proc/<pid>/smaps`) makes a cold row cost one
page. Measured: cold 8k prefill 3.41–4.37 s → 3.26–3.33 s, 32k 13.7–14.6 s → 13.2 s, and the KV
pool grew by ~28k tokens because less cache was resident when vLLM profiled memory. `PREWARM`
reads the file through a separate descriptor and is unaffected.

## Hybrid mode: NVFP4 experts + blockwise-fp8 side layers

Both NVFP4 checkpoints (NVIDIA's and RadixArk's) quantize only the routed experts (ModelOpt NVFP4) and leave
the dense side layers — GDN `in_proj`/`out_proj`, QSA `q/k/v/o_proj`, shared experts,
~15 GiB — in bf16. Every decoded token reads all of them, so they set the decode
bandwidth floor. `scripts/prepare-hybrid.sh` rewrites those 300 tensors as blockwise
fp8-e4m3 with a per-128×128 fp32 `weight_scale_inv` (the DeepSeek-V3 layout that vLLM's
`Fp8LinearMethod` already loads), in a sibling snapshot directory made of relative
symlinks — only the 4 rewritten shards are real files.

Serving them needs a small dispatch shim (`src/vllm_fp8_hybrid_modelopt.py`,
`VLLM_FP8_HYBRID=1`), a port of @Saren-Arterius's int4+fp8 dispatch to the ModelOpt
config: it scans the safetensors metadata for `F8_E4M3` weights that have a
`weight_scale_inv` sibling and, for those (fused) modules, returns vLLM's blockwise-fp8
linear method instead of the bf16 path, leaving the NVFP4 MoE path alone. One
model-specific wrinkle: `qsa.py` builds the QSA `qkv_proj` with
`without_modelopt_fp4(quant_config)` — i.e. with **no** quant config at all — so the
shim is never consulted there and loading dies on `'QKVParallelLinear' object has no
attribute 'data'`. The Dockerfile redirects that call to a proxy config that dispatches
to fp8 when the checkpoint has fp8 q/k/v for that layer and to bf16 otherwise.

Measured on the GX10 (YaRN 500k, MTP=2, exact top-k, prefix caching, greedy, real
prompts):

| | NVFP4 | hybrid | Intel int4 + fp8 (fork) |
|---|---|---|---|
| Tournament (17 agentic scenarios × 3, ok/51) | 45 | 45 | 44 |
| Deterministic at T=0 | yes | yes | **no** (also with Marlin atomic adds off) |
| Decode, 400-token answers (median of 6) | 25.7 tok/s | **30.8 tok/s** | 34.3 tok/s |
| Prefill 32k | ~1,900 tok/s | ~1,800 tok/s | ~1,900 tok/s |
| Needle at 92k | 64 s | 69 s | 69 s |
| TTFT, 2nd+ turn, 8 concurrent 20k-token conversations | 5.9 s | **4.1 s** | 8.7 s |
| KV cache (`GPU_MEM=0.80`) | 582k tokens | **633k** | 762k |
| Resident weights | ~84 GiB | ~77 GiB | ~70 GiB |

The six failed tournament passes are the same two scenarios (`b6_reconcile`,
`c5_inventory_reconcile`) in every configuration we have ever run, including the
unquantized-side-layer NVFP4 without any of our changes — they are the model, not the
quantization. Before we raised the harness's turn budget the hybrid lost a few extra
passes by spending one more tool call (it checks balances before acting), which is a
behavioural nuance rather than a precision loss; every tool call it made was correct.

The Intel AutoRound variant is fastest at raw decode but could not be made
deterministic and has the worst cached-TTFT (its prefill is the slowest), so we did not
adopt it; the numbers are here for completeness.

### The NVFP4 MTP draft graft (`MODE=hybrid-mtp`)

The hybrid keeps the checkpoint's MTP draft head as published: routed experts **fused
BF16** (`mtp.layers.0.mlp.experts.{down_proj,gate_up_proj}`, shapes `(512,1280,2560)` and
`(512,2560,640)`, ~4.7 GiB). The draft is read on every speculation step, so it is the
same bandwidth story as the side layers — but it is also 3 GiB of card that spec decoding
pays for in KV. [Inferact's](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)
checkpoint of the same base quantizes the draft experts per-expert NVFP4
(6,144 tensors: `…experts.{E}.{gate_proj,up_proj,down_proj}.{weight,weight_scale,
weight_scale_2,input_scale}`, 1.4 GiB). `scripts/prepare-mtp-graft.sh` grafts that block
onto the `-fp8hybrid` snapshot (see the README section for the mechanics and the two
exclude-list traps). Mechanically the swap is safe because the engine's
`RoutedExperts.load_weights` builds its mapping with `include_fused=True` and dispatches
per tensor on rank (3D = fused, 2D = per-expert), so both layouts load through one code
path; the draft's quant config is resolved from the checkpoint's own `hf_quant_config.json`,
renumbered by vLLM (`mtp.layers.0` → `mtp.layers.48` on the engine side) and matched with
substring rules, which is why the 29 explicit module names work in both spellings.

Three properties hold, all verified on the GX10:

- **The target model is untouched.** Its loader skips everything under `mtp.`
  (`skip_substrs=["mtp."]`), so the 6,144 new tensors never reach it; the fp8 dispatch
  still detects exactly the same 300 side layers.
- **Every emitted token is the target's argmax.** Greedy verification accepts a draft
  token only when it matches the target's argmax, so the output text is a property of the
  target alone. `scripts/greedy-probe.sh` against the BF16-MTP arm and the graft produces
  byte-identical text (5/5 prompts × 400 tokens, first-token logprobs identical to 4
  decimals) — a silently mispaired drafter would fail this.
- **Acceptance stays in the same band** (~2.6 vs ~2.7 mean acceptance length at MTP=2):
  the NVFP4 draft proposes marginally differently, rejected proposals cost nothing extra,
  and the decode win comes from the ~4x smaller draft reads.

Measured (same box, same defaults as the table above, MTP=2): weights 77.83 → 74.75 GiB,
KV pool 625,669 → 734,292 tokens (+17%), greedy decode +20–27% on the 5-prompt probe.
One caveat inherited from the graft design: the graft directory holds symlinks into *both*
parent snapshots plus the Inferact blob — HF cache tooling cannot see that, so do not
prune either parent.


## Weight loading: the per-expert H2D copy (patch 14)

About 9 of the ~11 minutes of a boot were "Loading weights": main model 450–541 s, MTP drafter
~46 s, on `nvidia/Qwen3.8-Flash-Next-NVFP4` in the hybrid layout. Storage was not the bottleneck.
Local NVMe and an NFS share loaded within the ±40 s boot-to-boot noise of each other, iowait
stayed around 0.5%, and one core was busy.

A py-spy profile of the EngineCore during loading (60 s, 99% of samples in `load_weights`):

| self time | frame |
|---|---|
| 59.6% | `RoutedExperts._load_w13` → `expert_data.copy_(loaded_weight)` |
| 29.6% | `RoutedExperts._load_w2` → `expert_data.copy_(loaded_weight)` |
| ~5% | `FusedMoE.load_weights` |
| ~4% | linear layers (`load_merged_column_weight`, `load_row_parallel_weight`, `load_qkv_weight`) |

Every routed expert arrives as separate tensors: 48 layers × 512 experts × 3 projections, each an
800 KiB NVFP4 weight plus a 100 KiB fp8 block scale, about 149k tensors in all. vLLM copies each
one to the GPU on its own, straight from the safetensors mmap view. In the native profile the time
is `cuMemcpyHtoDAsync_v2` from pageable memory, with the CPU spinning in `libcuda` through each
small synchronous copy.

A micro-benchmark on one real 9 GiB expert shard (`tools/bench_moe_load.py`). Each variant ran in
a fresh process on a shard no earlier run had touched, with the client page cache dropped, and
every variant produced byte-identical weights:

| variant | ms / tensor |
|---|---|
| **A** vLLM today: mmap view → `param.copy_()` | 1.74 |
| **E** the shard read into the page cache first, then A | 1.86–1.89 |
| **F** `.clone()` the mmap view, then the same `copy_` | **0.23** |
| **B** a reused pinned bounce buffer | 0.22–0.26 |
| **C** pinned staging per 3,072 tensors, one H2D + a GPU scatter | 0.25 |

The copy is slow whenever its source is a file-backed page, cached (E) or not (A). From ordinary
anonymous memory it is fast (F), and a plain clone does as well as pinned staging (B, C). The cause
is not verified. The likely place is the driver's pageable-copy path on GB10's unified memory, which
appears to handle file-backed pages far more expensively than anonymous ones. Reading the data is
not the cost: copying the same mmap view into host memory takes 0.03–0.7 ms depending on the
cache state, and a `pread()` 0.04–0.07 ms.

The fix is `src/patch_moe_load_clone.py`. In `_load_w13` and `_load_w2`, when the source is a CPU
tensor that is not pinned and the destination is not on the CPU, the copy now reads from
`loaded_weight.clone()`. Weights and block scales both go through these two sites, and so do the
MTP drafter's block-fp8 experts. The memory cost is one transient tensor of at most 800 KiB.
`VLLM_LOAD_CLONE=0` restores the stock copy.

One boot on the preview image (`MODE=hybrid`, NVIDIA checkpoint, weights on NFS, compile cache
reused):

| | before | patch 14 |
|---|---|---|
| Loading weights, main model | 450–541 s | **150 s** |
| Loading weights, MTP drafter | 46 s | **32 s** |
| startup to "Application startup complete" | ~11 min | **4 min 32 s** |

Model memory (74.9 GiB), the KV pool (718k tokens, within the usual boot-to-boot range) and
`scripts/smoke-test.sh` (prefix-cache hit, identical first-token logprobs across runs, 35 tok/s
decode) are unchanged. RadixArk's checkpoint has the same per-expert layout and should benefit the
same way, but it has not been measured.

Ruled out:

- `--safetensors-load-strategy prefetch`. vLLM skips it because the checkpoint is larger than free
  RAM, and E shows that a warm page cache would not help anyway.
- `eager` and `enable_multithread_load`. Both read whole files into RAM, and one file is the 50 GiB
  PLE shard, with only 2–5 GiB free during loading.

The remaining 150 s is covered in the next section.

## The rest of weight loading (patches 15–18)

With patch 14 in, one boot was profiled end to end (`tools/profile_boot.sh`: py-spy on every
process in the container, 10 s slices, 100 Hz; summarized with `tools/pyspy_slices.py`). It
matched the unprofiled boot, 149 + 32 s of loading and 4 min 32 s to ready. The main load:

| s (≈) | where | cause |
|---|---|---|
| 45 | patch 14's `.clone()` | page-faulting cold file pages through the mmap view, ~1.4 GiB/s |
| 26 | `_load_w13` / `_load_w2` `copy_` and the loader chain | the per-tensor H2D copy and its Python overhead |
| 25 | `FusedMoE.load_weights` | pure Python: every tensor is substring-matched against all 1,536 expert-mapping entries (103 µs × 297k tensors = 30 s in isolation; same loop on vLLM `main`) |
| 25 | linear layers (`parameter.py`, `linear.py`) | `param.copy_` straight from the mmap view, the slow path patch 14 fixed for the experts |
| 16 | `vocab_parallel_embedding.py` | `embed_tokens` + `lm_head`, 2 × 1.2 GiB H2D straight from the mmap (~150 MB/s) |
| 13–15 | `PREWARM=1` | streams the 47.7 GiB PLE table into the page cache |

The MTP drafter's 32 s: ~15 s loading its own `embed_tokens` + `lm_head`, which vLLM then deletes
to share the target's, and ~13 s walking all 11 files / 299,845 tensors to keep ~3,100
(`remap_weight_names` filters after each tensor was created).

Reading one whole 9 GiB expert shard in the loader's order (`f.keys()`), client page cache cold
(checked with `mincore`), CPU only, bytes identical:

| read path | 9 GiB shard |
|---|---|
| mmap `get_tensor` + `.clone()` (patch 14) | 3.95–4.13 s = 2.2 GiB/s |
| `pread` each tensor into a fresh CPU tensor | **0.82 s = 10.9 GiB/s** |

The four patches, each on by default with its own switch:

| patch | switch | what |
|---|---|---|
| 15 `src/patch_load_pread.py` | `VLLM_LOAD_PREAD=0` | `safetensors_weights_iterator`: tensors ≤ 64 MiB are `pread` into ordinary memory, and their storage is tagged so patch 14 skips its now redundant clone. Larger tensors and every `ngram_embedding.shard_*` stay mmap views, so the PLE table is never read into RAM. Needs 14. |
| 16 `src/patch_moe_name_index.py` | `VLLM_MOE_NAME_INDEX=0` | the expert loader looks names up in an index (`{len: {name: [idx]}}` at each `experts.` position) instead of scanning all 1,536 entries; same entries, same order, and the original's `break` on fused names. Falls back to the full scan if an entry does not start with `experts.`. |
| 17 `src/patch_embed_chunked_copy.py` | `VLLM_LOAD_EMBED_CHUNK=0` | `VocabParallelEmbedding.weight_loader` copies CPU → GPU in 64 MiB row blocks, each cloned into ordinary memory first. |
| 18 `src/patch_mtp_name_prefilter.py` | `VLLM_MTP_NAME_PREFILTER=0` | wraps vLLM's `should_skip_weight` (the hook the iterator asks before reading each tensor) with an optional keep-filter; the drafter's `load_weights` sets it to `_remap_mtp_weight_name(n) is not None` while it loads, so it skips the other 296,742 tensors unread. Not used when the model has secondary weight sources. |

Boots on the preview image (`MODE=hybrid`, NVIDIA checkpoint, weights on NFS over RDMA, compile
cache reused):

| | patch 14 | + 15–17 | + 18 |
|---|---|---|---|
| Loading weights, main model | 149 s | 35 s | 35.5 s |
| Loading weights, MTP drafter | 32 s | 12 s | **1.2 s** |
| startup to "Application startup complete" | 4 min 32 s | 2 min 13 s | **2 min 8 s** |

- CPU tests, in the image: `src/test_moe_name_index_cpu.py` (22,516 name/config cases, including
  EPLB redundant experts, LoRA prefixes and fused names, identical to the original loop, 73.5 →
  1.1 µs per tensor); `src/test_load_patches_cpu.py` (every small tensor byte-identical to the stock
  iterator on three real files, embeddings and all 128 PLE shards stay unread views);
  `src/test_mtp_prefilter_cpu.py` (on the whole snapshot the filtered iterator yields exactly the
  3,103 names the drafter keeps).
- Greedy first-token log-probs on five reference prompts: identical to the patch 14 boot after 15–17
  and again after 18. `scripts/smoke-test.sh`: deterministic, prefix-cache hit, 35.4 tok/s decode.
- The drafter: a greedy probe cannot catch a bad drafter (the target verifies every token), so
  `vllm:spec_decode_num_{draft,accepted}_tokens_total` were read around one probe. 1,208 drafted /
  864 accepted with patch 18, and the same on an A/B boot with `VLLM_MTP_NAME_PREFILTER=0`.
- Model memory 74.9 GiB, unchanged; the KV pool stays inside the usual boot-to-boot range.

On vLLM v0.30 the call sites are the same; patch 18 finds `mtp.py` under the `qwen4_exp` package
and keeps its `mapper=` argument.

Two notes. Most of patch 15's gain is the read path, so it should matter more where the checkpoint
pages come off local disk rather than out of an NFS server's RAM, as here; that has not been
measured. And the weight stream still pushes most of the prewarmed PLE table out of the page cache
(8.5% of it resident at "Application startup complete"); that affects the first requests, not the
boot.


## vLLM v0.30 specifics

Three things in vLLM v0.30.0 shape how the patches hook in. (How they were ported from the
earlier bases is in [HISTORY.md](HISTORY.md).)

**PLE lookup.** The n-gram module left `ple_layer.py` for `ngram_embedding.py`. The hashing is a
Triton kernel (`ops/ple.py`) called through `compute_ngram_ids`, and the lookup goes through
`Qwen4ExpPLEDeviceEmbedding`, or `Qwen4ExpPLEPinnedHostEmbedding` with engram CPU offload.
`src/vllm_ple_mmap.py` swaps both embedding classes for the mmap placeholder during `__init__`,
keeps the stock hashing, and sends the lookup through the `ple_mmap_lookup_ids` op
([The patch](#the-patch-srcvllm_ple_mmappy)).

**CUDA graphs.** v0.30 lists Qwen4Exp among the architectures that get *breakable* CUDA graphs by
default (`VLLM_USE_BREAKABLE_CUDAGRAPH`): torch.compile is off and one capture drives the whole
forward, ending a graph segment around every op that must run eagerly. Our gather is CPU work plus
a host-to-device copy, so it cannot be captured. Three consequences, each hit on a real boot:

1. The gather ends the segment itself through the capture's `add_eager` (a synchronize inside a
   capture is `cudaErrorStreamCaptureUnsupported`).
2. It writes into a persistent per-layer output buffer, sliced per batch. An eager segment must
   write to the same address on every replay.
3. At capture time the kernels queued before it, the n-gram hashing among them, are recorded and
   not run, so the ids hold garbage. While a capture context is open, even paused, the eager
   segment only zero-fills its output. At replay the segments run in order and the ids are real.

`-cc.splitting_ops` in `scripts/serve.sh` carries a list for the FX path
(`VLLM_USE_BREAKABLE_CUDAGRAPH=0`), which has not been measured.

**QSA top-k.** vllm#54513 split the indexer into prefill and decode paths that share one `_topk`
helper in `ops/qsa_indexer.py`. `patch_qsa_exact_topk.py` patches that helper, and
`src/patch_qsadet.py` wires the deterministic kernel into it (the upstream wiring script only
knew the older file). The stock kernel changed too (vllm#54110, vllm#56346); `scripts/smoke-test.sh`
and the tournament's determinism probe re-measure GB10 determinism on every boot, and both pass with
our kernel.

Patches 12 and 13 target the v0.30 parser engine; the 30-case test module passes. Patches 3, 9
and 11 of the earlier bases are upstream fixes that v0.30 ships (vllm#50729, vllm#52775, vllm#55513).

## fp8 KV cache on the QSA path (opt-in)

`src/patch_qsa_fp8_kv.py`, after [@Nanetnounou](https://github.com/Nanetnounou)'s original for the
preview image ([issue #6](https://github.com/blazux/qwen3.8-Flash-DGX/issues/6)). vLLM quantizes on
the write side already (`do_kv_cache_update` with the layer's `_k_scale`/`_v_scale`), so the patch
only touches the read side: the split-K kernel dequantizes K and V with vLLM's own `_cast_kv_tile`
(fp8 per-tensor), BLOCK_N is halved under quantization to fit sm_121's 101,376-byte shared memory,
the warmup compiles the fp8 specialization, uint8 storage is reinterpreted as fp8 (a bit view), and
the bf16-only guards are widened, the one inherited from `FlashAttentionImpl` included, since QSA
never uses its kernels. The QSA indexer's caches are v0.30's own: the raw-key ring stays bf16 and the
compressed keys have their own dtype (`indexer_kv_dtype`, native fp8, vllm#54890). The read passes
the layer's real scales (the preview version read with a fixed 1.0, which only matched because the
writes used 1.0 as well). With `--kv-cache-dtype auto` the branch is compiled out: first-token
log-probs 5/5 identical to the image without the patch. `tests/test_fp8_kv_read.py` checks on a GPU
that the fp8 read path gives bit-identical attention output.

Measured on our GX10 (NVIDIA checkpoint, hybrid, MTP=2, YaRN, `GPU_MEM=0.80`): at `CTX=1000000` a
1,039k-token pool, needles found at 196k, 413k, 635k and 931k tokens; decode −4%, prefill −3 to
−17%. At 500k context the tournament scored 88.4% over 3 runs, bf16 87.8%. vLLM raises the attention
block to 3,184 tokens in this mode to keep attention and Mamba pages equal, so prefix-cache hits are
twice as coarse. The fp8 saving applies to the attention K/V only: the GDN/PLE recurrent states, the
QSA compressed keys and the raw-key ring stay as they are.
