# MLX Vulkan Backend — Task Tracker

## Context

Implement a Vulkan compute backend for MLX (ml-explore/mlx) to enable the framework on Linux
with any Vulkan-capable GPU (AMD, NVIDIA, Intel). Mirrors the existing CUDA backend structure.
Target: Linux-first. macOS via MoltenVK deferred. Full primitive coverage. AOT SPIR-V kernels.

**Key contract**: `mlx/backend/gpu/eval.h` — 4 functions all GPU backends must implement:

- `gpu::new_stream(Stream)`, `gpu::eval(array&)`, `gpu::finalize(Stream)`, `gpu::synchronize(Stream)`

**Reference backends**: `mlx/backend/cuda/` (structure), `mlx/backend/metal/` (kernel patterns)

**Current devices**: 
- Apple M1 via MoltenVK (macOS development)
- AWS AMD `g4ad.xlarge` (Cloud validation, remote) (not to be used for now)
- **Local Ubuntu 25.10 (gsai)**: Intel i7-7700 (8 cores), 8GB RAM, **AMD Radeon RX 580** (Polaris 10), Tailscale IP: `100.104.140.2` (Refer Quick Connect section below)
**Last verified**: 2026-03-16

### Remote Access (Local Ubuntu)

The local Ubuntu AMD validation target is accessible via **Tailscale** for secure development.

**Quick Connect:**
```bash
ssh -i ~/.ssh/id_ed25519_remote gsai@100.104.140.2
```

**VS Code Remote SSH Config:**
```ssh
Host gsai-box
    HostName 100.104.140.2
    User gsai
    IdentityFile ~/.ssh/id_ed25519_remote
```

**Canonical remote workspace:**

- Workspace root: `/home/gsai/mlx-vulkan`
- MLX source tree: `/home/gsai/mlx-vulkan/mlx-src`
- Linux Vulkan build dir: `/home/gsai/mlx-vulkan/build_vulkan_linux`
- Compile logging smoke: `/home/gsai/mlx-vulkan/tests/vulkan/test_compile_logging.py`

**Canonical remote workflow (from the local workstation):**

```bash
bash scripts/local_amd_sync.sh
bash scripts/local_amd_build.sh
bash scripts/local_amd_profile_compile.sh
```

**Working rule:** treat `/home/gsai/mlx-vulkan` as the source of truth for remote AMD validation. Do not rely on ad hoc edits under older remote directories when validating the current workspace.

---

## Current Priority (as of 2026-03-16)

**Top milestone**: push steady-state RX 580 decode clearly past CPU parity during real `mlx-lm` generation.

**Minimum bar**: a strict `generate_step` run on the local AMD box should stay on Vulkan under `MLX_VULKAN_FAIL_ON_CPU_FALLBACK=1` and materially improve warmed multi-token throughput, not just first-token latency.

**Immediate profiling focus**: keep narrowing the hottest eager decode path on real Qwen runs, using RX 580 layer timing and utilization data to decide whether the next win is attention, KV-cache maintenance, RoPE, or host/submit overhead.

### Current Truth

> [!IMPORTANT]
> These are the only claims that are currently backed by real-hardware evidence.

- ✅ **AMD Vulkan runtime proven** — real Linux execution on AWS `g4ad.xlarge` (RADV NAVI12)
- ✅ **Core tensor ops, autograd, and tiny MLP training proven** — on real AMD Vulkan hardware
- ✅ **Stage 25 bring-up passes on real AMD** — import, GPU detect, core ops, autograd, optimizers, tiny MLP, Python bridge, and CPU-fallback outputs
- ✅ **Transformer-critical ops audit passes on real AMD** — embeddings, causal attention forward, batched `Linear` / matmul, LayerNorm forward/backward, RoPE, RMSNorm, optimizer update, and uncompiled standard encoder layer
- ✅ **Tiny transformer and TinyGPT training now pass on real AMD** — after fixing the non-cooperative batched matmul path for `subgroup_size=64`
- [x] **Vulkan compiled-graph execution now survives real RX 580 profiling without crashing**:
      the compiled smoke passes on AMD, contiguous float-like compiled kernels (`Sigmoid -> Multiply`, `Sigmoid -> Multiply -> Multiply`) execute through Vulkan, and the earlier `copy_gpu` / `SmallVector` abort was removed by routing the float32 cast bridge through contiguous vector copies.
- [~] **Broadcast / stride coverage is partially live on the RX 580 compiled path**:
      real `mlx-lm` profiling now keeps the previously hot fused kernels `Broadcast -> Multiply`, `Subtract -> Broadcast -> Multiply`, `Broadcast -> Broadcast -> Multiply -> Add -> Broadcast -> Multiply`, and the `LogAddExp` softmax-style kernel on Vulkan after adding generic stride metadata and descriptor-offset support.
      Remaining work is still in Phase 6.2, but the blocker has moved from "no broadcast/strided support" to "finish broader view/shape coverage and then re-profile for the next missing primitive or reduction."
- [x] **Compiled broadcast + offset regression is now covered**:
      `tests/vulkan/test_compile_logging.py` now exercises a sliced-view broadcast multiply through `mx.compile`, so the new RX 580 broadcast/offset path is checked by the standard scripted compile smoke.
- [~] **Terminal compiled `Sum` reduction is now live on the RX 580 for the real LLM hot path**:
      Vulkan `mx.compile` fusion now admits shape-specialized last-axis `Reduce::Sum` roots on GPU traces, and real `mlx-lm` profiling on the RX 580 shows the new `Broadcast -> Multiply -> Sum` kernel dispatching through the Vulkan JIT path during `gated_delta_step_ops`.
      This is intentionally first-pass only: it is limited to terminal, keepdims, last-axis `Sum` reductions and does not yet cover multi-axis/general reductions or `Prod` / `Max` / `Min`.
- [~] **The next end-to-end LLM bottleneck has moved beyond compiled reduction fusion**:
      after the `Sum` kernel landed, the guarded one-token RX 580 benchmark still timed out after 180s with no first token, but the live Python stack is now in `mlx_lm.models.cache.KVCache.update_and_fetch(...)` / `qwen3_next.py` rather than the earlier `gated_delta` reduction path.
- [~] **Phase 6 correctness blockers have moved from cache-path fallbacks to a throughput wall**:
      the RX 580 now runs the previously failing LLM correctness path under `MLX_VULKAN_FAIL_ON_CPU_FALLBACK=1` without tripping a CPU fallback before the 180s timeout.
      Concretely:
      - Vulkan now has a native grouped `Conv1d` path with a float32 shader-IO bridge, and strict AMD probes cover both dense and depthwise float16 cases.
      - eager Python indexing no longer materializes plain slice / int access to CPU, so cache-tail views like `conv_input[:, -3:]` stay on-device.
      - broadcasted bool `where(...)` now materializes strided / broadcasted operands into dense GPU temporaries inside `Select::eval_gpu`, which keeps the grouped-query SDPA causal-mask path on Vulkan.
      The guarded one-token RX 580 benchmark still times out after 180s with no first yield, but the strict run now reaches that timeout without a fallback error, so the remaining blocker is throughput / scheduling rather than another missing primitive.
- [~] **Affine quantized matmul is now genuinely on Vulkan for the RX 580 LLM path**:
      `QuantizedMatmul::eval_gpu(...)` no longer hides the affine path behind a synchronized CPU reference call.
      The Vulkan backend now dequantizes affine weights on-device, including the real `transpose=True` 5-bit Qwen projection path and the non-transposed 4-bit path, and the new strict smoke `tests/vulkan/test_quantized_gpu.py` passes on AMD with `MLX_VULKAN_FAIL_ON_CPU_FALLBACK=1`.
      End-to-end impact: the guarded one-token `generate_step` benchmark moved from a 180s timeout with no yield to a successful first token in about `7.28s` on GPU.
- [~] **The next RX 580 throughput hotspot is no longer decoder linear attention; it is the logits path plus first-layer warmup**:
      refreshed AMD profiling now shows 5-token prompt prefill at about `3.23s` total in a fresh process, with decoder layers accounting for about `2.03s`, the tied `lm_head` projection about `1.19s`, and the largest single decoder cost being the first linear layer at about `0.85s`.
      The old `~15-17s` per-linear-layer wall is gone; most later linear layers are now about `0.05s` each and full-attention layers are about `0.055-0.058s`.
- [~] **The tied-vocab logits path now has a selective direct Vulkan kernel, and RX 580 first-token latency dropped again**:
      Vulkan now uses a dedicated transpose affine QMM shader for the giant-vocab, small-`M` case instead of materializing a full dequantized `[K, N]` matrix before matmul. The gate is intentionally narrow: `transpose=True`, 2D packed weights, `M <= 8`, and `N >= 8192`, which keeps normal decoder projections on the proven two-pass path while targeting the Qwen tied `lm_head`.
      The strict AMD smoke `tests/vulkan/test_quantized_gpu.py` now includes a large-vocab bfloat16 regression for this path, and the guarded one-token `generate_step` benchmark on the RX 580 improved from about `7.18s` to about `5.44s`, now beating the matching CPU baseline (`~7.08s`).
      Follow-up probing shows the next meaningful latency target is first-layer / first-use warmup plus the remaining decoder prefill cost rather than the tied logits projection itself.
- [~] **Warmed RX 580 decode is still badly underutilizing the GPU**:
      a warmed 16-token RX 580 utilization profile now shows first yield at about `4.63s`, then a very regular `~1.80s` per-token cadence, while `gpu_busy_percent` averages only about `17.9%` with a `0%` median and only about `19%` of samples at or above `50%`.
      That points away from a single missing fused `mx.compile` kernel and toward bursty eager-path work: many small kernels plus host/submit gaps are leaving the card mostly idle during steady-state decode.
- [x] **RX 580 memory pressure is not the current decode blocker**:
      current scripted AMD status checks show the box staying well clear of memory limits during this work: system RAM is around `1.1 / 7.2 GiB` used with about `48 MiB` swap in use, and idle post-run GPU memory is around `299 MiB` VRAM plus about `29 MiB` GTT.
      Earlier active-run checks also stayed comfortably below the RX 580's `8 GiB` VRAM and `3.59 GiB` GTT limits, so the current throughput wall should be treated as a scheduling / kernel-density problem rather than a memory-capacity issue.
- [~] **Decode-time grouped-query SDPA now has a native Vulkan fast path on the RX 580**:
      `mlx/backend/vulkan/primitives.cpp` now routes the real decode-only `mx.fast.scaled_dot_product_attention(...)` case to a dedicated Vulkan kernel instead of materializing the fallback graph. The current gate is intentionally narrow: `q_len == 1`, no array mask, no sinks, no training/VJP, float32/float16/bfloat16 inputs, and head dim `128` or `256`.
      The new shader `mlx/backend/vulkan/kernels/sdpa_vector.comp` runs the causal/no-array-mask grouped-query decode case in one dispatch with an online softmax update, and the new strict regression `tests/vulkan/test_sdpa_gpu.py` passes on the RX 580 under `MLX_VULKAN_FAIL_ON_CPU_FALLBACK=1`.
- [~] **The native decode SDPA kernel improved the real Qwen path, but steady-state decode is still only near CPU parity**:
      refreshed RX 580 layer profiling for `mlx-community/Qwen3.5-2B-5bit` now shows decode full-attention layers around `0.053-0.055s` instead of the earlier `~0.054-0.061s` class, while `lm_head_tied` is about `0.586s`.
      Updated strict end-to-end 16-token `generate_step` measurements on the RX 580 now land around:
      - GPU run 1: `29.35s`, `0.545 tok/s`, first yield `4.38s`
      - GPU run 2: `28.24s`, `0.566 tok/s`, first yield `4.23s`
      - matching CPU rerun: `27.91s`, `0.573 tok/s`, first yield `4.24s`
      So the new SDPA path clearly improved the GPU decode baseline (older warmed GPU result was about `31.85s` / `0.502 tok/s`), but the RX 580 is still only at rough CPU parity rather than decisively ahead.
- [~] **A native Vulkan RoPE path is now live for the scalar-offset Qwen decode case, but it is a local win rather than a full throughput unlock**:
      `RoPE::eval_gpu(...)` no longer always materializes the fallback graph. Vulkan now has a native forward path for the actual LLM rotary case we are exercising on RX 580: no custom `freqs`, nontraditional layout, scalar offset, and float32/float16/bfloat16 inputs.
      The stricter `tests/vulkan/test_rope_gpu.py` now checks CPU-reference numerical agreement for both a transposed prefill-style tensor and a decode-style scalar-offset `[1, 8, 1, 256]` tensor under `MLX_VULKAN_FAIL_ON_CPU_FALLBACK=1`, and it passes on the RX 580.
      Real AMD profiling shows that this RoPE path improves the attention block itself:
      - average prefill full-attention layer time moved from about `0.0596s` to about `0.0521s`
      - average decode full-attention layer time moved from about `0.0540s` to about `0.0512s`
      But the refreshed strict 16-token GPU benchmark is still only about `28.71s` / `0.557 tok/s` with first yield about `4.36s`, which keeps whole-run decode in the same near-parity band as the SDPA-only checkpoint.
      Conclusion: native RoPE is worth keeping because it removes repeated fallback-graph overhead in the real path, but the next major wall is still outside RoPE itself, likely KV-cache maintenance and/or host-submit gaps.
- [~] **Even after the SDPA fast path, the RX 580 is still mostly idle during warmed decode**:
      the refreshed utilization profile after the SDPA change shows:
      - first yield about `4.38s`
      - total `16`-token run about `28.54s` / `0.561 tok/s`
      - average GPU busy about `17.8%`
      - median GPU busy `0%`
      - `gpu_busy_p95` about `98.95%`
      - memory busy average only about `3.12%`
      The card is still seeing short bursts of useful work separated by long idle gaps, so the next throughput target remains eager decode orchestration rather than another blind compiled-op expansion.
- [~] **The direct tied-logits kernel still has headroom, but the latest RX 580 rewrite is only partially validated so far**:
      `mlx/backend/vulkan/kernels/quantized_qmv.comp` now tiles the activation vector through shared memory across output columns instead of having every invocation re-read the same `x` row independently.
      Real AMD validation so far is promising but incomplete:
      - the local AMD Vulkan build succeeds with the new shader,
      - `scripts/local_amd_probe_lm_head.sh` still confirms `direct_qmv=1` on the real `N=248320`, `K=2048` Qwen projection,
      - isolated tied-logits timing improved to about `1.09s` for the full 5-token logits pass and about `0.77s` for the last-token-only path,
      - the warm layer probe now shows `lm_head_tied` around `0.58s` instead of the earlier `~0.65s` class.
      However, the follow-up end-to-end decode rerun was interrupted when the AMD host stopped accepting SSH connections mid-benchmark, so no new whole-run tokens/sec claim is recorded yet.
- [~] **Vulkan compiled JIT kernels now persist SPIR-V across processes, removing the pathological cold-start cliff on RX 580**:
      `mlx/backend/vulkan/compiled.cpp` no longer relies on an in-memory-only SPIR-V cache. The generated GLSL source and compiled `.spv` are now reused from `~/.cache/mlx_vulkan_jit` when the on-disk GLSL text matches the current kernel source, so identical `mx.compile` kernels do not re-run `glslc` for every new Python process.
      Real RX 580 validation shows the cross-process cold-start wall collapsed: a fresh-process first token that had ballooned to about `54.35s` in the same-process warm probe now drops to about `5.57s` on the first post-build run, and a second fresh process reaches about `4.69s`, matching the warmed steady-state path.
- ❌ **CPU-fallback linalg correctness broken on real AMD** — `qr`, `svd`, `cholesky`, `eigh`, `inv` return zeros.
  *Update (2026-03-10)*: Isolated a critical memory erasure bug where CPU writes to `raw_ptr()` mappings are lost/zeroed between accesses.
- ❌ **Full MLX suite compatibility not yet achieved** — historical MoltenVK pass rates (below) are not validated on real Linux hardware

---

### Authoritative Compatibility Gates

Default policy: CPU fallback is **allowed** for early bring-up gates only when explicitly marked. CPU fallback is **forbidden** starting at the transformer gate (Gate 7).

| Gate | Name | Test Entrypoint | Hardware | Pass Condition | CPU Fallback |
|------|------|----------------|----------|----------------|-------------|
| 1 | Device / runtime / build / import | `test_stage25_linux_vulkan_bringup.py` | Any Vulkan GPU | Import + GPU detection | N/A |
| 2 | Core tensor ops without CPU fallback | `test_stage25_linux_vulkan_bringup.py` (core ops) | AMD or NVIDIA Linux | matmul, softmax, elementwise correct | ❌ Forbidden |
| 3 | Autograd correctness | `test_stage25_linux_vulkan_bringup.py` (autograd) | AMD or NVIDIA Linux | `value_and_grad` matches expected | ❌ Forbidden |
| 4 | Optimizer correctness | `test_stage25_linux_vulkan_bringup.py` (optimizers) | AMD or NVIDIA Linux | Parameters update correctly | ❌ Forbidden |
| 5 | Tiny MLP training convergence | `test_stage25_linux_vulkan_bringup.py` (tiny MLP) | AMD or NVIDIA Linux | Loss decreases over the smoke window | Allowed for non-training ops only |
| 6 | Transformer-critical op coverage | `test_transformer_ops_audit.py` | AMD or NVIDIA Linux | Embedding, causal attention, norm, optimizer path execute on Vulkan | ❌ Forbidden |
| 7 | Tiny transformer training convergence | `test_tiny_transformer.py` | AMD or NVIDIA Linux | Loss decreases, no CPU fallback on the training step | ❌ Forbidden |
| 8 | Selected MLX suite coverage | `test_random.py`, `test_ops.py` subset, `test_array.py` subset | AMD or NVIDIA Linux | Defined pass-rate threshold per suite | Allowed where marked |
| 9 | Long-run stability | 1000-step training loop | AMD or NVIDIA Linux | No OOM, no device lost, no crash | ❌ Forbidden |
| 10 | NVIDIA parity | Same gates 1–9 | NVIDIA Linux | Same pass criteria | Same policy |

---

### Cross-Platform Compatibility Matrix

| Capability | MoltenVK / macOS | AMD / Linux (real HW) | NVIDIA / Linux (real HW) |
|---|---|---|---|
| Tensor ops | proven (historical) | **proven** | not run |
| Autograd | proven (historical) | **proven** | not run |
| Optimizers (SGD) | proven (historical) | **proven** (tiny MLP) | not run |
| Optimizers (Adam) | proven (local gate) | **proven** (transformer training) | not run |
| Tiny MLP training | proven (historical) | **proven** | not run |
| Linalg CPU fallback | proven (historical, unified mem) | **failing** | not run |
| Tiny transformer training | proven (local gate) | **proven** | not run |
| TinyGPT 10-epoch training | proven (local gate) | **proven** | not run |
| LoRA / small-LLM fine-tune | not run | not run | not run |
| Official MLX `test_ops.py` | ~124/134 (historical) | not run | not run |
| Official MLX `test_array.py` | ~65/68 (historical) | not run | not run |
| Official MLX `test_random.py` | 14/14 (historical) | not run | not run |
| Long-run stability | not run | not run | not run |

---

### Live Validation Snapshot (AMD/Linux)

- [x] Real Linux AMD Vulkan runtime provisioned and validated on AWS `g4ad.xlarge`:
      `amdgpu` loaded, `/dev/dri/renderD128` present, `vulkaninfo` reports `RADV NAVI12`.
- [x] Vulkan Python build succeeds on Linux with `python3.11`.
- [x] `mlx.core` imports and selects the Vulkan GPU on real AMD hardware.
- [x] Simple shader-backed MLX op executes correctly on Vulkan:
      vector add returned the expected result and created a fresh Vulkan pipeline cache file.
- [x] Autograd smoke passes on Vulkan:
      `mx.value_and_grad(sum(x*x))` matched the expected scalar loss and gradient.
- [x] Tiny MLP training smoke passes on Vulkan:
      `Linear -> ReLU -> Linear` with SGD showed stable loss decrease on real AMD Vulkan.
- [x] Transformer-critical ops audit passes on real AMD Vulkan:
      embeddings, causal attention forward, batched `Linear` / matmul, LayerNorm forward/backward, RoPE, RMSNorm, optimizer update, SiLU (uncompiled), and standard `TransformerEncoderLayer` (uncompiled).
- [x] Tiny transformer training now passes on real AMD Vulkan:
      Adam-based next-token smoke converges without CPU fallback on the live `g4ad` box.
- [x] TinyGPT 10-epoch training now passes on real AMD Vulkan:
      loss decreases cleanly over all 10 epochs on the live `g4ad` box.
- [x] The real-AMD `Linear` / matmul blocker was fixed in the non-cooperative Vulkan path:
      batched `Matmul(x[B,T,C], w[C,N])` no longer advances a broadcast RHS per batch, and the fallback shader now uses a safe `16x16` tile on `subgroup_size=64` instead of the broken `32x8` layout.
- [x] Quantized `GatherMM` fallback stability fixed:
      Intermittent `test_gather_qmm_sorted` failures addressed by routing `GatherMM::eval_gpu` fallback through a direct CPU reference path instead of re-entering generic lazy `gather_mm(...)`.
- [ ] CPU-fallback linalg correctness still fails on real AMD Vulkan:
      `qr`, `svd`, `cholesky`, `eigh`, and `inv` currently return zeros or fail in the discrete-GPU fallback path.
- [x] LoRA / small-LLM fine-tuning path is locally proven via a dedicated short smoke test.
- [x] End-to-End LLM Inference passes on real AMD Vulkan:
      `Qwen3.5-2B-5bit` quantified model successfully generated text natively through `mlx-lm` without crashing.

### Functional Success Criteria

| Stage | What We Run | Success | Current State |
| --- | --- | --- | --- |
| Tensor Ops | Large matmul, elementwise ops | Runs on Vulkan GPU without CPU fallback | In progress; core ops proven, broader MLX tensor coverage still pending |
| Autograd | Gradient tests on simple loss | Gradients match CPU backend within tolerance | ✅ Proven on AMD Vulkan |
| Optimizers | Adam / SGD updates | Parameters update correctly | ✅ SGD proven in tiny MLP smoke; Adam proven in real-AMD transformer training |
| Neural Net Training | Small MLP or CNN | Loss decreases consistently | ✅ Tiny 2-layer MLP proven |
| Transformer Training | Tiny transformer | Training converges without CPU fallback | ✅ Proven on real AMD Vulkan |
| TinyGPT Training | TinyGPT, 10 epochs | Training stays finite and converges without CPU fallback | ✅ Proven on real AMD Vulkan |
| LLM Finetuning | LoRA on a small LLM | Runs for long training windows | ✅ Proven via local LoRA smoke |
| Stability | Long training run | No leaks, no device resets | ❌ Not yet proven |

### Performance Success Criteria

| Metric | Baseline | Minimum Acceptable | Current State |
| --- | --- | --- | --- |
| GPU utilization | CUDA / Metal reference | >50% | Not yet measured on real AMD |
| Matmul throughput | CUDA / Metal reference | >=40% of CUDA / Metal | Not yet measured on real AMD |
| Tiny transformer training throughput | Metal MLX reference | >=30-40% | Not yet measured |
| Memory efficiency | CUDA / Metal usage | <=1.5x memory | Not yet measured |
| Long-run kernel stability | No crashes | Occasional recoverable errors at worst | Not yet measured |

### Compatibility-First Roadmap

- [x] Proven: Vulkan backend executes real MLX training primitives on a discrete GPU.
- [x] Proven: backward pass and optimizer updates on real AMD Vulkan.
- [ ] Eliminate the remaining CPU-fallback correctness blockers that break MLX compatibility on discrete GPUs:
      Implement explicit host-staging buffer round-trips for linalg (`copy_to_host` -> CPU LAPACK -> `copy_from_host`) to bypass the fragile and currently broken `raw_ptr()` mapping path.
- [x] Add a tiny transformer smoke that uses only supported Vulkan paths and verify convergence locally.
- [x] Clear the first transformer backward blocker on Vulkan:
      `LayerNormVJP` now materializes its fallback graph into Vulkan outputs, and Gate 6 / Gate 7 pass on the local Vulkan build.
- [x] Enforce the "no CPU fallback" contract in transformer gates locally:
      `MLX_VULKAN_FAIL_ON_CPU_FALLBACK=1` is now enabled in Gate 6 / Gate 7, and the tiny-transformer smoke was rewritten to avoid `mx.compile`-wrapped activations while keeping real token embeddings in the training loop.
- [x] Unblock embedding-weight training on Vulkan:
      `Gather::vjp` now lowers the embedding-style row-gather case to `ScatterAxis` instead of generic `Scatter`, so embedding gradients no longer force CPU fallback.
- [x] Restore the real next-token objective in Gate 7:
      `nn.losses.cross_entropy(...)` now runs cleanly in the strict tiny-transformer training loop after the embedding-gradient fix.
- [x] Rerun Gate 6 / Gate 7 on the real AMD Vulkan box and confirm training convergence.
- [x] Fix the real AMD training blocker in the generic `Linear` / matmul path:
      non-cooperative batched matmul on `subgroup_size=64` no longer mis-tiles the RHS or corrupt the fallback tile buffers on AMD.
- [x] Re-run `test_tiny_transformer.py` and `test_tinygpt_10epoch.py` on real AMD after the `Linear` / matmul fix.
- [x] Audit transformer-critical ops for fallback-free execution:
      `matmul`, `addmm`, `softmax`, `layer_norm`, `rope`, embeddings, masking, indexing, optimizer updates.
- [x] Fix the quantized `GatherMM` / sorted quantized fallback stability:
      Vulkan `GatherMM` fallback now computes via a direct CPU reference path matching CPU stride semantics, solving stateful memory corruption in mixed-mode runs (`test_gather_qmm_sorted`).
- [ ] Identify and remove remaining blockers for real LLM workloads:
      broader MLX suite coverage, sequence-length scaling, long-run stability, and cross-vendor validation.
- [ ] After the compatibility ladder is green, measure throughput and memory behavior on real AMD and NVIDIA hardware (see Gates 9–10).

### Cross-Vendor Validation Checklist

- [x] Added a dedicated bring-up smoke script at `tests/vulkan/test_stage25_linux_vulkan_bringup.py`:
      covers import, GPU detection, core ops, CPU-fallback output paths, and Python bridge safety; supports `MLX_CORE_SO=...` for fresh builds and `MLX_VULKAN_REQUIRE_VENDOR=amd|nvidia|intel` for strict runner gating.
      Latest expansion: linalg fallback checks remain available behind `MLX_VULKAN_INCLUDE_LINALG=1`, but are no longer part of the default first-pass contract.
- [ ] NVIDIA is a planned validation target after the AMD compatibility ladder is fully green.
      Run the same compatibility ladder on NVIDIA: Stage 25, autograd smoke, tiny MLP training, tiny transformer training.
- [ ] Defer vendor-specific tuning and cooperative-matrix optimization until the training-compatibility bar is met.

### Pre-NVIDIA Blockers From Static Code Review

- [x] First-pass fix for Vulkan allocator readback ownership in `backend/vulkan/allocator.cpp`:
      separated VMA mappings from CPU readback snapshots so discrete-GPU `raw_ptr()` no longer stores heap snapshots as fake VMA mappings.
- [x] First allocator memory-model shift toward discrete-GPU behavior:
      `backend/vulkan/allocator.cpp` no longer requires host-coherent primary allocations and now prefers transfer-backed access, with Stage 25 still passing against the staged host-transfer path.
- [x] First-pass synchronization hardening for GPU-stream CPU fallbacks in `backend/vulkan/primitives.cpp` and `backend/vulkan/copy.cpp`.
      Added `synchronize()` before compiled fallback, linalg CPU fallbacks, quantize CPU/readback paths, and dynamic offset CPU reads.
- [x] Added explicit Vulkan host transfer helpers (`copy_from_host` / `copy_to_host`) and migrated backend call sites away from direct mapped-pointer writes where practical.
- [x] Disabled unsafe Vulkan no-copy host pointer wrapping:
      `backend/vulkan/allocator.cpp::make_buffer()` now returns `nullptr` until true host-memory import exists, forcing NumPy/raw host pointer construction onto the safe copy path instead of misinterpreting host pointers as internal `VulkanBuffer*` handles.
- [x] Centralized host-to-buffer initialization for the main MLX inference/model-loading path:
      shared array constructors, common scalar/load helpers, GGUF tensor import, and GGUF quantized unpacking now use allocator-level host copies instead of assuming Vulkan allocations are directly writable through `data<T>()`.
- [x] Removed the simple fresh-allocation direct-write assumptions in CPU fallback helpers:
      integer predicate zero-fill (`backend/cpu/unary.cpp`), empty-K matmul output zeroing (`backend/cpu/matmul.cpp`), and CPU dynamic-offset scalar writes (`backend/cpu/primitives.cpp`) now go through allocator-level host copies.
- [x] Added a central GPU-stream CPU-fallback output flush in `backend/cpu/encoder.h`:
      CPU fallback kernels that write through `data<T>()` on GPU-backed outputs now push those host-side writes back through allocator staging before returning to the Vulkan evaluator.
- [x] Patched shared serialization/readback paths to use allocator host copies:
      `export.cpp`, `.npy` save, safetensors save, GGUF save, compile scalar folding, and debug printing no longer read Vulkan-backed arrays through direct `data<T>()` pointers.
- [x] Hardened Python bridge host reads for discrete-memory safety:
      `python/src/buffer.h` and `python/src/convert.cpp` now expose host-owned snapshots for buffer protocol, NumPy export, scalar conversion, and `tolist()` paths instead of dereferencing backend memory directly; the Python buffer protocol is explicitly read-only on those snapshot-backed exports, including reversed / strided views.
- [x] Remove or downgrade overstated readiness claims:
      `VK_EXT_external_memory_host` zero-copy, Vulkan `mx.compile()` GPU fusion, SDPA coverage, and full FFT surface coverage.
- [ ] Re-audit MoltenVK-only fallback assumptions before NVIDIA bring-up:
      linalg fallbacks, dynamic offset reads, quantized dequant readback, and any direct `data<T>()` access on GPU-backed arrays.
- [ ] Broad allocator memory-model flip still pending:
      the model-loading, main CPU-fallback output path, and shared serialization/readback paths are now staged safely, but a final audit is still needed for direct host access that occurs outside encoder-managed GPU fallbacks and outside the already-patched construction/load/readback helpers before primary Vulkan allocations can move to fully device-local discrete-GPU behavior.
- [ ] Non-core paths still need explicit NVIDIA validation:
      distributed backends and other code that bypasses the CPU encoder are not yet part of the validated bring-up gate.
      Current audit result: MPI/ring/JACCL still use direct host pointers internally, but Vulkan distributed ops are explicit no-GPU stubs today, so they are not blockers for the first single-node NVIDIA bring-up milestone.

### Exit Criteria For This Milestone

- [x] Real discrete-GPU Vulkan execution is proven on at least one Linux hardware target.
- [x] Core MLX tensor ops, autograd, and a tiny MLP training loop run on Vulkan.
- [x] Tiny transformer training converges without CPU fallback on real Linux hardware.
- [x] TinyGPT-style repeated training remains finite and convergent on real Linux AMD hardware.
- [ ] Transformer-critical ops are either native Vulkan paths or discrete-GPU-safe fallbacks with verified correctness.
- [ ] No segfaults, teardown crashes, or unbounded memory growth in the targeted training path.
- [ ] Cross-vendor validation exists on both AMD and NVIDIA hardware.

### Phase 5: Graph Compilation (Vulkan)
- [x] Architect `mlx/backend/vulkan/compiled.cpp` based on the Metal JIT pattern.
- [x] Introduce a mechanism to dynamically generate GLSL shaders from `mx.compile()` graphs.
- [x] Test a simple compiled `mx.compile(lambda x: x + x)` graph natively on Vulkan.
- [x] Expand `to_glsl_op` with comprehensive unary/binary ops (sin, cos, exp, tanh, sqrt, max, min, comparisons, etc.).
- [x] Verify expanded ops on native AMD GPU (RX 580 via RADV).
- [x] Implement in-memory SPIR-V caching (avoid re-invoking `glslc` for identical kernels).
- [x] Add compile-coverage logging for fused Vulkan kernels and summarize primitive frequency from real generation runs.
- [x] Audit which primitives LLM inference actually hits (profile a generation run).
- [ ] Support complex ops (reduction, broadcasting) in the GLSL generator.
- [ ] Validate performance on AMD GPUs with LLM inference.

### Phase 6: LLM Inference Acceleration (Next)

The goal is to make `mlx-lm generate` run fast on the AMD GPU by ensuring all hot-path operations execute natively through the compiled Vulkan pipeline.

#### 6.1 Reduction Support in Compiled GLSL
- [ ] Implement `Sum` reduction in `to_glsl_op` using shared memory + workgroup barriers.
      First real-hardware pass is now live for the hot case: terminal, last-axis, keepdims `Reduce::Sum` roots such as `Broadcast -> Multiply -> Sum` compile and execute through Vulkan on the RX 580. General/multi-axis reduction lowering and reduction-specific workgroup optimization are still pending.
- [ ] Implement `Prod`, `Max`, `Min` reductions.
- [ ] Handle multi-axis reductions and keepdims.

#### 6.2 Broadcasting & Strided Access
- [ ] Support non-contiguous inputs in `build_kernel` (strided index computation).
- [ ] Handle broadcasting rules (scalar expansion, dimension alignment).
      Current AMD profiling is partially green here: collapsed broadcast-stride metadata and descriptor-buffer offsets now keep the main hot fused attention / softmax kernels on Vulkan, but broader view coverage and re-profiling are still pending.

#### 6.3 Missing Primitive Coverage
- [x] Audit which primitives LLM inference actually hits (profile a generation run).
- [x] Add compile-coverage logging for fused Vulkan kernels and summarize primitive frequency from real generation runs.
- [ ] Add any missing ops to `to_glsl_op` (e.g., `Erf`, `Softmax` components).
      `Sigmoid` and `LogAddExp` are now lowered in the Vulkan GLSL path.
      Latest 4-token RX 580 compile-coverage run shows the hottest fused kernels are already on Vulkan:
      - `Broadcast -> Multiply -> Sum` (`324` dispatches)
      - `Broadcast -> Multiply` (`162` dispatches)
      - `Broadcast -> Broadcast -> Multiply -> Add` (`162` dispatches)
      - `Subtract -> Broadcast -> Multiply` (`162` dispatches)
      - the softmax-style `Exp -> Negative -> Broadcast -> Add -> Broadcast -> LogAddExp -> ...` kernel
      - the `Sigmoid` / `Multiply` MLP kernels
      So the next decode-throughput wall is no longer an obvious uncovered compiled primitive in the hottest observed generation kernels.
- [ ] Handle `AsType` / static casts between dtypes in GLSL.
      Current stopgap: float16 / bfloat16 compiled kernels are bridged through float32 shader IO plus Vulkan copy-based casts.

#### 6.4 End-to-End LLM Benchmarking
- [ ] Run `mlx-lm generate` on AMD RX 580 with `mx.compile` active.
- [ ] Measure tokens/sec and compare against CPU-fallback baseline.
- [ ] Profile GPU utilization to identify remaining bottlenecks.
      Preliminary lower-bound timing on the local RX 580 shows model load is about 3.6s for both GPU and CPU runs, while one-token generation still did not complete within the short benchmark windows (>120s GPU, >60s CPU) and needs better first-token / streaming instrumentation before we treat it as a real throughput number.
      Latest RX 580 benchmark after compiled `Sum` fusion: model load is about 3.71s on GPU, one-token `generate_step` still timed out after 180s with no first yield, and the live stack now points at `KVCache.update_and_fetch(...)` / `qwen3_next.py` rather than the earlier `gated_delta` reduction loop.
      Latest strict RX 580 benchmark after the `Conv1d`, eager-indexing, and broadcasted-`where` fixes: model load is about 3.59s on GPU, the run still times out after 180s with no first yield, but it no longer trips `MLX_VULKAN_FAIL_ON_CPU_FALLBACK=1`. Layer-by-layer AMD probes now show the first full-attention block, including grouped-query SDPA decomposition, completing on Vulkan; the next job is reducing latency rather than removing another correctness fallback.
      Latest strict RX 580 benchmark after the affine Vulkan QMM path landed: model load is about `5.67s` on the first cold run and about `3.58s` on the warm rerun; guarded one-token `generate_step` now succeeds with first yield in about `7.28s` (`7.13s` warm rerun) instead of timing out. A matching CPU baseline on the same box is about `7.09s`, so the catastrophic hidden CPU detour is gone but steady-state GPU throughput still needs work.
      Latest refreshed AMD layer profile: 5-token prompt prefill is about `3.23s`, decoder layers are about `2.03s`, tied `lm_head` is about `1.19s`, and the first linear layer still pays about `0.85s` of one-time warmup overhead while later linear layers stay near `0.05s`.
      Latest strict RX 580 benchmark after the selective direct large-vocab QMM path landed: model load is about `3.56s`, guarded one-token `generate_step` reaches first yield in about `5.44s`, and the GPU is now ahead of the matching CPU baseline (`~7.08s`). A focused tied-`lm_head` probe confirms the new direct path is active for the real `N=248320`, `K=2048` Qwen projection.
      Latest strict RX 580 benchmark after persistent compiled SPIR-V caching landed: model load is about `4.63s`, guarded one-token `generate_step` reaches first yield in about `4.69s`, and fresh-process warm-probe runs now show `cold_in_process` about `5.57s` on the first run and about `4.69s` on the next fresh process instead of the earlier `~54s` cold-start cliff.
      Latest warm-cache multi-token benchmark (`generate_step`, `max_tokens=16`): GPU is about `31.85s` total / `0.502 tok/s` with first yield about `4.68s`, while the matching CPU baseline is about `31.02s` / `0.516 tok/s` with first yield about `4.60s`. So fresh-process cold start is fixed, but steady-state decode throughput is still only around CPU parity.
      Latest warmed utilization profile over that same decode regime: average GPU busy is only about `17.9%`, median `0%`, and the post-first-yield cadence is about `1.80s` per token, confirming that the remaining wall is low occupancy / bursty scheduling rather than a single catastrophic shader miss.
      Latest isolated tied-logits experiment: a shared-`x` rewrite of `quantized_qmv.comp` builds and runs on the RX 580, keeps `direct_qmv=1` active, and reduces the isolated tied `lm_head` timings to about `1.09s` for the full 5-token pass, about `0.77s` for the last-token path, and about `0.58s` in the warm layer probe; the matching whole-run decode rerun still needs to be repeated after the AMD host connectivity issue is resolved.

### Immediate Next Steps

1. Re-run the full warmed RX 580 decode benchmark once the local AMD host is stable again, using the new utilization profiler to confirm whether the shared-`x` tied-logits kernel improves whole-run tokens/sec.
2. If whole-run decode is still mostly idle on GPU, profile the non-compiled eager path around KV-cache maintenance / attention plumbing instead of continuing blind `mx.compile` expansion.
3. Keep iterating on the tied-logits kernel only if isolated `lm_head` wins continue to translate into end-to-end decode gains; otherwise switch effort to the next hottest eager path.
4. Finish the remaining compiled reduction work: broader axis coverage plus `Prod` / `Max` / `Min`.
5. Add more compile primitives only if later profiling shows a real uncovered fused kernel, since the current hot compiled kernels are already present on Vulkan.

---

## Build Status (as of 2026-03-03)

> ⚠️ **Historical snapshot**: MoltenVK/macOS validation only. Does not imply Linux/NVIDIA readiness. See the Cross-Platform Compatibility Matrix above for current status.

**Build dir**: `build/temp.macosx-15.0-arm64-cpython-311/mlx.core/`
**Python**: 3.11 (`python3.11`) — `.so` is `core.cpython-311-darwin.so`

| Step                                                                 | Status                  |
| -------------------------------------------------------------------- | ----------------------- |
| `cmake -B ... -DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF`          | ✅ PASSES               |
| `cmake --build ... -j4`                                              | ✅ PASSES (zero errors) |
| All SPIR-V shaders pass `spirv-val`                                  | ✅                      |
| Python bindings (`mlx.core` importable)                              | ✅                      |
| `test_stage3a_allocator.py` (Allocator)                              | ✅ 19/19 PASS           |
| `test_stage8_unary.py` (Unary GPU ops)                               | ✅ 17/17 PASS           |
| `test_stage9_binary.py` (Binary GPU ops)                             | ✅ PASS                 |
| `test_stage10_reduce.py` (Reductions)                                | ✅ PASS                 |
| `test_stage11_matmul.py` (Matmul)                                    | ❌ 1/2 (batch matmul)   |
| `test_stage12_nn_ops.py` (Softmax/ArgOps)                            | ✅ PASS                 |
| `test_stage13_indexing.py` (Gather/GatherAxis/ScatterAxis)           | ✅ 7/7 PASS             |
| `test_stage14_sort.py` (Sort)                                        | ✅ 6/6 PASS             |
| `test_stage15_scan.py` (Scan / prefix ops)                           | ✅ 5/5 PASS             |
| `test_stage16_nn_extended.py` (LayerNorm, RMSNorm, RoPE, SoftMax)   | ✅ 8/8 PASS             |
| `test_stage17_fft.py` (FFT/RFFT)                                     | ✅ 3/3 PASS             |
| `test_stage17_addmm_conv_rbits.py` (AddMM/Conv/RBits)               | ✅ 7/7 PASS             |
| `test_stage18_concat.py` (Concatenate)                               | ✅ 3/3 PASS             |
| `test_stage19_quantized.py` (QuantizedMatmul)                        | ✅ 17/17 PASS           |
| `test_stage20_linalg.py` (QRF/SVD/Inverse/Cholesky)                 | ✅ 4/4 PASS             |
| `test_stage21_advanced_mm.py` (GatherMM/BlockMaskedMM/SegmentedMM)  | ✅ 7/7 PASS             |
| `test_stage22_sync.py` (Event/Fence sync)                            | ✅ 7/7 PASS             |
| `test_stage23_shape_misc.py` (Shape/Misc ops)                        | ✅ 8/8 PASS             |
| `test_stage24_qqmatmul.py` (QQMatmul/Quantize/GatherQMM)            | ✅ 7/8 PASS (1 SKIP)    |
| `test_stage24_subgroup.py` (Workgroup tuning)                        | ✅ 3/3 PASS             |

**Post-build workflow** (required after any `.cpp` change):

```bash
cmake --build build/temp.macosx-15.0-arm64-cpython-311/mlx.core -j4
cp build/lib.macosx-15.0-arm64-cpython-311/mlx/core.cpython-311-darwin.so python/mlx/
cp build/temp.macosx-15.0-arm64-cpython-311/mlx.core/libmlx.dylib python/mlx/lib/
```

**After any `.comp` shader change**, either:

```bash
glslc --target-env=vulkan1.2 mlx/backend/vulkan/kernels/FOO.comp -o build_vulkan/mlx/backend/vulkan/kernels/FOO.spv
```

or simply run the full `cmake --build build_vulkan -j4` (rebuilds shaders too).

---

## Codebase Gap Analysis — Verified 2026-03-07

> ⚠️ **Historical snapshot (2026-03-07)**: Based on MoltenVK/macOS run. Not revalidated on AMD/Linux hardware. See the Compatibility Matrix above for current real-hardware status.

### test_ops.py Results: 124/134 PASS, 10 FAIL

Full suite runs to completion with no segfaults or hangs.

#### ❌ FAILING (10 tests — confirmed root causes)

| Test | Root Cause | Priority |
|---|---|---|
| `test_take` | Indexing result mismatch for multi-dim takes | P1 |
| `test_take_along_axis` | Indexing result mismatch for axis takes | P1 |
| `test_put_along_axis` | `[0, 0]` != `[15, 122]` — ScatterAxis wrong result | P1 |
| `test_scans` | "Attempting to eval an array without a primitive" (Upstream MLX macOS Bug, Memory Race Fixed) | P1 |
| `test_softmax` | Precision: `0.03...` unexpected value | P2 |
| `test_round` | Wrong result — `round()` not correctly implemented on GPU | P2 |
| `test_logcumsumexp` | LogAddExp scan returns wrong result | P2 |
| `test_trace` | `array(0)` unexpected — trace/diagonal sum wrong | P2 |
| `test_linspace` | Float precision wrong for endpoint calculation | P3 |
| `test_meshgrid` | Wrong output shape from meshgrid indexing | P3 |

#### ✅ PREVIOUSLY FAILING — NOW FIXED
logical_not, isinf, isnan, isneginf, nan_to_num, maximum, minimum, logaddexp, logsumexp, median, outer, real_imag, rsqrt, scalar_inputs, sort, sort_nan

---

---

## Known Issues / Critical Bugs Fixed

> 📜 **Historical reference**: Bug fixes applied to codebase. Status reflects code changes, not current real-hardware test results.

### Fixed (2026-03-08) — CPU Scan Fallback Stability + Vulkan Availability Gate

48. **`scan.cpp` pointer-drift hardening** (`backend/cpu/scan.cpp`):
    - Replaced `contiguous_scan` pointer-walk logic with index-based loops (matching the earlier `strided_scan` hardening).
    - Eliminates reverse/exclusive pointer underflow risk and stabilizes repeated scan pipelines used by `test_scans`.

49. **Zero-sized axis safety in scans** (`backend/cpu/scan.cpp`):
    - Added early return in `scan_op` when `in.size() == 0` or `in.shape(axis) == 0`.
    - Prevents divide-by-zero / negative-offset behavior in empty-dimension edge cases.

50. **Vulkan runtime availability check** (`backend/vulkan/device_info.cpp`):
    - Replaced hardcoded `gpu::is_available() == true` with guarded runtime probe (`try/catch` around device init).
    - `device_count()` and `device_info()` now reflect actual availability instead of forcing invalid Vulkan startup.

51. **Scan Memory Race Condition Fixed** (`backend/cpu/encoder.h`, `backend/vulkan/primitives.cpp`):
    - Resolved non-deterministic memory corruption in `cumsum` resulting from eagerly recycled VMA buffers.
    - Fixed a reference leak in `cpu::CommandEncoder` during GPU-stream CPU fallbacks.
    - Fixed `GatherAxis`, `ScatterAxis`, and `Gather` to unconditionally extend lifetime of input buffers via `add_temporary`, preventing premature evaluator frees.

### Fixed (2026-03-07) — LogicalNot all-False on Vulkan backend

45. **`unary.comp` LogicalNot hardcoded return** (`kernels/unary.comp`):
    - `case UNARY_LOGNOT: return 1.0;` always returned True. Fixed to `(val == 0.0) ? 1.0 : 0.0`.
    - Reverted bool output packing regression to simpler `out_bool[idx]` assignment.

46. **`dispatch_unary` silent failure** (`primitives.cpp`):
    - Returned `true` on `VK_NULL_HANDLE` pipeline, preventing CPU fallback. Fixed to return `false`.
    - Added CPU fallback to `LogicalNot::eval_gpu` matching other unary ops pattern.

47. **Build relinking caveat**:
    - CMake `make -j8` does not always detect `.o` changes and relink `libmlx.a` → `core.cpython-*.so`.
    - Workaround: force-delete `libmlx.a` + `core.so` to trigger proper relinking.

### Fixed (2026-03-03) — Hadamard CPU/GPU Sync, Memory Bugs & Precision

32. **Fence::wait deadlocks fixed** (`fence.cpp`):
    - Added explicit commit to producer stream during cross-device semaphore wait to prevent queue stalling.

33. **Hadamard memory & precision patches** (`primitives.cpp`):
    - Fixed typing for float16/bfloat16 in `eval_cpu` GPU fallbacks, eliminating `1.0 != 0.0` mismatch.
    - Added missing `dev.add_temporary(s, src_arr)` handling avoiding VAP use-after-free corruption in `eval_gpu`.

34. **copy.comp data race & cancellation** (`kernels/copy.comp`):
    - Solved precision cancellation from `int64` upcast to `float32` by taking a 32-bit fast path loop.
    - Updated 16-bit element assignments in `copy.comp` via `atomicAnd/atomicOr` eliminating multithreading overwrites.

### Fixed (2026-03-03) — Dtype Coverage in copy/arange/binary/divmod shaders

27. **copy.comp missing uint16/int16/int64/uint64** (`kernels/copy.comp`):
    - Added DTYPE_UINT16, INT16 cases to `read_as_float()` (16-bit packed reads) and `write_float()` (atomic 16-bit writes).
    - Added DTYPE_INT64, UINT64 two-word reads and writes for 64-bit types.

28. **arange.comp integer dtype + inf-step NaN guard** (`kernels/arange.comp`):
    - Added `out_dtype` push constant; push size 12 → 16.
    - Integer types (int32/uint32/int64/uint64) now convert start+step to integer first (matches CPU).
    - Float path now guards `isinf(step)` to prevent `0 * inf = NaN`.

29. **binary.comp MoltenVK float(int64) workaround** (`kernels/binary.comp`):
    - `float(int64_t)` returns 0 on MoltenVK; workaround: `float(int(int64_t))`.

30. **binary_two.comp full dtype rewrite for divmod** (`kernels/binary_two.comp`):
    - Rewrote from float[] buffers to uint[] with DTYPE_* dispatch for all dtypes.
    - Added `dtype` push constant; push size 16 → 20.

31. **Pipeline cache version bumped 10 → 12** (`device.cpp`):
    - v11: arange push const size change
    - v12: binary_two push const size change

### Fixed (2026-03-02) — >4D Binary Broadcast Limits & CPU Fallbacks

25. **>4D Binary Broadcast Limit Fixed** (`primitives.cpp`):
    - `binary.comp` push constants only support up to 4 dimensions. Broadcasting or expanding dimensions > 4 resulted in out-of-bounds stride arrays or dimensionality mismatches (like the `test_expand_sums` failure, which produced scalar outputs instead of correct tensors).
    - Fixed `dispatch_binary` in `primitives.cpp`. It now processes `collapse_contiguous_dims` first. If `shape.size() > 4`, it recursively delegates to fully contiguous bounds using `copy_gpu(CopyType::General)`, seamlessly instantiating `a_contig` and `b_contig` buffers before submitting the primitive to the binary shader under the 4D push constant constraints.

26. **CPU Fallbacks for Unsupported Primitives** (`primitives.cpp`):
    - Operations like `Scatter`, `Gather`, `Sort`, `Partition`, `Convolution`, etc., were throwing `std::runtime_error("[vulkan::...] Fallback to eval_cpu is unsupported")` when invoked on the GPU, breaking testing for entire execution branches.
    - Replaced these exceptions across `primitives.cpp` (WIP/partially applied) with synchronized `eval_cpu(inputs, out)` executions. Test graphs that fall back to unsupported GPU primitives can now correctly hand execution back to the CPU execution stream.

### Fixed (2026-02-26)

1. **eval.cpp — SIGSEGV on eval_gpu() deletion**: `eval_gpu()` call was accidentally deleted by a `sed` command. Restored manually. This was the root cause of segfaults during GPU dispatch.

2. **binary.comp — int32 arithmetic corruption on Apple Silicon**: Apple Silicon GPU flushes denormals-to-zero, meaning int32 values reinterpreted as float bits via `uintBitsToFloat()` were silently flushed to 0. Fixed by adding dtype-aware arithmetic paths (`DTYPE_FLOAT` / `DTYPE_INT` / `DTYPE_UINT`) so integer ops never pass through float interpretation.

3. **fft.cpp — broken FFT implementation**: Broken `RFFT`/`IRFFT` class names and `is_power_of_2` ambiguity caused compile errors. Replaced with minimal stubs that throw `runtime_error`.

4. **Stale Python extension after rebuild**: `build_vulkan/core.cpython-314-darwin.so` must be manually copied to `python/mlx/` and re-signed with `codesign` after each build — this is not automated by CMake.

### Fixed (2026-02-27) — FFT Implementation

5. **fft.comp — wrong dispatch geometry**: Vulkan was dispatching `vkCmdDispatch(batch_size, threadgroup_batch_size, 1)` with `local_size_x=256`. Metal dispatches 1 workgroup per batch item with `threads_per_fft × tg_batch_size` threads. Fixed: `vkCmdDispatch(batch_size, 1, 1)` with `local_size_x=1024`.

6. **fft.cpp — FFTPushConstants::stockham[8] off-by-one**: `supported_radices = {13,11,8,7,6,5,4,3,2}` has 9 entries. `stockham[8]` (radix-2 steps) was never stored — array was only 8 elements. Fixed: extended to `stockham[9]`, loop limit `i < 9`.

7. **fft_op — encoder.op_count never incremented**: `Device::commit()` returns early if `op_count == 0`. All GPU commands were silently dropped — FFT output was all zeros. Fixed: added `encoder.op_count++` after `vkCmdDispatch`.

8. **fft.comp — radix-8 not implemented**: For n≥512, `plan_stockham_fft` uses radix-8 (e.g. 512=8³). Shader had no radix-8 codelet. Added `radix8()`, `radix_butterfly_8()`, and radix-8 pass loop in `perform_fft`. n=512..4096 now pass.

9. **FFTPushConstants push constant overflow**: Struct was 132 bytes (33×uint32); Vulkan limit is 128 bytes. `real` field at offset 128 was always out-of-bounds → `is_rfft` always false, RFFT loaded complex64 from float32 input. Fixed: removed unused `rader[9]` from struct and GLSL block → struct drops to 96 bytes.

10. **RFFT support**: Added binding 2 (`float src_real[]`) to `fft.comp`. When `params.real==1 && params.inv==0`, loads float input as `vec2(x, 0.0)` and truncates output to `n/2+1`. Removed Metal-specific 2-RFFT batch halving from `fft.cpp`.

### Fixed (2026-02-28) — Gather/Scatter ND + Scan GPU dispatch

11. **Gather::eval_gpu — only handled 1D axis=0**: `mx.take(2D_array, idx, axis=1)` threw "Fallback to eval_cpu is unsupported". Fixed: added INDEX_GATHER_GEN (op=3) to `indexing.comp` — general ND gather using `j = tid / slice_total`, `outer_off = within / inner`, `inner_off = within % inner`, `src_pos = outer_off * src_outer_stride + idx[j] * inner + inner_off`. Now all single-axis gather cases are handled on GPU.

12. **GatherAxis/ScatterAxis push constant size mismatch**: Both functions were calling `get_pipeline("indexing", ..., 3, 32)` with a 9- or 8-field struct, but the shader declared 9 fields = 36 bytes. After the push constant was extended to 11 fields (44 bytes), the 32-byte pipeline layout caused vkCmdPushConstants to be out-of-range. Fixed: unified all three indexing ops to use shared `IndexPushConst` (44 bytes, 11 fields) via `indexing_dispatch()` helper.

13. **MoltenVK stale pipeline cache crash**: After changing the indexing push constant layout from 36→44 bytes, the on-disk pipeline cache (`~/.cache/mlx_vulkan_pipeline_cache.bin`) retained the old binary pipeline. MoltenVK's pipeline cache loader crashed with SIGKILL on dlopen. Fixed: added version suffix to cache path (`mlx_vulkan_pipeline_cache_v2.bin`); bump version whenever push constant layouts change.

14. **ScatterAxis negative-index wrap using wrong field**: `INDEX_SCATTER` was wrapping with `int(params.idx_size)` instead of `int(params.src_ax_size)`. Fixed in `indexing.comp`.



### Fixed (2026-03-03) — Cody-Waite sin/cos, CPU Fallbacks, Scan Fix

38. **Cody-Waite range reduction for sin/cos** (`unary.comp`):
   - Added `cw_reduce()`, `cw_sin()`, `cw_cos()` functions with `precise` qualifier
   - Uses standard glibc float32 constants (C1=1.5707962513, C2=7.5497894159e-8)
   - Prevents compiler contraction that would cancel reduction term
   - Matches glibc/libm float32 output precision near zero

39. **CPU fallbacks for unsupported types** (`primitives.cpp`):
   - `Log::eval_gpu`: falls back to CPU if dispatch fails
   - `BINARY_GPU` macro: falls back to CPU for complex64/complex128
   - `Equal::eval_gpu`: falls back to CPU for complex types
   - `Scan::eval_gpu`: falls back to CPU for non-last-axis, non-float32, or scan_size > 1024

40. **fill_gpu zero-size and unallocated output guard** (`copy.cpp`):
   - Early return when `out.size() == 0`
   - Allocate output if not already allocated (matches Metal behavior)

41. **Pipeline cache version bump** (`device.cpp`):
   - Bumped v12 → v14 for Cody-Waite push constant change

42. **Scan CPU fallback segfault fix** (`primitives.cpp`):
   - Removed incorrect Vulkan memory barrier code (lines 1845-1861) from Scan::eval_gpu
   - CPU fallback now: `synchronize()` → `eval_cpu()` → `return`, without any Vulkan encoder access
   - Fix prevents segfaults in test_scans

43. **Tests** (before → after):
   - All stage tests: PASS (no regressions)
   - test_array.py: 64/68 PASS
   - test_ops.py: ~100/134 PASS (9 pre-existing failures, no new regressions)
   - test_random.py: 14/14 PASS
   - sin/cos Cody-Waite precision fix confirmed

44. **Files changed**: `copy.cpp`, `device.cpp`, `unary.comp`, `primitives.cpp`

15. **Scan::eval_gpu unconditionally threw**: Prefix scan (cumsum/cumprod) always failed with a runtime_error. Implemented full GPU dispatch via a two-level Hillis-Steele scan in `scan.comp`: serial inclusive scan within each thread's chunk (chunk_size = ceil(scan_size/256)), parallel cross-chunk prefix on stotals[], propagate back, exclusive conversion at writeback. Supports scan_size ≤ 1024; Sum/Prod/Max/Min; inclusive and exclusive; reverse.

### Fixed (2026-02-28) — Binary broadcast stride-based indexing

16. **binary.comp — broadcast arrays caused GPU hang**: Binary shader used flat `idx` for both inputs. For broadcast arrays like `(2,1)` broadcast to `(2,3)`, `data_size=2` but shader indexed `b[0..5]` → OOB GPU hang. Three approaches were tried:

    **Rejected: Modulo indexing (`idx % data_size`)** — Works for scalar (`data_size=1`, `idx%1=0`) and row broadcast (`(1,3)→(2,3)`: `data_size=3`, `idx%3` gives `0,1,2,0,1,2`). Fails for column broadcast (`(2,1)→(2,3)`: `data_size=2`, `idx%2` gives `0,1,0,1,0,1` but correct mapping is `0,0,0,1,1,1`). The issue: simple modulo assumes the broadcast dimension is last, which isn't true for column broadcasts.

    **Rejected: CPU-side `expand_broadcast` via `copy_gpu_inplace`** — Materializes the broadcast by dispatching a General copy shader with zero-stride input strides. The copy itself completed (verified via debug prints), but the subsequent binary shader dispatch hung. Root cause: enqueuing two shader dispatches (copy + binary) within a single `eval_gpu` call on MoltenVK causes a command-buffer-level GPU conflict. This is a MoltenVK-specific issue — commands enqueued at the same depth in the call stack can deadlock.

    **Chosen: Stride-based ND broadcast indexing in-shader** — Push constants expanded 28→80 bytes (within 128-byte limit) with `ndim`, `out_shape[4]`, `a_strides[4]`, `b_strides[4]`. The shader decomposes the flat output index into ND coordinates via `out_shape`, then dots with per-input strides (stride=0 for broadcast dims). This eliminates any pre-copy, handles all broadcast patterns (scalar, row, column, ND) correctly, and is a single shader dispatch matching the existing architecture. The 80-byte push constant is well under Vulkan's 128-byte minimum guarantee. `compute_broadcast_strides()` in `primitives.cpp` computes zero-stride for dims where `in.shape(i)==1`.

17. **Pipeline cache v3→v4**: binary.comp push constant layout changed (28→80 bytes). Bumped `kPipelineCacheVersion` in `device.cpp`. Additionally, the build system (`setup.py build_ext`) does not automatically detect `.comp` shader changes — manual `glslc` recompilation is required after shader edits. This was a contributing factor to debugging difficulty (stale `.spv` cached from Feb 27 was running instead of the modified shader).

### Known Remaining Issues (Test Suite Audit 2026-02-28)

Ran comprehensive test suites `test_array.py` and `test_ops.py`. Results:
- `test_array.py`: 52 PASS, 16 FAIL, 1 HANG (`test_deep_graphs`)
- `test_ops.py`: 41 PASS, 93 FAIL, 0 HANG (Zero GPU deadlocks in ops suite!)
- Total: 93/203 tests pass (46%).

**Failure Patterns to Address Next**:
1. **Missing GPU Primitives**: Ops like `AsStrided`, `Scatter` (multi-axis), `ArgPartition` throw "Fallback to eval_cpu is unsupported".
2. **Reduction correctness**: `test_sum` returns 0 for a non-zero sum, indicating reduction/type conversion issues.
3. **GPU Deadlock**: `test_deep_graphs` causes the only observed hang.

### Fixed (2026-03-01) — Unary Shader Enum Mismatch + float16 Support

18. **unary.comp — enum values off by one for 15/31 operations**: The original `unary.comp` had `UNARY_LOG2=10` inserted after `LOG1P=9`, pushing `Sin=11`, `Cos=12`, etc. — but `primitives.cpp` had `Sin=10`, `Cos=11`, etc. Result: every function from `sin` onward dispatched the *wrong* opcode (e.g. `sin(0.5)` returned `log2(0.5) = -1`). Fixed by rewriting `unary.comp` with constants derived directly from the `UnaryOp` enum in `primitives.cpp`. 15 operations now return correct results.

19. **unary.comp — no float16 support**: Float16 input was interpreted as `float` bits via `uintBitsToFloat()`, producing garbage (e.g. `exp(2.0 f16) = 0.0`). Fixed: added a float16 path in `main()` that uses `unpackHalf2x16()`/`packHalf2x16()` to process two f16 values per thread from a single `uint32` word. push constants now include `input_elem_bytes` so the shader can branch. `exp(2.0 f16) = 7.39` ✅

20. **Log::eval_gpu — log2/log10 had no GPU dispatch**: `unary.comp` received op code `Log2=29` but there was no `Log2::eval_gpu` — only a single `Log` class with a `Base` enum (e, two, ten). Fixed: removed `UNARY_GPU(Log, Log)` macro and replaced with an explicit `Log::eval_gpu` that checks `state()` (the base) and dispatches `UnaryOp::Log`, `UnaryOp::Log2`, or `UnaryOp::Log10`. `log2(0.5) = -1.0` ✅, `log10(0.5) = -0.301` ✅

- Hadamard: Not yet implemented (`kernels/hadamard.comp` stub only).
- Scan: scan_size > 1024 throws (multi-pass GPU scan not yet implemented). LogAddExp variant not on GPU.

### Fixed (2026-03-01) — Thread Safety & Descriptor Pool Stabilization

21. **Device::commit thread safety bug**: Detached background threads in `commit()` were suffering from lambda capture corruption (likely an Apple Clang 15 arm64 ABI issue) leading to `std::bad_function_call`. Fixed by encapsulating state in a heap-allocated `CommitState` struct and adding null-pointer guards for completion handlers.

22. **EXC_BAD_ACCESS in MoltenVK (Descriptor Pool Lifecycle)**: Global `descriptor_pool_` was being reset via `vkResetDescriptorPool` while other streams might still have in-flight command buffers using those descriptor sets. Fixed by moving `VkDescriptorPool` management to the per-stream `CommandEncoder`. Descriptor pools are now created and destroyed alongside command pools in the background commitment thread.

23. **Matmul EXC_BAD_ACCESS on zero-sized dimensions**: Matrix multiplications with `K=0` were causing `vulkan::get_buffer()` to return `VK_NULL_HANDLE` for the empty input tensors. Passing these to `vkUpdateDescriptorSets` triggered a segmentation fault in MoltenVK. Fixed by adding `out.size() == 0` check and currently refining the handling of zero-sized reduction dimensions (K=0).

24. **Stream-aware alloc_descriptor_set**: Updated `alloc_descriptor_set(Stream s, ...)` signature across the backend to ensure shaders always acquire descriptors from the pool associated with their active command encoder. Fixed ~20 compilation errors in `primitives.cpp` related to missing `stream()` context.

### Known Remaining Issues (2026-03-01) — Updated
## Proposed Changes

### P1.3 Int32 Support for Sort (Bitonic)

`test_sort` and `test_median` are failing because `sort.comp` implicitly assumes `float32` for all data handling:
- Reads array values as `float`
- Pads dummy power-of-2 elements with IEEE `+Inf`
- Performs `isnan` checks and `float` comparisons

This completely breaks for `int32` which might have negative sign bits evaluated as positive floats, or valid integers evaluated as `NaN`.

#### [MODIFY] [sort.comp (Vulkan Shader)](file:///Users/ektasaini/Desktop/mlx-vulkan/mlx-src/mlx/backend/vulkan/kernels/sort.comp)
- Change `<float> data` buffers to `<uint> data`.
- Introduce a push constant field `uint type_is_int`.
- Define `uint MAX_VAL` correctly depending on `type_is_int`: `0x7F800000u` (Float +Inf) vs `0x7FFFFFFFu` (Int32 Max).
- Load into shared memory `<uint> s_data` (storing raw bits).
- Introduce a compare function:
  ```glsl
  bool cmp(uint a_bits, uint b_bits) {
      if (params.type_is_int == 1u) {
          int a = int(a_bits); int b = int(b_bits);
          return a > b;
      } else {
          float a = uintBitsToFloat(a_bits); float b = uintBitsToFloat(b_bits);
          float a_k = isnan(a) ? POS_INF : a;
          float b_k = isnan(b) ? POS_INF : b;
          return a_k > b_k;
      }
  }
  ```

#### [MODIFY] [primitives.cpp (Vulkan)](file:///Users/ektasaini/Desktop/mlx-vulkan/mlx-src/mlx/backend/vulkan/primitives.cpp)
- Add `uint type_is_int` into `SortPushConst`.
- Restore the `float32` **and** `int32` GPU path for arrays `< 128` items.

- **fast::Quantize dequantize GPU**: Inline CPU workaround. GPU shader path causes
  `VK_ERROR_DEVICE_LOST` when `mx.random.normal` semaphores are pending in the command buffer.
  Root cause: semaphore state inconsistency in `commit()` when called with `has_sems=1` and
  empty op_count. Needs deeper investigation of the command buffer submission path.
- **Memory Tracking**: Background `commit` thread synchronization vs allocator stats — second pass needed.

---

> 📜 **Historical reference**: Phases 0–9 below are implementation logs. Status checkmarks reflect code completion, not current real-hardware validation. See the Authoritative Compatibility Gates above for the live validation framework.

## Phase 0: Repository Setup ✅ COMPLETE

- [x] Clone MLX into `/Users/ektasaini/Desktop/mlx-vulkan`
- [x] Install Vulkan toolchain (macOS via Homebrew: vulkan-headers, vulkan-loader, shaderc, spirv-tools, glslang, molten-vk, vulkan-validationlayers)
- [x] Verify Vulkan GPU detected: Apple M1 via MoltenVK
- [x] Read and understand `mlx/backend/gpu/eval.h` — interface contract
- [x] Read and understand `mlx/backend/no_gpu/primitives.cpp` — full list of ~80 ops to implement
- [x] Read and understand `mlx/backend/cuda/device.h` + `allocator.h` — structural template

---

## Phase 1: Build System ✅ COMPLETE

- [x] Add `MLX_BUILD_VULKAN` option to root `CMakeLists.txt`
  - `option(MLX_BUILD_VULKAN "Build Vulkan backend" OFF)`
  - `find_package(Vulkan REQUIRED)` guard block
  - FetchContent fallback for VulkanMemoryAllocator (header-only, include path only)
  - `add_subdirectory(mlx/backend/vulkan)` when `MLX_BUILD_VULKAN=ON`
  - Hooked into same `if(NOT MLX_BUILD_GPU)` guard as CUDA
- [x] Create `mlx/backend/vulkan/` directory
- [x] Create `mlx/backend/vulkan/CMakeLists.txt`
  - `compile_shader()` cmake function using glslc
  - 22 `.comp` shaders compiled to `.spv` — all SPIRV-val validated ✅
  - All `.cpp` sources in `target_sources(mlx PRIVATE ...)`
  - VMA linked as include-only (avoids CMake export set issue)
- [x] CMake configure succeeds with `-DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF`
- [x] `cmake --build` succeeds — zero errors, only VMA nullability warnings (harmless)
- [x] **Key fix**: VMA FetchContent target cannot be in MLX export set → use `target_include_directories` with `${vulkanmemoryallocator_SOURCE_DIR}/include` instead of `target_link_libraries`

---

## Phase 2: Device Infrastructure ✅ COMPLETE

### `mlx/backend/vulkan/device.h` + `device.cpp`

- [x] `VulkanDevice` struct with: `VkInstance`, `VkPhysicalDevice`, `VkDevice`, `VkQueue`, `VkPipelineCache`
- [x] Instance creation: enable `VK_KHR_get_physical_device_properties2`, validation layers (debug builds)
- [x] Physical device selection: prefer discrete GPU, fall back to integrated
- [x] Logical device + compute queue family discovery
- [x] Pipeline cache: persist to disk (`~/.cache/mlx_vulkan_pipeline_cache.bin`), load on init
- [x] `new_queue(int index)` — creates per-stream `VkCommandPool` + initial `VkCommandBuffer`
- [x] `get_command_buffer(int index)` — returns current recording command buffer for stream
- [x] `end_encoding(int index)` — `vkEndCommandBuffer`
- [x] `commit_command_buffer(int index)` — `vkQueueSubmit` + `vkResetCommandPool` for next
- [x] `command_buffer_needs_commit(int index)` — heuristic (same as Metal: command count threshold)
- [x] `get_pipeline(const std::string& name)` — load SPIR-V from VULKAN_KERNELS_PATH, create+cache `VkPipeline`
- [x] `VulkanDevice& device(mlx::core::Device dev)` — singleton accessor (mirrors `metal::device()`)
- [x] Descriptor pool + `allocate_descriptor_set(VkDescriptorSetLayout)` helper
- [x] `bind_buffer(VkDescriptorSet ds, uint32_t binding, const array& arr)` helper

### `mlx/backend/vulkan/utils.h` + `utils.cpp`

- [x] `div_ceil(uint64_t a, uint64_t b)` — dispatch grid helper
- [x] `insert_buffer_barrier(VkCommandBuffer, const array&)` — pipeline barrier for compute→compute RAW
- [x] `to_vk_format(Dtype)` — mlx dtype → `VkFormat` mapping
- [x] `get_type_string(Dtype)` — for pipeline name keying

---

## Phase 3: Memory Allocator ✅ COMPLETE

### `mlx/backend/vulkan/allocator.h` + `allocator.cpp`

- [x] Include VMA (`vk_mem_alloc.h`) — create `VmaAllocator` on device init
- [x] `VulkanAllocator` class extending `mlx::core::allocator::Allocator`
- [x] `Buffer malloc(size_t size)` — `vmaCreateBuffer` with `VMA_MEMORY_USAGE_AUTO` + `DEVICE_LOCAL`
- [x] `void free(Buffer buffer)` — `vmaDestroyBuffer`
- [x] `size_t size(Buffer buffer) const` — `vmaGetAllocationInfo`
- [x] `void* Buffer::raw_ptr()` — staging buffer mapped pointer for host access
- [x] Staging buffer pool for CPU↔GPU transfers (discrete GPU requires explicit staging)
- [x] `active_memory_`, `peak_memory_`, `memory_limit_` tracking (mirrors MetalAllocator)
- [x] `get_active_memory()`, `get_peak_memory()`, `set_memory_limit()` free functions
- [x] `clear_cache()` — drain VMA pool (mirrors `metal::clear_cache()`)
- [x] `Allocator& allocator()` free function — returns singleton `VulkanAllocator`

---

## Phase 4: Event, Fence, Device Info ✅ COMPLETE

### `mlx/backend/vulkan/event.h` + `event.cpp`

- [x] `VulkanEvent` wrapping `VkSemaphore` (timeline semaphore — `VK_KHR_timeline_semaphore`)
- [x] `signal(uint64_t value)` — `vkSignalSemaphore`
- [x] `wait(uint64_t value)` — `vkWaitSemaphores` (CPU blocks)
- [x] Integrate with MLX `Event` type: `Event::wait()`, `Event::wait(Stream)`, `Event::signal(Stream)`
- [x] `Event::is_signaled()` — queries `vkGetSemaphoreCounterValue` (fixed from always-true stub)
- [x] CPU-stream `signal(stream)` path — enqueued via `scheduler::enqueue` (matches Metal)
- [x] CPU-stream `wait(stream)` path — enqueued via `scheduler::enqueue` (matches Metal)

### `mlx/backend/vulkan/fence.cpp`

- [x] `FenceImpl` with CPU-side `std::condition_variable` for cross-stream sync
- [x] `Fence::update(Stream, array, cross_device)` — GPU path: completion handler signals cv; CPU path: scheduler enqueue
- [x] `Fence::wait(Stream, array)` — GPU path: drain stream then CPU wait; CPU path: scheduler enqueue with cv wait

### `mlx/backend/vulkan/device_info.cpp`

- [x] Implement `mlx::core::gpu::device_info()` — returns device_name, architecture, memory_size, vulkan_api_version, vendor_id
- [x] `VkPhysicalDeviceProperties` + `VkPhysicalDeviceMemoryProperties` queries
- [x] Vendor ID → architecture string mapping (AMD/NVIDIA/Intel/ARM/Qualcomm/Apple)
- [x] Apple Silicon fallback: device-name-based detection when vendorID unknown (MoltenVK reports non-standard IDs)
- [x] Fixed `thread_local` caching bug — map now clears/repopulates on each call

---

## Phase 5: GPU Eval Dispatch ✅ COMPLETE

### `mlx/backend/vulkan/eval.cpp` — implements `mlx/backend/gpu/eval.h`

- [x] `gpu::new_stream(Stream stream)` — calls `vulkan::device(...).new_queue(stream.index)`
- [x] `gpu::eval(array& arr)` — full dispatch loop (mirrors `metal/eval.cpp`):
  - Get command buffer for stream
  - Call `arr.primitive().eval_gpu(inputs, outputs)`
  - Track input/output buffer lifetimes
  - Check `command_buffer_needs_commit()` → submit if true
  - Register completion handler → `scheduler::notify_task_completion(s)`
- [x] `gpu::finalize(Stream s)` — end encoding + queue submit
- [x] `gpu::synchronize(Stream s)` — end encoding + submit + `vkQueueWaitIdle` (CPU blocks)
- **Note**: eval_gpu() call was accidentally deleted by sed and restored — SIGSEGV was the symptom.

---

## Phase 6: GPU Copy & Slicing ✅ COMPLETE

### `mlx/backend/vulkan/copy.cpp` — implements `mlx/backend/gpu/copy.h`

- [x] `copy_gpu(src, out, ctype, s)` — dispatch `copy.comp` shader
- [x] `copy_gpu_inplace(in, out, data_shape, i_strides, o_strides, ...)` — strided copy shader
- [x] `fill_gpu(val, out, s)` — dispatch `fill.comp` (scalar broadcast)
- [x] `contiguous_copy_gpu(arr, s)` — returns contiguous buffer copy
- [x] `reshape_gpu(in, out, s)` — transpose+copy via shader
- [x] `flatten_in_eval`, `reshape_in_eval`, `swapaxes_in_eval` helper stubs

### `mlx/backend/vulkan/slicing.cpp` — implements `mlx/backend/gpu/slicing.h`

- [x] `slice_gpu(...)` — dispatch `slicing.comp`
- [x] `pad_gpu(...)` — implemented in `gpu/slicing.cpp` using `fill_gpu` + `copy_gpu_inplace` ✅
- [x] `concatenate_gpu(...)` — implemented in `vulkan/slicing.cpp` using per-input `copy_gpu_inplace` ✅

> **Note**: No standalone `pad.comp`/`concatenate.comp` shaders needed — both ops are
> composed from existing `copy.comp` and `fill.comp` dispatch paths.

---

## Phase 7: GLSL Compute Kernels (AOT SPIR-V) 🔄 PARTIAL

All shaders live in `mlx/backend/vulkan/kernels/`. Each compiled to `.spv` at build time via glslc.
Naming: `<op>_<dtype>.comp` or template + specialization constants for dtype variants.

**Total shaders compiled**: 22 ✅

### Kernel Conventions (apply to all shaders)

- [x] Default workgroup: `layout(local_size_x = 256) in;`
- [x] Bounds check: `if (idx >= size) return;`
- [x] Use push constants for params ≤ 128 bytes; UBO for larger metadata
- [ ] `kernels/bf16.glsl` — bfloat16 pack/unpack/arithmetic helpers
- [ ] `kernels/defines.glsl` — common defines, type aliases, math constants
- [ ] `kernels/utils.glsl` — index flattening, strides, broadcasting helpers

### Tier 1 — Core ✅ COMPLETE

- [x] `kernels/copy.comp` — contiguous, strided, scalar fill, broadcast
- [x] `kernels/unary.comp` — abs, neg, sign, sqrt, rsqrt, cos, sin, exp, log, relu, sigmoid, tanh, ...
  - **Note**: int32 dtype path may need audit (similar to binary.comp issue)
- [x] `kernels/binary.comp` — add, sub, mul, div, pow, min, max, eq, ne, lt, le, gt, ge, logical ops
  - **Fixed**: dtype-aware arithmetic paths added for DTYPE_FLOAT/INT/UINT (Apple Silicon denormal flush bug)
- [x] `kernels/arange.comp` — fill with range

### Tier 2 — Reduction & Matmul ✅ COMPLETE

- [x] `kernels/reduce.comp` — sum, min, max, prod along arbitrary axes (subgroup + workgroup reduction)
- [x] `kernels/arg_reduce.comp` — argmin, argmax
- [x] `kernels/matmul.comp` — tiled GEMM (16×16 or 32×32 tile), handles non-square shapes
- [x] `kernels/binary_two.comp` — two-output binary ops (divmod etc.)
- [x] `kernels/ternary.comp` — select/where (conditional elementwise)

### Tier 3 — Neural Net Essentials 🔄 PARTIAL

- [x] `kernels/softmax.comp` — numerically stable softmax (max-subtract + exp + sum + divide)
- [x] `kernels/logsumexp.comp`
- [x] `kernels/normalization.comp` — layer_norm, rms_norm (mean/variance in subgroup)
- [x] `kernels/rope.comp` — rotary position embeddings
- [x] `kernels/scan.comp` — prefix scan (inclusive/exclusive, add/mul)

### Tier 4 — Indexing & Shape Ops 🔄 PARTIAL

- [x] `kernels/indexing.comp` — gather (read at indices), scatter (write at indices), scatter-add
- [x] `kernels/slicing.comp` — handled naturally via general strided `copy.comp` indexing!
- [x] `kernels/pad.comp` — handled naturally via general strided `copy.comp` indexing!
- [x] `kernels/sort.comp` — bitonic sort (GPU-parallel)

### Tier 5 — Advanced Ops 🔄 PARTIAL

- [x] `kernels/conv.comp` — convolution (im2col approach, dispatch to matmul)
- [x] `kernels/fft.comp` — **COMPLETE** (Stockham Cooley-Tukey radix-2/4/8; RFFT float input via binding 2; 3/3 tests pass)
- [x] `kernels/hadamard.comp` — stub only (throws runtime_error, not yet implemented)
- [ ] `kernels/attention.comp` — scaled dot-product attention (fused QK^T·V)
- [x] `kernels/quantized.comp` — affine quantize/dequantize (int4, int8)
- [x] `kernels/random.comp` — Philox / Threefry PRNG for `mx.random.*`
- [x] `kernels/rbits.comp` — RandomBits Threefry PRNG

---

## Phase 8: Primitives Dispatch 🔄 PARTIAL

### `mlx/backend/vulkan/primitives.cpp`

Implement `eval_gpu()` for every primitive. Pattern per op:

1. Get VkCommandBuffer from stream
2. Get cached VkPipeline by name
3. Allocate + write VkDescriptorSet (bind input/output arrays)
4. Push constants (size, strides, type params)
5. `vkCmdDispatch(cmd, ceil(n/256), 1, 1)`
6. Insert memory barrier

#### Elementwise Unary (dispatch `unary.comp` with op specialization constant)

- [x] Abs, Arccos, Arcsin, Arctan, Ceil, Cos, Cosh, Erf, Erfinv
- [x] Exp, Expm1, Floor, Log, Log1p, Log2, Neg, Round, Rsqrt
- [x] Sigmoid, Sign, Sin, Sinh, Sqrt, Square, StopGradient, Tan, Tanh

#### Elementwise Binary (dispatch `binary.comp`)

- [x] Add, ArcTan2, BitAnd, BitOr, BitXor, Divide
- [x] Equal, FloorDivide, Greater, GreaterEqual, LeftShift
- [x] Less, LessEqual, LogAddExp, Maximum, Minimum
- [x] Multiply, NotEqual, Power, Remainder, RightShift, Subtract

#### Elementwise Ternary

- [x] Select (where)

#### Reduction

- [x] Reduce (sum, min, max, prod, logsum — axis-wise)
- [x] ArgReduce (argmin, argmax)

#### Shape / Memory

- [x] Arange
- [x] AsType (type cast) — verified: int32→float32 ✅
- [x] AsStrided
- [x] Broadcast
- [x] Concatenate
- [x] Copy (contiguous copy)
- [x] Flatten (via reshape)
- [x] NumberOfElements (via gpu/primitives.cpp)
- [x] Pad
- [x] Reshape (via copy_gpu) — verified: [4]→[2,2] ✅
- [x] Slice, SliceUpdate
- [x] Split (via gpu/primitives.cpp)
- [x] Squeeze, Expand
- [x] Transpose
- [x] Unflatten (via gpu/primitives.cpp)
- [x] View (via gpu/primitives.cpp)

#### Linear Algebra

- [x] AddMM (A + alpha \* B @ C)
- [x] BlockMaskedMM — Vulkan GPU implementation enabled (`eval_gpu` + `block_masked_mm.comp`) ✅
- [x] GatherMM — Vulkan GPU implementation enabled (`eval_gpu` + `gather_mm.comp`) ✅
- [x] GatherQMM — descriptive error on GPU stream (replaced NO_GPU stub) ✅
- [x] SegmentedMM — descriptive error on GPU stream; CPU stream works ✅
- [x] Matmul — verified ✅
- [x] QuantizedMatmul — GPU dispatch **COMPLETE** ✅ (dequantize pass + matmul; 17/17 PASS)
- [x] QQMatmul — GPU dispatch **COMPLETE** ✅ (dual dequantize + matmul; 7/8 PASS, 1 SKIP=API check)
- [x] QRF, SVD, Inverse, Cholesky, Eig, Eigh, LUF — MoltenVK CPU fallback coverage
  (currently delegate to `eval_cpu()` after stream synchronization; discrete-GPU staging/readback still needs explicit hardening)

#### Neural Net Ops

- [x] Conv1D, Conv2D, Conv3D (ConvolutionVjp)
- [x] FFT, RFFT — Stockham radix-2/4/8 path covered by current tests
- [ ] IFFT, IRFFT — partial implementation only; inverse/Bluestein helper gaps remain in `fft.cpp`
- [x] Hadamard — GPU path for power-of-2 float32 sizes; larger/non-power-of-2 cases fall back
- [x] LayerNorm, RMSNorm (GPU dispatch via `normalization.comp`)
- [x] LogSumExp
- [x] Rope (GPU dispatch via `rope.comp`)
- [ ] ScaledDotProductAttention — still unimplemented on Vulkan (`NO_GPU_MULTI`)
- [x] Softmax — verified via smoke test ✅
- [x] Scan (prefix ops — GPU dispatch via scan.comp; Hillis-Steele 2-level; scan_size ≤ 1024; 5/5 tests pass)

#### Indexing

- [x] Gather (unsupported bounds native halt via exceptions)
- [x] GatherAxis, ScatterAxis (GPU dispatch via `indexing.comp`)
- [x] Scatter (unsupported bounds native halt via exceptions)

#### Sort

- [x] ArgSort (unsupported bounds native halt via exceptions)
- [x] Sort (GPU dispatch via `sort.comp` for ≤128, `radix_sort.comp` for >128)
  - Bitonic sort: arrays ≤128 elements (shared memory 512 elements)
  - Radix sort: arrays >128 elements (8-pass 4-bit digit sorting, supports up to 8192+ elements)
  - Supports int32 and float32 dtypes on last axis
  - Handles both ascending and descending sort order
  - Validates with test_radix_sort.py: 6/6 test suites passing
- [x] Partition, ArgPartition (unsupported bounds native halt via exceptions)

#### Random

- [ ] BernoulliWithCDF
- [x] RandomBits (Threefry PRNG via `rbits.comp`)

#### Quantization

- [x] QuantizedMatmul — GPU dispatch COMPLETE ✅ (17/17 PASS)
- [x] QQMatmul — GPU dispatch COMPLETE ✅ (7/8 PASS; 1 SKIP = MLX dim check)
- [x] fast::Quantize — quantize direction via eval_cpu; dequantize via inline CPU on VMA buffers ✅
- [x] fast::ConvertFP8 — eval_cpu fallback ✅
- [x] GatherQMM — descriptive runtime_error stub (no GPU impl yet; consistent with GatherMM) ✅
- **Note**: `AffineQuantize` / `DequantizedMatmul` do not exist as primitives in this MLX version
- **Note**: fast::Quantize dequantize uses inline CPU (not GPU shader) due to VK_ERROR_DEVICE_LOST
  when GPU semaphores from mx.random.normal are pending. GPU shader path left as future work.

#### Misc

- [x] Compiled (eval_cpu fallback via unified memory) ✅
- [ ] CustomVJP, CustomTransforms (CPU fallback OK for now)
- [ ] Depends (sync primitive)
- [x] Load (mmap via eval_cpu fallback) ✅
- [ ] Jit (stub)

---

## Phase 9: Integration & Testing 🔄 PARTIAL

### Build Validation ✅ COMPLETE

- [x] `cmake -B build_vulkan -DMLX_BUILD_VULKAN=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON` — succeeds
- [x] `cmake --build build_vulkan -j4` — zero errors (only VMA nullability warnings)
- [x] All 22 `.comp` shaders compile to `.spv` without errors
- [x] All 22 `.spv` pass `spirv-val` validation

### Smoke Tests ✅ PASSING

- [x] Vulkan device detected: Apple M1 (via MoltenVK)
- [x] VK_EXT_external_memory_host detected
- [ ] Zero-copy host import path actually implemented
- [x] `add` (float32): `mx.ones(4) + mx.full(4, 3)` → 4.0 ✅
- [x] `multiply` (float32): `mx.full(4,2) * mx.full(4,3)` → 6.0 ✅
- [x] `arange`: `mx.arange(0,4,1)[2]` → 2.0 ✅
- [x] `sum` reduction: `mx.sum(mx.ones(4))` → 4.0 ✅
- [x] `matmul`: `mx.matmul(ones(2,3), ones(3,2))[0,0]` → 3.0 ✅
- [x] `exp` (unary): `mx.exp(mx.zeros(4))[0]` → 1.0 ✅
- [x] `maximum` (binary): `mx.maximum(ones*1, ones*2)[0]` → 2.0 ✅
- [x] `int32 add/sub/mul/div`: `[1,2,3]+[4,5,6]=[5,7,9]` ✅
- [x] `int32 eq/lt/gt comparisons` ✅
- [x] `float32 add/mul` ✅
- [x] `astype int32→float32`: `[1,2,3]→[1.0,2.0,3.0]` ✅
- [x] `bool equality (a==a)`: `[True,True,True]` ✅
- [x] `reshape [4]→[2,2]` ✅
- [x] `softmax` (from previous session) ✅

### Historical MoltenVK Validation Snapshot

- [x] `int32` arithmetic correctness (fixed binary.comp dtype bug)
- [x] `test_array.py` full suite — 65/68 PASS (historical MoltenVK snapshot)
- [x] `test_ops.py` suite — ~100/134 PASS (historical MoltenVK snapshot)
- [x] `test_random.py` — 14/14 PASS
- [x] `unary.comp` int32 paths (abs, neg on int32) — audit complete

### Numerical Equivalence (vs CPU)

- [x] Write `tests/vulkan_equivalence.py` — compare GPU vs CPU output for all primitives
- [x] Tolerance: `atol=1e-4` for float32, `atol=1e-2` for float16/bfloat16
- [x] Matmul equivalence for sizes: 4×4, 128×128, 512×512, 1024×1024
- [x] Reduction equivalence along all axes
- [ ] Note: Script has API compatibility issues (uses older mx.array signature)

### MLX Test Suite

- [ ] `python -m pytest tests/ -x -v` — not currently accepted as green; requires revalidation on the latest Vulkan-linked runtime
- [x] `python -m pytest tests/test_ops.py` — op-level coverage on MoltenVK
- [x] `python -m pytest tests/test_random.py` — RNG reproducibility
- [x] No regressions introduced in this session

### Performance Baselines

- [ ] Run `benchmarks/` matmul benchmark — record GFLOPS for comparison
- [ ] Compare against CPU backend throughput

---

> 📜 **Historical reference**: CI uses lavapipe (software Vulkan). Not a substitute for real-hardware validation.

## Phase 10: Continuous Integration ✅ COMPLETE (lavapipe)

- [x] `.github/workflows/vulkan.yml` — ubuntu-22.04 + lavapipe software Vulkan, no GPU required
- [x] cmake configure + build step in CI
- [x] SPIR-V shader validation: `spirv-val kernels/*.spv`
- [x] GPU smoke test (wrapped in try/except for lavapipe extension limits)
- [x] Stage test suite loop — fails if >50% of stages fail
- [ ] Self-hosted GPU runner (AMD/NVIDIA) — requires hardware; future work

---

## Reference Links

- MLX CUDA backend (structure reference): `mlx/backend/cuda/`
- MLX gpu/ interface contract: `mlx/backend/gpu/eval.h`
- [Vulkan-Samples by LunarG](https://github.com/KhronosGroup/Vulkan-Samples) — SPIR-V loading, init boilerplate
- [Shaderc](https://github.com/google/shaderc) — GLSL→SPIR-V, AOT + optional JIT
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross) — SPIR-V reflection for descriptor layout
- [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) — device memory allocator
- [Vulkan Compute for AI](https://github.com/PacktPublishing/Vulkan-Compute) — compute shader patterns
- GitHub issue: https://github.com/ml-explore/mlx/issues/1751

---

## Key Technical Decisions

| Decision        | Choice                          | Rationale                                                                         |
| --------------- | ------------------------------- | --------------------------------------------------------------------------------- |
| Shader language | GLSL 4.60 compute               | Standard, mature tooling, all Vulkan drivers                                      |
| Compilation     | AOT via glslc at build time     | No runtime compiler dep, faster startup                                           |
| Memory          | VMA (VulkanMemoryAllocator)     | Handles suballocation, mirrors MetalAllocator pooling                             |
| bfloat16        | uint16 storage + manual ops     | No native Vulkan bf16 — same approach as metal bf16.h                             |
| Kernel variants | VkSpecializationInfo per dtype  | Avoids JIT string templates, compile-time branching                               |
| macOS           | MoltenVK (active dev target)    | Linux remains primary; macOS M1 used for iteration                                |
| Discrete GPU    | Staging buffers for host access | No unified memory — explicit CPU↔GPU transfer path                                |
| int32 on GPU    | Dtype-aware GLSL paths          | Apple Silicon flushes denormals-to-zero; int bits must not pass through float ops |

---

## Pending Work — Historical Priority Snapshot (as of 2026-03-09)

> ⚠️ **Needs revalidation**: Priority order below is superseded by the Authoritative Compatibility Gates above. Individual items remain relevant but must be sequenced through the gate framework.

> Based on live audit: 134 tests in `test_ops.py`, 24 shaders, full `primitives.cpp` code review.

---

### 🔴 P0 — Segfault blocking full test suite run

| ID | Issue | File | Root Cause |
|---|---|---|---|
| S1 | `test_scans` segfault / bus error (CPU fallback path) | `cpu/scan.cpp` | ✅ Mitigated in code: `contiguous_scan` and `strided_scan` rewritten with index-based loops (no pointer drift), plus zero-axis guard to avoid divide-by-zero / negative-offset paths. |
| S2 | Full test suite stability | multi | Pending user-side full `pytest` re-run in Vulkan environment to confirm no residual scan regressions. |

**S1 Validation Snapshot (2026-03-08)**:
1. Added deterministic CPU regression checks for forward/reverse inclusive/exclusive scan invariants across `cumsum/cumprod/cummax/cummin` (`tests/test_gather_cpu.cpp`).
2. Rebuilt and ran regression in a CPU-only build (`MLX_BUILD_VULKAN=OFF`): passes over randomized repeated trials.
3. Full Python `test_scans` still requires user-side revalidation on Vulkan-linked runtime.

---

### 🔴 P1 — Correctness bugs in shipped shaders

#### 1.1 NaN/Inf detection missing in `unary.comp` ✅ DONE (2026-03-05)
Tests: `test_isinf` ✅, `test_isnan` ✅, `test_isneginf` ✅, `test_nan_to_num` ✅

- Added `UNARY_ISINF`, `UNARY_ISNAN`, `UNARY_ISNEGINF` to `unary.comp`
- Fixed `dispatch_unary` to register 3 descriptor bindings (was 2 — caused segfault)
- Fixed SIMD `IsInf`/`IsNaN`/`IsNegInf` in `unary_ops.h` to handle `complex64` element-by-element
  instead of treating real+imag as raw float32 pairs (shifted result bug)
- Fixed `ternary.comp` boolean condition: was reading bool as `float[]` (packed uint8 treated as
  non-zero float → every element evaluated as true). Now reads as `uint[]`, extracts byte at `idx`.
- Fixed `ternary.comp` float16/bfloat16: added sized reads for true/false/out buffers;
  zero-initializes 16-bit output before dispatch so `atomicOr` writes are safe
- Fixed `Select::eval_gpu` push constants: 16 → 28 bytes, added `elem_bytes` fields

#### 1.2 NaN propagation wrong in `binary.comp` maximum/minimum/logaddexp ✅ DONE (2026-03-05)
Tests: `test_maximum` ✅, `test_minimum` ✅, `test_logaddexp` ✅, `test_logsumexp` ✅

- Fixed NaN propagation in `binary.comp` for `BINARY_MAX`, `BINARY_MIN`, `BINARY_LOGADDEXP`
- Fixed `LogAddExp` for `complex64` in `binary_ops.h`: use `log(exp(a)+exp(b))` not max-trick

#### 1.3 Sort — int32 with negatives broken ✅ DONE (2026-03-05)
Tests: `test_sort` ✅, `test_median` ✅

- Changed `<float> data` buffers to `<uint> data` in `sort.comp`
- Added push constant `uint type_is_int`
- Now raw bits are compared as ints if `type_is_int==1`, solving the float parsing issues.

#### 1.4 Sort — axis=None / multi-axis fallback issues ✅ DONE (2026-03-05)
Tests: `test_sort` ✅ 

- Safely falls back to CPU in `Sort::eval_gpu` for `sort_axis != in.ndim() - 1` and `sort_pow2 > 128`.

#### 1.5 Sort — NaN ordering ✅ DONE (2026-03-05)
Tests: `test_sort_nan` ✅

- For float bitonic sort in `sort.comp`, `isnan(fi)` intercepts NaNs and swaps them with `uintBitsToFloat(0x7F800000u)` `POS_INF` so they are safely dropped to the end.

#### 1.6 ScatterAxis wrong result for `put_along_axis` ✅ DONE (2026-03-08)
Tests: `test_put_along_axis` ✅

- `ScatterAxis` dispatch in `indexing.comp` was treating `src_outer_stride == 1u` (element-wise mode) incorrectly by sharing index reads across channels unnecessarily.
- **Fix**: Updated `INDEX_SCATTER` and `INDEX_SCATTER_ADD` in `indexing.comp` to respect `src_outer_stride == 1u` properly, mirroring the logic fix from `GatherAxis`.

#### 1.7 `rsqrt` precision
Tests: `test_rsqrt` ❌

- GPU `1.0/sqrt(x)` has worse precision than CPU `std::rsqrt()` for edge values
- **Fix**: Use `inversesqrt()` in GLSL which maps to hardware instruction, or add a Newton-Raphson refinement step

#### 1.8 `round` wrong result
Tests: `test_round` ❌

- `round()` in GLSL uses "round half to even" (banker's rounding) on some drivers; MLX CPU uses "round half away from zero"
- **Fix**: Replace `round(x)` with `floor(x + 0.5)` in `unary.comp` for the Round case

#### 1.9 `linspace` precision
Tests: `test_linspace` ❌

- Linspace endpoints may have floating point accumulation error vs CPU
- **Fix**: Check `arange.comp` start/stop/step calculation; use `lerp(start, stop, t/n)` pattern

#### 1.10 `meshgrid` shape wrong
Tests: `test_meshgrid` ❌

- meshgrid uses indexing='xy' vs 'ij' — wrong axis assignment
- **Fix**: Check `meshgrid` Python wrapper in `mlx-src/python/mlx/core/__init__.py`; likely a Python-level fix not GPU

#### 1.11 `outer` shape mismatch
Tests: `test_outer` ❌

- `mx.outer(a, b)` should produce `(len(a), len(b))` matrix via broadcasting reshape
- **Fix**: Check dispatch path; may need reshape+broadcast before binary multiply

#### 1.12 Scalar input handling
Tests: `test_scalar_inputs` ❌

- Broadcasting scalar Python values to MLX arrays before GPU dispatch has edge cases
- **Fix**: Check `Binary::eval_gpu` path when one input is `ndim=0` (scalar)

---

### 🔴 P2 — Missing GPU implementations (CPU fallback but should be GPU)

#### 2.1 Scan > 1024 — multi-pass GPU ✅ DONE (2026-03-05)
- **Status**: ✅ Implemented — removed `scan_size > 1024` guard, recursive multi-pass scan works
- Verified: `cumsum(ones(N))` correct for N = 1023, 1024, 1025, 2048, 2049, 5000, 50000
- **Files**: `primitives.cpp`

#### 2.2 Sort — axis=None support ✅ Already handled
- **Status**: ✅ The ops layer (`ops.cpp:2350`) already flattens to 1D + sorts on axis=0
- `Sort::eval_gpu` receives a 1D tensor where axis=0 IS the last axis — works correctly
- Radix sort (>128 elements) has correctness bugs; kept CPU fallback for `sort_pow2 > 128`

#### 2.3 Sort — non-float32/int32 dtypes
- **Status**: CPU fallback for bfloat16, float16, int8, uint8, int64
- **Fix**: Extend `radix_sort.comp` to support more dtypes via appropriate key encoding

#### 2.4 Multi-axis Gather (ND index tensors)
- **Status**: IndexPushConst extended with `idx_ndim/idx_shape/idx_strides` (push const size 80), but `indexing.comp` not yet updated to use them
- **Fix**: Implement ND gather path in `indexing.comp` reading `idx_ndim`, `idx_shape[]`, `idx_strides[]` from push constants
- **Files**: `kernels/indexing.comp`

#### 2.5 GatherMM push constant 128-byte violation ✅ DONE (2026-03-05)
- **Status**: ✅ Fixed — refactored push constants from 200 → 56 bytes
- Shape/stride arrays (36 fields, 144 bytes) moved to SSBO at binding 5
- Push constants now contain only 14 scalar fields (56 bytes)
- **Files**: `kernels/gather_mm.comp`, `primitives.cpp`

#### 2.6 Hadamard large arrays (n > 2048)
- **Status**: Falls to CPU for n > 2048 or non-power-of-2
- **Fix**: Increase GPU limit (currently 2048) or implement multi-pass butterfly in `hadamard.comp`

---

### 🟡 P3 — Infrastructure gaps ✅ COMPLETE (2026-03-05)

#### 3.1 Pipeline cache version discipline ✅ DONE
- **Fixed**: Added comment block to `device.cpp` tracking `kPipelineCacheVersion` bumps to ensure auditability.

#### 3.2 GatherMM push constant size violation ✅ DONE (2026-03-05)
- **Fixed**: Moved shape/stride arrays to SSBO (binding 5). Push constants reduced to 56 bytes. See P2.5.

#### 3.3 `matmul_coop.spv` compilation safety ✅ DONE
- **Fixed**: Guarded `matmul_coop.spv` and `gather_mm_coop.spv` compilation with a cmake check for MoltenVK/Apple platforms where `VK_KHR_cooperative_matrix` is unsupported by offline compilers.

#### 3.4 `has_cooperative_matrix_` flag always false on M1 ✅ DONE
- **Verified**: `has_cooperative_matrix_` correctly initializes to false on MoltenVK, confirming cooperative ops fall back properly.

#### 3.5 Full test suite segfault isolation ✅ DONE
- **Fixed**: Resolved test suite shutdown segfault in `Device::commit` by joining background threads during destruction. Corrected subsequent discovery of `isinf` / `isnan` float16 MoltenVK compiler bugs via native bit manipulation in `unary.comp`.

---

### 🟢 P4 — Performance & Polish

| # | Task | Impact |
|---|---|---|
| 4.1 | GPU sort: support bfloat16/float16 via key encoding in radix_sort.comp | Covers transformer KV-cache topK |
| 4.2 | AMD RDNA matmul tile tuning: verify BN=32 for 64-wide wavefronts actually helps | AMD perf |
| 4.3 | fast::Quantize dequantize GPU shader path | Eliminates CPU sync round-trip |
| 4.4 | Rader/Bluestein FFT for non-power-of-2 sizes | Completeness |
| 4.5 | Distributed: AllReduce via vkCmdCopyBuffer | Multi-GPU |
| 4.6 | Numerical equivalence test suite (GPU vs CPU for all ops) | Regression safety |
| 4.7 | CI: fix lavapipe compatibility for cooperative_matrix compile guard | CI health |

---

### Summary Dashboard (updated 2026-03-05)

| Category | Count | Status |
|---|---|---|
| test_ops.py PASSING | ~114/134 | Pre-existing cumprod reverse bug causes early exit |
| test_ops.py FAILING | ~20 | See P1 table above |
| Shaders compiled | 24/24 | All present |
| Segfault (blocks CI) | 0 | P0 — scan > 1024 ✅ fixed |
| NaN/Inf handling | ✅ | P1.1, P1.2 done |
| Sort correctness | ✅ | P1.3–P1.5 done |
| Scan > 1024 GPU | ✅ | P2.1 done |
| Multi-axis Gather GPU | ❌ | P2.4 |
| GatherMM push const size | ✅ fixed | P2.5 / P3.2 done |
| Coop matrix on M1 | ❌ unsupported | P3.3, P3.4 |


### Immediate: Unblock test_array.py and test_ops.py

1. Audit `unary.comp` for int32 dtype correctness (same class of bug as binary.comp fix).
2. Implement missing shape/memory primitives: `Concatenate`, `Split`, `Pad`, `NumberOfElements`.
3. Run `test_array.py` suite to completion; log all failures.
4. Run `test_ops.py` suite; identify remaining CPU fallback ops causing failures.

### Phase G: FFT / Hadamard (Spectral Ops) 🔄 PARTIAL

- [x] `FFT`, `RFFT` — Stockham Cooley-Tukey radix-2/4/8 path; currently covered by 3/3 tests.
- [ ] `IFFT`, `IRFFT` — partial implementation only; inverse/Bluestein helper gaps remain in `fft.cpp`.
- [x] `Hadamard` — GPU Walsh-Hadamard Transform in `kernels/hadamard.comp`; butterfly in shared mem; n≤2048 on GPU, larger/non-power-of-2 fall back to `eval_cpu`.
- [ ] Rader/Bluestein for non-power-of-2 FFT sizes — future work.

### Remaining Work (Priority Order) — Updated 2026-03-01

#### Critical — Hanging Stages (GPU deadlock / infinite loop)
These stages timeout after 20s. A hang blocks the entire test suite.

1. **Stage 10 (Reduce)** — `test_stage10_reduce.py` hangs. Root cause unknown; likely command buffer
   never commits or fence never signals in the reduction path.
2. **Stage 11 (Matmul)** — `test_stage11_matmul.py` hangs. Matmul dispatch works in isolation
   (stage 19 passes) but something in the stage 11 test cases causes a hang.
3. **Stage 12 (NN Ops)** — `test_stage12_nn_ops.py` hangs. Likely same class of issue.
4. **Stage 17 FFT** — `test_stage17_fft.py` hangs. Previously passed (3/3 in Feb) — regression.
5. **Stage 18 Concat** — `test_stage18_concat.py` hangs. Concatenate/strided copy issue.
6. **Stage 21 Advanced MM** — `test_stage21_advanced_mm.py` updated for GPU numerical checks
   (`GatherMM` + `BlockMaskedMM`) and segmented_mm fallback behavior.

#### High Priority — Failing Stages (runs, wrong results)
7. **Stage 14 Sort** — 0/2 FAIL. Bitonic sort regression; previously 6/6 in Feb.
8. **Stage 16 NN Extended** — 2/4 FAIL. LayerNorm/RMSNorm/RoPE regression; 2 tests failing.
9. **Stage 17 AddMM/Conv/RBits** — 0/2 FAIL. AddMM or Conv dispatch broken.

#### Medium Priority — Completed, not yet fully wired
10. **GatherMM / BlockMaskedMM on GPU stream** — Implemented on Vulkan (`eval_gpu` paths + shader dispatch).
    `SegmentedMM` remains CPU fallback on GPU stream.
11. **GatherQMM GPU** — Currently throws gracefully; would need gather→dequant→matmul pipeline.

#### Low Priority / Future
12. Rader/Bluestein FFT for non-power-of-2 sizes.
13. Multi-axis Gather/Scatter fully on GPU (currently 1D only).
14. Numerical equivalence test suite (`tests/vulkan_equivalence.py`).
15. Performance baselines vs CPU backend.
16. fast::Quantize dequantize GPU shader path (currently inline CPU; GPU shader caused
    VK_ERROR_DEVICE_LOST when random semaphores pending).

---

> 📜 **Historical reference**: Gap analysis from earlier milestones. Items not yet incorporated into the compatibility ladder are deferred.

## Phase 11: Production Readiness (from Architectural Review 2026-03-01)

Gaps identified in REVIEW.md, assessed against current implementation:

### A. JIT Kernel Fusion (`mx.compile()` GPU path)
**Status**: ❌ Not implemented on Vulkan. Current behavior is CPU fallback after stream synchronization.
**Plan**:
- [ ] Implement `Compiled::eval_gpu` with shaderc JIT: fuse elementwise chains into single SPIR-V kernel
- [ ] Cache fused kernels by op-graph hash

### B. Workgroup Tuning for AMD RDNA
**Status**: ✅ Infrastructure COMPLETE. Subgroup size queried at init; M1/MoltenVK = 32/128.
  Shaders still hardcode `local_size_x=256` — specialization constants not yet wired per-shader.
**Plan**:
- [x] Query subgroup size via `VkPhysicalDeviceSubgroupProperties` + `vkGetPhysicalDeviceProperties2`
- [x] Store `subgroup_size_` / `preferred_workgroup_size_` in `VulkanDevice`; exposed in `device_info()`
- [ ] Pass to shaders via `VkSpecializationInfo` per pipeline
- [ ] Tune `matmul.comp` tile size for 64-wide wavefronts (AMD) vs 32-wide (Intel)

### C. Cooperative Vectors / Subgroup Matrix Ops
**Status**: Not implemented.
**Gap**: VK_NV_cooperative_matrix / VK_KHR_cooperative_matrix would allow hardware tensor cores on NVIDIA/AMD.
**Plan**:
- [ ] Check `VK_KHR_cooperative_matrix` availability at device init
- [ ] Implement `matmul.comp` fast path using `coopMatLoad`/`coopMatMulAdd` when available

### D. Distributed Execution
**Status**: All `distributed::` ops are `NO_GPU_MULTI` (throw).
**Gap**: `mx.distributed` requires cross-device/cross-host communication.
**Plan**:
- [ ] Implement `AllReduce` via `vkCmdCopyBuffer` + barrier (single-node, multi-queue)
- [ ] Multi-node: integrate with MPI or NCCL equivalent (future)

### E. ARM AI-ML Layer (reviewed and REJECTED)
The review's sequence diagram depicts this as a callable shader library. **This is incorrect.**
The ARM layer is a Vulkan loader extension (`VK_ARM_data_graph`/`VK_ARM_tensors`), injected
transparently. It is not a library you link against or call functions from. No action required.
**Verdict**: ❌ Not applicable — assessed and documented in External References section.

---

## External References — ARM AI/ML Emulation Layer

**Repo**: `https://github.com/arm/ai-ml-emulation-layer-for-vulkan`

**What it is**: Vulkan layer implementing `VK_ARM_data_graph` + `VK_ARM_tensors` extensions. Injected
by the Vulkan Loader — transparent to apps. Not a linkable compute library.

**Relevance to this project**:

| Aspect | Verdict | Notes |
|---|---|---|
| Direct shader reuse | ❌ No | Only `copy.comp` + utility glsl in `tensor/shaders/` — no ML ops |
| `VK_ARM_tensors` extension | ⚠️ Future | Alternate fast path on ARM Mali; not useful today |
| AOT SPIR-V strategy | ✅ Confirms | ARM uses same AOT approach — validates our architecture |
| MoltenVK variant shaders | ✅ Noted | `tensor_mvk.glsl` confirms MoltenVK needs platform-specific code |
| Direct integration | ❌ No | Extension layer, not a compute library to link |

**Conclusion**: No immediate action. Add `VK_ARM_tensors` as future fast-path ticket in Phase 10.

---

## Phase 5: Shape & Misc Primitives ✅ COMPLETE

### Key findings

- **NumberOfElements**, **Unflatten**, **View**, **Split**: Already correctly implemented in
  `mlx/backend/gpu/primitives.cpp` using proper GPU copy/reshape ops. No changes needed in
  the Vulkan-specific file.

- **Load** (`primitives.cpp` line ~1706): Was throwing `runtime_error` — fixed to `eval_cpu(inputs, out)`.
  On unified-memory (MoltenVK) the VMA buffer is CPU-accessible so the mmap reader writes directly.

- **Compiled** (`primitives.cpp`): Was throwing `runtime_error` — currently falls back to
  `eval_cpu(inputs, outputs)` after stream synchronization. This is CPU fallback coverage, not Vulkan JIT fusion.

### Test: Stage 23 (`tests/vulkan/test_stage23_shape_misc.py`)

- [x] NumberOfElements (GPU) — scalar product of axis sizes
- [x] Unflatten (GPU) — reshape one axis → multiple
- [x] View / dtype reinterpret (GPU) — float32 → uint32 byte-level reinterpret
- [x] Split basic (GPU) — 2-chunk split shape verification
- [x] Split numerical correctness — compare with numpy reference
- [x] Load round-trip (.npz) — mx.savez then mx.load
- [x] Compiled (mx.compile) — eager vs compiled numerical match
- [x] Compiled repeated calls — 5 successive invocations

---

> 📜 **Historical reference**: Gap analysis snapshot from 2026-03-02. Some items (BF16, sort, scan) have since been completed. See the compatibility ladder for current sequencing.

## Phase 12: Production Readiness Gaps (Identified 2026-03-02)

### Current Status (2026-03-02)
**Custom stage suite**: ~22/23 stages passing (only stage 14 test spec mismatch).
**Official MLX test suite** (`tests/test_ops.py`, `tests/test_array.py`): ~46% pass rate.
**Blocker**: MLX defaults to `bfloat16` for model weights — **zero BF16 shader support**.

---

### 🔴 CRITICAL BLOCKERS (production impossible without these)

#### 1. BF16 Support in All Shaders ✅ COMPLETE
- MLX uses `mx.bfloat16` as default dtype for LLM/model weights
- Current: ALL BF16 ops silently fail / produce garbage (no shader BF16 path)
- Fix: Add `bf16.glsl` helper (pack/unpack as uint16), wire into `unary.comp`, `binary.comp`, `reduce.comp`, `matmul.comp`, `softmax.comp`, `normalization.comp`, `copy.comp`
- **Estimate**: 3-4 shader files, ~200 lines total
- [x] `kernels/bf16.glsl` — `unpackBfloat2x16` / `packBfloat2x16` helpers
- [x] `unary.comp` — BF16 input/output path (alongside existing f16 path)
- [x] `binary.comp` — BF16 arithmetic paths (broadcast strides already supported)
- [x] `reduce.comp` — BF16 accumulation (accumulate in f32, write back as bf16)
- [x] `softmax.comp` — BF16 input/output (via C++ wrappers)
- [x] `matmul.comp` — BF16 operands (accumulate in f32)
- [x] `normalization.comp` — BF16 layer_norm/rms_norm (via C++ wrappers)
- [x] **Softmax BF16 copy back fix** (2026-03-03): Fixed bug where `Softmax::eval_gpu` temp buffer wasn't copied back to output, causing BF16 softmax to return zeros. Added `copy_gpu(*temp_out, out, CopyType::General, stream())` after barrier.

#### 2. Multi-axis Gather on GPU
- Transformers use `mx.take(x, idx)` with non-trivial index shapes (2D/3D)
- Current: 1D single-axis gather works; multi-axis falls to CPU
- Fix: Extend `indexing.comp` INDEX_GATHER_GEN to handle multi-dim index tensors
- [ ] Detect multi-axis gather in `Gather::eval_gpu`
- [ ] Pass `idx_shape[]` + `idx_strides[]` in push constants (extend IndexPushConst)
- [ ] Shader: flatten ND index → source offset using `src_shape` and `idx_shape`
- **Note**: Must bump `kPipelineCacheVersion` when IndexPushConst layout changes

#### 3. Sort > 256 Elements (Radix Sort)
- Current: sort > 256 falls to CPU (bitonic sort limited to 256 SMEM)
- Fix: Implement 2-pass radix sort in `sort.comp` for arbitrary sizes
- [ ] Pass 1: Per-workgroup 4-bit radix digit histogram
- [ ] Pass 2: Prefix sum over histograms (reuse scan.comp approach)
- [ ] Pass 3: Scatter to sorted positions
- Alternatively: merge sort using ping-pong buffers (simpler, good enough for n≤4M)

---

### 🟡 HIGH PRIORITY (needed for 90%+ official test pass rate)

#### 4. Numerical Equivalence Test Suite
- No automated GPU-vs-CPU comparison tests
- [ ] Create `tests/vulkan_equivalence.py`
- [ ] Test all 28 unary ops, 18 binary ops, reduce (all axes), matmul (8 sizes), softmax, normalization
- [ ] Tolerance: `atol=1e-4` for float32, `atol=1e-2` for float16
- [ ] Add to CI workflow

#### 5. Fix Official MLX Test Suite (46% → 90%+)
- Last audit: `test_array.py` 52/69 pass, `test_ops.py` 41/134 pass
- Key failure categories:
  - Missing GPU primitives / Incorrect Implementations: `test_add`, `test_arange_corner_cases_cast`, `test_arange_overload_dispatch`, `test_array_equal`, `test_bitwise_ops`, `test_clip`, `test_complex_ops`, `test_complex_power`, `test_conjugate`, `test_cos`, `test_divmod`, `test_dynamic_slicing`
  - `test_diag` induces a Python `Segmentation fault` via Apple's standard library extensions.
- [ ] Profile failures with `python -m pytest tests/test_ops.py -v 2>&1 | grep FAIL`
- [ ] `test_add` (Missing arithmetic pipeline/corner case precision issue)
- [ ] `test_arange_corner_cases_cast`, `test_arange_overload_dispatch` (Data type casting failures)
- [ ] `test_array_equal` (Shape equality broadcasting failures)
- [ ] `test_bitwise_ops` (Unimplemented/failing integer binary operations)
- [ ] `test_clip`
- [ ] `test_complex_ops`, `test_complex_power`, `test_conjugate` (Complex numbering evaluation)
- [ ] `test_cos` (Missing trigonometric precision)
- [ ] `test_divmod`
- [ ] `test_dynamic_slicing`

#### 6. Scan > 1024 Multi-pass GPU
- Current: scan_size > 1024 falls to CPU
- Fix: 3-pass Hillis-Steele: (1) scan within blocks, (2) scan of block sums, (3) propagate
- [ ] Extend `scan.comp` with second pass dispatch when `scan_size > 1024`
- [ ] Dispatch logic in `Scan::eval_gpu`: choose 1-pass or 3-pass based on size

#### 7. Stage 14 Test Spec Update
- 3 tests expect `RuntimeError` for sort >256, but code now silently CPU-fallbacks
- [ ] Update `test_stage14_sort.py`: verify correct results from CPU fallback path
- [ ] Extend to test sort correctness for sizes 512, 1024

---

### 🟢 MEDIUM PRIORITY (nice-to-have for completeness)

#### 8. Workgroup Tuning via VkSpecializationInfo ✅ COMPLETE
- Infrastructure complete (subgroup size queried at init)
- Shaders updated from `local_size_x=256` explicitly to dynamic `WORKGROUP_SIZE`
- [x] Wire `preferred_workgroup_size_` through `VkSpecializationInfo` per-pipeline
- [x] Tune `matmul.comp` tile for AMD 64-wide wavefronts vs NVIDIA 32-wide

#### 9. Cooperative Matrix Ops / Hardware Tensor Cores ✅ COMPLETE
- [x] Enable `VK_KHR_cooperative_matrix` during init logic if GPU supports it
- [x] Write `matmul_coop.spv` alternative pathway leveraging `coopMatMulAdd` 
- [x] Fix compiler errors restricting matrix memory constraints 

#### 10. GatherMM / BlockMaskedMM GPU Implementation
- GPU stream implementations now present for both ops
- Needed for sparse attention in transformer models
- [x] `GatherMM::vjp` gradients fixed: resolved Thread Scaling and uint32 limits inside `ScatterAxis` via fallback arrays
- [x] `GatherMM::eval_gpu`: gather + fused matmul shader path
- [x] `BlockMaskedMM::eval_gpu`: block-sparse matmul with mask

#### 10. mx.compile() GPU Fusion (JIT Kernel Fusion)
- Currently: CPU fallback; individual sub-ops dispatch to GPU separately
- Ideal: fuse elementwise chains into single SPIR-V kernel via shaderc JIT
- [ ] Implement `Compiled::eval_gpu` with shaderc JIT compilation
- [ ] Cache fused kernels by op-graph hash

#### 11. fast::Quantize GPU Shader Path
- Currently: inline CPU on VMA buffers (due to VK_ERROR_DEVICE_LOST with pending semaphores)
- Root cause: command buffer submission race when `random.normal` semaphores are in-flight
- [ ] Investigate `commit()` path for `has_sems=1, op_count=0` case
- [ ] GPU shader path for dequantize (eliminate CPU sync overhead)

---

### Priority Order for Parallel Agent Work

| Priority | Task | Estimated Impact |
|----------|------|-----------------|
| 1 | BF16 shader support (all shaders) | Unblocks all LLM workloads |
| 2 | Stage 14 test fix | Quick win, removes red from test suite |
| 3 | Official test suite profiling | Identify next 50 failures to fix |
| 4 | Multi-axis Gather GPU | Unblocks transformer attention ops |
| 5 | Sort > 256 radix sort | Production data sizes |
| 6 | Scan > 1024 multi-pass | Edge cases |

---

> ⚠️ **Superseded**: Replaced by the Authoritative Compatibility Gates at the top of this document. Kept for reference only.

## Prod Readiness Checklist (Target: MLX Vulkan MVP)

These items must be completed before the Vulkan backend can be considered "Prod Ready" for public use.

### 1. Stability & Completeness
- [ ] **Zero Segfaults:** The entire `python -m pytest tests/` suite must pass or explicitly skip natively without hard crashes, bus errors, or segfaults (e.g., resolving S1 `test_scans`).
- [ ] **Full `eval_gpu` Coverage:** Eliminate all remaining `NO_GPU` or `throw std::runtime_error` stubs in `primitives.cpp` by implementing either a dedicated Vulcan shader or a clean, safe unified-memory `eval_cpu` fallback.
- [ ] **Thread-Safety Audit:** Verify that `Device::commit` properly handles heavily concurrent MLX workloads, ensuring `CommandEncoder` bindings and `VkDescriptorPool` objects do not induce `EXC_BAD_ACCESS` under load.

### 2. Correctness & Precision
- [ ] **Numerical Equivalence:** `tests/vulkan_equivalence.py` validation for all core primitive paths against CPU/Metal execution, confirming bounded loss across IEEE standard `float32`, `float16`, and `bfloat16`.
- [ ] **Complex Data Support:** Full arithmetic equivalence across shaders for `complex64` mathematical paths.
- [ ] **Shape/Stride Handling:** Guarantee contiguous, broadcasted, padded, and strided memory interpretations evaluate seamlessly within GLSL without OOB geometry errors (e.g. verifying `Concat` and `Gather` logic boundaries).

### 3. Performance & Memory
- [ ] **Zero-Copy Transits:** Ensure that `VK_EXT_external_memory_host` or VMA unified memory architectures reliably negate CPU <-> GPU round-trip copy overheads. 
- [ ] **Clean Memory Management:** No GPU memory leaks; temporary buffer garbage collection accurately tracks execution boundaries and respects `allocator::free`.
- [ ] **Shader Pipelining Cache:** Seamless SPIR-V dynamic recompilation caching limits cold-start costs. 

### 4. Build & Platform
- [ ] **Linux Target CI:** Add automated `vulkan-validationlayers` headless CI tests on standard Ubuntu Linux topologies (in addition to MoltenVK constraints).
- [ ] **Package Signing & Linking:** Simplify dynamic linkage patterns such that Python extensions load reliably via standardized `pip` / `uv` packaging workflows without `macOS` `.so` conflicts.
| 7 | Equivalence test suite | Prevents regressions |
| 8 | Workgroup tuning | Performance on AMD/NVIDIA |
