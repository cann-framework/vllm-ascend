# Kimi MLA request-parallel decode

This opt-in feature partitions whole decode requests across the existing TP
ranks. Each rank computes all MLA heads for its request shard; an AllToAll
restores TP head order before the output gate/projection. KV allocation, block
IDs and PD transfer remain framework-managed. The producer stays native.

## Enable

Use the existing production D-side launch configuration and add:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

--additional-config '{"enable_kimi_mla_decode_dp":true}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,32,64,128,256]}'
```

The flag defaults to false. Enabling it also enables the communication-overlap
paths; there are no separate overlap switches. Merge existing additional-config
keys into the same JSON. This top-level flag uses the name from the 0.27
implementation; the old nested `kimi_mla_compute` configuration is removed.

Requirements: Kimi hybrid AttnRes, pure KV consumer, Runner V2, EP/SP, TP2/4/8,
PP=PCP=DCP=1, BF16/FP16 ND attention caches, no LoRA or dynamic speculation.
Retain the deployment's model/quantization/connector settings. In particular,
Mooncake/HIXL endpoint and transport configuration is environment-specific.

The example is TP8 with one query token per request. With fixed DSpark seven
draft tokens, q=8: retain the validated draft configuration on both arms and
use token buckets divisible by TP*q, for example `[64,128,192,256,320,384,512]`.
T256 then represents 32 global request slots and four slots per TP8 rank.
Actual requests and padded capacity are tracked separately; empty rank slots
have zero used-query/KV lengths and negative write slots.

## Execution and backend

Decode prepares local Q/current KV, then submits hidden-state AG while running
local attention. Value projection produces packed A2A output; gate computation
can overlap A2A. Canonical current KV is refreshed before readiness publication.
This adds no KV AllGather. Private full-head weights and supported KV-only
refresh weights consume memory and are prepared before KV sizing.

Backend selection remains framework-owned. For a 96-head model on TP8, native
FlashMLA uses 12 heads/rank and request-parallel FlashMLA uses all 96 heads over
local requests. FlashMLA additionally requires matching hardware/operator APIs,
latent width 512, 64 merged RoPE lanes, one KV head and compatible cache/CP/spec
settings. The feature does not expand native head support for TP2/TP4's 48/24
heads. Unsupported compute steps retain native execution; invalid capture
geometry is rejected at configuration time.

The implementation reuses the framework FlashMLA reader, token-fused 576 cache,
native fused prolog where eligible and stable metadata buffers. FIA fallback
requires internally dense component rows. No conversion of the entire history
cache is introduced to bridge incompatible readers. Scheme-specific KV-only
refresh supports native floating-point and MXFP8 weights; other schemes retain
the original projection.

## Validate

Start native and compute sequentially on the same cards, changing only
`enable_kimi_mla_decode_dp=false/true`. Use identical model, dtype, draft, cache
budget, FULL_DECODE_ONLY buckets and prefix warmup. Confirm the expected backend
and query/head geometry before interpreting throughput.

Check short deterministic outputs/candidates first, then q1/q8, B1, partial
buckets, empty ranks and cross-bucket replay. Compare repeated long-context
requests using actual response completion-token counts. Fixed-token generation
and a successful HTTP response alone are not correctness acceptance.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/ut/attention/test_kimi_mla_compute_partition.py \
  tests/ut/attention/test_kimi_mla_compute_backend.py \
  tests/ut/attention/test_flashmla_tp8_compute.py \
  tests/ut/test_ascend_config.py::TestKimiMlaDecodeDP
```

Use the deployment's existing launch and load-generation tools for the native
and compute comparison. Keep the fastest supported native backend and identical
Graph, model, draft and request settings. FIA measurements do not establish
FlashMLA-versus-FlashMLA gains. Multi-stream submission permits overlap but does
not guarantee it or a workload-independent speedup.
