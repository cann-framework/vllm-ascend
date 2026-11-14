# Kimi K3 static FP8 MLA cache

Static C8 stores the Kimi K3 MLA latent and auxiliary key in E4M3 PA_NZ
pages. The scale is one positive FP32 checkpoint parameter per MLA layer;
it is not stored with each token. KDA/Mamba and DSpark draft caches retain
their own formats and quantization.

## Configuration

Enable the path explicitly in Ascend additional configuration:

```json
{"mla_cache_mode": "C8"}
```

The default, `C16`, keeps the existing MLA path and its cache configuration.
The new C8 mode works with global `cache_dtype="auto"`; it does not use a
global FP8 cache flag to select MLA kernels. The checkpoint must provide a
FAKQuant recipe and a calibrated, finite, positive scalar `fa_k.scale` for
every target MLA layer. Nonzero KV offsets are rejected.

The supported configuration is Kimi K3 NoPE MLA with latent/auxiliary
dimensions 512/64, Runner V2, PCP=1, and DCP=1 or DCP=8. DCP=8 requires
interleave=1536. P/D with DCP=8 also requires D's manager block size 1536.
P/D uses `MooncakeConnectorV2` or `MooncakePullConnector`.

The prefill producer uses eager execution and DCP=1. Decode supports eager
and `FULL_DECODE_ONLY` Graph configuration. Other Graph modes are rejected.
DSpark's draft layers are identified before their attention constructors
run, so they do not inherit the target's C8 constraints or causal plan.

## Physical layout

Each physical kernel page has 128 tokens and 576 one-byte elements per
token: 73728 bytes. Its cache view is `[pages, 1, 18, 128, 32]`.
A 1536-token manager block represents 12 logical kernel pages; it is not
one `[18, 1536, 32]` PA_NZ page.

The existing hybrid allocator may pad manager pages to accommodate
KDA/Mamba state. Common page strides are aligned to 73728 bytes. Let
`R=manager_tokens/128` and `P=physical_page_bytes/73728`. The C8 address
adapter maps logical kernel ID `b*R+j` to physical ID `b*P+j`, and maps
slots using the same manager-page base. Padding remains part of the
original manager page and is never compressed across neighboring pages.
CoW and zeroing operate on the complete physical manager page.

Startup logs containing `Kimi C8 PA_NZ` report payload, physical page
bytes, manager blocks, physical kernel pages, and backing allocation.
One byte per MLA element does not imply that total model memory halves.

## Execution

Prefill uses the current BF16 data. Chunk/prefix history is gathered from
PA_NZ and dequantized once with the layer scale. Only new tokens are
quantized and written into the persistent cache.

The delivered fused Prolog has fixed 7168/1536/96-head MXFP8 geometry.
Eligible eager, non-DCP decode prepares its weights and norm buffers
during model loading, before KV budgeting. Other valid configurations,
including TP-local 12 heads, use the existing quantized projections plus
merged-Q quantization and a PA_NZ writer. Neither path changes KV format.
If the external Prolog package is needed, set `KIMI_C8_PROLOG_ROOT` to its
package parent or `ops` directory. There is no machine-specific default.

The current native DCP metadata interface accepts interleave 128. The
production adapter therefore computes each query's exact visible local
prefix at the configured interleave and submits causal Q1 rows to quant
FA. It does not report 1536 as 128. Multi-token verification expands the
schedule and may have a performance cost that needs measurement.

Quant FA returns `[T,H,512]` BF16 and `[H,T]` FP32 LSE. DCP exchanges Q
bytes and FP32 descales, converts LSE to `[T,H,1]`, and uses the existing
output exchange/combine before the local V projection.

Matching MLA layers share one schedule in the model step. Metadata can
run on a dedicated stream with input-ready and output-ready events.
Graph uses persistent V2 input views, fixed-size device computations and
a masked byte-scatter writer. Its layers do not enter legacy FIA task
updates.

## Transfer and validation status

Mooncake publishes the actual physical geometry and per-layer C8
layout/FP32 descale contract. Transfer planning uses logical token
geometry, then maps each endpoint to physical kernel IDs. Only valid
kernel payload is copied. Missing or mismatched C8 contracts fail before
the transfer engine reads the cache. Existing all-group/all-worker
completion and failure handling remains in use.

CPU tests cover address mappings, native projection assembly, metadata,
Graph input logic and events, DCP causal/LSE math, actual transfer address
emission with CPU memory copies, and DSpark construction. They do not
establish device accuracy, Graph capture/replay, HCCL behavior or throughput.
Those require separate NPU validation with the intended model and topology.
