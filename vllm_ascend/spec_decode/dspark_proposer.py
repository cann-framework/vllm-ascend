# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import enable_pcp
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer, _compute_num_programs
from vllm_ascend.spec_decode.utils import DynamicSpecScheduler, SlidingWindowAdapter

_REDHAT_WINDOW_GQA_NATIVE_WIDTH = 8
_REDHAT_WINDOW_GQA_VERIFY_WIDTH = 7


def _get_native_window_gqa_width(config: Any, verify_width: int) -> int | None:
    """Return the trained proposal width for RedHat's window GQA checkpoint.

    The checkpoint describes an eight-row anchor block while Kimi K3's target
    Verify ABI consumes seven draft tokens (anchor + seven rows).  Other
    DSpark checkpoints use one width throughout and must keep the generic path.
    """
    block_size = getattr(config, "block_size", None)
    has_target_layers = bool(
        getattr(config, "aux_hidden_state_layer_ids", None)
        or getattr(config, "eagle_aux_hidden_state_layer_ids", None)
        or getattr(config, "target_layer_ids", None)
        or (getattr(config, "dflash_config", None) or {}).get("target_layer_ids")
    )
    if (
        (
            getattr(config, "model_type", None) == "qwen3"
            or "Qwen3DSparkModel" in (getattr(config, "architectures", None) or ())
            or "DSparkDraftModel" in (getattr(config, "architectures", None) or ())
        )
        and bool(getattr(config, "use_sliding_window", False))
        and bool(getattr(config, "sample_from_anchor", False))
        and has_target_layers
        and isinstance(block_size, int)
        and block_size == _REDHAT_WINDOW_GQA_NATIVE_WIDTH
        and verify_width == _REDHAT_WINDOW_GQA_VERIFY_WIDTH
    ):
        return block_size
    return None


class AscendDSparkProposer(AscendDflashProposer):
    """DSpark block proposer.

    DSpark uses vLLM's ``mtp`` method in user config, but its execution shape is
    closer to DFlash: target hidden states prepopulate draft K/V, then one
    anchor-first query block emits all speculative tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        self.sample_from_anchor = getattr(self.draft_model_config.hf_config, "sample_from_anchor", True)
        self._native_num_speculative_tokens = _get_native_window_gqa_width(
            self.draft_model_config.hf_config,
            self.num_speculative_tokens,
        )
        draft_width = self._native_num_speculative_tokens or self.num_speculative_tokens
        if self.sample_from_anchor:
            self.num_query_per_req = draft_width
        else:
            self.num_query_per_req = 1 + draft_width

        blk = 1 + draft_width
        self._dspark_draft_buffer = torch.zeros((self.max_batch_size, blk), dtype=torch.int64, device=device)
        self._dspark_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        # Replace the target-sized DFlash buffers with the draft model's hidden
        # size. Assignment releases the old tensors without an explicit del.
        self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        dynamic_spec_config = get_ascend_config().dynamic_spec_config
        self.dynamic_spec = None

        if dynamic_spec_config.method == "dspark":
            self.dynamic_spec = DynamicSpecScheduler(
                method="dspark",
                method_params=dynamic_spec_config.method_params,
                max_batch_size=self.max_batch_size,
                num_speculative_tokens=self.num_speculative_tokens,
                device=device,
            )
        # The RedHat graph is trained and compiled for a fixed eight-row
        # block. Dynamic verification changes the public K and its confidence
        # buffers, so leave this special path on the fixed K=7 Verify ABI.
        if self._native_num_speculative_tokens is not None:
            self.dynamic_spec = None
        # DSpark runs eager only (Ascend cudagraph unsupported on this path).
        self.use_cuda_graph = False
        # Max query tokens depend on whether sampling from anchor or not.
        self.max_query_tokens = self.max_batch_size * self.num_query_per_req
        self.max_positions = self.max_num_tokens + self.max_query_tokens
        self.arange_dflash = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )
        # Position ids for the draft query block [max_query_tokens].
        # Overrides dflash:49; v2 uses input_buffers.positions.
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        # Primary-group query slot mapping buffer [max_query_tokens].
        # Overrides dflash:37; v2 uses BlockTables.slot_mappings. Per-non-
        # primary-gid buffers live in _per_group_query_slot_mapping_buffers.
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        # The common proposer allocates these buffers from the Verify width.
        # RedHat's native GQA graph has one extra query row, so grow only this
        # DSpark-owned path; the target runner and rejection sampler continue
        # to use the configured Verify width.
        if self._native_num_speculative_tokens is not None:
            max_num_reqs = getattr(self.runner, "max_num_reqs", self.max_batch_size)
            max_query_slots = self.max_num_tokens + (draft_width - 1) * max_num_reqs
            self.slot_mapping_group = [
                torch.zeros(max_query_slots, dtype=torch.int32, device=device)
                for _ in range(draft_width)
            ]
            runner_max_num_tokens = getattr(self.runner, "max_num_tokens", self.max_num_tokens)
            metadata_len = runner_max_num_tokens + 2 * max_num_reqs
            self.seq_lens_group = [
                torch.zeros(metadata_len, dtype=torch.int32, device=device)
                for _ in range(draft_width)
            ]
            self.query_start_loc_group = [
                torch.zeros(metadata_len, dtype=torch.int32, device=device)
                for _ in range(draft_width)
            ]
            required_indices = self.max_batch_size * draft_width
            if self.token_indices_to_sample.numel() < required_indices:
                self.token_indices_to_sample = torch.zeros(
                    required_indices, dtype=torch.int32, device=device
                )

        # The v1 runner owns block tables and slot mappings. Keep per-group
        # references here because K3 draft layers can span multiple cache
        # groups with different logical block sizes.
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        # Per-gid logical block size used to expand slot mappings. The KV
        # manager's physical page can be larger when hybrid cache groups share
        # one allocation, so kv_cache_spec.block_size is not interchangeable
        # with the attention kernel's block size.
        self._per_group_kernel_block_sizes: dict[int, int] = {}

        self._per_group_block_table_buffers: dict[int, torch.Tensor] = {}
        self._per_group_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._per_group_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None

    def _compute_confidence(
        self,
        last_hidden_states: torch.Tensor,
        draft_token_ids: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        num_tokens = num_reqs * self.num_speculative_tokens
        flat_hidden = last_hidden_states.reshape(num_tokens, last_hidden_states.shape[-1])
        # Markov embeddings of the draft input tokens (cheap lookup, so they
        # are recomputed here instead of being captured in the drafting loop).
        markov_embs = self.model.markov_embed(draft_token_ids[:, : self.num_speculative_tokens])
        # The confidence head concatenates both inputs, so their dtypes must
        # match; it upcasts to float32 internally.
        flat_markov = markov_embs.reshape(num_tokens, markov_embs.shape[-1]).to(flat_hidden.dtype)
        conf_raw = self.model.compute_confidence(flat_hidden, flat_markov)
        confidence = self._dspark_confidence_logits_buffer[:num_reqs]
        confidence.copy_(conf_raw.reshape(num_reqs, self.num_speculative_tokens))
        return confidence

    def _propose(self, num_speculative_tokens: int, *args, **kwargs) -> torch.Tensor:
        """Run the native eight-row graph and expose the seven-row Verify ABI."""
        native_width = self._native_num_speculative_tokens
        if native_width is None or num_speculative_tokens == 0:
            return super()._propose(num_speculative_tokens, *args, **kwargs)
        if num_speculative_tokens > native_width - 1:
            raise ValueError(
                "Window GQA DSpark supports at most native_width - 1 draft "
                f"tokens ({native_width - 1}), got {num_speculative_tokens}."
            )

        # Target verification remains expressed in the seven-token width.
        # Only the draft graph sees its native eight-row width; the final row
        # is discarded before returning.
        verify_width = num_speculative_tokens
        try:
            draft_token_ids = super()._propose(native_width, *args, **kwargs)
            if self._last_draft_probs is not None:
                self._last_draft_probs = self._last_draft_probs[:, :verify_width].contiguous()
            return draft_token_ids[:, :verify_width]
        finally:
            # The base implementation records the width passed to it. Restore
            # the public Verify width after the fixed-width native graph run;
            # scheduler/rejection-sampling state is always expressed in K=7.
            self.num_speculative_tokens = verify_width

    def load_model(self, model: torch.nn.Module) -> None:
        """Load DSpark and infer the window from a native GQA checkpoint.

        The Ascend option ``draft_window_size`` remains an explicit override.
        RedHat's GQA checkpoint already carries the trained window in its
        ``transformer_layer_config``; use it only for that DSpark format so
        existing DFlash/MLA and non-window DSpark models keep their behavior.
        """
        super().load_model(model)
        config = self.draft_model_config.hf_config
        if self._native_num_speculative_tokens is None or self.draft_window_size is not None or not (
            (
                getattr(config, "model_type", None) == "qwen3"
                or "Qwen3DSparkModel" in (getattr(config, "architectures", None) or ())
            )
            and bool(getattr(config, "use_sliding_window", False))
            and getattr(config, "sliding_window", None) is not None
        ):
            return

        # ``seq_lens`` already includes the native query block.  Preserve the
        # trained history [anchor-W, anchor) as well as all native query rows,
        # so the paged table must retain W + native_width keys.  FIA then
        # applies ``dflash_config.swa_window_size = W + native_width - 1``
        # query-relatively, which recreates the anchor-aligned mask.
        history_window = int(config.sliding_window)
        cache_window = history_window + self._native_num_speculative_tokens
        self.draft_window_size = cache_window
        self.sliding_window = SlidingWindowAdapter(
            cache_window,
            self.kernel_block_size,
            self.runner.max_num_reqs,
            future_offset=0,
            device=self.device,
        )

    def initialize_attn_backend(
        self,
        kv_cache_config,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        self._draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())
        self.attn_layer_names = list(sorted(self._draft_attn_layer_names))
        self._per_group_kernel_block_sizes = {}
        self.draft_attn_groups: list[AttentionGroup] = []

        for kv_cache_gid, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            draft_layer_names_in_group = set(kv_cache_group_spec.layer_names) & self._draft_attn_layer_names
            if not draft_layer_names_in_group:
                continue

            attention_groups: dict[tuple[str, Any], AttentionGroup] = {}
            # iterate in a way like vllm's llm_base_proposer
            for layer_name in draft_layer_names_in_group:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (attn_backend.full_cls_name(), layer_kv_cache_spec)

                if key not in attention_groups:
                    kernel_block_size = int(
                        kernel_block_sizes[kv_cache_gid]
                        if kernel_block_sizes is not None and kv_cache_gid < len(kernel_block_sizes)
                        else layer_kv_cache_spec.block_size
                    )
                    attn_group = AttentionGroup(
                        attn_backend,
                        [layer_name],
                        layer_kv_cache_spec,
                        kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    self._per_group_kernel_block_sizes[kv_cache_gid] = kernel_block_size
                    attention_groups[key] = attn_group
                else:
                    attention_groups[key].layer_names.append(layer_name)

            self.draft_attn_groups.extend(attention_groups.values())

        if (
            getattr(self.runner, "device_metadata_executor", None) is not None
            and self.dcp_size == 1
            and not enable_pcp()
        ):
            for attn_group in self.draft_attn_groups:
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, AscendDSAMetadataBuilder):
                    builder.enable_dspark_device_metadata(self.max_query_tokens)
                else:
                    from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder

                    if isinstance(builder, AscendDSAV41MetadataBuilder):
                        builder.enable_device_metadata()

        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        self.kernel_block_size = self._per_group_kernel_block_sizes[self.kv_cache_gid]

        name_to_gid = {
            ln: gid
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            for ln in group.layer_names
            if ln in self.attn_layer_names
        }
        self._layer_group_idx = [name_to_gid[name] for name in self.attn_layer_names]

        # some buffers need information of groups
        self._per_group_query_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_query_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }
        self._per_group_context_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # The initial input token of markovHead is the next token
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        self._dspark_seed_buffer[n:].fill_(0)
        batch_size = cad.num_reqs
        num_query_total = batch_size * self.num_query_per_req
        num_sample_total = batch_size * self.num_speculative_tokens
        has_num_rejected = num_rejected_tokens_gpu is not None
        dcp_size = getattr(self, "dcp_size", 1)
        dcp_rank = getattr(self, "dcp_rank", 0)
        cp_interleave_size = self.vllm_config.parallel_config.cp_kv_cache_interleave_size if dcp_size > 1 else 1
        long_seq_args = None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        self._per_group_block_table_buffers = {
            attn_group.kv_cache_group_id: self._per_group_block_tables[attn_group.kv_cache_group_id]
            for attn_group in self.draft_attn_groups
        }
        self._context_slot_mapping_buffers = None
        self._dflash_num_context = int(cad.query_start_loc_cpu[batch_size])
        self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states[: self._dflash_num_context]

        token_indices_to_sample = torch.empty(
            num_sample_total,
            dtype=torch.int32,
            device=self.device,
        )

        # Query block: reuse the DFlash inputs kernel logic (host-side ref)
        # per kv-cache-group to fill positions / input_ids / query slot_mapping
        # / token_indices.
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            gid_block_table = self._per_group_block_table_buffers[gid]
            kernel_block_size = self._per_group_kernel_block_sizes[gid]
            copy_and_expand_dflash_and_dspark_inputs_kernel[
                (_compute_num_programs(self._dflash_num_context, num_query_total),)
            ](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                context_slot_mapping_ptr=self._per_group_slot_mappings[gid],
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=self._per_group_context_slot_mapping_buffers[gid],
                out_query_slot_mapping_ptr=self._per_group_query_slot_mapping_buffers[gid],
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=gid_block_table,
                block_table_stride=gid_block_table.stride(0),
                # Metadata
                query_start_loc_ptr=cad.query_start_loc,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=num_rejected_tokens_gpu,
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=kernel_block_size,
                num_query_per_req=self.num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
                DCP_SIZE=dcp_size,
                DCP_RANK=dcp_rank,
                CP_INTERLEAVE_SIZE=cp_interleave_size,
            )
        # to compute self._context_slot_mapping_buffers from dict to list
        self._context_slot_mapping_buffers = [
            self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
        ]

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dflash[: batch_size + 1] * self.num_query_per_req
        cad.seq_lens = effective_seq_lens + self.num_query_per_req
        # The model runner has already corrected this canonical host mirror
        # with the accepted-token count. Extend it on CPU alongside the device
        # lengths, without another reject D2H copy or attention-side wait.
        if cad._seq_lens_cpu is not None:
            draft_seq_lens_cpu = cad._seq_lens_cpu.clone()
            draft_seq_lens_cpu[:batch_size].add_(self.num_query_per_req)
            cad._seq_lens_cpu = draft_seq_lens_cpu
            if getattr(cad, "seq_lens_cpu", None) is not None:
                cad.seq_lens_cpu = draft_seq_lens_cpu
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * self.num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [self.num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = self.num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = self.num_query_per_req
        cad.max_seq_len = cad.max_seq_len + self.num_query_per_req
        cad.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_query_total]
        cad.positions = self.positions  # this would be sliced in attention backend
        if hasattr(self.model, "get_draft_attn_causal"):
            # Currently, attention causality across draft layers are uniform.
            cad.causal = self.model.get_draft_attn_causal()[0]
        elif self._native_num_speculative_tokens is not None:
            # AscendQwen3DSpark inherits the upstream Qwen3 model and does not
            # expose K3's get_draft_attn_causal() hook.  RedHat Window GQA is
            # causal; keep this flag true so the metadata builder retains the
            # causal mask required by FIA's sparse_mode=4 window path.
            dflash_config = getattr(self.draft_model_config.hf_config, "dflash_config", {}) or {}
            cad.causal = bool(dflash_config.get("causal", True))
        else:
            cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        if dcp_size > 1:
            assert self.runner is not None
            dcp_manager = getattr(self.runner, "dcp_manager", None)
            assert dcp_manager is not None
            dcp_manager.prepare_parallel_draft_metadata(cad, self.draft_attn_groups)

        return num_query_total, token_indices_to_sample, cad, long_seq_args

    def _clear_dummy_slot_mappings(self) -> None:
        for buf in self._per_group_query_slot_mapping_buffers.values():
            buf.fill_(-1)
        for buf in self._per_group_context_slot_mapping_buffers.values():
            buf.fill_(-1)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        num_query_total = num_reqs * self.num_query_per_req
        num_query_tokens = min(num_query_total if num_reqs > 0 else num_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        self.token_indices_to_sample.fill_(0)
        self._clear_dummy_slot_mappings()

        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            model_instance=self.model,
            draft_attn_metadatas=[],
            eplb_heat_collection_status=(
                self.runner.eplb_heat_collection_status if self.runner.dynamic_eplb else False
            ),
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                self._dflash_num_context = num_input_tokens
                # The native RedHat graph always executes all eight rows.
                # Keep profiling/DP dummy runs consistent with the real
                # proposal path even though the public Verify width is seven.
                public_width = self.num_speculative_tokens
                native_width = self._native_num_speculative_tokens
                if native_width is not None:
                    self.num_speculative_tokens = native_width
                try:
                    self._runnable(
                        num_input_tokens=num_input_tokens,
                        batch_size=num_reqs,
                        token_indices_to_sample=self.token_indices_to_sample[
                            : num_reqs * self.num_speculative_tokens
                        ],
                        target_positions=self._get_positions(num_input_tokens),
                        inputs_embeds=None,
                        multi_steps_attn_metadata=[],
                        num_tokens=num_input_tokens,
                    )
                finally:
                    if native_width is not None:
                        self.num_speculative_tokens = public_width
