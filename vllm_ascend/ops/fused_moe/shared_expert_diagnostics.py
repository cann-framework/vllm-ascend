import logging
from contextlib import contextmanager

import torch
import torch_npu
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from vllm_ascend.ascend_config import get_ascend_config

logger = init_logger(__name__)


@contextmanager
def shared_expert_core_limit(stage, core_limit):
    if not get_ascend_config().multistream_overlap_shared_expert:
        yield
        return
    with torch_npu.npu.npugraph_ex.scope.limit_core_num(*core_limit):
        logger.info_once(
            "[SharedExpertCoreLimit] stage=%s aic_num=%d aiv_num=%d",
            stage,
            *core_limit,
        )
        yield


class SharedExpertDiagnostics:
    def __init__(self, layer_name, vllm_config):
        self.layer_name = layer_name
        transfer = vllm_config.kv_transfer_config
        model = vllm_config.model_config
        self.enabled = (
            logger.isEnabledFor(logging.INFO)
            and getattr(model, "runner_type", None) != "draft"
            and transfer is not None
            and (
                getattr(transfer, "kv_role", None) == "kv_consumer"
                or (getattr(transfer, "is_kv_consumer", False) and not getattr(transfer, "is_kv_producer", False))
                or getattr(transfer, "kv_connector", None) == "DecodeBenchConnector"
            )
        )
        self._seen = set()

    def record(self, stage, **fields):
        if not self.enabled or torch.compiler.is_compiling():
            return
        context = get_forward_context()
        if getattr(context, "is_draft_model", False):
            return
        capturing = torch.npu.is_current_stream_capturing()
        graph_mode = getattr(context, "cudagraph_runtime_mode", None)
        key = (
            stage,
            capturing,
            str(graph_mode),
            fields.get("schedule"),
            fields.get("overlap"),
            fields.get("comm_type"),
            fields.get("rejected"),
            fields.get("input_path"),
            getattr(context, "in_profile_run", False),
        )
        if key in self._seen:
            return
        self._seen.add(key)
        stream = torch.npu.current_stream()
        logger.info_once(
            "[SharedExpertDebug] layer=%s stage=%s device=%s stream=%s capturing=%s graph_mode=%s profile_run=%s %s",
            self.layer_name,
            stage,
            stream.device,
            stream.npu_stream,
            capturing,
            graph_mode,
            getattr(context, "in_profile_run", False),
            " ".join(f"{name}={value}" for name, value in fields.items()),
        )
