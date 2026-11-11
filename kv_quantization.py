# kv_quantization.py
import torch
from typing import Tuple

class KVQuantizer:
    """
    Controls whether and how to quantize old KV-cache entries.
    """
    def __init__(
        self,
        enabled: bool = False,
        quantize_type: str = "int8",
        sink_size: int = 0,
        window_size: int = 0,
    ):
        self.enabled = enabled
        self.quantize_type = quantize_type
        self.sink_size = sink_size
        self.window_size = window_size

    def true_quantize_tensor(self, x: torch.Tensor) -> Tuple[torch.IntTensor, torch.FloatTensor]:
        # Compute per-token max over channels (dim=-1)
        max_abs = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, H, T, 1]
        scale = max_abs / 127.0
        qx = torch.clamp((x / scale).round(), -127, 127).to(torch.int8)
        return qx, scale
