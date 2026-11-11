# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
import time
from typing import Optional
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    flex_attention,
    create_block_mask
)
from kv_compression import KVCompressor
from kv_quantization import KVQuantizer

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None
    has_qkv_bias: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),

    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "llama-3.1-8b": dict(block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "llama-3.1-70b": dict(block_size=131072, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "llama-3.1-405b": dict(block_size=131072, n_layer=126, n_head=128, n_local_heads=8, dim=16384, intermediate_size=53248, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "Qwen2.5-7B-Instruct": dict(block_size=32768, n_layer=28, n_head=28, n_local_heads=4, dim=3584, intermediate_size=18944, vocab_size=152064, rope_base=1000000, 
        norm_eps=1e-6, has_qkv_bias=True),
    "Qwen2-0.5B-Instruct": dict(block_size=131072, n_layer=24, n_head=14, n_local_heads=2, dim=896, intermediate_size=4864, vocab_size=151936, rope_base=1000000, 
        norm_eps=1e-6, has_qkv_bias=True), 
}

import torch
from torch import nn, Tensor

class KVCache(nn.Module):
    """
    Unified KV cache supporting both full and sliding-window (sink+window) modes.

    Args:
        max_batch_size: maximum batch size (usually 1 for autoregressive inference)
        max_seq_length: total sequence length for full cache
        n_heads: number of attention heads
        head_dim: dimension per head
        compress: whether to use sink+window compression
        sink_size: number of prefix tokens to always retain (only used if compress=True)
        window_size: sliding-window size of recent tokens (only used if compress=True)
        dtype: tensor dtype
    """
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        compress: bool = False,
        sink_size: int = 0,
        window_size: int = None,
        dtype: torch.dtype = torch.bfloat16,
        quantize: bool = False,
        quantizer: KVQuantizer = None
    ):
        super().__init__()
        self.compress = compress
        self.quantize = quantize
        self.quantizer = quantizer
        
        if not compress and not quantize:
            # Full KV cache: preallocate for max_seq_length
            cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
            self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
            self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

        elif compress:
            assert window_size is not None, "window_size must be set when compress=True or quantize=True"
            self.sink = sink_size
            self.window = window_size
            self.total = sink_size + window_size
            # compressed buffers
            self.register_buffer(
                'k_cache', torch.zeros(max_batch_size, n_heads, self.total, head_dim, dtype=dtype)
            )
            self.register_buffer(
                'v_cache', torch.zeros_like(self.k_cache)
            )
            # track original positions for masking
            self.register_buffer(
                'kv_positions', torch.zeros(self.total, dtype=torch.long),
                persistent=False
            )
            self.ptr = 0
            self.prefill_done = False
        
        elif quantize:
            assert window_size is not None, "window_size must be set when quantize=True"
            self.sink = sink_size
            self.window = window_size
            self.total_fp = sink_size + window_size
            self.max_seq = max_seq_length

            shape_fp = (max_batch_size, n_heads, self.total_fp, head_dim)
            self.register_buffer("k_cache", torch.zeros(shape_fp, dtype=dtype))
            self.register_buffer("v_cache", torch.zeros_like(self.k_cache))

            quant_cap = max_seq_length - self.total_fp
            assert quant_cap > 0, "max_seq_length must exceed sink+window"

            # int8 data
            shape_q = (max_batch_size, n_heads, quant_cap, head_dim)
            self.register_buffer("k_quant", torch.zeros(shape_q, dtype=torch.int8))
            self.register_buffer("v_quant", torch.zeros_like(self.k_quant))

            # per‑slot scale (float32, one per key vector)
            scale_shape = (max_batch_size, n_heads, quant_cap, 1)
            self.register_buffer("k_scale", torch.ones(scale_shape, dtype=torch.float32))
            self.register_buffer("v_scale", torch.ones_like(self.k_scale))

            # map absolute positions → buffer slot index
            #   ≥0 & < total_fp : index into k_cache/v_cache  
            #   <0 & ≥−quant_cap : `−1−slot` → index into k_quant/v_quant
            self.register_buffer("kv_map", torch.zeros(max_seq_length, dtype=torch.int64))

            # pointers & length
            self.ptr = 0
            self.fp_ptr = 0
            self.q_ptr = 0
            self.cache_len = 0
            self.mid = max_seq_length - self.total_fp  # tokens that go to quant cache


    def prefill_update(self, k_new: Tensor, v_new: Tensor):
        """
        Bulk‑load prompt KV into cache.  Handles three mutually exclusive modes:
            • self.compress == True: sliding‑window compression
            • self.quantize == True: partial int8 quantization of “middle” tokens
            • else: no compression, no quantization (full precision)

        Args:
            k_new, v_new: [B, H, T_prefill, D]
        Returns:
            (k_cache_slice, v_cache_slice) to be used by attention
        """
        T = k_new.size(2)
        
        if self.compress:
            if T <= self.total:
                self.k_cache[:, :, :T] = k_new
                self.v_cache[:, :, :T] = v_new
                self.kv_positions[:T] = torch.arange(T, device=k_new.device)
                compressed_k = self.k_cache[:, :, :T]
                compressed_v = self.v_cache[:, :, :T]
            else:
                # keep sink + last window
                self.k_cache[:, :, :self.sink] = k_new[:, :, :self.sink]
                self.k_cache[:, :, self.sink:] = k_new[:, :, -self.window:]
                self.v_cache[:, :, :self.sink] = v_new[:, :, :self.sink]
                self.v_cache[:, :, self.sink:] = v_new[:, :, -self.window:]
                # set positions
                self.kv_positions[:self.sink] = torch.arange(self.sink, device=k_new.device)
                self.kv_positions[self.sink:] = torch.arange(
                    T - self.window, T, device=k_new.device
                )
                total = self.sink + self.window
                compressed_k = self.k_cache[:, :, :total]
                compressed_v = self.v_cache[:, :, :total]

            self.ptr = 0
            self.prefill_done = True
            return compressed_k, compressed_v

        elif self.quantize:
            # (a) if we haven't yet exceeded sink+window, just fill FP cache
            self.ptr = 0
            self.q_ptr = 0
            self.fp_ptr = 0
            self.cache_len = 0
            self.kv_map[:] = 0 

            if T <= self.total_fp:
                self.k_cache[:, :, :T] = k_new
                self.v_cache[:, :, :T] = v_new
                # map absolute pos → FP slot
                self.kv_map[:T] = torch.arange(T, device=k_new.device)
                self.cache_len = T
                self.prefill_done = True
                # attention will use FP slice
                return self.k_cache[:, :, :T], self.v_cache[:, :, :T]

            # (b) else: we have more tokens than sink+window → quantize the “middle”
            mid_len = T - self.total_fp
            device = k_new.device

            # 1) sink (full‑precision)
            self.k_cache[:, :, :self.sink] = k_new[:, :, :self.sink]
            self.v_cache[:, :, :self.sink] = v_new[:, :, :self.sink]
            self.kv_map[:self.sink] = torch.arange(self.sink, device=device)

            # 2) middle (quantize into int8 + scale)
            k_mid = k_new[:, :, self.sink : self.sink + mid_len]
            v_mid = v_new[:, :, self.sink : self.sink + mid_len]
            qk, scale_k = self.quantizer.true_quantize_tensor(k_mid)
            qv, scale_v = self.quantizer.true_quantize_tensor(v_mid)
            # write into pre‑allocated quant buffers
            self.k_quant[:, :, :mid_len] = qk
            self.k_scale[:, :, :mid_len, :] = scale_k
            self.v_quant[:, :, :mid_len] = qv
            self.v_scale[:, :, :mid_len, :] = scale_v
            # map absolute positions → quant slots via negative coding
            #   slot i in quant buffer is referenced as kv_map[p] = -1 - i
            indices = -1 - torch.arange(mid_len, device=device)
            self.kv_map[self.sink : self.sink + mid_len] = indices

            # 3) window (full‑precision of last `window` tokens)
            start_win = T - self.window
            self.k_cache[:, :, self.sink :] = k_new[:, :, start_win : T]
            self.v_cache[:, :, self.sink :] = v_new[:, :, start_win : T]
            # map those last tokens
            win_slots = torch.arange(self.window, device=device)
            self.kv_map[start_win : T] = self.sink + win_slots

            # finalize
            self.cache_len = T
            self.prefill_done = True
            
            # attention will pull from both FP and dequantized int8 via kv_map lookup
            return self.k_cache, self.v_cache

        else:
            return k_new, v_new


    def update(self, input_pos: Tensor, k: Tensor, v: Tensor):
        """
        k, v: [B, H, 1, D] during decode
        input_pos: [1] absolute position
        """
        if not self.compress and not self.quantize:
            self.k_cache[:, :, input_pos] = k  # no .squeeze() or .item()
            self.v_cache[:, :, input_pos] = v
            return self.k_cache, self.v_cache

        elif self.compress:
            assert self.prefill_done, "Must call prefill_update before update"
            idx = self.sink + self.ptr
            self.k_cache[:, :, idx] = k.squeeze(2)
            self.v_cache[:, :, idx] = v.squeeze(2)
            self.kv_positions[idx] = input_pos.item()
            self.ptr = (self.ptr + 1) % self.window
            return self.k_cache, self.v_cache

        elif self.quantize and self.quantizer is not None:
            assert self.prefill_done, "Must call prefill_update before update"
            pos = input_pos.item()
            step = self.ptr  # how many tokens we've decoded since prefill

            # (a) fill out the “mid” quantized region first
            if step < self.mid:
                # squeeze out the length‑1 dim
                k_slice = k.squeeze(2)  # [B,H,D]
                v_slice = v.squeeze(2)

                # true quantize → int8 + per‑vector scale
                # we unsqueeze back to shape [B,H,1,D] so true_quantize_tensor returns
                # (qx: [B,H,1,D] int8, scale: [B,H,1,1] float32)
                qk, scale_k = self.quantizer.true_quantize_tensor(k_slice.unsqueeze(2))
                qv, scale_v = self.quantizer.true_quantize_tensor(v_slice.unsqueeze(2))
                # store into your pre‑allocated quant buffers
                self.k_quant[:, :, step]    = qk.squeeze(2)            # [B,H,D]
                self.k_scale[:, :, step, 0] = scale_k.squeeze().unsqueeze(0)  # shape [1, 8]
                self.v_quant[:, :, step]    = qv.squeeze(2)
                self.v_scale[:, :, step, 0] = scale_v.squeeze().unsqueeze(0)  # shape [1, 8]

                # record in kv_map so attention knows to look in quant cache
                self.kv_map[pos] = -1 - step

            # (b) once the “mid” is full, roll into your sliding‑window in fp
            else:
                window_idx = (step - self.mid) % self.window
                full_idx = self.sink + window_idx

                self.k_cache[:, :, full_idx] = k.squeeze(2)
                self.v_cache[:, :, full_idx] = v.squeeze(2)

                # map this position into the fp cache
                self.kv_map[pos] = full_idx

            # advance counters
            self.ptr       += 1
            self.cache_len += 1

            # return the fp cache placeholders; attention will reconstruct the full keys/values
            return self.k_cache, self.v_cache
        else:
            # should never get here
            raise RuntimeError("KVCache.update: invalid mode")


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.get_mask_mod = get_mask_mod

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for layer in self.layers:
            attn = layer.attention
            attn.kv_cache = KVCache(
                max_batch_size=max_batch_size,
                max_seq_length=max_seq_length,
                n_heads=attn.n_local_heads,
                head_dim=attn.head_dim,
                dtype=attn.wqkv.weight.dtype,
                compress=attn.kv_compressor.enabled if attn.kv_compressor else False,
                quantize=attn.kv_quantizer.enabled if attn.kv_quantizer else False,
                quantizer=attn.kv_quantizer if attn.kv_quantizer else None,
                sink_size=attn.kv_compressor.sink_size if attn.kv_compressor and attn.kv_compressor.enabled else (
                    attn.kv_quantizer.sink_size if attn.kv_quantizer and attn.kv_quantizer.enabled else 0
                ),
                window_size=attn.kv_compressor.window_size if attn.kv_compressor and attn.kv_compressor.enabled else (
                    attn.kv_quantizer.window_size if attn.kv_quantizer and attn.kv_quantizer.enabled else 0
                ),
            )

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype, self.config.rope_scaling)

    def forward(self, mask: BlockMask, idx: Tensor, input_pos: Optional[Tensor] = None, prefill_mode: bool = False) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask, prefill_mode=prefill_mode)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: BlockMask, prefill_mode: bool = False) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos, prefill_mode=prefill_mode)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=getattr(config, "has_qkv_bias", False))
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)
        
        self.kv_compressor = KVCompressor(
            enabled=False,
            window_size=None,
            sink_size=0
        )
        self.kv_quantizer = KVQuantizer(
            enabled=False,
            quantize_type="int8",
            sink_size=0,
            window_size=0
        )


    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
        if prefix + "bqkv" in state_dict:
            self.wqkv.bias = torch.nn.Parameter(state_dict.pop(prefix + "bqkv"))

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: BlockMask,
        input_pos: Optional[Tensor] = None,
        prefill_mode: bool = False
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        # 1) project & split into q, k, v
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # 2) apply rotary embeddings
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # 3) move head dim to [B, H, L, D]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        compress = getattr(self.kv_cache, "compress", False)
        quantize = getattr(self.kv_cache, "quantize", False)

        # 4) populate KV cache
        if self.kv_cache is not None:
            if prefill_mode:
                if compress or quantize:
                    k,v = self.kv_cache.prefill_update(k, v)
                else:
                    pos = input_pos.view(-1)
                    self.kv_cache.k_cache[:, :, pos] = k
                    self.kv_cache.v_cache[:, :, pos] = v
                    k = self.kv_cache.k_cache
                    v = self.kv_cache.v_cache

            else:
                # One‐token sliding update
                k, v = self.kv_cache.update(input_pos, k, v)

        # if we are actually in sliding‐window mode, crop the mask to match
        if compress:
            # 5) adjust mask to match (q_len, kv_len)
            q_len = q.shape[2]
            kv_len = k.shape[2]
            if mask.seq_lengths != (q_len, kv_len):
                mask.seq_lengths = (q_len, kv_len)
                mask._adjust(q_len, kv_len)

            # 6) override mask.mask_mod if compressed
            if self.kv_cache is not None and getattr(self.kv_cache, "kv_positions", None) is not None:
                orig_pos = self.kv_cache.kv_positions      # [kv_len]
                # compute absolute position of the first key for this call
                if prefill_mode:
                    # Use default causal behavior in prefill
                    mask.mask_mod = lambda b, h, q_idx, kv_idx: kv_idx <= q_idx
                else:
                    cur_base = input_pos[0].item() - q_len + 1
                    def new_mask_mod(b, h, q_idx, kv_idx):
                        return orig_pos[kv_idx] <= cur_base + q_idx
                    mask.mask_mod = new_mask_mod

        elif quantize:
            # how many total slots have been filled so far?
            kv_map = self.kv_cache.kv_map           # [max_seq_length]
            kv_len = min(self.kv_cache.cache_len, kv_map.size(0))
            slot = kv_map[:kv_len]                  # e.g. tensor([0,1,2,-1,-2,3,...])

            # clamp positives → [0..total_fp-1], and compute quant‑indices
            pos_fp = slot.clamp(min=0)              # full‑precision indices
            pos_q  = (-1 - slot).clamp(min=0)       # quantized indices

            # gather full‑precision keys & values
            k_fp = self.kv_cache.k_cache[:, :, pos_fp, :]        # [B,H,kv_len,D]
            v_fp = self.kv_cache.v_cache[:, :, pos_fp, :]

            # gather quantized and dequantize: int8 → float * scale
            target_dtype = q.dtype  # e.g., torch.bfloat16 or torch.float16
            # Dequantize and cast
            k_q = (self.kv_cache.k_quant[:, :, pos_q, :].to(torch.float32) *
                self.kv_cache.k_scale[:, :, pos_q, :]).to(target_dtype)

            v_q = (self.kv_cache.v_quant[:, :, pos_q, :].to(torch.float32) *
                self.kv_cache.v_scale[:, :, pos_q, :]).to(target_dtype)

            # now select between the two per‑slot
            selector = (slot >= 0).view(1, 1, -1, 1)               # broadcast mask
            k = torch.where(selector, k_fp, k_q)
            v = torch.where(selector, v_fp, v_q)

            # adjust mask lengths
            mask.seq_lengths = (q.shape[2], kv_len)
            mask._adjust(q.shape[2], kv_len)

        # 7) perform attention
        y = flex_attention(
            q, k, v,
            block_mask=mask,
            enable_gqa=(self.n_head != self.n_local_heads)
        )

        # 8) project out and return
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
