"""Compact Gemma4 model for prime-rl training."""

from __future__ import annotations

from typing import NamedTuple, cast

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, FlexKernelOptions, create_block_mask, flex_attention
from torch.nn.attention.varlen import varlen_attn
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.gemma4.converting_gemma4 import run_on_flat_language_model_keys
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput

# Compile flex_attention with inductor autotuning for best kernel selection
_compiled_flex_attention = torch.compile(
    flex_attention,
    options={
        # "max_autotune": True,
        # "coordinate_descent_tuning": True,
        "triton.cudagraphs": False,
    },
)

_compiled_create_block_mask = torch.compile(create_block_mask)

# For head_dim=512 (full attention layers), default block sizes OOM on registers.
# BLOCK_M/N=32 fits within B200's 232448-byte register limit.
# Backward block sizes must also be pinned — without them the autotuner explores
# 64x64 backward configs that exceed the register limit and crash with
# IS_DIVISIBLE=False (any seq_len not perfectly aligned).
# BLOCK_M1=16 (phase 1 / dK,dV) is ~6% faster than 32 for the backward pass.
_FLEX_HD512_KERNEL_OPTIONS: FlexKernelOptions = {
    "BLOCK_M": 32, "BLOCK_N": 32,
    "BLOCK_M1": 16, "BLOCK_N1": 32,
    "BLOCK_M2": 32, "BLOCK_N2": 32,
}


class _RMSNorm(nn.RMSNorm):
    """RMSNorm that preserves input dtype regardless of weight dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).to(x.dtype)


def _default_position_ids(x: torch.Tensor) -> torch.Tensor:
    return torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat([x_rot, x_pass], dim=-1)


def _get_kv_sharing_info(config: Gemma4TextConfig, layer_idx: int) -> tuple[int | None, bool]:
    num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
    if not num_kv_shared_layers:
        return None, False

    source_layers = config.layer_types[:-num_kv_shared_layers]  # ty:ignore[not-subscriptable]
    source_idx = len(source_layers) - 1 - source_layers[::-1].index(config.layer_types[layer_idx])  # ty:ignore[not-subscriptable]
    if layer_idx >= len(source_layers):
        return source_idx, False
    return None, layer_idx == source_idx


def _layer_head_dim(config: Gemma4TextConfig, layer_type: str) -> int:
    return config.global_head_dim if layer_type == "full_attention" else config.head_dim


class _PackedAttentionMetadata(NamedTuple):
    doc_ids: torch.Tensor
    total_len: int
    cu_seq: torch.Tensor | None = None
    max_seq_len: int | None = None


def _build_packed_attention_metadata(
    position_ids: torch.Tensor,
    *,
    need_cu_seq: bool,
) -> _PackedAttentionMetadata:
    """Build packed-sequence metadata for flex and varlen attention.

    A new document begins wherever positions reset, i.e. positions[i] <= positions[i-1].
    For flex attention we use per-token document IDs. For varlen attention we also
    derive cumulative sequence offsets over the packed documents.
    """
    flat = position_ids.reshape(-1)
    boundaries = torch.zeros_like(flat, dtype=torch.bool)
    boundaries[1:] = flat[1:] <= flat[:-1]
    doc_ids = boundaries.int().cumsum(0)

    if not need_cu_seq:
        return _PackedAttentionMetadata(doc_ids=doc_ids, total_len=flat.numel())

    starts = torch.cat(
        (flat.new_zeros((1,), dtype=torch.int64), boundaries.nonzero(as_tuple=False).flatten()),
    )
    cu_seq = torch.empty(starts.numel() + 1, device=flat.device, dtype=torch.int32)
    cu_seq[:-1] = starts.to(torch.int32)
    cu_seq[-1] = flat.numel()
    max_seq_len = int(torch.diff(cu_seq).max().item())
    return _PackedAttentionMetadata(doc_ids=doc_ids, total_len=flat.numel(), cu_seq=cu_seq, max_seq_len=max_seq_len)


def _build_flex_block_masks(
    doc_ids: torch.Tensor,
    total_len: int,
    sliding_window: int | None,
    layer_types: set[str],
) -> dict[str, BlockMask]:
    """Build BlockMasks for flex_attention: one per layer type (sliding vs full).

    Packed sequences require a same-document check so tokens don't attend across
    document boundaries.
    """

    def _causal_packed(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])

    masks: dict[str, BlockMask] = {}
    if "full_attention" in layer_types:
        masks["full_attention"] = _compiled_create_block_mask(
            _causal_packed, B=None, H=None, Q_LEN=total_len, KV_LEN=total_len, device=doc_ids.device,
        )

    if sliding_window is not None and "sliding_attention" in layer_types:

        def _sliding_packed(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
            return (kv_idx <= q_idx) & (q_idx - kv_idx < sliding_window) & (doc_ids[q_idx] == doc_ids[kv_idx])

        masks["sliding_attention"] = _compiled_create_block_mask(
            _sliding_packed, B=None, H=None, Q_LEN=total_len, KV_LEN=total_len, device=doc_ids.device,
        )

    return masks


class Gemma4ScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        return super().forward(input_ids) * self.weight.new_tensor(self.embed_scale)


class Gemma4DualRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_position_embeddings
        self.config.standardize_rope_params()
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _inv_freq(config: Gemma4TextConfig, layer_type: str, device: torch.device) -> torch.Tensor:
        rope_params = config.rope_parameters[layer_type]  # ty:ignore[not-subscriptable]
        rope_type = rope_params["rope_type"]
        if rope_type != "default":
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"
            return ROPE_INIT_FUNCTIONS[rope_type](config, **kwargs)[0]

        head_dim = _layer_head_dim(config, layer_type)
        rotary_dim = int(head_dim * rope_params.get("partial_rotary_factor", 1.0))
        steps = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float)
        return 1.0 / (rope_params["rope_theta"] ** (steps / rotary_dim))

    def _cos_sin_cache(self, layer_type: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self._cache.get(layer_type)
        if cos_sin is not None and cos_sin[0].device == device:
            return cos_sin

        inv_freq = self._inv_freq(self.config, layer_type, device)
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_sin = (emb.cos(), emb.sin())
        self._cache[layer_type] = cos_sin
        return cos_sin

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_cache, sin_cache = self._cos_sin_cache(layer_type, x.device)
        return cos_cache[position_ids].to(dtype=x.dtype), sin_cache[position_ids].to(dtype=x.dtype)


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]  # ty:ignore[not-subscriptable]
        self.head_dim = _layer_head_dim(config, self.layer_type)
        self.num_key_value_heads = config.num_key_value_heads
        self.kv_shared_layer_index, self.store_full_length_kv = _get_kv_sharing_info(config, layer_idx)
        self._kernel_options = _FLEX_HD512_KERNEL_OPTIONS if self.head_dim > 256 else FlexKernelOptions()
        self._use_varlen_attn = getattr(config, "_attn_implementation", None) == "varlen" and self.head_dim == 256
        if self.layer_type == "sliding_attention":
            self._varlen_window_size = (cast(int, config.sliding_window), 0)
        else:
            self._varlen_window_size = (-1, 0)

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False)

    def compute_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute K/V for this layer. Used to pre-populate shared KV outside checkpointed regions."""
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        cos, sin = position_embeddings
        k = _apply_rotary_pos_emb(self.k_norm(self.k_proj(hidden_states).view(hidden_shape)), cos, sin)
        v = self.v_norm(self.v_proj(hidden_states).view(hidden_shape))
        return k, v

    def attn_projections(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        cos, sin = position_embeddings
        q = _apply_rotary_pos_emb(self.q_norm(self.q_proj(hidden_states).view(hidden_shape)), cos, sin)
        k, v = self.compute_kv(hidden_states, position_embeddings)
        return q, k, v

    def output_proj(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask | None = None,
        packed_attention: _PackedAttentionMetadata | None = None,
        shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if self.kv_shared_layer_index is not None and shared_kv is not None:
            hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
            cos, sin = position_embeddings
            q = _apply_rotary_pos_emb(self.q_norm(self.q_proj(hidden_states).view(hidden_shape)), cos, sin)
            k, v = shared_kv[self.kv_shared_layer_index]
        else:
            q, k, v = self.attn_projections(hidden_states, position_embeddings)

        if self._use_varlen_attn:
            if packed_attention is None or packed_attention.cu_seq is None or packed_attention.max_seq_len is None:
                raise ValueError("Varlen attention requires packed sequence metadata.")
            attn_output = varlen_attn(
                q.contiguous().flatten(0, 1),
                k.contiguous().flatten(0, 1),
                v.contiguous().flatten(0, 1),
                packed_attention.cu_seq,
                packed_attention.cu_seq,
                packed_attention.max_seq_len,
                packed_attention.max_seq_len,
                scale=1.0,
                window_size=self._varlen_window_size,
            )
        else:
            # (batch*seq, heads, dim) -> (1, heads, batch*seq, dim) for flex_attention
            q = q.contiguous().flatten(0, 1).unsqueeze(0).transpose(1, 2)
            k = k.contiguous().flatten(0, 1).unsqueeze(0).transpose(1, 2)
            v = v.contiguous().flatten(0, 1).unsqueeze(0).transpose(1, 2)

            attn_output = _compiled_flex_attention(
                q, k, v, block_mask=block_mask, scale=1.0,
                enable_gqa=True, kernel_options=self._kernel_options,
            )  # ty:ignore[invalid-assignment]

            attn_output = attn_output.transpose(1, 2).squeeze(0)

        attn_output = attn_output.contiguous().view(*hidden_states.shape[:-1], -1)
        return self.output_proj(attn_output)


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        use_double_wide_mlp = getattr(config, "use_double_wide_mlp", False)
        is_kv_shared_layer = layer_idx >= config.num_hidden_layers - num_kv_shared_layers > 0
        intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp and is_kv_shared_layer else 1)

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx)

        self.input_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        if config.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(config.hidden_size, config.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(config.hidden_size_per_layer_input, config.hidden_size, bias=False)
            self.post_per_layer_input_norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        per_layer_input: torch.Tensor | None = None,
        shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        packed_attention: _PackedAttentionMetadata | None = None,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        # Pre-compute source KV before attention so the model loop can store it.
        # This avoids relying on dict mutation side effects inside checkpointed regions.
        source_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.self_attn.store_full_length_kv:
            source_kv = self.self_attn.compute_kv(normed, position_embeddings)

        hidden_states = self.self_attn(
            normed,
            position_embeddings,
            block_mask=block_mask,
            packed_attention=packed_attention,
            shared_kv=shared_kv,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.post_feedforward_layernorm(self.mlp(hidden_states))
        hidden_states = residual + hidden_states

        if per_layer_input is not None:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states) * per_layer_input
            hidden_states = self.post_per_layer_input_norm(self.per_layer_projection(hidden_states))
            hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar.to(hidden_states.dtype)

        if source_kv is not None:
            # Return KV alongside hidden_states; the model loop stores it into shared_kv
            return hidden_states, source_kv[0], source_kv[1]
        return hidden_states


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config
        self._attn_implementation = getattr(config, "_attn_implementation", "flex_attention")
        pad_token_id = cast(int, config.pad_token_id)
        self.embed_tokens = Gemma4ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            pad_token_id,
            config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4DualRotaryEmbedding(config)
        self.layer_types = cast(list[str], config.layer_types)
        self._need_varlen_metadata = self._attn_implementation == "varlen" and any(
            _layer_head_dim(config, layer_type) == 256 for layer_type in set(self.layer_types)
        )
        self._flex_layer_types = {
            layer_type
            for layer_type in set(self.layer_types)
            if not (self._attn_implementation == "varlen" and _layer_head_dim(config, layer_type) == 256)
        }

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            total_dim = config.num_hidden_layers * config.hidden_size_per_layer_input
            self.embed_tokens_per_layer = Gemma4ScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                total_dim,
                pad_token_id,
                config.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_model_projection = nn.Linear(config.hidden_size, total_dim, bias=False)
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = _RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)
            self.per_layer_input_scale = 2.0**-0.5

    def _build_per_layer_inputs(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        token_inputs = self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        projected = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        projected = projected.reshape(*inputs_embeds.shape[:-1], self.config.num_hidden_layers, -1)
        projected = self.per_layer_projection_norm(projected)
        return (projected + token_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = _default_position_ids(inputs_embeds)
        else:
            # Normalize to 0-based: inject_prime_lm_head's patched forward generates 1-indexed position_ids,
            # but RoPE cache lookup and cu_seqlens reset detection expect 0-indexed.
            position_ids = position_ids - position_ids[:, :1]
        if self.hidden_size_per_layer_input and per_layer_inputs is None:
            assert input_ids is not None
            per_layer_inputs = self._build_per_layer_inputs(input_ids, inputs_embeds)

        packed_attention = _build_packed_attention_metadata(position_ids, need_cu_seq=self._need_varlen_metadata)
        block_masks = _build_flex_block_masks(
            packed_attention.doc_ids,
            packed_attention.total_len,
            self.config.sliding_window,
            self._flex_layer_types,
        )

        position_embeddings = {
            layer_type: self.rotary_emb(inputs_embeds, position_ids, layer_type)
            for layer_type in set(self.layer_types)
        }

        hidden_states = inputs_embeds
        shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            layer_type = self.layer_types[i]
            # .contiguous() is required for activation offloading compatibility:
            # the slice is a view with large strides from the parent tensor, and
            # offloading via saved_tensors_hooks can't preserve those strides on
            # the CPU round-trip, causing compiled backward to fail on stride checks.
            pli = None if per_layer_inputs is None else per_layer_inputs[:, :, i, :].contiguous()
            out = layer(
                hidden_states,
                position_embeddings[layer_type],
                pli,
                shared_kv,
                packed_attention,
                block_masks.get(layer_type),
            )
            # Source layers return (hidden_states, k, v); store KV outside the checkpointed region
            if isinstance(out, tuple):
                hidden_states, k, v = out
                shared_kv[i] = (k, v)
            else:
                hidden_states = out

        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states))


class Gemma4MultimodalModel(nn.Module):
    """Wrap the custom language model in Gemma4's multimodal checkpoint structure."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        text_config = cast(Gemma4TextConfig, config.text_config)
        self.language_model = Gemma4Model(text_config)

        vision_config = getattr(config, "vision_config", None)
        audio_config = getattr(config, "audio_config", None)
        if vision_config is not None or audio_config is not None:
            from transformers.models.gemma4.modeling_gemma4 import (
                Gemma4AudioModel,
                Gemma4MultimodalEmbedder,
                Gemma4VisionModel,
            )

            self.vision_tower = Gemma4VisionModel._from_config(vision_config) if vision_config is not None else None
            self.audio_tower = Gemma4AudioModel._from_config(audio_config) if audio_config is not None else None
            self.embed_vision = (
                Gemma4MultimodalEmbedder(vision_config, text_config) if vision_config is not None else None
            )
            self.embed_audio = (
                Gemma4MultimodalEmbedder(audio_config, text_config) if audio_config is not None else None
            )
        else:
            self.vision_tower = None
            self.audio_tower = None
            self.embed_vision = None
            self.embed_audio = None

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.embed_tokens = cast(Gemma4ScaledWordEmbedding, value)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> BaseModelOutputWithPast:
        return self.language_model(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)


class Gemma4PreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Gemma4Config
    base_model_prefix = "model"
    _no_split_modules = ["Gemma4DecoderLayer"]
    _supports_flex_attn = True

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return run_on_flat_language_model_keys(state_dict)

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return run_on_flat_language_model_keys(state_dict)

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return run_on_flat_language_model_keys(state_dict)

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return run_on_flat_language_model_keys(state_dict)


class Gemma4ForCausalLM(Gemma4PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Gemma4TextConfig | Gemma4Config, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        if isinstance(config, Gemma4Config):
            text_config = cast(Gemma4TextConfig, config.text_config)
            self.model: Gemma4Model | Gemma4MultimodalModel = Gemma4MultimodalModel(config)
            self._tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
            # Surface text-config attributes so inject_prime_lm_head can find them
            config.final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
        else:
            text_config = config
            self.model = Gemma4Model(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if isinstance(self.model, Gemma4MultimodalModel):
            return self.model.get_input_embeddings()
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        if isinstance(self.model, Gemma4MultimodalModel):
            self.model.set_input_embeddings(value)
            return
        self.model.embed_tokens = cast(Gemma4ScaledWordEmbedding, value)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **_: object,
    ) -> PrimeLmOutput:
        """Pre-injection inference forward. Replaced by inject_prime_lm_head at training time."""
        if position_ids is None:
            source = inputs_embeds if inputs_embeds is not None else input_ids
            assert source is not None
            position_ids = _default_position_ids(source)

        outputs = self.model(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        return PrimeLmOutput(logits=self.lm_head(outputs.last_hidden_state))

    def init_buffers_post_meta(self) -> None:
        # Gemma4DualRotaryEmbedding uses a lazy cache, no persistent buffers to init
        pass
