"""Compact Gemma4 model for prime-rl training."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.gemma4.converting_gemma4 import run_on_flat_language_model_keys
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None  # ty:ignore[invalid-assignment]

# Compile flex_attention for performance — this is the intended usage per PyTorch docs
_compiled_flex_attention = torch.compile(flex_attention)


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


def _compute_cu_seqlens(
    position_ids: torch.Tensor,
    use_flash_attn: bool,
) -> tuple[torch.Tensor | None, int | None]:
    batch_size, seq_len = position_ids.shape
    flat_position_ids = position_ids.reshape(-1)
    has_resets = (flat_position_ids[1:] == 0).any().item()

    if not use_flash_attn and batch_size == 1 and not has_resets:
        return None, None

    if not has_resets:
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            device=position_ids.device,
            dtype=torch.int32,
        )
        return cu_seqlens, seq_len

    seqlens = torch.cat(
        [
            flat_position_ids[:1],
            flat_position_ids[:-1][(flat_position_ids == 0)[1:]] + 1,
            flat_position_ids[-1:] + 1,
        ]
    )
    return seqlens.cumsum(dim=0, dtype=torch.int32), int(seqlens.max().item())


def _build_flex_block_masks(
    cu_seqlens: torch.Tensor,
    total_len: int,
    sliding_window: int | None,
) -> dict[str, BlockMask]:
    """Build BlockMasks for flex_attention: one per layer type (sliding vs full)."""

    def _causal_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        causal = q_idx >= kv_idx
        # Same-sequence check: both q and kv must be in the same packed sequence
        seq_id_q = (q_idx.unsqueeze(-1) >= cu_seqlens).sum(-1) - 1
        seq_id_kv = (kv_idx.unsqueeze(-1) >= cu_seqlens).sum(-1) - 1
        same_seq = seq_id_q == seq_id_kv
        return causal & same_seq

    full_mask = create_block_mask(_causal_mask, B=1, H=None, Q_LEN=total_len, KV_LEN=total_len, device=cu_seqlens.device)
    masks: dict[str, BlockMask] = {"full_attention": full_mask}

    if sliding_window is not None:

        def _sliding_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
            causal = q_idx >= kv_idx
            within_window = (q_idx - kv_idx) < sliding_window
            seq_id_q = (q_idx.unsqueeze(-1) >= cu_seqlens).sum(-1) - 1
            seq_id_kv = (kv_idx.unsqueeze(-1) >= cu_seqlens).sum(-1) - 1
            same_seq = seq_id_q == seq_id_kv
            return causal & within_window & same_seq

        masks["sliding_attention"] = create_block_mask(
            _sliding_mask, B=1, H=None, Q_LEN=total_len, KV_LEN=total_len, device=cu_seqlens.device
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

        head_dim = config.global_head_dim if layer_type == "full_attention" else config.head_dim
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
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.head_dim = config.global_head_dim if self.layer_type == "full_attention" else config.head_dim
        self.num_key_value_heads = config.num_key_value_heads

        self.kv_shared_layer_index, self.store_full_length_kv = _get_kv_sharing_info(config, layer_idx)
        self._attn_impl = getattr(config, "_attn_implementation", "sdpa")
        self._flash_attn_func = flash_attn_varlen_func if self._attn_impl == "flash_attention_2" else None

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
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False)

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        flash_attn = self._flash_attn_func
        assert flash_attn is not None
        kwargs: dict[str, object] = {"causal": True, "softmax_scale": 1.0}
        if self.sliding_window is not None:
            kwargs["window_size"] = (self.sliding_window - 1, 0)
        return flash_attn(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        if k.shape[1] != q.shape[1]:
            repeat_factor = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        if self.sliding_window is None and (cu_seqlens is None or len(cu_seqlens) <= 2):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
            return output.transpose(1, 2).squeeze(0)

        total_len = q.shape[2]
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, total_len], device=q.device, dtype=torch.int32)
        mask = torch.full((total_len, total_len), float("-inf"), device=q.device, dtype=q.dtype)
        for i in range(len(cu_seqlens) - 1):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            seq_len = end - start
            local_mask = torch.zeros(seq_len, seq_len, device=q.device, dtype=q.dtype)
            local_mask = local_mask.masked_fill(
                torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1),
                float("-inf"),
            )
            if self.sliding_window is not None:
                local_mask = local_mask.masked_fill(
                    torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), -self.sliding_window),
                    float("-inf"),
                )
            mask[start:end, start:end] = local_mask

        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        return output.transpose(1, 2).squeeze(0)

    def _flex_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        # flex_attention expects (B, H, S, D)
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        if k.shape[1] != q.shape[1]:
            repeat_factor = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        output = _compiled_flex_attention(q, k, v, block_mask=block_mask, scale=1.0)
        return output.transpose(1, 2).squeeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
        block_mask: BlockMask | None = None,
        shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        cos, sin = position_embeddings

        q = _apply_rotary_pos_emb(self.q_norm(self.q_proj(hidden_states).view(hidden_shape)), cos, sin)
        if self.kv_shared_layer_index is not None and shared_kv is not None:
            k, v = shared_kv[self.kv_shared_layer_index]
        else:
            k = _apply_rotary_pos_emb(self.k_norm(self.k_proj(hidden_states).view(hidden_shape)), cos, sin)
            v = self.v_norm(self.v_proj(hidden_states).view(hidden_shape))
            if self.store_full_length_kv and shared_kv is not None:
                shared_kv[self.layer_idx] = (k, v)

        q = q.contiguous().flatten(0, 1)
        k = k.contiguous().flatten(0, 1)
        v = v.contiguous().flatten(0, 1)

        if block_mask is not None and self.head_dim <= 256:
            attn_output = self._flex_attention(q, k, v, block_mask)
        elif self._flash_attn_func is not None and cu_seqlens is not None and max_seqlen is not None and self.head_dim <= 256:
            attn_output = self._flash_attention(q, k, v, cu_seqlens, max_seqlen)
        else:
            attn_output = self._sdpa_attention(q, k, v, cu_seqlens)

        attn_output = attn_output.contiguous().view(*hidden_states.shape[:-1], -1)
        return self.o_proj(attn_output)


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

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        if config.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(config.hidden_size, config.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(config.hidden_size_per_layer_input, config.hidden_size, bias=False)
            self.post_per_layer_input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
        per_layer_input: torch.Tensor | None = None,
        shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings, cu_seqlens, max_seqlen, block_mask, shared_kv
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

        return hidden_states * self.layer_scalar


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config
        pad_token_id = cast(int, config.pad_token_id)
        self.embed_tokens = Gemma4ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            pad_token_id,
            config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4DualRotaryEmbedding(config)

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
            self.per_layer_projection_norm = nn.RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)
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

        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        use_flash_attn = attn_impl == "flash_attention_2"
        use_flex_attn = attn_impl == "flex_attention"

        # Always compute cu_seqlens: needed for flash, SDPA fallback (large head_dim), and flex mask construction
        cu_seqlens, max_seqlen = _compute_cu_seqlens(position_ids, use_flash_attn or use_flex_attn)
        if use_flash_attn and cu_seqlens is not None:
            torch._dynamo.mark_dynamic(cu_seqlens, 0)

        flex_block_masks: dict[str, BlockMask] | None = None
        if use_flex_attn and cu_seqlens is not None:
            total_len = position_ids.shape[0] * position_ids.shape[1]
            flex_block_masks = _build_flex_block_masks(cu_seqlens, total_len, self.config.sliding_window)

        layer_types = cast(list[str], self.config.layer_types)
        position_embeddings = {
            layer_type: self.rotary_emb(inputs_embeds, position_ids, layer_type)
            for layer_type in set(layer_types)
        }

        hidden_states = inputs_embeds
        shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            layer_type = layer_types[i]
            block_mask = flex_block_masks[layer_type] if flex_block_masks is not None else None
            hidden_states = layer(
                hidden_states,
                position_embeddings[layer_type],
                cu_seqlens,
                max_seqlen,
                None if per_layer_inputs is None else per_layer_inputs[:, :, i, :],
                shared_kv,
                block_mask,
            )

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
