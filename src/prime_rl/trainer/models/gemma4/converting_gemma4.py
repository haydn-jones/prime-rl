from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

_MULTIMODAL_MODEL_PREFIXES = (
    "model.language_model.",
    "model.vision_tower.",
    "model.embed_vision.",
    "model.audio_tower.",
    "model.embed_audio.",
)


def has_nested_language_model_keys(state_dict: dict[str, Tensor]) -> bool:
    return any(key.startswith("model.language_model.") for key in state_dict)


def remap_language_model_keys(state_dict: dict[str, Tensor], *, to_flat: bool) -> None:
    """Move text weights between flat and multimodal Gemma4 checkpoint layouts."""
    if to_flat:
        keys = [key for key in list(state_dict) if key.startswith("model.language_model.")]
        for key in keys:
            state_dict["model." + key.removeprefix("model.language_model.")] = state_dict.pop(key)
        return

    keys = [
        key
        for key in list(state_dict)
        if key.startswith("model.") and not any(key.startswith(prefix) for prefix in _MULTIMODAL_MODEL_PREFIXES)
    ]
    for key in keys:
        state_dict["model.language_model." + key.removeprefix("model.")] = state_dict.pop(key)


def run_on_flat_language_model_keys(
    state_dict: dict[str, Tensor],
    transform: Callable[..., object] | None = None,
    *args: object,
) -> dict[str, Tensor]:
    has_nested = has_nested_language_model_keys(state_dict)
    if has_nested:
        remap_language_model_keys(state_dict, to_flat=True)
    if transform is not None:
        transform(state_dict, *args)
    if has_nested:
        remap_language_model_keys(state_dict, to_flat=False)
    return state_dict
