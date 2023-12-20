from typing import Callable, Dict, List, Literal, Optional, Union
import torch
from torch import Tensor
from jaxtyping import Float, Int
import numpy as np
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache
from transformer_lens.utils import get_act_name
from datasets import Dataset, Features, Sequence, Value
import einops
from summarization_utils.cache import resid_names_filter


def handle_position_argument(
    pos: Union[Literal["each"], int, List[int], Int[Tensor, "batch pos ..."]],
    component: Int[Tensor, "batch pos ..."],
) -> Int[Tensor, "subset_pos"]:
    """Handles the position argument for ablation functions"""
    if isinstance(pos, int):
        pos = torch.tensor([pos])
    elif isinstance(pos, list):
        pos = torch.tensor(pos)
    elif pos == "each":
        pos = torch.tensor(list(range(component.shape[1])))
    return pos


def resample_ablate_component(
    component: Float[Tensor, "batch ..."], seed: int = 77
) -> Float[Tensor, "batch ..."]:
    """Resample-ablates a batch tensor according to the index of the first dimension"""
    batch_size = component.shape[0]
    # set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Resample the batch
    indices = np.random.choice(batch_size, batch_size, replace=True)
    component = component[indices]
    return component


def mean_ablate_component(
    component: Float[Tensor, "batch ..."]
) -> Float[Tensor, "batch ..."]:
    """
    Mean-ablates a batch tensor

    :param component: the tensor to compute the mean over the batch dim of
    :return: the mean over the cache component of the tensor
    """
    copy_to_ablate = component.clone()
    batch_mean = component.mean(dim=0)
    # make every batch item a copy of the batch mean
    for row in range(component.shape[0]):
        copy_to_ablate[row] = batch_mean
    assert copy_to_ablate.shape == component.shape
    return copy_to_ablate


def zero_ablate_component(component: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Zero-ablates a batch tensor"""
    return torch.zeros_like(component)


def zero_ablate_attention_pos_hook(
    pattern: Float[Tensor, "batch head seq_Q seq_K"],
    hook: HookPoint,
    pos_by_batch: List[List[int]],
    head_idx: int = 0,
) -> Float[Tensor, "batch head seq_Q seq_K"]:
    """Zero-ablates an attention pattern tensor at a particular position"""
    assert hook.name is not None and "pattern" in hook.name

    batch_size = pattern.shape[0]
    assert len(pos_by_batch) == batch_size

    for i in range(batch_size):
        for p in pos_by_batch[i]:
            pattern[i, head_idx, p, p] = 0

    return pattern


def freeze_attn_pattern_hook(
    pattern: Float[Tensor, "batch head seq_Q seq_K"],
    hook: HookPoint,
    cache: ActivationCache,
    layer: int = 0,
    head_idx: int = 0,
) -> Float[Tensor, "batch head seq_Q seq_K"]:
    """Freeze the attention pattern for a given position, layer and head"""
    assert hook.name is not None and "pattern" in hook.name
    pattern[:, head_idx, :, :] = cache[f"blocks.{layer}.attn.hook_pattern"][
        :, head_idx, :, :
    ]
    pattern[:, head_idx, :, :] = cache[f"blocks.{layer}.attn.hook_pattern"][
        :, head_idx, :, :
    ]
    return pattern


def freeze_layer_pos_hook(
    component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    cache: ActivationCache,
    component_type: str = "hook_resid_post",
    pos: Union[Literal["each"], int, List[int]] = -1,
    layer: int = 0,
) -> Float[Tensor, "batch pos ..."]:
    """Base function to freeze the layer for a given position, layer and head"""
    assert hook.name is not None and component_type in hook.name
    pos_t = handle_position_argument(pos, component)
    for p in pos_t:
        component[:, p, :] = cache[f"blocks.{layer}.{component_type}"][:, p, :]
    return component


def freeze_mlp_pos_hook(
    component: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    cache: ActivationCache,
    component_type: str = "hook_post",
    pos: Union[Literal["each"], int, List[int]] = -1,
    layer: int = 0,
):
    """Freeze the mlp for a given position, layer and head"""
    return freeze_layer_pos_hook(
        component, hook, cache, f"mlp.{component_type}", pos, layer
    )


def freeze_attn_head_pos_hook(
    component: Float[Tensor, "batch pos head d_head"],
    hook: HookPoint,
    cache: ActivationCache,
    component_type: str = "hook_z",
    pos: Union[Literal["each"], int, List[int]] = -1,
    layer: int = 0,
    head_idx: int = 0,
):
    """Freeze the attention head for a given position, layer and head"""
    assert hook.name is not None and component_type in hook.name
    pos_t = handle_position_argument(pos, component)
    for p in pos_t:
        component[:, p, head_idx, :] = cache[f"blocks.{layer}.attn.{component_type}"][
            :, p, head_idx, :
        ]
    return component


def ablate_layer_pos_hook(
    component: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    cache: ActivationCache,
    ablation_func: Optional[
        Callable[[Float[Tensor, "batch ..."]], Float[Tensor, "batch ..."]]
    ] = None,
    component_type: str = "hook_resid_post",
    pos: Union[Literal["each"], int, List[int]] = -1,
    layer: int = 0,
) -> Float[Tensor, "batch pos d_mlp"]:
    """Base function to ablate the layer for a given position, layer and head"""
    assert hook.name is not None and component_type in hook.name
    pos_t = handle_position_argument(pos, component)
    if ablation_func is None:
        ablation_func = lambda x: x
    for p in pos_t:
        component[:, p, :] = ablation_func(
            cache[f"blocks.{layer}.{component_type}"][:, p, :]
        )
    return component


def ablate_mlp_pos_hook(
    component: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    cache: ActivationCache,
    ablation_func: Optional[
        Callable[[Float[Tensor, "batch ..."]], Float[Tensor, "batch ..."]]
    ] = None,
    component_type: str = "hook_post",
    pos: Union[Literal["each"], int, List[int]] = -1,
    layer: int = 0,
) -> Float[Tensor, "batch pos d_mlp"]:
    return ablate_layer_pos_hook(
        component, hook, cache, ablation_func, f"mlp.{component_type}", pos, layer
    )


def ablate_attn_head_pos_hook(
    component: Float[Tensor, "batch pos d_head"],
    hook: HookPoint,
    cache: ActivationCache,
    ablation_func: Optional[
        Callable[[Float[Tensor, "batch ..."]], Float[Tensor, "batch ..."]]
    ] = None,
    component_type: str = "hook_z",
    pos: Union[Literal["each"], int, List[int]] = -1,
    layer: int = 0,
    head_idx: int = 0,
) -> Float[Tensor, "batch pos d_head"]:
    assert hook.name is not None and component_type in hook.name
    pos_t = handle_position_argument(pos, component)
    if ablation_func is None:
        ablation_func = lambda x: x
    for p in pos_t:
        component[:, p, head_idx, :] = ablation_func(
            cache[f"blocks.{layer}.attn.{component_type}"][:, p, head_idx, :]
        )
    return component


def _ablate_resid_with_precalc_mean(
    component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    cached_means: Float[Tensor, "layer d_model"],
    pos_mask: Int[Tensor, "batch pos"],
    layer: int = 0,
) -> Float[Tensor, "batch ..."]:
    """
    Mean-ablates a batch tensor

    Args:
        component: the tensor to compute the mean over the batch dim of
        hook: the hook point

    Returns:
        the mean over the cache component of the tensor
    """
    assert hook.name is not None and "resid" in hook.name

    # Identify the positions where pos_by_batch is 1
    batch_indices, sequence_positions = torch.where(pos_mask == 1)

    # Replace the corresponding positions in component with cached_means[layer]
    component[batch_indices, sequence_positions] = cached_means[layer].to(
        device=component.device
    )

    return component


def _ablate_resid_with_direction(
    component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    direction_vector: Float[Tensor, "layer d_model"],
    pos_mask: Int[Tensor, "batch pos"],
    multiplier: float = 1.0,
    layer: int = 0,
) -> Float[Tensor, "batch ..."]:
    """
    Ablates a batch tensor by removing the influence of a direction vector from it.

    Args:
        component: the tensor to compute the mean over the batch dim of
        direction_vector: the direction vector to remove from the component
        multiplier: the multiplier to apply to the direction vector
        pos_by_batch: the positions to ablate
        layer: the layer to ablate

    Returns:
        the ablated component
    """
    assert hook.name is not None and "resid" in hook.name

    # Normalize the direction vector to make sure it's a unit vector
    direction_normalized = (
        direction_vector[layer] / torch.norm(direction_vector[layer])
    ).to(device=component.device)

    # Calculate the projection of component onto direction_vector
    proj = (
        einops.einsum(component, direction_normalized, "b s d, d -> b s").unsqueeze(-1)
        * direction_normalized
    )

    # Ablate the direction from component
    component_ablated = (
        component.clone()
    )  # Create a copy to ensure original is not modified
    batch_indices, sequence_positions = torch.where(pos_mask == 1)
    component_ablated[batch_indices, sequence_positions] = (
        component[batch_indices, sequence_positions]
        - multiplier * proj[batch_indices, sequence_positions]
    )

    # Check that positions not in (batch_indices, sequence_positions) were not ablated
    check_mask = torch.ones_like(component, dtype=torch.bool)
    check_mask[batch_indices, sequence_positions] = 0
    if not torch.all(component[check_mask] == component_ablated[check_mask]):
        raise ValueError(
            "Positions outside of specified (batch_indices, sequence_positions) were ablated!"
        )

    return component_ablated


def ablation_hook_base(
    component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    pos_mask: Int[Tensor, "batch pos"],
    cached_means: Optional[Float[Tensor, "layer d_model"]] = None,
    direction_vectors: Optional[Float[Tensor, "layer d_model"]] = None,
    multiplier: float = 1.0,
    layer: int = 0,
) -> Float[Tensor, "batch ..."]:
    """
    Ablates a batch tensor by removing the influence of a direction vector from it.

    Args:
        component: the tensor to compute the mean over the batch dim of
        hook: the hook point
        pos_by_batch: the positions to ablate
        cached_means: the cached means to use for ablation
        direction_vector: the direction vector to remove from the component
        multiplier: the multiplier to apply to the direction vector
        layer: the layer to ablate

    Returns:
        the ablated component
    """
    assert hook.name is not None and "resid" in hook.name

    if cached_means is not None:
        return _ablate_resid_with_precalc_mean(
            component, hook, cached_means, pos_mask, layer
        )
    elif direction_vectors is not None:
        return _ablate_resid_with_direction(
            component, hook, direction_vectors, pos_mask, multiplier, layer
        )
    else:
        raise ValueError("Must specify either cached_means or direction_vector")


def convert_to_tensors(
    dataset: Union[Dataset, List[Dict[str, List[int]]]], column_name="tokens"
):
    final_batches = []

    for batch in dataset:
        assert isinstance(batch, dict)
        trimmed_batch = batch[column_name]
        final_batches.append(trimmed_batch)

    # Convert list of batches to tensors
    final_batches = [torch.tensor(batch, dtype=torch.long) for batch in final_batches]
    # Create a new dataset with specified features
    features = Features({"tokens": Sequence(Value("int64"))})
    final_dataset = Dataset.from_dict({"tokens": final_batches}, features=features)

    final_dataset.set_format(type="torch", columns=["tokens"])

    return final_dataset
