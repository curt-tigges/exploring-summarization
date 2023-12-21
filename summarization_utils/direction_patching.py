from typing import Optional
from jaxtyping import Float
import torch
from torch import Tensor
import einops
from transformer_lens import HookedTransformer, ActivationCache

from typeguard import typechecked


@typechecked
def project_to_subspace(
    vectors: Float[Tensor, "... d_model"],
    subspace: Float[Tensor, "d_model d_subspace"],
) -> Float[Tensor, "... d_model"]:
    assert vectors.shape[-1] == subspace.shape[0]
    basis_projections = einops.einsum(
        vectors, subspace, "... d_model, d_model d_subspace -> ... d_subspace"
    )
    summed_projections = einops.einsum(
        basis_projections, subspace, "... d_subspace, d_model d_subspace -> ... d_model"
    )
    return summed_projections


def create_cache_for_dir_patching(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    sentiment_dir: Float[Tensor, "d_model *d_das"],
    model: HookedTransformer,
    device: Optional[torch.device] = None,
) -> ActivationCache:
    """
    We patch the sentiment direction from corrupt to clean
    """
    if device is None:
        device = sentiment_dir.device
    if sentiment_dir.ndim == 1:
        sentiment_dir = sentiment_dir.unsqueeze(1)
    assert sentiment_dir.ndim == 2
    sentiment_dir = sentiment_dir / sentiment_dir.norm(dim=0, keepdim=True)
    cache_dict = dict()
    for act_name, clean_value in clean_cache.items():
        is_result = act_name.endswith("result")
        is_resid = (
            act_name.endswith("resid_pre")
            or act_name.endswith("resid_post")
            or act_name.endswith("attn_out")
            or act_name.endswith("mlp_out")
        )
        if is_resid or is_result:
            clean_value = clean_value.to(device)
            corrupt_value = corrupted_cache[act_name].to(device)

            corrupt_proj: Float[Tensor, "... d_model"] = project_to_subspace(
                corrupt_value,
                sentiment_dir,
            )
            clean_proj: Float[Tensor, "... d_model"] = project_to_subspace(
                clean_value,
                sentiment_dir,
            )
            cache_dict[act_name] = corrupt_value + (clean_proj - corrupt_proj)
        else:
            # Only patch the residual stream
            cache_dict[act_name] = corrupted_cache[act_name].to(device)

    return ActivationCache(cache_dict, model)
