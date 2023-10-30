import einops
from functools import partial
import torch
import numpy as np
import datasets
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import (
    load_dataset,
    concatenate_datasets,
    Features,
    Sequence,
    Value,
    Dataset,
)
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Optional, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import (
    get_dataset,
    tokenize_and_concatenate,
    get_act_name,
    test_prompt,
)
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd
from circuitsvis.activations import text_neuron_activations
from utils.store import (
    load_array,
    save_html,
    save_array,
    is_file,
    get_model_name,
    clean_label,
    save_text,
)
from utils.cache import resid_names_filter
from utils.circuit_analysis import get_logit_diff
from utils.ablation import (
    ablation_hook_base,
    freeze_attn_pattern_hook,
    convert_to_tensors,
)


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- GENERAL UTILS --------------------
def find_positions(
    batch_tensor: Float[Tensor, "batch seq_len"], token_ids: List = [13]
) -> List[List[int]]:
    """Finds positions of specified token ids in a tensor of shape (batch, sequence_position)

    Args:
        batch_tensor: Tensor of shape (batch, sequence_position)
        token_ids: List of token ids to find positions of

    Returns:
        positions: [batch, pos] nested list
    """
    positions = []
    for batch_item in batch_tensor:
        batch_positions = []
        for position, token in enumerate(batch_item):
            if token.item() in token_ids:
                batch_positions.append(position)
        positions.append(batch_positions)
    return positions


# -------------------- DIRECTION LOADING UTILS --------------------


# TODO: standardize the way we load directions
def load_directions(
    model: HookedTransformer,
    direction_folder: str = "directions",
    direction_prefix: str = "das_simple_train_ADJ",
    device: torch.device = DEFAULT_DEVICE,
) -> Float[Tensor, "layer d_model"]:
    """Loads a list of direction vectors of shape (n_layers, d_model)"""
    directions = []
    for i in range(model.cfg.n_layers):
        dir = np.load(f"{direction_folder}/{direction_prefix}{i}.npy")
        if len(dir.shape) == 2:
            dir = dir[:, 0]
        directions.append(torch.tensor(dir))

    # convert to tensor
    directions = torch.stack(directions).to(device)

    return directions


def get_random_directions(
    model: HookedTransformer,
    device: torch.device = DEFAULT_DEVICE,
) -> Float[Tensor, "layer d_model"]:
    """Returns a list of random direction vectors of shape (n_layers, d_model)"""
    directions = []
    num_layers = model.cfg.n_layers
    for _ in range(num_layers):
        dir = torch.randn(model.cfg.d_model).to(device)
        directions.append(dir)

    # convert to tensor
    directions = torch.stack(directions).to(device)

    return directions


def get_zeroed_dir_vector(
    model: HookedTransformer,
    device: torch.device = DEFAULT_DEVICE,
) -> Float[Tensor, "layer d_model"]:
    """Returns a list of zeroed direction vectors of shape (n_layers, d_model)"""
    return torch.zeros((model.cfg.n_layers, model.cfg.d_model)).to(device)


# TODO: Add metrics for loss


# -------------------- ACTIVATION UTILS --------------------
def get_layerwise_token_mean_activations(
    model: HookedTransformer,
    data_loader: DataLoader,
    token_id: int,
    device: torch.device = DEFAULT_DEVICE,
) -> Float[Tensor, "layer d_model"]:
    """Get the mean value of a particular token id across a dataset for each layer of a model

    Args:
        model: HookedTransformer model
        data_loader: DataLoader for the dataset
        token_id: Token id to get the mean value of

    Returns:
        token_mean_values: Tensor of shape (num_layers, d_model) containing the mean value of token_id for each layer
    """
    num_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    activation_sums: Float[Tensor, "layer d_model"] = torch.stack(
        [torch.zeros(d_model) for _ in range(num_layers)]
    ).to(device=device, dtype=torch.float32)
    token_counts = [0] * num_layers

    token_mean_values = torch.zeros((num_layers, d_model))
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value["tokens"].to(device)

        # get positions of all specified token ids in batch
        token_pos = find_positions(batch_tokens, token_ids=[token_id])

        _, cache = model.run_with_cache(batch_tokens, names_filter=resid_names_filter)

        for i in range(batch_tokens.shape[0]):
            for p in token_pos[i]:
                for layer in range(num_layers):
                    activation_sums[layer] += cache[f"blocks.{layer}.hook_resid_post"][
                        i, p, :
                    ]
                    token_counts[layer] += 1

    for layer in range(num_layers):
        token_mean_values[layer] = activation_sums[layer] / token_counts[layer]

    return token_mean_values


# -------------------- ABLATION UTILS --------------------
def zero_attention_pos_hook(
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


# -------------------- EXPERIMENTS --------------------
def compute_ablation_modified_logit_diff(
    model: HookedTransformer,
    data_loader: DataLoader,
    layers_to_ablate: List[int],
    heads_to_freeze: List[Tuple[int, int]],
    cached_means: Optional[Float[Tensor, "layer d_model"]] = None,
    frozen_attn_variant: bool = False,
    direction_vectors: Optional[Float[Tensor, "layer d_model"]] = None,
    multiplier=1.0,
    all_positions: bool = False,
    device: torch.device = DEFAULT_DEVICE,
) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    """
    Computes the change in logit difference (between two answers) when the activations of
    a particular token are mean-ablated.

    Args:
        model: HookedTransformer model
        data_loader: DataLoader for the dataset
        layers_to_ablate: List of layers to ablate
        heads_to_freeze: List of heads to freeze
        cached_means:
            List of tensors of shape (layer, d_model)
            containing the mean value of a given token for each layer

    Returns:
        orig_ld_list:
            List of tensors of shape (batch,)
            containing the logit difference for each item in the batch before ablation
        ablated_ld_list:
            List of tensors of shape (batch,)
            containing the logit difference for each item in the batch after ablation
        freeze_ablated_ld_list:
            List of tensors of shape (batch,)
            containing the logit difference for each item in the batch after ablation with attention frozen
    """
    orig_ld_list = []
    ablated_ld_list = []
    freeze_ablated_ld_list = []

    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value["tokens"].to(device)
        if all_positions:
            batch_pos = batch_value["attention_mask"].to(device)
        else:
            batch_pos = batch_value["positions"].to(device)

        # get the logit diff for the last token in each sequence
        orig_logits, clean_cache = model.run_with_cache(
            batch_tokens, return_type="logits", prepend_bos=False
        )
        assert isinstance(orig_logits, Tensor)
        orig_ld = get_logit_diff(
            orig_logits,
            mask=batch_value["attention_mask"],
            answer_tokens=batch_value["answers"],
        )
        orig_ld_list.append(orig_ld)

        # repeat with tokens ablated
        for layer in layers_to_ablate:
            hook = partial(
                ablation_hook_base,
                cached_means=cached_means,
                direction_vectors=direction_vectors,
                multiplier=multiplier,
                pos_mask=batch_pos,
                layer=layer,
            )
            model.blocks[layer].hook_resid_post.add_hook(hook)

        ablated_logits = model(batch_tokens, return_type="logits", prepend_bos=False)
        # check to see if ablated_logits has any nan values
        if torch.isnan(ablated_logits).any():
            print("ablated logits has nan values")
        ablated_ld = get_logit_diff(
            ablated_logits,
            mask=batch_value["attention_mask"],
            answer_tokens=batch_value["answers"],
        )
        ablated_ld_list.append(ablated_ld)

        model.reset_hooks()

        if frozen_attn_variant:
            # repeat with attention frozen and tokens ablated
            for layer, head in heads_to_freeze:
                freeze_attn = partial(
                    freeze_attn_pattern_hook,
                    cache=clean_cache,
                    layer=layer,
                    head_idx=head,
                )
                model.blocks[layer].attn.hook_pattern.add_hook(freeze_attn)

            for layer in layers_to_ablate:
                hook = partial(
                    ablation_hook_base,
                    cached_means=cached_means,
                    direction_vectors=direction_vectors,
                    multiplier=multiplier,
                    pos_by_batch=batch_pos,
                    layer=layer,
                )
                model.blocks[layer].hook_resid_post.add_hook(hook)

            freeze_ablated_logits = model(
                batch_tokens, return_type="logits", prepend_bos=False
            )
            freeze_ablated_ld = get_logit_diff(
                freeze_ablated_logits,
                mask=batch_value["attention_mask"],
                answer_tokens=batch_value["answers"],
            )
            freeze_ablated_ld_list.append(freeze_ablated_ld)

            model.reset_hooks()

    return (
        torch.cat(orig_ld_list),
        torch.cat(ablated_ld_list),
        torch.cat(freeze_ablated_ld_list),
    )


def compute_zeroed_attn_modified_loss(
    model: HookedTransformer,
    data_loader: DataLoader,
    heads_to_ablate: List[Tuple[int, int]],
    token_ids: List[int] = [13],
    device: torch.device = DEFAULT_DEVICE,
) -> Float[Tensor, "batch"]:
    loss_list = []
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value["tokens"].to(device)

        batch_pos = find_positions(batch_tokens, token_ids=token_ids)

        # get the loss for each token in the batch
        initial_loss = model(
            batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True
        )

        # add hooks for the activations of the 11 and 13 tokens
        for layer, head in heads_to_ablate:
            ablate_tokens = partial(
                zero_attention_pos_hook,
                pos_by_batch=batch_pos,
                head_idx=head,
            )
            model.blocks[layer].attn.hook_pattern.add_hook(ablate_tokens)

        # get the loss for each token when run with hooks
        hooked_loss = model(
            batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True
        )

        # compute the percent difference between the two losses
        loss_diff = (hooked_loss - initial_loss) / initial_loss

        loss_list.append(loss_diff)

    model.reset_hooks()
    return torch.cat(loss_list)


def compute_mean_ablation_modified_loss(
    model: HookedTransformer,
    data_loader: DataLoader,
    layers_to_ablate: List[int],
    cached_means: Float[Tensor, "layer d_model"],
    debug: bool = False,
    device: torch.device = DEFAULT_DEVICE,
) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    """Computes the change in loss when the activations of a particular token are mean-ablated.

    Args:
        model: HookedTransformer model
        data_loader: DataLoader for the dataset
        layers_to_ablate: List of layers to ablate
        cached_means: List of tensors of shape (layer, d_model) containing the mean value of a given token for each layer

    Returns:
        loss_diff_list: List of tensors of shape (batch,) containing the loss difference for each item in the batch
        orig_loss_list: List of tensors of shape (batch,) containing the original loss for each item in the batch
    """
    loss_diff_list = []
    orig_loss_list = []
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value["tokens"].to(device)
        batch_pos = batch_value["positions"].to(device)

        # get the loss for each token in the batch
        initial_loss = model(
            batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True
        )
        # concatenate column of 0s
        initial_loss = torch.cat(
            [torch.zeros((initial_loss.shape[0], 1)).to(device), initial_loss], dim=1
        )

        if debug:
            print(f"initial loss shape: {initial_loss.shape}")
            print(initial_loss[0])
        orig_loss_list.append(initial_loss)

        # add hooks for the activations of the relevant tokens
        for layer in layers_to_ablate:
            mean_ablate_token = partial(
                ablate_resid_with_precalc_mean,
                cached_means=cached_means,
                pos_by_batch=batch_pos,
                layer=layer,
            )
            model.blocks[layer].hook_resid_post.add_hook(mean_ablate_token)

        # get the loss for each token when run with hooks
        hooked_loss = model(
            batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True
        )
        # concatenate column of 0s
        hooked_loss = torch.cat(
            [torch.zeros((hooked_loss.shape[0], 1)).to(device), hooked_loss], dim=1
        )

        if debug:
            print(f"hooked loss shape: {hooked_loss.shape}")
            print(hooked_loss[0])

        # compute the difference between the two losses
        loss_diff = hooked_loss - initial_loss

        # set all positions right after batch_pos to zero
        if debug:
            print(f"batch pos: {batch_pos}")
        # use the batch_pos tensor to set all positions right after batch_pos==1 to zero
        # enter code here
        # Step 1: Shift batch_pos tensor
        # We use roll to shift the tensor. We pad the first column with zeros after the roll since roll is circular.
        shifted_batch_pos = torch.roll(batch_pos, shifts=1, dims=1)
        shifted_batch_pos[
            :, 0
        ] = 0  # Set the first column to zero because roll is circular

        # Step 2: Zero out loss_diff positions
        # Use the shifted_batch_pos tensor to mask loss_diff and set those positions to zero.
        loss_diff[shifted_batch_pos == 1] = 0

        # set all masked positions to zero
        loss_diff[batch_value["attention_mask"] == 0] = 0

        loss_diff_list.append(loss_diff)

        model.reset_hooks()

    return torch.cat(loss_diff_list), torch.cat(orig_loss_list)
