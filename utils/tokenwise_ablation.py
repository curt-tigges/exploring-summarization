import os
import einops
from functools import partial
import torch
import numpy as np
import datasets
from torch import Tensor
from datasets import (
    load_dataset,
    concatenate_datasets,
    Features,
    Sequence,
    Value,
    Dataset,
)
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
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
from utils.cache import resid_names_filter
from utils.circuit_analysis import get_logit_diff
from utils.ablation import (
    ablation_hook_base,
    freeze_attn_pattern_hook,
    convert_to_tensors,
)
from utils.datasets import ExperimentDataLoader
from utils.store import create_file_name, ResultsFile

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
    return torch.randn((model.cfg.n_layers, model.cfg.d_model)).to(device)


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
    data_loader: ExperimentDataLoader,
    token_id: int,
    device: torch.device = DEFAULT_DEVICE,
    cached: bool = True,
) -> Float[Tensor, "layer d_model"]:
    """Get the mean value of a particular token id across a dataset for each layer of a model

    Args:
        model: HookedTransformer model
        data_loader: ExperimentDataLoader for the dataset
        token_id: Token id to get the mean value of

    Returns:
        token_mean_values: Tensor of shape (num_layers, d_model) containing the mean value of token_id for each layer
    """
    file = ResultsFile(
        "mean_token_acts",
        model_name=model.cfg.model_name,
        dataset=data_loader.name,
        token_id=token_id,
        extension="pt",
    )
    if cached and file.exists():
        return torch.load(file.path)

    num_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    activation_sums: Float[Tensor, "layer d_model"] = torch.zeros(
        (num_layers, d_model), device=device, dtype=torch.float32
    )
    token_count: int = 0

    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value["tokens"].to(device)

        # Get binary mask of positions where token_id matches in the batch of tokens
        token_mask = batch_tokens == token_id
        token_count += token_mask.sum()

        _, cache = model.run_with_cache(batch_tokens, names_filter=resid_names_filter)

        for layer in range(num_layers):
            layer_activations = cache[f"blocks.{layer}.hook_resid_post"]
            activation_sums[layer] += einops.einsum(
                layer_activations,
                token_mask.float(),
                "batch seq d_model, batch seq -> d_model",
            )

    token_mean_values = (activation_sums / token_count).cpu()

    torch.save(token_mean_values, file.path)
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


def mask_loss_at_next_positions(
    loss: Float[Tensor, "batch seq_len"],
    positions: Float[Tensor, "batch seq_len"],
    attention_mask: Float[Tensor, "batch seq_len"],
) -> Float[Tensor, "batch seq_len"]:
    """
    We want to ignore loss changes immediately following the token of interest.
    """
    shifted_batch_pos = torch.roll(positions, shifts=1, dims=1)
    shifted_batch_pos[:, 0] = 0  # Set the first column to zero because roll is circular

    # Zero out loss_diff positions
    # Use the shifted_batch_pos tensor to mask loss_diff and set those positions to zero.
    loss[shifted_batch_pos == 1] = 0

    # set all masked positions to zero
    loss[attention_mask == 0] = 0
    return loss


# -------------------- EXPERIMENTS --------------------
def compute_ablation_modified_metric(
    model: HookedTransformer,
    data_loader: ExperimentDataLoader,
    layers_to_ablate: List[int] | Literal["all"] = "all",
    heads_to_freeze: List[Tuple[int, int]] | Literal["all"] = "all",
    cached_means: Optional[Float[Tensor, "layer d_model"]] = None,
    frozen_attn_variant: bool = False,
    direction_vectors: Optional[Float[Tensor, "layer d_model"]] = None,
    multiplier=1.0,
    all_positions: bool = False,
    metric: Literal["logits", "loss"] = "logits",
    device: torch.device = DEFAULT_DEVICE,
    cached: bool = True,
) -> Float[Tensor, "experiment batch *pos"]:
    """
    Computes the change in metric (between two answers) when the activations of
    a particular token are mean-ablated.

    If cached_means is specified, then we ablate the full residual stream at every layer.
    if direction_vectors is specified, then we only mean-ablate those directions.

    Args:
        model: HookedTransformer model
        data_loader: ExperimentDataLoader for the dataset
        layers_to_ablate: List of layers to ablate
        heads_to_freeze: List of heads to freeze
        cached_means:
            List of tensors of shape (layer, d_model)
            containing the mean value of a given token for each layer
        frozen_attn_variant:
            If True, repeat the experiment with attention frozen
        direction_vectors:
            List of tensors of shape (layer, d_model)
            containing the direction vector for each layer
        multiplier:
            Multiplier for the direction vector
        all_positions:
            If True, ablate all positions in the sequence.
            Otherwise, only ablate the positions specified in the data loader.
        metric:
            Metric to compute. Either "logits" or "loss".
            N.B. we only mask subsequent positions for loss.
            Also note that we compute the logit diff for only the last token in each sequence.
        device:
            Device to run the experiment on

    Returns:
        output: Tensor of shape (num_experiments, batch_size, sequence_length)
    """
    assert cached_means is not None or direction_vectors is not None
    assert metric in ["logits", "loss"]
    file = ResultsFile(
        "ablation_modified_metric",
        model_name=model.cfg.model_name,
        dataset=data_loader.name,
        layers_to_ablate=layers_to_ablate,
        heads_to_freeze=heads_to_freeze,
        metric=metric,
        direction_vectors=direction_vectors,
        multiplier=multiplier,
        all_positions=all_positions,
        frozen_attn_variant=frozen_attn_variant,
        extension="pt",
    )
    if cached and file.exists():
        return torch.load(file.path)
    if layers_to_ablate == "all":
        layers_to_ablate = list(range(model.cfg.n_layers))
    if heads_to_freeze == "all":
        heads_to_freeze = [
            (layer, head)
            for layer in range(model.cfg.n_layers)
            for head in range(model.cfg.n_heads)
        ]
    experiment_names = [
        "orig",
        "ablated",
    ]
    if frozen_attn_variant:
        experiment_names.append("freeze_ablated")
    experiment_index = {name: i for i, name in enumerate(experiment_names)}
    if metric == "logits":
        out_shape = (len(experiment_names), data_loader.dataset.num_rows)
    else:
        assert metric == "loss"
        out_shape = (
            len(experiment_names),
            data_loader.dataset.num_rows,
            model.cfg.n_ctx,
        )
    output = torch.zeros(
        out_shape,
        device="cpu",
        dtype=torch.float32,
    )
    batch_size = data_loader.batch_size
    assert batch_size is not None
    for batch_idx, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value["tokens"].to(device)
        if all_positions:
            batch_pos = batch_value["attention_mask"].to(device)
        else:
            batch_pos = batch_value["positions"].to(device)

        # Step 1: original metric without hooks

        if metric == "logits":
            # get the logit diff for the last token in each sequence
            orig_logits, clean_cache = model.run_with_cache(
                batch_tokens, return_type="logits", prepend_bos=False
            )
            assert isinstance(orig_logits, Tensor)
            orig_metric = get_logit_diff(
                orig_logits,
                mask=batch_value["attention_mask"],
                answer_tokens=batch_value["answers"],
                per_prompt=True,
            )
        else:
            assert metric == "loss"
            # get the loss for each token in the batch
            orig_loss, clean_cache = model.run_with_cache(
                batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True
            )
            assert isinstance(orig_loss, Tensor)
            # concatenate column of 0s
            orig_metric = torch.cat(
                [torch.zeros((orig_loss.shape[0], 1)).to(device), orig_loss], dim=1
            )
        output[
            experiment_index["orig"],
            batch_idx * batch_size : (batch_idx + 1) * batch_size,
        ] = orig_metric.cpu()

        # Step 2: repeat with tokens ablated

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

        if metric == "logits":
            ablated_logits = model(
                batch_tokens, return_type="logits", prepend_bos=False
            )
            # check to see if ablated_logits has any nan values
            if torch.isnan(ablated_logits).any():
                print("ablated logits has nan values")
            ablated_metric = get_logit_diff(
                ablated_logits,
                mask=batch_value["attention_mask"],
                answer_tokens=batch_value["answers"],
                per_prompt=True,
            )
        else:
            assert metric == "loss"
            # get the loss for each token when run with hooks
            hooked_loss = model(
                batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True
            )
            # concatenate column of 0s
            hooked_loss = torch.cat(
                [torch.zeros((hooked_loss.shape[0], 1)).to(device), hooked_loss], dim=1
            )
            loss_diff = hooked_loss - orig_metric
            ablated_metric = mask_loss_at_next_positions(
                loss_diff,
                positions=batch_pos,
                attention_mask=batch_value["attention_mask"],
            )

        output[
            experiment_index["ablated"],
            batch_idx * batch_size : (batch_idx + 1) * batch_size,
        ] = ablated_metric.cpu()
        model.reset_hooks()

        # Step 3: repeat with ablation and frozen attention
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
                    pos_mask=batch_pos,
                    layer=layer,
                )
                model.blocks[layer].hook_resid_post.add_hook(hook)

            if metric == "logits":
                freeze_ablated_logits = model(
                    batch_tokens, return_type="logits", prepend_bos=False
                )
                freeze_ablated_metric = get_logit_diff(
                    freeze_ablated_logits,
                    mask=batch_value["attention_mask"],
                    answer_tokens=batch_value["answers"],
                    per_prompt=True,
                )
            else:
                assert metric == "loss"
                # get the loss for each token when run with hooks
                hooked_loss = model(
                    batch_tokens,
                    return_type="loss",
                    prepend_bos=False,
                    loss_per_token=True,
                )
                # concatenate column of 0s
                hooked_loss = torch.cat(
                    [torch.zeros((hooked_loss.shape[0], 1)).to(device), hooked_loss],
                    dim=1,
                )
                loss_diff = hooked_loss - orig_metric
                freeze_ablated_metric = mask_loss_at_next_positions(
                    loss_diff,
                    positions=batch_pos,
                    attention_mask=batch_value["attention_mask"],
                )

            output[
                experiment_index["freeze_ablated"],
                batch_idx * batch_size : (batch_idx + 1) * batch_size,
            ] = freeze_ablated_metric.cpu()
            model.reset_hooks()

    torch.save(output, file.path)
    return output


def compute_zeroed_attn_modified_loss(
    model: HookedTransformer,
    data_loader: ExperimentDataLoader,
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
