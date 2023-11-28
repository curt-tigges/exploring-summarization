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
    lm_cross_entropy_loss,
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
    overwrite: bool = False,
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
    if not overwrite and file.exists():
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


# -------------------- LOSS UTILS --------------------
def subset_cross_entropy_loss(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    tokens: Int[torch.Tensor, "batch pos"],
    vocab_mask: Optional[Bool[torch.Tensor, "d_vocab"]] = None,
    inf: float = float("inf"),
) -> Float[torch.Tensor, "batch pos"]:
    """Wrapper around `utils.lm_cross_entropy_loss`.

    Similar to forward() with return_type=="loss".

    N.B. vocab_mask specifies which tokens to include not exclude.
    """
    if tokens.device != logits.device:
        tokens = tokens.to(logits.device)
    if vocab_mask is not None:
        logits = logits.masked_fill(~vocab_mask, -inf)
    loss = lm_cross_entropy_loss(logits, tokens, per_token=True)
    loss = torch.where(
        torch.isclose(loss, torch.tensor(inf)),
        torch.zeros_like(loss),
        loss,
    )
    return loss


def loss_fn(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch pos"],
    vocab_mask: Optional[Bool[Tensor, "d_vocab"]] = None,
) -> Float[Tensor, "batch pos"]:
    """Computes the cross entropy loss for a batch of tokens"""
    logits = model(tokens, return_type="logits", prepend_bos=False)
    return subset_cross_entropy_loss(logits, tokens, vocab_mask)


# -------------------- EXPERIMENTS --------------------


def compute_ablation_modified_loss(
    model: HookedTransformer,
    data_loader: ExperimentDataLoader,
    layers_to_ablate: List[int] | Literal["all"] = "all",
    cached_means: Optional[Float[Tensor, "layer d_model"]] = None,
    direction_vectors: Optional[Float[Tensor, "layer d_model"]] = None,
    multiplier=1.0,
    all_positions: bool = False,
    vocab_mask: Optional[Bool[Tensor, "d_vocab"]] = None,
    device: torch.device = DEFAULT_DEVICE,
    overwrite: bool = False,
) -> Float[Tensor, "experiment batch pos"]:
    """
    Computes the change in loss when the activations of
    a particular token are mean-ablated.

    If cached_means is specified, then we ablate the full residual stream at every layer.
    if direction_vectors is specified, then we only mean-ablate those directions.

    cf. compute_ablation_modified_logit_diff, key differences:
        - compute cross entropy loss instead of logit diff
        - uses all token positions
        - does not require dataset to have an "answers" column

    Args:
        model: HookedTransformer model
        data_loader: ExperimentDataLoader for the dataset
        layers_to_ablate: List of layers to ablate
        cached_means:
            List of tensors of shape (layer, d_model)
            containing the mean value of a given token for each layer
        direction_vectors:
            List of tensors of shape (layer, d_model)
            containing the direction vector for each layer
        multiplier:
            Multiplier for the direction vector
        all_positions:
            If True, ablate all positions in the sequence.
            Otherwise, only ablate the positions specified in the data loader.
        vocab_mask:
            Inclusion mask of shape (vocab_size,) to apply to the logits
        device:
            Device to run the experiment on

    Returns:
        output: Tensor of shape (num_experiments, batch_size, sequence_length)
    """
    assert cached_means is not None or direction_vectors is not None
    file = ResultsFile(
        "ablation_modified_loss",
        model_name=model.cfg.model_name,
        dataset=data_loader.name,
        layers_to_ablate=layers_to_ablate,
        direction_vectors=direction_vectors,
        multiplier=multiplier,
        all_positions=all_positions,
        vocab_mask=(None if vocab_mask is None else vocab_mask.sum()),
        extension="pt",
    )
    if not overwrite and file.exists():
        return torch.load(file.path)
    if layers_to_ablate == "all":
        layers_to_ablate = list(range(model.cfg.n_layers))
    experiment_names = [
        "orig",
        "ablated",
    ]
    # get length of prompts in dataset
    item_len = len(data_loader.dataset[0]["tokens"])
    experiment_index = {name: i for i, name in enumerate(experiment_names)}
    out_shape = (
        len(experiment_names),
        data_loader.dataset.num_rows,
        item_len,
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

        # get the loss for each token in the batch
        orig_loss = loss_fn(
            model,
            batch_tokens,
            vocab_mask,
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

        # get the loss for each token when run with hooks
        hooked_loss = loss_fn(model, batch_tokens, vocab_mask)
        # concatenate column of 0s
        hooked_loss = torch.cat(
            [torch.zeros((hooked_loss.shape[0], 1)).to(device), hooked_loss], dim=1
        )
        ablated_metric = hooked_loss - orig_metric

        output[
            experiment_index["ablated"],
            batch_idx * batch_size : (batch_idx + 1) * batch_size,
        ] = ablated_metric.cpu()

        model.reset_hooks()

    torch.save(output, file.path)
    return output


def compute_ablation_modified_logit_diff(
    model: HookedTransformer,
    data_loader: ExperimentDataLoader,
    layers_to_ablate: List[int] | Literal["all"] = "all",
    cached_means: Optional[Float[Tensor, "layer d_model"]] = None,
    direction_vectors: Optional[Float[Tensor, "layer d_model"]] = None,
    multiplier=1.0,
    all_positions: bool = False,
    device: torch.device = DEFAULT_DEVICE,
    overwrite: bool = False,
) -> Float[Tensor, "experiment batch"]:
    """
    Computes the change in logit diff when the activations of
    a particular token are mean-ablated.

    If cached_means is specified, then we ablate the full residual stream at every layer.
    if direction_vectors is specified, then we only mean-ablate those directions.

    cf. compute_ablation_modified_loss, key differences:
        - compute logit diff instead of cross entropy loss
        - uses last token position only
        - requires dataset to have an "answers" column

    Args:
        model: HookedTransformer model
        data_loader: ExperimentDataLoader for the dataset
        layers_to_ablate: List of layers to ablate
        cached_means:
            List of tensors of shape (layer, d_model)
            containing the mean value of a given token for each layer
        direction_vectors:
            List of tensors of shape (layer, d_model)
            containing the direction vector for each layer
        multiplier:
            Multiplier for the direction vector
        all_positions:
            If True, ablate all positions in the sequence.
            Otherwise, only ablate the positions specified in the data loader.
        device:
            Device to run the experiment on
    """
    assert cached_means is not None or direction_vectors is not None
    file = ResultsFile(
        "ablation_modified_logit_diff",
        model_name=model.cfg.model_name,
        dataset=data_loader.name,
        layers_to_ablate=layers_to_ablate,
        direction_vectors=direction_vectors,
        multiplier=multiplier,
        all_positions=all_positions,
        extension="pt",
    )
    if not overwrite and file.exists():
        return torch.load(file.path)
    if layers_to_ablate == "all":
        layers_to_ablate = list(range(model.cfg.n_layers))
    experiment_names = [
        "orig",
        "ablated",
    ]
    # get length of prompts in dataset
    item_len = len(data_loader.dataset[0]["tokens"])
    experiment_index = {name: i for i, name in enumerate(experiment_names)}
    out_shape = (len(experiment_names), data_loader.dataset.num_rows)
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

        # get the logit diff for the last token in each sequence
        orig_logits = model(batch_tokens, return_type="logits", prepend_bos=False)
        assert isinstance(orig_logits, Tensor)
        orig_metric = get_logit_diff(
            orig_logits,
            mask=batch_value["attention_mask"],
            answer_tokens=batch_value["answers"],
            per_prompt=True,
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

        ablated_logits = model(batch_tokens, return_type="logits", prepend_bos=False)
        # check to see if ablated_logits has any nan values
        if torch.isnan(ablated_logits).any():
            print("ablated logits has nan values")
        ablated_metric = get_logit_diff(
            ablated_logits,
            mask=batch_value["attention_mask"],
            answer_tokens=batch_value["answers"],
            per_prompt=True,
        )

        output[
            experiment_index["ablated"],
            batch_idx * batch_size : (batch_idx + 1) * batch_size,
        ] = ablated_metric.cpu()
        model.reset_hooks()

    torch.save(output, file.path)
    return output


def compute_zeroed_attn_modified_loss(
    model: HookedTransformer,
    data_loader: ExperimentDataLoader,
    heads_to_ablate: List[Tuple[int, int]],
    token_ids: List[int] = [13],
    vocab_mask: Optional[Bool[Tensor, "d_vocab"]] = None,
    device: torch.device = DEFAULT_DEVICE,
) -> Float[Tensor, "batch"]:
    loss_list = []
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value["tokens"].to(device)

        batch_pos = find_positions(batch_tokens, token_ids=token_ids)

        # get the loss for each token in the batch
        initial_loss = loss_fn(model, batch_tokens, vocab_mask)

        # add hooks for the activations of the 11 and 13 tokens
        for layer, head in heads_to_ablate:
            ablate_tokens = partial(
                zero_attention_pos_hook,
                pos_by_batch=batch_pos,
                head_idx=head,
            )
            model.blocks[layer].attn.hook_pattern.add_hook(ablate_tokens)

        # get the loss for each token when run with hooks
        hooked_loss = loss_fn(model, batch_tokens, vocab_mask)

        # compute the percent difference between the two losses
        loss_diff = (hooked_loss - initial_loss) / initial_loss

        loss_list.append(loss_diff)

    model.reset_hooks()
    return torch.cat(loss_list)
