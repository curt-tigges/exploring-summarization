from enum import Enum
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from jaxtyping import Float, Int
from typeguard import typechecked
import einops

from fancy_einsum import einsum

import plotly.graph_objs as go
import torch
import ipywidgets as widgets
from IPython.display import display
from transformers import PreTrainedTokenizerBase
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import get_attention_mask


def get_final_non_pad_token(
    logits: Float[Tensor, "batch pos vocab"],
    attention_mask: Int[Tensor, "batch pos"],
) -> Float[Tensor, "batch vocab"]:
    """Gets the final non-pad token from a tensor.

    Args:
        logits (torch.Tensor): Logits to use.
        attention_mask (torch.Tensor): Attention mask to use.

    Returns:
        torch.Tensor: Final non-pad token logits.
    """
    # Get the last non-pad token
    position_index = einops.repeat(
        torch.arange(logits.shape[1], device=logits.device),
        "pos -> batch pos",
        batch=logits.shape[0],
    )
    masked_position = torch.where(
        attention_mask == 0, torch.full_like(position_index, -1), position_index
    )
    last_non_pad_token = einops.reduce(
        masked_position, "batch pos -> batch", reduction="max"
    )
    assert (last_non_pad_token >= 0).all()
    # Get the final token logits
    final_token_logits = logits[torch.arange(logits.shape[0]), last_non_pad_token, :]
    return final_token_logits


# =============== METRIC UTILS ===============
def get_final_token_logits(
    logits: Float[Tensor, "batch *pos vocab"],
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    mask: Optional[Int[Tensor, "batch pos"]] = None,
) -> Float[Tensor, "batch vocab"]:
    if tokenizer is not None:
        assert tokens is not None
        mask = get_attention_mask(tokenizer, tokens, prepend_bos=False)
        final_token_logits = get_final_non_pad_token(logits, mask)
    elif mask is not None:
        final_token_logits = get_final_non_pad_token(logits, mask)
    elif logits.ndim == 3:
        final_token_logits = logits[:, -1, :]
    else:
        final_token_logits = logits
    return final_token_logits


@typechecked
def get_logit_diff(
    logits: Float[Tensor, "batch *pos vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
    per_prompt: bool = False,
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    mask: Optional[Int[Tensor, "batch pos"]] = None,
) -> Float[Tensor, "*batch"]:
    """
    Gets the difference between the logits of the provided tokens
    e.g., the correct and incorrect tokens in IOI

    Args:
        logits (torch.Tensor): Logits to use.
        answer_tokens (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
        May or may not have batch dimension depending on `per_prompt`.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    n_pairs = answer_tokens.shape[1]
    final_token_logits: Float[Tensor, "batch vocab"] = get_final_token_logits(
        logits, tokens=tokens, tokenizer=tokenizer, mask=mask
    )
    repeated_logits: Float[Tensor, "batch n_pairs d_vocab"] = einops.repeat(
        final_token_logits, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    left_logits: Float[Tensor, "batch n_pairs"] = repeated_logits.gather(
        -1, answer_tokens[:, :, 0].unsqueeze(-1)
    ).squeeze(-1)
    right_logits: Float[Tensor, "batch n_pairs"] = repeated_logits.gather(
        -1, answer_tokens[:, :, 1].unsqueeze(-1)
    ).squeeze(-1)
    left_logits_batch: Float[Tensor, "batch"] = left_logits.mean(dim=1)
    right_logits_batch: Float[Tensor, "batch"] = right_logits.mean(dim=1)
    if per_prompt:
        return left_logits_batch - right_logits_batch

    return (left_logits_batch - right_logits_batch).mean()


@typechecked
def get_prob_diff(
    logits: Float[Tensor, "batch *pos vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
    per_prompt: bool = False,
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Float[Tensor, "*batch"]:
    """
    Gets the difference between the softmax probabilities of the provided tokens
    e.g., the correct and incorrect tokens in IOI

    Args:
        logits (torch.Tensor): Logits to use.
        answer_tokens (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: Difference between the softmax probs of the provided tokens.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    n_pairs = answer_tokens.shape[1]
    final_token_logits: Float[Tensor, "batch vocab"] = get_final_token_logits(
        logits, tokens=tokens, tokenizer=tokenizer
    )
    repeated_logits: Float[Tensor, "batch n_pairs d_vocab"] = einops.repeat(
        final_token_logits, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    probs: Float[Tensor, "batch n_pairs vocab"] = repeated_logits.softmax(dim=-1)
    left_probs: Float[Tensor, "batch n_pairs"] = probs.gather(
        -1, answer_tokens[:, :, 0].unsqueeze(-1)
    ).squeeze(-1)
    right_probs: Float[Tensor, "batch n_pairs"] = probs.gather(
        -1, answer_tokens[:, :, 1].unsqueeze(-1)
    ).squeeze(-1)
    left_probs_batch: Float[Tensor, "batch"] = left_probs.mean(dim=1)
    right_probs_batch: Float[Tensor, "batch"] = right_probs.mean(dim=1)
    if per_prompt:
        return left_probs_batch - right_probs_batch

    return (left_probs_batch - right_probs_batch).mean()


def get_log_probs(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    per_prompt: bool = False,
) -> Float[Tensor, "batch n_pairs"]:
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    n_pairs = answer_tokens.shape[1]
    logits: Float[Tensor, "batch vocab"] = get_final_token_logits(
        logits, tokens=tokens, tokenizer=tokenizer
    )
    assert len(answer_tokens.shape) == 2

    # convert logits to log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # get the log probs for the answer tokens
    log_probs_repeated = einops.repeat(
        log_probs, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    answer_log_probs: Float[Tensor, "batch n_pairs"] = log_probs_repeated.gather(
        -1, answer_tokens.unsqueeze(-1)
    )
    # average over the answer tokens
    answer_log_probs_batch: Float[Tensor, "batch"] = answer_log_probs.mean(dim=1)
    if per_prompt:
        return answer_log_probs_batch
    else:
        return answer_log_probs_batch.mean()


def log_prob_diff_noising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Float[Tensor, ""]:
    """
    Linear function of log prob, calibrated so that it equals 0 when performance is
    same as on clean input, and 1 when performance is same as on flipped input.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    log_prob = get_log_probs(logits, answer_tokens, tokens=tokens, tokenizer=tokenizer)
    ld = (log_prob - clean_value) / (flipped_value - clean_value)
    if return_tensor:
        return ld
    else:
        return ld.item()


def log_prob_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Float[Tensor, ""]:
    """
    Linear function of log prob, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    log_prob = get_log_probs(logits, answer_tokens, tokens=tokens, tokenizer=tokenizer)
    ld = (log_prob - flipped_value) / (clean_value - flipped_value)
    if return_tensor:
        return ld
    else:
        return ld.item()


def logit_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Float[Tensor, ""]:
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    patched_logit_diff = get_logit_diff(
        logits, answer_tokens, tokens=tokens, tokenizer=tokenizer
    )
    ld = (patched_logit_diff - flipped_value) / (clean_value - flipped_value)
    if return_tensor:
        return ld
    else:
        return ld.item()


def prob_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> float:
    """
    Linear function of prob diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    patched_logit_diff = get_prob_diff(
        logits, answer_tokens, tokens=tokens, tokenizer=tokenizer
    )
    ld = ((patched_logit_diff - flipped_value) / (clean_value - flipped_value)).item()
    if return_tensor:
        return ld
    else:
        return ld.item()


def center_logit_diffs(
    logit_diffs: Float[Tensor, "batch"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
) -> Tuple[Float[Tensor, "batch"], float]:
    """
    Useful to debias a model when using as a binary classifier
    """
    device = logit_diffs.device
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    is_positive = (answer_tokens[:, 0, 0] == answer_tokens[0, 0, 0]).to(device=device)
    bias = torch.where(is_positive, logit_diffs, -logit_diffs).mean().to(device=device)
    debiased = logit_diffs - torch.where(is_positive, bias, -bias)
    return debiased, bias.item()


def get_accuracy_from_logit_diffs(logit_diffs: Float[Tensor, "batch"]):
    return (logit_diffs > 0).float().mean()


def logit_flip_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Float[Tensor, ""]:
    """
    Linear function of accuracy, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    Moves in discrete jumps based on whether logit diffs are closer to clean or corrupted.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    patched_logit_diffs = get_logit_diff(
        logits, answer_tokens, per_prompt=True, tokens=tokens, tokenizer=tokenizer
    )
    centered_logit_diffs = center_logit_diffs(patched_logit_diffs, answer_tokens)[0]
    accuracy = get_accuracy_from_logit_diffs(centered_logit_diffs)
    lf = (accuracy - flipped_value) / (clean_value - flipped_value)
    if return_tensor:
        return lf
    else:
        return lf.item()


def logit_diff_noising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> float:
    """
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    patched_logit_diff = get_logit_diff(
        logits, answer_tokens, tokens=tokens, tokenizer=tokenizer
    )
    ld = (patched_logit_diff - clean_value) / (clean_value - flipped_value)

    if return_tensor:
        return ld
    else:
        return ld.item()


# =============== LOGIT LENS UTILS ===============


def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    answer_tokens: Int[Tensor, "batch pair correct"],
    model: HookedTransformer,
    pos: int = -1,
    biased: bool = False,
):
    scaled_residual_stack: Float[Tensor, "... batch d_model"] = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=pos
    )
    answer_residual_directions: Float[
        Tensor, "batch pair correct d_model"
    ] = model.tokens_to_residual_directions(answer_tokens)
    answer_residual_directions = answer_residual_directions.mean(dim=1)
    logit_diff_directions: Float[Tensor, "batch d_model"] = (
        answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
    )
    batch_logit_diffs: Float[Tensor, "... batch"] = einops.einsum(
        scaled_residual_stack,
        logit_diff_directions,
        "... batch d_model, batch d_model -> ... batch",
    )
    if not biased:
        diff_from_unembedding_bias: Float[Tensor, "batch"] = (
            model.b_U[answer_tokens[:, :, 0]] - model.b_U[answer_tokens[:, :, 1]]
        ).mean(dim=1)
        batch_logit_diffs += diff_from_unembedding_bias
    return einops.reduce(batch_logit_diffs, "... batch -> ...", "mean")


def cache_to_logit_diff(
    cache: ActivationCache,
    answer_tokens: Int[Tensor, "batch pair correct"],
    model: HookedTransformer,
    pos: int = -1,
):
    final_residual_stream: Float[Tensor, "batch pos d_model"] = cache["resid_post", -1]
    token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[
        :, pos, :
    ]
    return residual_stack_to_logit_diff(
        token_residual_stream,
        answer_tokens=answer_tokens,
        model=model,
        cache=cache,
        pos=pos,
    )


class PatchingMetric(Enum):
    LOGIT_DIFF_DENOISING = logit_diff_denoising
    LOGIT_DIFF_NOISING = logit_diff_noising
    LOGIT_FLIP_DENOISING = logit_flip_denoising
    PROB_DIFF_DENOISING = prob_diff_denoising
