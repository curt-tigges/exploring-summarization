from jaxtyping import Float
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import torch
from torch import Tensor
import einops
from transformer_lens import HookedTransformer, ActivationCache
from typing import Dict, List, Literal
from summarization_utils.patching_metrics import get_logit_diff
from summarization_utils.path_patching import act_patch, IterNode, Node
from summarization_utils.toy_datasets import CounterfactualDataset


def get_position_dict(
    tokens: Float[Tensor, "batch seq_len"], model: HookedTransformer, sep=","
) -> Dict[str, List[List[int]]]:
    """
    Returns a dictionary of positions of clauses and separators
    e.g. {"clause_0": [[0, 1, 2]], "sep_0": [[3]], "clause_1": [[4, 5, 6]]}
    """
    sep_id = model.to_single_token(sep)
    batch_size, seq_len = tokens.shape
    full_pos_dict = dict()
    for b in range(batch_size):
        batch_pos_dict = {}
        sep_count = 0
        current_clause = []
        for s in range(seq_len):
            if tokens[b, s] == sep_id:
                batch_pos_dict[f"clause_{sep_count}"] = current_clause
                batch_pos_dict[f"sep_{sep_count}"] = [s]
                sep_count += 1
                current_clause = []
                continue
            else:
                current_clause.append(s)
        if current_clause:
            batch_pos_dict[f"clause_{sep_count}"] = current_clause
        for k, v in batch_pos_dict.items():
            if k in full_pos_dict:
                full_pos_dict[k].append(v)
            else:
                full_pos_dict[k] = [v]
    for pos_key, pos_values in full_pos_dict.items():
        assert len(pos_values) == batch_size, (
            f"Position {pos_key} has {len(pos_values)} values, "
            f"expected {batch_size}"
        )
    return full_pos_dict


def patch_prompt_base(
    prompt: str,
    answer: str,
    cf_prompt: str,
    cf_answer: str,
    model: HookedTransformer,
    prepend_bos: bool = True,
    layers: Literal["all", "each"] = "each",
) -> Float[np.ndarray, "*layer pos"]:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    cf_tokens = model.to_tokens(cf_prompt, prepend_bos=prepend_bos)
    answer_id = model.to_single_token(answer)
    cf_answer_id = model.to_single_token(cf_answer)
    answer_tokens = torch.tensor(
        [answer_id, cf_answer_id], dtype=torch.int64, device=model.cfg.device
    ).unsqueeze(0)
    assert prompt_tokens.shape == cf_tokens.shape, (
        f"Prompt and counterfactual prompt must have the same shape, "
        f"for prompt {prompt} "
        f"got {prompt_tokens.shape} and {cf_tokens.shape}"
    )
    model.reset_hooks(including_permanent=True)
    base_logits_by_pos: Float[Tensor, "1 seq_len d_vocab"] = model(
        prompt_tokens,
        prepend_bos=False,
        return_type="logits",
    )
    base_logits: Float[Tensor, "... d_vocab"] = base_logits_by_pos[:, -1, :]
    base_ldiff = get_logit_diff(base_logits, answer_tokens=answer_tokens)
    cf_logits, cf_cache = model.run_with_cache(
        cf_tokens, prepend_bos=False, return_type="logits"
    )
    assert isinstance(cf_logits, Tensor)
    cf_ldiff = get_logit_diff(cf_logits, answer_tokens=answer_tokens)
    metric = lambda logits: (
        get_logit_diff(logits, answer_tokens=answer_tokens) - base_ldiff
    ) / (cf_ldiff - base_ldiff)
    if layers == "each":
        nodes = IterNode(node_names=["resid_pre"], seq_pos="each")
        results = act_patch(
            model, prompt_tokens, nodes, metric, new_cache=cf_cache, verbose=True
        )[
            "resid_pre"
        ]  # type: ignore
        results = torch.stack(results, dim=0)  # type: ignore
        results = einops.rearrange(
            results,
            "(pos layer) -> layer pos",
            layer=model.cfg.n_layers,
            pos=prompt_tokens.shape[1],
        )
    elif layers == "all":
        results = [
            act_patch(
                model,
                prompt_tokens,
                [
                    Node("resid_pre", layer=layer, seq_pos=pos)
                    for layer in range(model.cfg.n_layers)
                ],
                metric,
                new_cache=cf_cache,
                verbose=True,
            )
            for pos in range(prompt_tokens.shape[1])
        ]
        results = torch.stack(results, dim=0)
    else:
        raise ValueError(f"Invalid layers {layers}")
    results = results.cpu().to(dtype=torch.float32).numpy()
    return results


def patch_prompt_by_layer(
    prompt: str,
    answer: str,
    cf_prompt: str,
    cf_answer: str,
    model: HookedTransformer,
    prepend_bos: bool = True,
) -> Float[np.ndarray, "layer pos"]:
    return patch_prompt_base(
        prompt,
        answer,
        cf_prompt,
        cf_answer,
        model=model,
        prepend_bos=prepend_bos,
        layers="each",
    )


def patch_prompt_by_position(
    prompt: str,
    answer: str,
    cf_prompt: str,
    cf_answer: str,
    model: HookedTransformer,
    prepend_bos: bool = True,
) -> Float[np.ndarray, "pos"]:
    return patch_prompt_base(
        prompt,
        answer,
        cf_prompt,
        cf_answer,
        model=model,
        prepend_bos=prepend_bos,
        layers="all",
    )


def patch_by_layer(
    dataset: CounterfactualDataset,
) -> List[Float[np.ndarray, "layer pos"]]:
    results_list = []
    for prompt, answer, cf_prompt, cf_answer in dataset:
        prompt_results = patch_prompt_by_layer(
            prompt,
            answer,
            cf_prompt,
            cf_answer,
            model=dataset.model,
            prepend_bos=True,
        )
        results_list.append(prompt_results)
    return results_list


def plot_layer_results_per_batch(
    dataset: CounterfactualDataset, results: List[Float[np.ndarray, "layer pos"]]
) -> go.Figure:
    fig = make_subplots(rows=len(results), cols=1)
    for row, (prompt, result) in enumerate(zip(dataset.prompts, results)):
        prompt_str_tokens = dataset.model.to_str_tokens(prompt)
        hm = go.Heatmap(
            z=result,
            x=[f"{i}: {t}" for i, t in enumerate(prompt_str_tokens)],
            y=[f"{i}" for i in range(dataset.model.cfg.n_layers)],
            colorscale="RdBu",
            zmin=0,
            zmax=1,
            hovertemplate="Layer %{y}<br>Position %{x}<br>Logit diff %{z}<extra></extra>",
        )
        fig.add_trace(hm, row=row + 1, col=1)
        # set x and y axes titles of each trace
        fig.update_xaxes(title_text="Position", row=row + 1, col=1)
        fig.update_yaxes(title_text="Layer", row=row + 1, col=1)
    fig.update_layout(
        title_x=0.5,
        title=f"Patching metric by layer and position, {dataset.model.cfg.model_name}",
        width=800,
        height=400 * len(results),
        overwrite=True,
    )
    return fig


def patch_by_position(
    dataset: CounterfactualDataset,
) -> List[Float[np.ndarray, "pos"]]:
    results_list = []
    for prompt, answer, cf_prompt, cf_answer in dataset:
        prompt_results = patch_prompt_by_position(
            prompt,
            answer,
            cf_prompt,
            cf_answer,
            model=dataset.model,
            prepend_bos=True,
        )
        results_list.append(prompt_results)
    return results_list


def plot_position_results_per_batch(
    dataset: CounterfactualDataset, results: List[Float[np.ndarray, "pos"]]
) -> go.Figure:
    fig = make_subplots(rows=len(results), cols=1)
    for row, (prompt, result) in enumerate(zip(dataset.prompts, results)):
        prompt_str_tokens = dataset.model.to_str_tokens(prompt)
        hm = go.Bar(
            y=result,
            x=[f"{i}: {t}" for i, t in enumerate(prompt_str_tokens)],
        )
        fig.add_trace(hm, row=row + 1, col=1)
        # set x and y axes titles of each trace
        fig.update_xaxes(title_text="Position", row=row + 1, col=1)
    fig.update_layout(
        title_x=0.5,
        title=f"Patching metric by position, {dataset.model.cfg.model_name}",
        width=800,
        height=400 * len(results),
        overwrite=True,
        showlegend=False,
    )
    return fig


def patch_by_position_group(
    dataset: CounterfactualDataset, sep=",", verbose: bool = True
) -> Float[pd.DataFrame, "batch group"]:
    if dataset.base_ldiff.shape[0] == 0:
        dataset.compute_logit_diffs(vectorized=True)
    assert (dataset.base_ldiff != dataset.cf_ldiff).all(), (
        f"Base logit diff {dataset.base_ldiff} and cf logit diff {dataset.cf_ldiff} "
        f"must be different"
    )
    metric = lambda logits: (
        get_logit_diff(
            logits,
            answer_tokens=dataset.answer_tokens,
            mask=dataset.mask,
            per_prompt=True,
        )
        - dataset.base_ldiff
    ) / (dataset.cf_ldiff - dataset.base_ldiff)
    pos_dict = get_position_dict(dataset.prompt_tokens, model=dataset.model, sep=sep)
    results_dict = dict()
    for pos_label, positions in pos_dict.items():
        nodes = [
            Node(node_name="resid_pre", layer=layer, seq_pos=positions)
            for layer in range(dataset.model.cfg.n_layers)
        ]
        pos_results = act_patch(
            dataset.model, dataset.prompt_tokens, nodes, metric, new_input=dataset.cf_tokens, verbose=verbose  # type: ignore
        )
        results_dict[pos_label] = pos_results.cpu().numpy()
    return pd.DataFrame(results_dict)


def patch_at_position(
    dataset: CounterfactualDataset,
    positions: List[int] | List[List[int]],
    verbose: bool = True,
) -> Float[Tensor, "batch"]:
    results = []
    for batch, (prompt, answer, cf_prompt, cf_answer) in enumerate(dataset):
        prompt_tokens = dataset.model.to_tokens(prompt, prepend_bos=True)
        cf_tokens = dataset.model.to_tokens(cf_prompt, prepend_bos=True)
        answer_id = dataset.model.to_single_token(answer)
        cf_answer_id = dataset.model.to_single_token(cf_answer)
        answer_tokens = torch.tensor(
            [answer_id, cf_answer_id], dtype=torch.int64, device=dataset.device
        ).unsqueeze(0)
        assert prompt_tokens.shape == cf_tokens.shape, (
            f"Prompt and counterfactual prompt must have the same shape, "
            f"for prompt {prompt} "
            f"got {prompt_tokens.shape} and {cf_tokens.shape}"
        )
        dataset.model.reset_hooks()
        base_logits = dataset.model(
            prompt_tokens, prepend_bos=False, return_type="logits"
        )
        base_ldiff = get_logit_diff(base_logits, answer_tokens=answer_tokens)
        cf_logits, cf_cache = dataset.model.run_with_cache(
            cf_tokens, prepend_bos=False, return_type="logits"
        )
        assert isinstance(cf_logits, Tensor)
        cf_ldiff = get_logit_diff(cf_logits, answer_tokens=answer_tokens)
        metric = lambda logits: (
            get_logit_diff(logits, answer_tokens=answer_tokens) - base_ldiff
        ) / (cf_ldiff - base_ldiff)
        nodes = [
            Node(node_name="resid_pre", layer=layer, seq_pos=positions[batch])
            for layer in range(dataset.model.cfg.n_layers)
        ]
        result = act_patch(
            dataset.model,
            prompt_tokens,
            nodes,
            metric,
            new_cache=cf_cache,
            verbose=verbose,
        )
        results.append(result)
    results = torch.stack(results, dim=0)
    return results
