import warnings
from jaxtyping import Float, Int
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import re
import torch
from torch import Tensor
import einops
from transformer_lens import HookedTransformer
from typing import Dict, List, Literal, Optional, Union
from summarization_utils.patching_metrics import get_logit_diff
from summarization_utils.path_patching import act_patch, IterNode, Node
from summarization_utils.toy_datasets import CounterfactualDataset


def extract_placeholders(template: str, prompt: str) -> Dict[str, str]:
    """
    Args:
    - template: the template string that contains placeholders
        e.g.  "{NAME_L} likes {OBJECT_L}. {NAME_R} likes {OBJECT_R}."
    - prompt: the prompt string that contains the actual values for the placeholders
        e.g. "John likes apples. Mary likes oranges."
    Returns:
    - A dictionary of placeholders and their values
        e.g. {"NAME_L": "John", "OBJECT_L": "apples", "NAME_R": "Mary", "OBJECT_R": "oranges"}
    """
    # Parse the template to identify the structure and placeholders
    # Assuming placeholders are in the format {PLACEHOLDER}
    placeholders = re.findall(r"\{(.*?)\}", template)

    # Replace placeholders in the template with regex patterns to match any word(s)
    regex_pattern = template
    regex_pattern = regex_pattern.replace(r"[", r"\[")
    regex_pattern = regex_pattern.replace(r"]", r"\]")
    regex_pattern = regex_pattern.replace(r"?", r"\?")
    regex_pattern = regex_pattern.replace(r"/", r"\/")
    regex_pattern = regex_pattern.replace(r".", r"\.")
    regex_pattern = regex_pattern.replace(r"|", r"\|")
    for placeholder in placeholders:
        regex_pattern = regex_pattern.replace(
            "{" + placeholder + "}",
            r"(?P<" + placeholder + r">.*?)",
            1,  # only replace 1st occurrence to avoid duplicate keys
        )
        regex_pattern = regex_pattern.replace(
            "{" + placeholder + "}",
            r"(.*?)",
        )

    # Use the generated regex pattern to match the given line and extract the placeholder
    match = re.search(regex_pattern, prompt, re.IGNORECASE)
    assert match is not None, (
        f"Prompt {prompt} does not match the template {template}. "
        f"Generated regex pattern: {regex_pattern}"
    )
    out = match.groupdict()
    for k, v in out.items():
        assert isinstance(v, str), (
            f"Placeholder {k} has value {v} of type {type(v)}, "
            f"expected a string.\n"
            f"Prompt: {prompt}\n"
            f"Template: {template}\n"
            f"Generated regex pattern: {regex_pattern}\n"
            f"GroupDict: {out}\n"
        )
    return out


def get_position_dict(
    tokens: Float[Tensor, "batch seq_len"],
    model: HookedTransformer,
    sep: Union[str, List[str]] = ",",
) -> Dict[str, List[List[int]]]:
    """
    Returns a dictionary of positions of clauses and separators
    e.g. {
        "clause_0": [[0, 1, 2]], "sep_0": [[3]],
        "clause_1": [[4, 5, 6]], "sep_1": [[7]],
        "clause_2": [[8, 9, 10]], "sep_2": [[11]],
        "sep_all": [[3, 7, 11]]
    }
    """
    batch_size, seq_len = tokens.shape
    full_pos_dict = dict()
    for b in range(batch_size):
        if isinstance(sep, str):
            sep_id = model.to_single_token(sep)
        else:
            sep_id = model.to_single_token(sep[b])
        batch_pos_dict = {}
        sep_count = 0
        current_clause = []
        sep_all = []
        for s in range(seq_len):
            if tokens[b, s] == sep_id:
                batch_pos_dict[f"clause_{sep_count}"] = current_clause
                batch_pos_dict[f"sep_{sep_count}"] = [s]
                sep_all.append(s)
                sep_count += 1
                current_clause = []
                continue
            else:
                current_clause.append(s)
        if current_clause:
            batch_pos_dict[f"clause_{sep_count}"] = current_clause
        batch_pos_dict["sep_all"] = sep_all
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


def is_negative(
    pos: Union[
        Literal["each"], int, List[int], List[List[int]], Int[Tensor, "batch *pos"]
    ]
) -> bool:
    if isinstance(pos, str):
        return False
    elif isinstance(pos, int):
        return pos < 0
    elif isinstance(pos, list):
        return all([is_negative(p) for p in pos])
    elif isinstance(pos, Tensor):
        return bool((pos < 0).all().item())


def patch_prompt_base(
    prompt: str,
    answer: str,
    cf_prompt: str,
    cf_answer: str,
    model: HookedTransformer,
    prepend_bos: bool = True,
    layers: Literal["all", "each"] = "each",
    node_name: str = "resid_pre",
    seq_pos: Union[
        Literal["each"], int, List[int], List[List[int]], Int[Tensor, "batch *pos"]
    ] = "each",
    verbose: bool = True,
    check_shape: bool = True,
) -> Float[np.ndarray, "*layer pos"]:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    cf_tokens = model.to_tokens(cf_prompt, prepend_bos=prepend_bos)
    answer_id = model.to_single_token(answer)
    cf_answer_id = model.to_single_token(cf_answer)
    answer_tokens = torch.tensor(
        [answer_id, cf_answer_id], dtype=torch.int64, device=model.cfg.device
    ).unsqueeze(0)
    if check_shape:
        assert prompt_tokens.shape == cf_tokens.shape, (
            f"Prompt and counterfactual prompt must have the same shape, "
            f"for prompt {prompt} "
            f"got {prompt_tokens.shape} and {cf_tokens.shape}"
        )
    model.reset_hooks(including_permanent=True)
    base_logits: Float[Tensor, "1 seq_len d_vocab"] = model(
        prompt_tokens,
        prepend_bos=False,
        return_type="logits",
    )
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
        nodes = IterNode(node_names=[node_name], seq_pos=seq_pos)
        if seq_pos == "each":
            n_pos = prompt_tokens.shape[1]
        elif seq_pos is None or isinstance(seq_pos, int):
            n_pos = 1
        elif isinstance(seq_pos, list):
            n_pos = len(seq_pos)
        elif isinstance(seq_pos, Tensor) and seq_pos.ndim == 1:
            n_pos = 1
        elif isinstance(seq_pos, Tensor) and seq_pos.ndim == 2:
            n_pos = seq_pos.shape[1]
        else:
            raise ValueError(f"Invalid seq_pos {seq_pos}")
        results = act_patch(
            model, prompt_tokens, nodes, metric, new_cache=cf_cache, verbose=verbose
        )[
            node_name
        ]  # type: ignore
        results = torch.stack(results, dim=0)  # type: ignore
        if node_name == "resid_pre":
            results = einops.rearrange(
                results,
                "(pos layer) -> layer pos",
                layer=model.cfg.n_layers,
                pos=n_pos,
            )
        elif node_name == "z":
            results = einops.rearrange(
                results,
                "(pos layer head) -> layer head pos",
                layer=model.cfg.n_layers,
                head=model.cfg.n_heads,
                pos=n_pos,
            )
    elif layers == "all":
        results = [
            act_patch(
                model,
                prompt_tokens,
                [
                    Node(node_name, layer=layer, seq_pos=pos)
                    for layer in range(model.cfg.n_layers)
                ],
                metric,
                new_cache=cf_cache,
                verbose=verbose,
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
    node_name: str = "resid_pre",
    seq_pos: Union[
        Literal["each"], int, List[int], List[List[int]], Int[Tensor, "batch *pos"]
    ] = "each",
    verbose: bool = True,
) -> Float[np.ndarray, "layer pos"]:
    return patch_prompt_base(
        prompt,
        answer,
        cf_prompt,
        cf_answer,
        model=model,
        prepend_bos=prepend_bos,
        layers="each",
        node_name=node_name,
        seq_pos=seq_pos,
        verbose=verbose,
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
    prepend_bos: bool = True,
    node_name: str = "resid_pre",
    seq_pos: Union[
        Literal["each"], int, List[int], List[List[int]], Int[Tensor, "batch *pos"]
    ] = "each",
    verbose: bool = True,
) -> List[Float[np.ndarray, "layer pos"]]:
    results_list = []
    for i, (prompt, answer, cf_prompt, cf_answer) in enumerate(dataset):
        prompt_results = patch_prompt_by_layer(
            prompt,
            answer,
            cf_prompt,
            cf_answer,
            model=dataset.model,
            prepend_bos=prepend_bos,
            node_name=node_name,
            seq_pos=(
                seq_pos[i]
                if isinstance(seq_pos, list) or isinstance(seq_pos, Tensor)
                else seq_pos
            ),
            verbose=verbose,
        )
        results_list.append(prompt_results)
    return results_list


def plot_layer_results_per_batch(
    dataset: CounterfactualDataset,
    results: List[Float[np.ndarray, "layer pos"]],
    seq_pos: Optional[Union[int, List[int]]],
) -> go.Figure:
    if isinstance(seq_pos, int):
        seq_pos = [seq_pos]
    fig = make_subplots(rows=len(results), cols=1)
    for row, (prompt, result) in enumerate(zip(dataset.prompts, results)):
        prompt_str_tokens = dataset.model.to_str_tokens(prompt)
        if seq_pos is None:
            seq_pos = list(range(len(prompt_str_tokens)))
        hm = go.Heatmap(
            z=result,
            x=[f"{i}: {t}" for i, t in zip(seq_pos, prompt_str_tokens)],
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


def plot_head_results_per_batch(
    dataset: CounterfactualDataset, results: List[Float[np.ndarray, "layer head *pos"]]
) -> go.Figure:
    fig = make_subplots(rows=len(results), cols=1, subplot_titles=dataset.prompts)
    for row, (prompt, result) in enumerate(zip(dataset.prompts, results)):
        if result.ndim == 3:
            result = result.squeeze(2)
        hm = go.Heatmap(
            z=result,
            x=[f"H{i}" for i in range(dataset.model.cfg.n_heads)],
            y=[f"L{i}" for i in range(dataset.model.cfg.n_layers)],
            colorscale="RdBu",
            zmin=0,
            zmax=1,
            hovertemplate="%{y}%{x}<br>Logit diff %{z}<extra></extra>",
            name=prompt,
        )
        fig.add_trace(hm, row=row + 1, col=1)
        # set x and y axes titles of each trace
        fig.update_xaxes(title_text="Head", row=row + 1, col=1)
        fig.update_yaxes(title_text="Layer", row=row + 1, col=1)
    fig.update_layout(
        title_x=0.5,
        title=f"Patching metric by layer and head, {dataset.model.cfg.model_name}",
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
    dataset: CounterfactualDataset, sep: str = ",", verbose: bool = True
) -> Float[pd.DataFrame, "batch group"]:
    if (
        sep not in dataset.prompts[0]
        and dataset.template is not None
        and sep in dataset.template
    ):
        # given sep needs to be interpreted as a placeholder
        sep_clean = [
            extract_placeholders(dataset.template, prompt)[sep]
            for prompt in dataset.prompts
        ]
        # Check that extracted results are strings
        for s in sep_clean:
            assert isinstance(s, str), (
                f"Separator {sep} must be a string, got {type(s)} {s}\n"
                f"Full list={sep_clean}\n"
                f"Template={dataset.template}\n"
            )
        sep_id = torch.tensor(
            [dataset.model.to_single_token(" " + s) for s in sep_clean],
            dtype=torch.int64,
            device=dataset.device,
        ).unsqueeze(
            1
        )  # [batch, 1]
        assert sep_id.shape == (len(dataset), 1), (
            f"Separator must have shape (batch, 1), " f"got {sep_id.shape}"
        )
        has_sep_mask = (dataset.prompt_tokens == sep_id).any(dim=1)  # [batch]
        sep_clean = [" " + s for s, m in zip(sep_clean, has_sep_mask) if m]
    else:
        sep_clean = sep.replace("_", " ")
        sep_id = dataset.model.to_single_token(sep_clean)
        has_sep_mask = (dataset.prompt_tokens == sep_id).any(dim=1)
    if dataset.base_ldiff.shape[0] == 0:
        dataset.compute_logit_diffs(vectorized=True)
    assert (dataset.base_ldiff != dataset.cf_ldiff).all(), (
        f"Base logit diff {dataset.base_ldiff} and cf logit diff {dataset.cf_ldiff} "
        f"must be different"
    )
    is_sep_mask = dataset.prompt_tokens == sep_id
    is_sep_cf_mask = dataset.cf_tokens == sep_id
    if not (is_sep_mask == is_sep_cf_mask).all():
        warnings.warn(
            f"Separators in prompt and counterfactual prompt are not at the same positions, "
            f"got {torch.where(dataset.prompt_tokens == sep_id)} and {torch.where(dataset.cf_tokens == sep_id)}"
        )
    metric = lambda logits: (
        get_logit_diff(
            logits,
            answer_tokens=dataset.answer_tokens[has_sep_mask],
            mask=dataset.mask[has_sep_mask],
            per_prompt=True,
        )
        - dataset.base_ldiff[has_sep_mask]
    ) / (dataset.cf_ldiff[has_sep_mask] - dataset.base_ldiff[has_sep_mask])
    pos_dict = get_position_dict(
        dataset.prompt_tokens[has_sep_mask], model=dataset.model, sep=sep_clean
    )
    results_dict = dict()
    for pos_label, positions in pos_dict.items():
        nodes = [
            Node(node_name="resid_pre", layer=layer, seq_pos=positions)
            for layer in range(dataset.model.cfg.n_layers)
        ]
        pos_results = act_patch(
            dataset.model, dataset.prompt_tokens[has_sep_mask], nodes, metric, new_input=dataset.cf_tokens[has_sep_mask], verbose=verbose  # type: ignore
        )
        results_dict[pos_label] = pos_results.to(dtype=torch.float32).cpu().numpy()
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
