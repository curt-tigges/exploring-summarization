# %%
import itertools
import random
import einops
from functools import partial
import numpy as np
import torch
import datasets
import os
import re
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union, Literal, Optional
from transformer_lens import HookedTransformer
from transformer_lens.utils import (
    get_dataset,
    tokenize_and_concatenate,
    get_act_name,
    test_prompt,
    get_attention_mask,
)
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd
from circuitsvis.activations import text_neuron_activations
from circuitsvis.topk_samples import topk_samples
from IPython.display import HTML, display
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from summarization_utils.circuit_analysis import get_logit_diff
from summarization_utils.tokenwise_ablation import (
    compute_ablation_modified_loss,
    load_directions,
    get_random_directions,
    get_zeroed_dir_vector,
    get_layerwise_token_mean_activations,
    ablation_hook_base,
    AblationHook,
    AblationHookIterator,
    get_batch_token_mean_activations,
    loss_fn,
    DEFAULT_DEVICE,
)
from summarization_utils.datasets import (
    OWTData,
    PileFullData,
    PileSplittedData,
    HFData,
    mask_positions,
    construct_exclude_list,
)
from summarization_utils.neuroscope import plot_top_onesided
from summarization_utils.store import ResultsFile, TensorBlockManager
from summarization_utils.path_patching import act_patch, Node, IterNode, IterSeqPos

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# %%
model = HookedTransformer.from_pretrained(
    "pythia-2.8b",
    torch_dtype=torch.float32,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
model = model.to(device)
assert model.tokenizer is not None
# %%
DATASET = [
    (
        "Known for being the first to walk on the moon, Neil",
        " Armstrong",
        "Known for being the star of the movie Jazz Singer, Neil",
        " Diamond",
    ),
    (
        "The first to walk on the moon is of course, Neil",
        " Armstrong",
        "The star of the movie Jazz Singer is of course, Neil",
        " Diamond",
    ),
    (
        "Known for being the first to cross Antarctica, Sir",
        " Ernest",
        "Known for being the first to summit Everest, Sir",
        " Edmund",
    ),
    (
        "The first to cross Antarctica was of course, Sir",
        " Ernest",
        "The first to summit Everest was of course, Sir",
        " Edmund",
    ),
    (
        "Known for being the fastest production car in the world, the",
        " McL",
        "Known for being the best selling car in the world, the",
        " Ford",
    ),
    (
        "The fastest production car in the world is of course, the",
        " McL",
        "The best selling car in the world is of course, the",
        " Ford",
    ),
    (
        "Known for being the most popular fruit in the world, the humble",
        " apple",
        "Known for being the most popular vegetable in the world, the humble",
        " potato",
    ),
    (
        "The most popular fruit in the world is of course, the humble",
        " apple",
        "The most popular vegetable in the world is of course, the humble",
        " potato",
    ),
    (
        "Known for being a wonder of the world located in Australia, the",
        " Great",
        "Known for being a wonder of the world located in India, the",
        " Taj",
    ),
    (
        "Known for being the most popular sport in Brazil, the game of",
        " soccer",
        "Known for being the most popular sport in India, the game of",
        " cricket",
    ),
    # (
    #     "Here are examples of US states: California, Texas, Florida, and",
    #     " New",
    #     "Here are examples of US presidents: Washington, Lincoln, Obama, and",
    #     " Trump",
    # ),
    # (
    #     "Here are examples of US cities: Chicago Illinois, Los Angeles, New York City, and",
    #     " Washington",
    #     "Here are examples of US senators: Bernie Sanders, Elizabeth Warren, Kamala Harris, and",
    #     " Cory",
    # ),
]
PREPEND_SPACE_TO_ANSWER = False
# %%
for prompt, _, cf_prompt, _ in DATASET:
    prompt_str_tokens = model.to_str_tokens(prompt)
    cf_str_tokens = model.to_str_tokens(cf_prompt)
    assert len(prompt_str_tokens) == len(cf_str_tokens), (
        f"Prompt and counterfactual prompt must have the same length, "
        f"for prompt \n{prompt_str_tokens} \n and counterfactual\n{cf_str_tokens} \n"
        f"got {len(prompt_str_tokens)} and {len(cf_str_tokens)}"
    )
# %%
i = 0
for prompt, answer, cf_prompt, cf_answer in DATASET:
    test_prompt(
        prompt,
        answer,
        model,
        top_k=10,
        prepend_space_to_answer=PREPEND_SPACE_TO_ANSWER,
    )
    test_prompt(
        cf_prompt,
        cf_answer,
        model,
        top_k=10,
        prepend_space_to_answer=PREPEND_SPACE_TO_ANSWER,
    )
    i += 2
    if i > 2:
        break
# %%
all_logit_diffs = []
cf_logit_diffs = []
for prompt, answer, cf_prompt, cf_answer in DATASET:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    cf_tokens = model.to_tokens(cf_prompt, prepend_bos=True)
    answer_id = model.to_single_token(answer)
    cf_answer_id = model.to_single_token(cf_answer)
    answer_tokens = torch.tensor(
        [answer_id, cf_answer_id], dtype=torch.int64, device=device
    ).unsqueeze(0)
    assert prompt_tokens.shape == cf_tokens.shape, (
        f"Prompt and counterfactual prompt must have the same shape, "
        f"for prompt {prompt} "
        f"got {prompt_tokens.shape} and {cf_tokens.shape}"
    )
    model.reset_hooks()
    base_logits = model(prompt_tokens, prepend_bos=False, return_type="logits")
    base_ldiff = get_logit_diff(base_logits, answer_tokens=answer_tokens)
    cf_logits = model(cf_tokens, prepend_bos=False, return_type="logits")
    assert isinstance(cf_logits, Tensor)
    cf_ldiff = get_logit_diff(cf_logits, answer_tokens=answer_tokens)
    all_logit_diffs.append(base_ldiff)
    cf_logit_diffs.append(cf_ldiff)
all_logit_diffs = torch.stack(all_logit_diffs, dim=0)
cf_logit_diffs = torch.stack(cf_logit_diffs, dim=0)
# %%
print(all_logit_diffs)
# %%
prompt_tokens = model.to_tokens([d[0] for d in DATASET], prepend_bos=True)
cf_tokens = model.to_tokens([d[2] for d in DATASET], prepend_bos=True)
mask = get_attention_mask(model.tokenizer, prompt_tokens, prepend_bos=False)
cf_mask = get_attention_mask(model.tokenizer, cf_tokens, prepend_bos=False)
assert prompt_tokens.shape == mask.shape
answer_tokens = torch.tensor(
    [(model.to_single_token(d[1]), model.to_single_token(d[3])) for d in DATASET],
    device=device,
)
base_logits = model(prompt_tokens, prepend_bos=False, return_type="logits")
base_ldiff = get_logit_diff(
    base_logits, answer_tokens=answer_tokens, per_prompt=True, mask=mask
)
cf_logits = model(cf_tokens, prepend_bos=False, return_type="logits")
assert isinstance(cf_logits, Tensor)
cf_ldiff = get_logit_diff(
    cf_logits, answer_tokens=answer_tokens, per_prompt=True, mask=cf_mask
)
print(base_ldiff)

# %%
print(f"Original mean: {all_logit_diffs.mean():.2f}")
print(f"Counterfactual mean: {cf_logit_diffs.mean():.2f}")
# %%
assert (all_logit_diffs > 0).all()
assert (cf_logit_diffs < 0).all()


# %%
# # ##############################################
# # ACTIVATION PATCHING
# # ##############################################
# %%
def plot_patch_by_layer(
    prompt: str,
    answer: str,
    cf_prompt: str,
    cf_answer: str,
    prepend_bos: bool = True,
) -> go.Figure:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    prompt_str_tokens = model.to_str_tokens(prompt_tokens, prepend_bos=prepend_bos)
    cf_tokens = model.to_tokens(cf_prompt, prepend_bos=prepend_bos)
    answer_id = model.to_single_token(answer)
    cf_answer_id = model.to_single_token(cf_answer)
    answer_tokens = torch.tensor(
        [answer_id, cf_answer_id], dtype=torch.int64, device=device
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
    nodes = IterNode(node_names=["resid_pre"], seq_pos="each")
    metric = lambda logits: (
        get_logit_diff(logits, answer_tokens=answer_tokens) - base_ldiff
    ) / (cf_ldiff - base_ldiff)
    results = act_patch(
        model, prompt_tokens, nodes, metric, new_cache=cf_cache, verbose=True
    )[
        "resid_pre"
    ]  # type: ignore
    results = torch.stack(results, dim=0)
    results = einops.rearrange(
        results,
        "(pos layer) -> layer pos",
        layer=model.cfg.n_layers,
        pos=prompt_tokens.shape[1],
    )
    results = results.cpu().numpy()

    fig = go.Figure(
        data=go.Heatmap(
            z=results,
            x=[f"{i}: {t}" for i, t in enumerate(prompt_str_tokens)],
            y=[f"{i}" for i in range(model.cfg.n_layers)],
            colorscale="RdBu",
            zmin=0,
            zmax=1,
            hovertemplate="Layer %{y}<br>Position %{x}<br>Logit diff %{z}<extra></extra>",
        ),
        layout=dict(
            title=f"Patching metric by layer, {model.cfg.model_name}",
            xaxis_title="Position",
            yaxis_title="Layer",
            title_x=0.5,
        ),
    )
    return fig


# %%
figs = []
for patch_idx, (prompt, answer, cf_prompt, cf_answer) in enumerate(DATASET):
    # if patch_idx <= 5:
    #     continue
    if patch_idx > 7:
        break
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    base_logits_by_pos: Float[Tensor, "1 seq_len d_vocab"] = model(
        prompt_tokens,
        prepend_bos=False,
        return_type="logits",
    )
    # if prompt_name == "sentiment inference":
    #     break
    fig = plot_patch_by_layer(prompt, answer, cf_prompt, cf_answer)
    figs.append(fig)
# Merge figures into subplots
fig = make_subplots(
    rows=len(figs), cols=1, subplot_titles=[f.layout.title.text for f in figs]
)
for row, f in enumerate(figs):
    fig.add_traces(f.data, rows=row + 1, cols=1)
fig.update_layout(
    title_x=0.5,
    xaxis_title="Position",
    yaxis_title="Layer",
    width=800,
    height=400 * len(figs),
)
fig.show()


# %%
def get_position_dict(
    tokens: Float[Tensor, "batch seq_len"], sep=","
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
    return full_pos_dict


# %%
PROMPTS = [d[0] for d in DATASET]
CF_PROMPTS = [d[2] for d in DATASET]
prompt_tokens = model.to_tokens(PROMPTS, prepend_bos=True)
cf_tokens = model.to_tokens(CF_PROMPTS, prepend_bos=True)
comma_id = model.to_single_token(",")
n_clauses = len(DATASET[0][0].split(","))
n_commas = DATASET[0][0].count(",")
pos_dict = get_position_dict(prompt_tokens, sep=",")
answer_tokens = torch.tensor(
    [(model.to_single_token(d[1]), model.to_single_token(d[3])) for d in DATASET],
    device=device,
)
mask = get_attention_mask(model.tokenizer, prompt_tokens, prepend_bos=False)
cf_mask = get_attention_mask(model.tokenizer, cf_tokens, prepend_bos=False)
assert prompt_tokens.shape == cf_tokens.shape
model.reset_hooks()
base_logits_by_pos: Float[Tensor, "1 seq_len d_vocab"] = model(
    prompt_tokens,
    attention_mask=mask,
    prepend_bos=False,
    return_type="logits",
)
print(base_logits_by_pos.shape)
base_ldiff = get_logit_diff(base_logits_by_pos, answer_tokens=answer_tokens, mask=mask)
cf_logits, cf_cache = model.run_with_cache(
    cf_tokens, prepend_bos=False, return_type="logits"
)
assert isinstance(cf_logits, Tensor)
cf_ldiff = get_logit_diff(cf_logits, answer_tokens=answer_tokens, mask=cf_mask)
metric = lambda logits: (
    get_logit_diff(logits, answer_tokens=answer_tokens, mask=mask) - base_ldiff
) / (cf_ldiff - base_ldiff)
print(pos_dict)
# %%
results_dict = dict()
for pos_label, positions in pos_dict.items():
    positions = pos_dict[pos_label]
    pos_tensor = torch.zeros_like(prompt_tokens, dtype=torch.bool)
    nodes = [
        Node(node_name="resid_pre", layer=layer, seq_pos=positions)
        for layer in range(model.cfg.n_layers)
    ]
    pos_results = act_patch(
        model, prompt_tokens, nodes, metric, new_cache=cf_cache, verbose=True  # type: ignore
    ).item()
    results_dict[pos_label] = pos_results
results_pd = pd.Series(results_dict)
# %%
fig = px.bar(results_pd, labels={"index": "Position", "value": "Patching metric"})
fig.update_layout(showlegend=False)
fig.show()
# %%
print(results_dict)
# results = torch.stack(results, dim=0)
# results = einops.rearrange(
#     results,
#     "(pos layer) -> layer pos",
#     layer=model.cfg.n_layers,
#     pos=prompt_tokens.shape[1],
# )
# results = results.cpu().numpy()

# fig = go.Figure(
#     data=go.Heatmap(
#         z=results,
#         x=[f"{i}: {t}" for i, t in enumerate(prompt_str_tokens)],
#         y=[f"{i}" for i in range(model.cfg.n_layers)],
#         colorscale="RdBu",
#         zmin=0,
#         zmax=1,
#         hovertemplate="Layer %{y}<br>Position %{x}<br>Logit diff %{z}<extra></extra>",
#     ),
#     layout=dict(
#         title=f"Patching metric by layer, {model.cfg.model_name}",
#         xaxis_title="Position",
#         yaxis_title="Layer",
#         title_x=0.5,
#     ),
# )


# %%
# ##############################################
# DIG INTO EXAMPLES
# ##############################################
# %%
# top_k = 10
# PROMPT_NAME = "knowledge extraction"
# ABLATION_TOKEN = ","
# layer = 0
# for prompt_name, prompt, answer, _, _ in list_of_prompts:
#     if prompt_name != PROMPT_NAME:
#         continue
#     model.reset_hooks()
#     my_test_prompt = partial(
#         test_prompt,
#         prompt=prompt,
#         answer=answer,
#         model=model,
#         top_k=top_k,
#         prepend_space_to_answer=False,
#         prepend_bos=False,
#     )
#     prompt_tokens = model.to_tokens(prompt, prepend_bos=False)
#     prompt_str_tokens = model.to_str_tokens(prompt_tokens, prepend_bos=False)
#     ablation_pos = prompt_str_tokens.index(ABLATION_TOKEN)  # type: ignore
#     ablation_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
#     ablation_mask[:, ablation_pos] = True
#     ablation_values = torch.zeros(
#         (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
#     )
#     ablation_hook = AblationHook(
#         model, ablation_mask, ablation_values=ablation_values, layers_to_ablate=layer
#     )
#     print("baseline")
#     my_test_prompt()
#     print(f"Zero Ablating '{ABLATION_TOKEN}'")
#     with ablation_hook:
#         my_test_prompt()

# %%
