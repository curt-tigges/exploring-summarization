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
from utils.circuit_analysis import get_logit_diff
from utils.tokenwise_ablation import (
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
from utils.datasets import (
    OWTData,
    PileFullData,
    PileSplittedData,
    HFData,
    mask_positions,
    construct_exclude_list,
)
from utils.neuroscope import plot_top_onesided
from utils.store import ResultsFile, TensorBlockManager
from utils.path_patching import act_patch, Node, IterNode, IterSeqPos

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
    "santacoder",
    torch_dtype=torch.float32,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
model = model.to(device)
assert model.tokenizer is not None
# %%
PREPEND_SPACE_TO_ANSWER = False
CODE_DATASET = [
    (
        "def count_words(string: str) -> int:\n    return len(string.",
        "split",
        "def count_lines(string: str) -> int:\n    return len(string.",
        "splitlines",
    ),
    (
        "def reverseorder_string(string: str) -> str:\n    return string",
        "[::-",
        "def halve_string(string: str) -> str:\n    return string",
        "[:",
    ),
    (
        "def is_uppercase(string: str) -> bool:\n    return string.is",
        "upper",
        "def is_lowercase(string: str) -> bool:\n    return string.is",
        "lower",
    ),
    (
        "def convert_to_celsius(temp: float) -> float:\n    return (temp",
        " -",
        "def convert_to_fahrenheit(temp: float) -> float:\n    return (temp",
        " *",
    ),
    (
        "def Factorial(n: int) -> int\n    if n < 2:\n        return 1\n    else:\n        return",
        " n",
        "def fibonacci(n: int) -> int\n    if n < 2:\n        return 1\n    else:\n        return",
        " fib",
    ),
    (
        "def find_min(array: List[int]) -> int:\n    return",
        " min",
        "def find_max(array: List[int]) -> int:\n    return",
        " max",
    ),
    (
        "def calculate_mean(array: List[int]) -> float:\n    return",
        " sum",
        "def calculate_mode(array: List[int]) -> float:\n    return",
        " max",
    ),
]
# %%
for prompt, answer, cf_prompt, cf_answer in CODE_DATASET:
    prompt_str_tokens = model.to_str_tokens(prompt)
    cf_str_tokens = model.to_str_tokens(cf_prompt)
    assert len(prompt_str_tokens) == len(cf_str_tokens), (
        f"Prompt and counterfactual prompt must have the same length, "
        f"for prompt \n{prompt_str_tokens} \n and counterfactual\n{cf_str_tokens} \n"
        f"got {len(prompt_str_tokens)} and {len(cf_str_tokens)}"
    )
    model.to_single_token(answer)
    model.to_single_token(cf_answer)
# %%
for prompt, answer, cf_prompt, cf_answer in CODE_DATASET:
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
# %%
model.tokenizer.padding_side = "left"
all_tokens = model.to_tokens([d[0] for d in CODE_DATASET], prepend_bos=True)
cf_tokens = model.to_tokens([d[2] for d in CODE_DATASET], prepend_bos=True)
attention_mask = get_attention_mask(model.tokenizer, all_tokens, prepend_bos=False)
answer_prefix = " " if PREPEND_SPACE_TO_ANSWER else ""
answer_tokens = torch.tensor(
    [
        (
            model.to_single_token(answer_prefix + answer),
            model.to_single_token(answer_prefix + cf_answer),
        )
        for _, answer, _, cf_answer in CODE_DATASET
    ],
    device=device,
    dtype=torch.int64,
)
assert all_tokens.shape == cf_tokens.shape
assert (all_tokens == model.tokenizer.pad_token_id).sum() == (
    cf_tokens == model.tokenizer.pad_token_id
).sum()
print(all_tokens.shape, answer_tokens.shape)
# %%
CENTERED = False
all_logits: Float[Tensor, "batch pos d_vocab"] = model(
    all_tokens, prepend_bos=False, return_type="logits", attention_mask=attention_mask
)
all_logit_diffs = get_logit_diff(
    all_logits, answer_tokens=answer_tokens, per_prompt=True
)
if CENTERED:
    all_logit_diffs -= all_logit_diffs.mean()
print(all_logit_diffs)
# %%
cf_logits: Float[Tensor, "batch pos d_vocab"] = model(
    cf_tokens, prepend_bos=False, return_type="logits", attention_mask=attention_mask
)
cf_logit_diffs = get_logit_diff(cf_logits, answer_tokens=answer_tokens, per_prompt=True)
if CENTERED:
    cf_logit_diffs -= cf_logit_diffs.mean()
print(cf_logit_diffs)
# %%
print(f"Original mean: {all_logit_diffs.mean():.2f}")
print(f"Counterfactual mean: {cf_logit_diffs.mean():.2f}")
# %%
print(f"Original accuracy: {(all_logit_diffs > 0).float().mean():.2f}")
print(f"Counterfactual accuracy: {(cf_logit_diffs < 0).float().mean():.2f}")
# %%
"""
TODO:
* Ensure cf prompts are same length
* Compute accuracy on original and flipped prompts
* Perform activation patching on periods and visualise results
"""


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
            zmin=-1,
            zmax=1,
            # set midpoint to 0
            zmid=0,
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
patch_idx = 0
for prompt, answer, cf_prompt, cf_answer in CODE_DATASET:
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
    patch_idx += 1
    if patch_idx > 5:
        break
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
