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
from summarization_utils.patching_metrics import get_logit_diff
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
from summarization_utils.toy_datasets import (
    CounterfactualDataset,
    ToyDeductionTemplate,
    ToyBindingTemplate,
    ToyProfilesTemplate,
)
from summarization_utils.counterfactual_patching import (
    patch_by_position_group,
    patch_by_layer,
    plot_layer_results_per_batch,
    plot_head_results_per_batch,
)
from summarization_utils.visualization import (
    plot_attention,
    scatter_attention_and_contribution,
    scatter_attention_and_contribution_sentiment,
    scatter_attention_and_contribution_simple,
    scatter_attention_and_contribution_logic,
)

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    "santacoder",
    torch_dtype=torch.float32,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device=device,
)
assert model.tokenizer is not None


# %%
def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) - 1


# %%
CODE_TUPLES = [
    (
        "def print_first_n_composites(n: int) -> None:\n    for num in range(2, n):\n        if num > 1:\n            for i in range(2, num):\n                if (num % i) == 0:\n                    ",
        " print",
        "def print_first_n_prime_numbers(n: int) -> None:\n    for num in range(2, n):\n        if num > 1:\n            for i in range(2, num):\n                if (num % i) == 0:\n                    ",
        " break",
    ),  # 52% at ":"
    (
        "def Factorial(n: int) -> int:\n    if n <= 1:\n        return ",
        "1",
        "def fibonacci(n: int) -> int:\n    if n <= 0:\n        return ",
        "0",
    ),  # 16% at ":"
    (
        "def power(a, b):\n    if b == 0:\n        return ",
        "1",
        "def divide(a, b):\n    if b == 0:\n        return ",
        "0",
    ),
]
patch_positions = torch.tensor(
    [
        listRightIndex(model.to_str_tokens(prompt), ":")
        for prompt, _, _, _ in CODE_TUPLES
    ],
    dtype=torch.long,
    device=device,
).unsqueeze(1)
# print(patch_positions)
# %%
# for i, (prompt, _, _, _) in enumerate(CODE_TUPLES):
#     print(model.to_str_tokens(prompt)[patch_positions[i]])
# print([f"{i}:{t}" for i, t in enumerate(model.to_str_tokens(prompt))])
# %%
dataset = CounterfactualDataset.from_tuples(CODE_TUPLES, model)
# %%
dataset.check_lengths_match()
# %%
dataset.test_prompts(max_prompts=20, top_k=10)
# %%
all_logit_diffs, cf_logit_diffs = dataset.compute_logit_diffs()
# %%
print(f"Original mean: {all_logit_diffs.mean():.2f}")
print(f"Counterfactual mean: {cf_logit_diffs.mean():.2f}")
# %%
assert (all_logit_diffs > 0).all()
assert (cf_logit_diffs < 0).all()
# %%
# ###################################################################################
# LAYER POSITION PATCHING
# ###################################################################################
# %%
pos_layer_results = patch_by_layer(dataset)
print(len(pos_layer_results), pos_layer_results[0].shape)
# %%
plot_layer_results_per_batch(dataset, pos_layer_results)
# %%
# ###################################################################################
# HEAD POSITION PATCHING
# ###################################################################################
# %%
head_layer_results = patch_by_layer(dataset, node_name="z", seq_pos=patch_positions)
print(len(head_layer_results), head_layer_results[0].shape)
# %%
plot_head_results_per_batch(dataset, head_layer_results)
# %%
heads: List[Tuple[int, int]] = [
    # (10, 7),
    (10, 15),
    # (11, 4)
]
# %%
for prompt, _, cf_prompt, _ in CODE_TUPLES:
    for p in (prompt, cf_prompt):
        if p == cf_prompt:
            continue
        html = plot_attention(
            model,
            p,
            heads,
            weighted=True,
            min_value=0.0,
            max_value=0.5,
        )
        display(HTML(html.local_src))
# %%
"""
Prompt 1:
10.7 attends to "0" and the other previous 3 tokens
10.15 attends to function name i.e. com (8) posit (9) and es (10) (and to "0")
11.4 attends to "0"

Prompt 2:
10.7 attends to "1"
10.15 attends to function name, i.e. "Factor" (2) but mostly "ial" (3)
11.4 attends to "1"

Prompt 3:
10.7 attends to "0"
10.15 attends to function name, i.e. "power" (2)
11.4 attends to "0"
"""

# %%
colon_positions = [56, 18, 14]
function_name_positions = [[8, 9, 10], [2, 3], [2]]
for batch, (prompt, _, _, _) in enumerate(CODE_TUPLES):
    _, cache = model.run_with_cache(prompt, names_filter=lambda n: "resid_post" in n)
    colon_pos = colon_positions[batch]
    for function_name_pos in function_name_positions[batch]:
        colon_residuals = []
        function_name_residuals = []
        for layer in range(model.cfg.n_layers):
            colon_residuals.append(cache["resid_post", layer][0, colon_pos])
            function_name_residuals.append(
                cache["resid_post", layer][0, function_name_pos]
            )
        colon_stack = torch.stack(colon_residuals)
        function_name_stack = torch.stack(function_name_residuals)
        # plot cosine sims
        cosine_sims: Float[Tensor, "layer layer"] = (
            colon_stack
            @ function_name_stack.T
            / (
                colon_stack.norm(dim=-1)[:, None]
                * function_name_stack.norm(dim=-1)[None, :]
            )
        )
        assert cosine_sims.shape == (model.cfg.n_layers, model.cfg.n_layers)
        fig = px.imshow(
            cosine_sims.cpu().numpy(),
            color_continuous_scale="Blues",
            labels={"x": "Function name layer", "y": "Colon layer"},
            origin="lower",
            title=f"Prompt {batch+1} cosine similarity of colon ({colon_pos}) and function name ({function_name_pos})",
        )
        fig.show()
# %%
model.set_use_attn_result(True)
result_act_name = get_act_name("result", 10)
for batch, (prompt, _, _, _) in enumerate(CODE_TUPLES):
    _, cache = model.run_with_cache(
        prompt, names_filter=lambda n: "resid_pre" in n or n == result_act_name
    )
    colon_pos = colon_positions[batch]
    attn_result = cache["result", 10][0, colon_pos, 15]
    for function_name_pos in function_name_positions[batch]:
        resid_pre = cache["resid_pre", 10][0, function_name_pos]
        # plot cosine sims
        cosine_sim: Float[
            Tensor, "layer layer"
        ] = torch.nn.functional.cosine_similarity(resid_pre, attn_result, dim=0)
        print(
            f"Prompt {batch+1} cosine similarity of colon ({colon_pos}) and function name ({function_name_pos}): {cosine_sim.item():.3f}"
        )

# %%
