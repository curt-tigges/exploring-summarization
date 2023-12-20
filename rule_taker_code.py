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
from typing import Callable, Dict, Iterable, List, Tuple, Union, Literal, Optional
from transformer_lens import ActivationCache, HookedTransformer
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
from summarization_utils.path_patching import (
    act_patch,
    Node,
    IterNode,
    IterSeqPos,
    _act_patch_single,
    get_batch_and_seq_pos_indices,
)

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PREPEND_SPACE_TO_ANSWER = False
# %%
model = HookedTransformer.from_pretrained(
    "santacoder",
    torch_dtype=torch.float32,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device=device,
)
assert model.tokenizer is not None
# %%
test_prompt(
    "x = 0\n x += 2\nx -= 1\nprint(x) # ",
    "1",
    model,
    prepend_space_to_answer=PREPEND_SPACE_TO_ANSWER,
)
# %%
test_prompt(
    "x = 1\n x += 2\nx -= 1\nprint(x) # ",
    "2",
    model,
    prepend_space_to_answer=PREPEND_SPACE_TO_ANSWER,
)
# %%
CODE_DATASET = [
    (
        "x = 0\nprint(x) # ",
        "0",
        "x = 1\nprint(x) # ",
        "1",
    ),
    (
        "x = 0\n x += 1\nprint(x) # ",
        "1",
        "x = 1\n x += 1\nprint(x) # ",
        "2",
    ),
    # (
    #     "x = 0\n x += 1\nx += 2\nprint(x) # ",
    #     "3",
    #     "x = 1\n x += 1\nx += 2\nprint(x) # ",
    #     "4",
    # ),
    # (
    #     "x = 0\nprint(x) # 0\n x += 1\nprint(x) # 1\nx *= 2print(x) # 2\nx += 1\nprint(x) # 3\nx -= 2\nprint(x) # ",
    #     "1",
    #     "x = 1\nprint(x) # 1\n x += 1\nprint(x) # 2\nx *= 2print(x) # 4\nx += 1\nprint(x) # 5\nx -= 2\nprint(x) # ",
    #     "3",
    # ),
    (
        "x = 'Hello World'\nprint(x) #",
        " Hello",
        "x = 'Hi Sys'\nprint(x) #",
        " Hi",
    ),
    # (
    #     "x = 'Hello World'\nx = x.upper()\nprint(x) #",
    #     " HEL",
    #     "x = 'Hi Sys'\nx = x.upper()\nprint(x) #",
    #     " H",
    # ),
    (
        "x = 'Hello World'\nx = x.upper()\nx = x.lower\nprint(x) #",
        " hello",
        "x = 'Hi Sys'\nx = x.upper()\nx = x.lower\nprint(x) #",
        " hi",
    ),
    (
        "x = 'Hello World'\nprint(x) # Hello World\nx = x.upper()\nprint(x) #",
        " HEL",
        "x = 'Hi Sys'\nprint(x) # Hi Sys\nx = x.upper()\nprint(x) #",
        " H",
    ),
    (
        "x = 'Hello World'\nprint(x) # Hello World\nx = x.upper()\nprint(x) # HELLO WORLD\nx = x.lower()\nprint(x) #",
        " hello",
        "x = 'Hi Sys'\nprint(x) # Hi Sys\nx = x.upper()\nprint(x) # HI SYS\nx = x.lower()\nprint(x) #",
        " hi",
    ),
    (
        "x = 'Hello World'\nprint(x) # Hello World\nx = x.upper()\nprint(x) # HELLO WORLD\nx = x.lower()\nprint(x) # hello world\nx *= 2\nprint(x) # hello worldhello world\nx = x.split()[0]\nprint(x) #",
        " hello",
        "x = 'Hi Sys'\nprint(x) # Hi Sys\nx = x.upper()\nprint(x) # HI SYS\nx = x.lower()\nprint(x) # hi sys\nx *= 2\nprint(x) # hi syshi sys\nx = x.split()[0]\nprint(x) #",
        " hi",
    ),
    (
        "def print_first_n_even_numbers(n: int) -> None:\n    for num in range(1, n + 1):\n        if num % 2 == ",
        "0",
        "def print_first_n_odd_numbers(n: int) -> None:\n    for num in range(1, n + 1):\n        if num % 2 == ",
        "1",
    ),
    (
        "def print_first_n_factorial_inorder(n: int) -> None:\n    x = 1\n    for num in range(1, n + 1):\n        x = x",
        " *",
        "def print_first_n_triangular_numbers(n: int) -> None:\n    x = 0\n    for num in range(1, n + 1):\n        x = x",
        " +",
    ),
    (
        "def print_first_n_multiples_of_3(n: int) -> None:\n    for num in range(1, n):\n        print(num * ",
        "3",
        "def print_first_n_multiples_of_5(n: int) -> None:\n    for num in range(1, n):\n        print(num * ",
        "5",
    ),
    (
        "def print_first_n_composites(n: int) -> None:\n    for num in range(2, n):\n        if num > 1:\n            for i in range(2, num):\n                if (num % i) == 0:\n                    ",
        " print",
        "def print_first_n_prime_numbers(n: int) -> None:\n    for num in range(2, n):\n        if num > 1:\n            for i in range(2, num):\n                if (num % i) == 0:\n                    ",
        " break",
    ),
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
        "def is_uppercase(string: str) -> bool:\n    # Check if string is in all caps using python's builtin isupper() method\n    return string.is",
        "upper",
        "def is_lowercase(string: str) -> bool:\n    # Check if string is in lower case using python's builtin islower() method\n    return string.is",
        "lower",
    ),
    (
        "def is_right_case(string: str) -> bool:\n    # Check if string is in all caps using python's builtin isupper() method\n    # This function will be useful later\n    return string.is",
        "upper",
        "def is_right_case(string: str) -> bool:\n    # Check if string is in lower case using python's builtin islower() method\n    # This function will be useful later\n    return string.is",
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
prompt_idx = 0
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
    prompt_idx += 2
    if prompt_idx >= 10:
        break
# %%
all_logit_diffs = []
cf_logit_diffs = []
for prompt, answer, cf_prompt, cf_answer in CODE_DATASET:
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
print(f"Original mean: {all_logit_diffs.mean():.2f}")
print(f"Counterfactual mean: {cf_logit_diffs.mean():.2f}")
# %%
assert (all_logit_diffs > 0).all()  # torch.where(all_logit_diffs <= 0)
assert (cf_logit_diffs < 0).all()  # torch.where(cf_logit_diffs >= 0)


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
    model.reset_hooks()
    base_logits, base_cache = model.run_with_cache(
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
    nodes = IterNode(node_names=["resid_pre"], seq_pos="each")
    metric = lambda logits: (
        get_logit_diff(logits, answer_tokens=answer_tokens) - base_ldiff
    ) / (cf_ldiff - base_ldiff)
    assert np.isclose(metric(base_logits).item(), 0.0)
    assert np.isclose(metric(cf_logits).item(), 1.0)
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
    assert not np.isnan(results).any(), (
        f"NaNs in results, " f"for prompt {prompt} " f"got {results}"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=results,
            x=[f"{i}: {t}" for i, t in enumerate(prompt_str_tokens)],
            y=[f"{i}" for i in range(model.cfg.n_layers)],
            colorscale="RdBu",
            zmin=0,
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
for patch_idx, (prompt, answer, cf_prompt, cf_answer) in enumerate(CODE_DATASET):
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    base_logits_by_pos: Float[Tensor, "1 seq_len d_vocab"] = model(
        prompt_tokens,
        prepend_bos=False,
        return_type="logits",
    )
    fig = plot_patch_by_layer(prompt, answer, cf_prompt, cf_answer)
    figs.append(fig)
    if patch_idx >= 4:
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
