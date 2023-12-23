# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

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
import pickle
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union, Literal, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
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
from summarization_utils.path_patching import act_patch, path_patch, Node, IterNode, IterSeqPos
from summarization_utils.tokenwise_ablation import patch_token_values_with_freezing_hooks

from summarization_utils.visualization import get_attn_head_patterns, imshow_p, plot_attention_heads, scatter_attention_and_contribution_simple
from summarization_utils.visualization import get_attn_pattern, plot_attention

from summarization_utils.counterfactual_patching import (
    get_position_dict,
    patch_prompt_by_layer,
    patch_by_position_group,
    patch_by_layer,
    plot_layer_results_per_batch
)

from summarization_utils.toy_datasets import (
    CounterfactualDataset,
    ToyCodeLoopTemplate,
)

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_checkpoint = "santacoder"
#model_checkpoint = "EleutherAI/pythia-2.8b"
# %%
model = HookedTransformer.from_pretrained(
    model_checkpoint,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    device=device,
)
#model = model.to(device)
assert model.tokenizer is not None
# %% [markdown]
# ## Dataset Setup

# %% [markdown]
# ### Old Testing Code

# %%
code_snippets = [
    "def print_first_n_factorial_inorder(n: int) -> None:\n    x = 1\n    for num in range(1, n + 1):\n        x = x * num",
    "def print_first_n_summation(n: int) -> None:\n    x = 0\n    for num in range(1, n + 1):\n        x = x + num",
    "def print_first_n_exponentiation(n: int) -> None:\n    x = 1\n    for num in range(1, n + 1):\n        x = x ** 2",
    "def print_first_n_subtraction(n: int) -> None:\n    x = 0\n    for num in range(1, n + 1):\n        x = x - num",
    "def print_first_n_division(n: int) -> None:\n    x = 1\n    for num in range(1, n + 1):\n        x = x / num"
]

# %%
OP_NAMES = [
    "factorial_inorder",
    "exponentiated_numbers",
    "subtracted_numbers",
    "divided_numbers",
]

# %%
for name in OP_NAMES:
    print(name)
    print(model.to_str_tokens(name, prepend_bos=False))

# %% [markdown]
# ### Create Dataset

# %%
dataset_template = ToyCodeLoopTemplate(model, dataset_size=25, max=25)
dataset = dataset_template.to_counterfactual()

# %%
all_logit_diffs, cf_logit_diffs = dataset.compute_logit_diffs()
print(f"Original mean: {all_logit_diffs.mean():.2f}")
print(f"Counterfactual mean: {cf_logit_diffs.mean():.2f}")

# %%
# for i in range(dataset.prompt_tokens.shape[0]):
#     print((dataset.prompts[i], dataset.cf_prompts[i]))
#     #print((len(dataset.prompt_tokens[i]), len(dataset.cf_tokens[i]), model.to_str_tokens(dataset.prompt_tokens[i]), model.to_str_tokens(dataset.cf_tokens[i])))

# %%
all_logit_diffs.mean(), (all_logit_diffs > 0).sum()/all_logit_diffs.shape[0], all_logit_diffs


# %%
cf_logit_diffs.mean(), (cf_logit_diffs < 0).sum()/cf_logit_diffs.shape[0], cf_logit_diffs

# %%
prompt_logits, prompt_cache = model.run_with_cache(dataset.prompt_tokens)

cf_logits, cf_cache = model.run_with_cache(dataset.cf_tokens)

# %%
model.to_str_tokens(dataset.answer_tokens[0])


# %% [markdown]
# ## Investigating the Circuit

# %%
def logit_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs 2"] = dataset.answer_tokens,
    flipped_logit_diff: float = cf_logit_diffs.mean(),
    clean_logit_diff: float = all_logit_diffs.mean(),
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    ld = ((patched_logit_diff - flipped_logit_diff) / (clean_logit_diff  - flipped_logit_diff))
    if return_tensor:
        return ld
    else:
        return ld.item()


def logit_diff_noising(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float = all_logit_diffs.mean(),
        corrupted_logit_diff: float = cf_logit_diffs.mean(),
        answer_tokens: Float[Tensor, "batch n_pairs 2"] = dataset.answer_tokens,
        return_tensor: bool = False,
    ) -> float:
        '''
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
        and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        '''
        patched_logit_diff = get_logit_diff(logits, answer_tokens)
        ld = ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff))

        if return_tensor:
            return ld
        else:
            return ld.item()


# %% [markdown]
# ### Activation Patching

# %%
results = act_patch(
    model=model,
    orig_input=dataset.cf_tokens,
    new_cache=prompt_cache,
    #patching_nodes=IterNode(["resid_pre"], seq_pos="each"),
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=logit_diff_denoising,
    verbose=True,
)

# %%
with open("results/tensors/santacoder_toycodeloop/act_patch_layer_outputs.pkl", "wb") as f:
    pickle.dump(results, f)

# %%
with open("results/tensors/santacoder_toycodeloop/act_patch_layer_outputs.pkl", "rb") as f:
    act_patch_resid_layer_output = pickle.load(f)

labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(dataset.prompt_tokens[0]))]
imshow_p(
    torch.stack([r.T for r in act_patch_resid_layer_output.values()]) * 100, # we transpose so layer is on the y-axis
    facet_col=0,
    #facet_labels=["resid_pre"],
    facet_labels=["resid_pre", "attn_out", "mlp_out"],
    title="Patching at resid stream & layer outputs (corrupted -> clean)",
    labels={"x": "Sequence position", "y": "Layer", "color": "Logit diff variation"},
    x=labels,
    xaxis_tickangle=45,
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=1500,
    height=600,
    zmin=-50,
    zmax=50,
    margin={"r": 100, "l": 100}
)

# %% [markdown]
# ### Patching Tokens

# %% [markdown]
# #### Phrase and Sum Token

# %%
ablation_pos = torch.tensor([8, 9, 39])

# %%
ablated_logit_diff, ablated_ld_list, clean_ld_list = patch_token_values_with_freezing_hooks(
    model=model,
    dataset=dataset,
    ablation_pos=ablation_pos,
    freeze_pos=torch.tensor([]),
    prompt_cache=prompt_cache,
    cf_cache=cf_cache
)
model.reset_hooks()

print(f"Original logit diff: {get_logit_diff(prompt_logits, dataset.answer_tokens).item():.4f}")
print(f"Post ablation logit diff: {ablated_logit_diff:.4f}")
print(f"Logit diff % change: {(ablated_logit_diff - get_logit_diff(prompt_logits, dataset.answer_tokens).item()) / get_logit_diff(prompt_logits, dataset.answer_tokens).item():.2%}")

# %%
model.reset_hooks()

# %% [markdown]
# #### Phrase Only

# %%
freeze_pos = torch.tensor([39])
ablation_pos = torch.tensor([8, 9])

# %%
ablated_logit_diff, ablated_ld_list, clean_ld_list = patch_token_values_with_freezing_hooks(
    model=model,
    dataset=dataset,
    ablation_pos=ablation_pos,
    freeze_pos=freeze_pos,
    prompt_cache=prompt_cache,
    cf_cache=cf_cache,
    freeze_attn_patterns=True,
    freeze_attn_values=True
)
model.reset_hooks()

print(f"Original logit diff: {get_logit_diff(prompt_logits, dataset.answer_tokens).item():.4f}")
print(f"Post ablation logit diff: {ablated_logit_diff:.4f}")
print(f"Logit diff % change: {(ablated_logit_diff - get_logit_diff(prompt_logits, dataset.answer_tokens).item()) / get_logit_diff(prompt_logits, dataset.answer_tokens).item():.2%}")

# %%
model.reset_hooks()

# %% [markdown]
# #### Sum Token Only

# %%
freeze_pos = torch.tensor([8, 9])
ablation_pos = torch.tensor([39])

# %%
ablated_logit_diff, ablated_ld_list, clean_ld_list = patch_token_values_with_freezing_hooks(
    model=model,
    dataset=dataset,
    ablation_pos=ablation_pos,
    freeze_pos=freeze_pos,
    prompt_cache=prompt_cache,
    cf_cache=cf_cache,
    freeze_attn_patterns=True,
    freeze_attn_values=False
)
model.reset_hooks()

print(f"Original logit diff: {get_logit_diff(prompt_logits, dataset.answer_tokens).item():.4f}")
print(f"Post ablation logit diff: {ablated_logit_diff:.4f}")
print(f"Logit diff % change: {(ablated_logit_diff - get_logit_diff(prompt_logits, dataset.answer_tokens).item()) / get_logit_diff(prompt_logits, dataset.answer_tokens).item():.2%}")

# %%
model.reset_hooks()

# %% [markdown]
# ### Path Patching

# %%
results = path_patch(
    model,
    orig_input=dataset.prompt_tokens,
    new_input=dataset.cf_tokens,
    sender_nodes=IterNode('z'), # This means iterate over all heads in all layers
    receiver_nodes=Node('resid_post', 23), # This is resid_post at layer 31
    patching_metric=logit_diff_noising,
    verbose=True
)
with open("results/tensors/santacoder_toycodeloop/path_patch_resid_post.pkl", "wb") as f:
    pickle.dump(results, f)

# %%
with open("results/tensors/santacoder_toycodeloop/path_patch_resid_post.pkl", "rb") as f:
    path_patch_resid_post = pickle.load(f)
imshow_p(
    path_patch_resid_post['z'] * 100,
    title="Direct effect on logit diff (patch from head output -> final resid)",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=600,
    margin={"r": 100, "l": 100}
)

# %%
