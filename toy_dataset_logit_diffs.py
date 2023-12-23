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
from summarization_utils.path_patching import act_patch, Node, IterNode, IterSeqPos

from summarization_utils.visualization import get_attn_head_patterns, imshow_p, plot_attention_heads, scatter_attention_and_contribution_simple
from summarization_utils.visualization import get_attn_pattern, plot_attention

from summarization_utils.toy_datasets import (
    CounterfactualDataset,
    ToyDeductionTemplate,
    ToyBindingTemplate,
    ToyProfilesTemplate,
    get_position_dict,
)

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_checkpoint = "EleutherAI/pythia-2.8b"
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
# ### Knowledge Dataset

# %%
dataset_template = ToyDeductionTemplate(model, dataset_size=10, max=10)
dataset = dataset_template.to_counterfactual()
dataset.check_lengths_match()

# %%
all_logit_diffs, cf_logit_diffs = dataset.compute_logit_diffs()
print(f"Original mean: {all_logit_diffs.mean():.2f}")
print(f"Counterfactual mean: {cf_logit_diffs.mean():.2f}")

# %%
for i in range(dataset.prompt_tokens.shape[0]):
    print((len(dataset.prompt_tokens[i]), len(dataset.cf_tokens[i]), model.to_str_tokens(dataset.prompt_tokens[i]), model.to_str_tokens(dataset.cf_tokens[i])))

# %%
GROUPS_DRAFT = [
        "human",
        "dog",
        "bird",
        "skunk",
        "badger",
        "bear",
        "lion",
        "tiger",
        "wolf",
        "plant",
        "flower",
        "doctor",
        "teacher",
        "scientist",
        "engineer",
        "writer",
        "artist",
    ]


# %%
for name in GROUPS_DRAFT:
    print(name)
    print(model.to_str_tokens(' ' + name, prepend_bos=False))
    print(model.to_str_tokens(' ' + name.capitalize()+'s', prepend_bos=False))


# %%
all_logit_diffs.mean(), all_logit_diffs


# %%
orig_logits, orig_cache = model.run_with_cache(dataset.prompt_tokens)
orig_logit_diff = get_logit_diff(orig_logits, dataset.answer_tokens, per_prompt=True)
orig_logit_diff.mean(), orig_logit_diff

# %%
cf_logit_diffs.mean(), cf_logit_diffs

# %%
flip_logits, flip_cache = model.run_with_cache(dataset.cf_tokens)
flip_logit_diff = get_logit_diff(flip_logits, dataset.answer_tokens, per_prompt=True)
flip_logit_diff.mean(), flip_logit_diff

# %%
test_prompt(model.to_string(dataset.prompt_tokens[0]), model.to_string(dataset.answer_tokens[0][0]), model, top_k=10)

# %%
