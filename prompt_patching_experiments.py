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

from summarization_utils.visualization import get_attn_head_patterns, imshow_p, plot_attention_heads, scatter_attention_and_contribution_simple
from summarization_utils.visualization import get_attn_pattern, plot_attention

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
PROMPTS = [
    ("Known for being the most popular fruit in the world, the", " apple"),
    ("Known for being the most popular vegetable in the world, the", " humble"),
    ("Known for being the most popular car in the world, the", " Volkswagen"),
    ("Known for being the most popular country in the world, the", " United"),
    ("Known for being the most popular city in the world, the", " Big"),
    ("Known for being the most popular animal in the world, the", " dog"),
]

# %%
orig_prompts = [p[0] for p in PROMPTS]
orig_tokens = model.to_tokens(orig_prompts, prepend_bos=True)
flip_tokens = orig_tokens.roll(-1, dims=0)

# %%
batch_size = len(orig_prompts)
n_pairs = 1

def create_patching_dataset(prompts_and_answers):
    orig_prompts = [p[0] for p in prompts_and_answers]
    orig_tokens = model.to_tokens(orig_prompts, prepend_bos=True)
    flip_tokens = orig_tokens.roll(-1, dims=0)
    orig_answers = [p[1] for p in prompts_and_answers]
    flip_answers = orig_answers[1:] + [orig_answers[0]]

    print(orig_tokens.shape)
    print(flip_tokens.shape)
    print(orig_answers)
    print(flip_answers)

    if isinstance(orig_answers[0], List):
        answer_tokens = torch.empty(
                (batch_size, min(n_pairs, len(orig_answers[0])), 2), 
                device=device, 
                dtype=torch.long
            )
    else:
        answer_tokens = torch.empty(
                (batch_size, 1, 2), 
                device=device, 
                dtype=torch.long
            )
        
    for i in range(len(orig_prompts)):
        if isinstance(orig_answers[i], List):
            for pair_idx in range(n_pairs):
                orig_ans_tok = model.to_tokens(orig_answers[i][pair_idx], prepend_bos=False)
                flip_ans_tok = model.to_tokens(flip_answers[i][pair_idx], prepend_bos=False)
                answer_tokens[i, pair_idx, 0] = orig_ans_tok
                answer_tokens[i, pair_idx, 1] = flip_ans_tok
        else:
            orig_ans_tok = model.to_tokens(orig_answers[i], prepend_bos=False)
            flip_ans_tok = model.to_tokens(flip_answers[i], prepend_bos=False)
            answer_tokens[i, 0, 0] = orig_ans_tok
            answer_tokens[i, 0, 1] = flip_ans_tok


    
    orig_tokens = orig_tokens.to(device)
    flip_tokens = flip_tokens.to(device)

    return orig_tokens, flip_tokens, answer_tokens

orig_prompt_toks, flip_prompt_toks, answer_tokens = create_patching_dataset(PROMPTS)

# %% [markdown]
# #### Activation Patching

# %%
for i in range(0, len(orig_prompt_toks)):
    logits, _ = model.run_with_cache(orig_prompt_toks[i])
    log_diff = get_logit_diff(logits, answer_tokens[i].unsqueeze(0))
    #if log_diff < 0.1:
    print(model.to_string(orig_prompt_toks[i]))
    print(model.to_string(flip_prompt_toks[i]))
    print(model.to_str_tokens(answer_tokens[i]))
    print(log_diff, "\n")

# %%
orig_logits, orig_cache = model.run_with_cache(orig_prompt_toks)
orig_logit_diff = get_logit_diff(orig_logits, answer_tokens, per_prompt=False)
orig_logit_diff

# %%
flip_logits, flip_cache = model.run_with_cache(flip_prompt_toks)
flip_logit_diff = get_logit_diff(flip_logits, answer_tokens, per_prompt=False)
flip_logit_diff


# %%
def logit_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs 2"] = answer_tokens,
    flipped_logit_diff: float = flip_logit_diff,
    clean_logit_diff: float = orig_logit_diff,
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
        clean_logit_diff: float = orig_logit_diff,
        corrupted_logit_diff: float = flip_logit_diff,
        answer_tokens: Float[Tensor, "batch n_pairs 2"] = answer_tokens,
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

logit_diff_denoising_tensor = partial(logit_diff_denoising, return_tensor=True)
logit_diff_noising_tensor = partial(logit_diff_noising, return_tensor=True)

# %%
# patching at each (layer, sequence position) for each of (resid_pre, attn_out, mlp_out) in turn

results = act_patch(
    model=model,
    orig_input=flip_prompt_toks,
    new_cache=orig_cache,
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=logit_diff_denoising,
    verbose=True,
)
with open("results/tensors/2_8b_comma_test/content_act_patch_resid_layer_output.pkl", "wb") as f:
    pickle.dump(results, f)

# %%
with open("results/tensors/2_8b_comma_test/content_act_patch_resid_layer_output.pkl", "rb") as f:
    act_patch_resid_layer_output = pickle.load(f)

assert act_patch_resid_layer_output.keys() == {"resid_pre", "attn_out", "mlp_out"}
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(orig_prompt_toks[0]))]
imshow_p(
    torch.stack([r.T for r in act_patch_resid_layer_output.values()]) * 100, # we transpose so layer is on the y-axis
    facet_col=0,
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
# ### Deduction Dataset

# %%
import itertools

# Lists of nouns, categories (both upper and lower case), and adjectives with opposites
nouns = ["Mountain", "Dog", "Computer", "Flower", "Horse", "Whale", "Snake", "Apple"]
categories = ["Animals", "Plants", "Foods", "Places"]

# Adjectives with their clear opposites
adjectives = {
    "fast": "slow",
    "heavy": "light",
    "bright": "dim",
    "old": "new",
    "slow": "fast",
    "light": "heavy",
    "dim": "bright",
    "new": "old",
}

# Function to create the formatted prompt
def create_prompt(noun_upper, noun_lower, category_upper, category_lower, stated_adj, queried_adj, is_true):
    answer = " Yes" if is_true else " No"
    return (f"{noun_upper}s are {category_lower}. {category_upper} are {stated_adj}. Q: Are {noun_lower}s {queried_adj}? A:", f"{answer}")

# Generating prompts based on logical mappings
prompts = []
for noun, category, adjective in itertools.product(nouns, categories, list(adjectives.keys())):
    # True variation
    prompts.append(create_prompt(noun, noun.lower(), category, category.lower(), adjective, adjective, is_true=True))

    # Alt variation
    prompts.append(create_prompt(noun, noun.lower(), category, category.lower(), adjective, adjectives[adjective], is_true=False))

    # Alt variation
    prompts.append(create_prompt(noun, noun.lower(), category, category.lower(), adjectives[adjective], adjective, is_true=False))

    # True variation
    prompts.append(create_prompt(noun, noun.lower(), category, category.lower(), adjectives[adjective], adjectives[adjective], is_true=True))

# Optional: print the prompts
for prompt in prompts[:10]:  # Just printing the first 10 for brevity
    print(prompt)

PROMPTS = prompts


# %%
def flip_list(lst, n):
    # Ensure n is valid
    if n <= 0 or n > len(lst):
        return "Invalid value of n"

    flipped_list = []
    # Process the list in chunks of 2n
    for i in range(0, len(lst), 2 * n):
        chunk = lst[i:i + 2 * n]
        # If the chunk is less than 2n but at least n, flip what we can
        if n <= len(chunk) < 2 * n:
            flipped_list.extend(chunk[n:] + chunk[:n])
        # If the chunk is exactly 2n, flip the two n-length sub-chunks
        elif len(chunk) == 2 * n:
            flipped_list.extend(chunk[n:] + chunk[:n])
        # If the chunk is less than n, just add the remaining elements
        else:
            flipped_list.extend(chunk)

    return flipped_list

n = 2
flipped_list = flip_list(PROMPTS, n)
for prompt in flipped_list[:12]:  # Just printing the first 10 for brevity
    print(prompt)


# %%
def check_words(words: List[str], model: HookedTransformer, is_noun: bool = False):
    for word in words:
        if is_noun:
            word = word + "s"
        print(word)
        print(model.to_str_tokens(model.to_tokens(word, prepend_bos=False)))
        print(f"Sentence case length: {model.to_tokens(word, prepend_bos=False).shape[1]} Lower case length: {model.to_tokens(word.lower(), prepend_bos=False).shape[1]}")

print("\nNouns:")
check_words(nouns, model, is_noun=True)
print("\nCategories:")
check_words(categories, model)
print("\nAdjectives:")
check_words(adjectives.keys(), model)

# %%
len(PROMPTS)

# %%
orig_prompts = [p[0] for p in PROMPTS]
orig_prompt_toks = model.to_tokens(orig_prompts, prepend_bos=True)
flip_prompts = [p[0] for p in flipped_list]
flip_prompt_toks = model.to_tokens(flip_prompts, prepend_bos=True)

answer_tokens = torch.stack([model.to_tokens([PROMPTS[i][1], flipped_list[i][1]], prepend_bos=False) for i in range(len(PROMPTS))])

# %%
answer_tokens = answer_tokens.transpose(1, 2)
answer_tokens.shape

# %%
from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

# Tokenize the prompts and calculate lengths
tokenized_lengths = [len(tokenizer.encode(prompt + answer)) for prompt, answer in PROMPTS]

# Print the tokenized lengths
for i, length in enumerate(tokenized_lengths):
    print(f"Prompt {i+1}: {length} tokens")

    if length > 18:
        for idx, tok in enumerate(model.to_str_tokens(orig_tokens[i])):
            print(idx, tok)

# %%
for i in range(0, len(orig_prompt_toks)):
    logits, _ = model.run_with_cache(orig_prompt_toks[i])
    log_diff = get_logit_diff(logits, answer_tokens[i].unsqueeze(0))
    #if log_diff < 0.1:
    print(model.to_string(orig_prompt_toks[i]))
    print(model.to_string(flip_prompt_toks[i]))
    print(model.to_str_tokens(answer_tokens[i]))
    print(log_diff, "\n")


# %%
def get_logit_diff_in_batches(model, orig_prompt_toks, answer_tokens, batch_size=32):
    orig_logits = []
    for i in range(0, len(orig_prompt_toks), batch_size):
        logits, _ = model.run_with_cache(orig_prompt_toks[i:i+batch_size])
        orig_logits.append(logits)
    orig_logits = torch.cat(orig_logits)
    orig_logit_diff = get_logit_diff(orig_logits, answer_tokens, per_prompt=True)
    return orig_logit_diff

orig_logit_diff = get_logit_diff_in_batches(model, orig_prompt_toks[:64], answer_tokens[:64])
flip_logit_diff = get_logit_diff_in_batches(model, flip_prompt_toks[:64], answer_tokens[:64])

# %%
print(f"Mean original logit diff: {orig_logit_diff.mean()} Accuracy: {(orig_logit_diff > 0).sum()/orig_logit_diff.shape[0]}")
print(f"Mean flipped logit diff: {flip_logit_diff.mean()} Accuracy: {(flip_logit_diff > 0).sum()/flip_logit_diff.shape[0]}")

# %%
orig_prompt_toks_small = orig_prompt_toks[:32]
flip_prompt_toks_small = flip_prompt_toks[:32]
answer_tokens_small = answer_tokens[:32]

# %%
orig_logits, orig_cache = model.run_with_cache(orig_prompt_toks_small)
orig_logit_diff_all = get_logit_diff(orig_logits, answer_tokens_small, per_prompt=True)
orig_logit_diff = orig_logit_diff_all.mean()
print(f"Mean original logit diff: {orig_logit_diff} Accuracy: {(orig_logit_diff_all > 0).sum()/orig_logit_diff_all.shape[0]}")

# %%
flip_logits, flip_cache = model.run_with_cache(flip_prompt_toks_small)
flip_logit_diff_all = get_logit_diff(flip_logits, answer_tokens_small, per_prompt=True)
flip_logit_diff = flip_logit_diff_all.mean()
print(f"Mean flipped logit diff: {flip_logit_diff} Accuracy: {(flip_logit_diff_all > 0).sum()/flip_logit_diff_all.shape[0]}")


# %%
def logit_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs 2"] = answer_tokens_small,
    flipped_logit_diff: float = flip_logit_diff,
    clean_logit_diff: float = orig_logit_diff,
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
        clean_logit_diff: float = orig_logit_diff,
        corrupted_logit_diff: float = flip_logit_diff,
        answer_tokens: Float[Tensor, "batch n_pairs 2"] = answer_tokens_small,
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


# %%
# patching at each (layer, sequence position) for each of (resid_pre, attn_out, mlp_out) in turn

results = act_patch(
    model=model,
    orig_input=flip_prompt_toks_small,
    new_cache=orig_cache,
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=logit_diff_denoising,
    verbose=True,
)
with open("results/tensors/2_8b_comma_test/content_act_patch_resid_layer_output.pkl", "wb") as f:
    pickle.dump(results, f)

# %%
with open("results/tensors/2_8b_comma_test/content_act_patch_resid_layer_output.pkl", "rb") as f:
    act_patch_resid_layer_output = pickle.load(f)

assert act_patch_resid_layer_output.keys() == {"resid_pre", "attn_out", "mlp_out"}
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(orig_prompt_toks[0]))]
imshow_p(
    torch.stack([r.T for r in act_patch_resid_layer_output.values()]) * 100, # we transpose so layer is on the y-axis
    facet_col=0,
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

# %%
