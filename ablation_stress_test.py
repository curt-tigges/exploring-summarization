# %%
import einops
from functools import partial
import numpy as np
import torch
import datasets
import re
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import (
    get_dataset,
    tokenize_and_concatenate,
    get_act_name,
    test_prompt,
)
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd
from circuitsvis.activations import text_neuron_activations
from circuitsvis.topk_samples import topk_samples
from IPython.display import HTML, display
import plotly.express as px
import plotly.graph_objects as go
from utils.circuit_analysis import get_logit_diff
from utils.tokenwise_ablation import (
    compute_ablation_modified_loss,
    load_directions,
    get_random_directions,
    get_zeroed_dir_vector,
    get_layerwise_token_mean_activations,
    AblationHook,
    get_batch_token_mean_activations,
    loss_fn,
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
from utils.store import ResultsFile

# %%
torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODEL_NAME = "gpt2-small"
# %%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
# %%
prompt_file = ResultsFile(
    "prompt",
    model=MODEL_NAME,
    data="armelr___the_pile_splitted_arxiv_train",
    result_type="cache",
    extension="txt",
)
prompt = prompt_file.load()
prompt_tokens = model.to_tokens(prompt, prepend_bos=False)
answer_position = -5
answer = model.to_single_str_token(prompt_tokens[:, answer_position].item())
prompt_tokens = prompt_tokens[:, :answer_position]
prompt = model.to_string(prompt_tokens)
# answer = " pancakes"
answer_id = model.to_single_token(answer)
prompt
# %%
TOKEN = ","
TOKEN_ID = model.to_single_token(TOKEN)
is_token = torch.where(prompt_tokens == TOKEN_ID)[1]
prompt_str_tokens = model.to_str_tokens(prompt_tokens)
top_k = 30
# %%
# ##############################################
# TEST_PROMPT
# ##############################################
# %%
print("No ablation baseline")
model.reset_hooks()
my_test_prompt = partial(
    test_prompt,
    prompt=prompt,
    answer=answer,
    model=model,
    top_k=top_k,
    prepend_space_to_answer=False,
    prepend_bos=False,
)
my_test_prompt()
# %%
random_position = 20
ablation_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
ablation_mask[:, 7] = True
ablation_values = torch.zeros(
    (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
)
ablation_hook = AblationHook(model, ablation_mask, cached_means=ablation_values)
print(f"Random position baseline")
with ablation_hook:
    my_test_prompt()
# %%
print(f"First {TOKEN} zero-ablation (pos {is_token[-1]})")
ablation_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
ablation_mask[:, is_token[0]] = True
ablation_values = torch.zeros(
    (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
)
ablation_hook = AblationHook(model, ablation_mask, cached_means=ablation_values)
with ablation_hook:
    my_test_prompt()
# %%
print(f"Second {TOKEN} zero-ablation (pos {is_token[-1]})")
ablation_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
ablation_mask[:, is_token[1]] = True
ablation_values = torch.zeros(
    (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
)
ablation_hook = AblationHook(model, ablation_mask, cached_means=ablation_values)
with ablation_hook:
    my_test_prompt()
# %%
print(f"Final {TOKEN} zero-ablation (pos {is_token[-1]})")
ablation_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
ablation_mask[:, is_token[-1]] = True
ablation_values = torch.zeros(
    (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
)
ablation_hook = AblationHook(model, ablation_mask, cached_means=ablation_values)
with ablation_hook:
    my_test_prompt()
# # %%
print(f"All {TOKEN} zero-ablation")
ablation_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
ablation_mask[:, is_token] = True
ablation_values = torch.zeros(
    (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
)
ablation_hook = AblationHook(model, ablation_mask, cached_means=ablation_values)
with ablation_hook:
    my_test_prompt()
# %%
print("Ablate random position with random values")
layer_scales = torch.tensor(
    # [
    #     40.7463,
    #     46.7377,
    #     49.6792,
    #     52.0449,
    #     56.4292,
    #     61.0445,
    #     66.7510,
    #     76.7875,
    #     90.8366,
    #     117.7684,
    #     198.9068,
    #     550.9730,
    # ]
    [
        53.04134735,
        62.52054975,
        79.28355995,
        86.63200742,
        92.7816774,
        100.65372648,
        109.84096658,
        125.45092181,
        142.45168816,
        172.55041634,
        239.6323185,
        464.36528878,
    ]
)
ablation_values = torch.randn(
    (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
)
ablation_values /= ablation_values.norm(dim=-1, keepdim=True)
ablation_values *= layer_scales[:, None]
ablation_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
ablation_mask[:, random_position] = True
ablation_hook = AblationHook(model, ablation_mask, cached_means=ablation_values)
with ablation_hook:
    my_test_prompt()
# %%
# ##############################################
# LOSS BY POSITION
# ##############################################
# %%
base_logits: Float[Tensor, "1 seq_len d_vocab"] = model.forward(
    prompt_tokens,
    prepend_bos=False,
    return_type="logits",
)[:, -1, :]
base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
base_loss = -base_log_probs[:, answer_id].item()
# %%
loss_by_pos = []
for pos in range(prompt_tokens.shape[1]):
    ablation_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
    ablation_mask[:, pos] = True
    ablation_values = torch.zeros(
        (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
    )
    ablation_hook = AblationHook(model, ablation_mask, cached_means=ablation_values)
    with ablation_hook:
        logits: Float[Tensor, "1 seq_len d_vocab"] = model.forward(
            prompt_tokens,
            prepend_bos=False,
            return_type="logits",
        )[:, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -log_probs[:, answer_id].item()
    loss_by_pos.append(loss)
print(loss_by_pos)
# %%
loss_chg_by_pos = np.array(loss_by_pos) - base_loss
fig = px.bar(
    y=loss_chg_by_pos,
    title="Loss increase by position",
    labels={"x": "Position", "y": "Loss"},
    # use prompt_str_tokens as x axis labels
    x=[f"{i}: {t}" for i, t in enumerate(prompt_str_tokens)],
)
fig.show()
# %%
# ##############################################
# Reconciling with compute_ablation_modified_loss
# ##############################################
# %%
exclude_regex = [
    r"\]",
    r"\[",
    r"\(",
    r"\)",
    r",",
    r":",
    r";",
    r"`",
    r"'",
    r"\.",
    r"!",
    r"\?",
    r"“",
    r"{",
    r"}",
    r"\{",
    r"\}",
    r"\^",
    r"\\",
    r"/",
    r"^g$",
    r"[0-9]",
    r"=",
    r"^\s+$",
    r"-",
    r"&",
    r"\&",
]
exclude_list = construct_exclude_list(model, exclude_regex)
vocab_mask = torch.ones(model.cfg.d_vocab, dtype=torch.bool, device=device)
vocab_mask[exclude_list] = False
print(len(exclude_list), vocab_mask.sum().item(), len(vocab_mask))
# %%
all_positions = False
layers_to_ablate = "all"
cached_means = torch.zeros((model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32)
ablation_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
ablation_mask[:, is_token] = True
batch_value = dict(
    tokens=torch.cat((prompt_tokens, torch.tensor([[answer_id]])), dim=1),
    positions=torch.ones_like(prompt_tokens) if all_positions else ablation_mask,
    attention_mask=torch.ones_like(prompt_tokens),
)
batch_tokens = batch_value["tokens"].to(device)
experiment_names = [
    "orig",
    "ablated",
]
experiment_index = {name: i for i, name in enumerate(experiment_names)}
# %%
if layers_to_ablate == "all":
    layers_to_ablate = list(range(model.cfg.n_layers))
if all_positions:
    batch_pos = batch_value["attention_mask"].to(device)
else:
    batch_pos = batch_value["positions"].to(device)

# Step 1: original metric without hooks

# get the loss for each token in the batch
orig_loss = loss_fn(
    model,
    batch_tokens,
    vocab_mask,
)
assert isinstance(orig_loss, Tensor)
# concatenate column of 0s
orig_metric = torch.cat(
    [torch.zeros((orig_loss.shape[0], 1)).to(device), orig_loss], dim=1
)

# Step 2: repeat with tokens ablated
ablation_hook = AblationHook(
    model,
    pos_mask=batch_pos,
    layers_to_ablate=layers_to_ablate,
    cached_means=cached_means,
    all_positions=all_positions,
    device=device,
)
with ablation_hook:
    # get the loss for each token when run with hooks
    hooked_loss = loss_fn(model, batch_tokens, vocab_mask)
# concatenate column of 0s
hooked_loss = torch.cat(
    [torch.zeros((hooked_loss.shape[0], 1)).to(device), hooked_loss], dim=1
)
ablated_metric = hooked_loss - orig_metric
print(hooked_loss[:, -1].item(), ablated_metric[:, -1].item())
# %%
# ##############################################
# CHECK RESIDUAL STREAM NORM
# ##############################################
# %%
_, cache = model.run_with_cache(prompt_tokens)
# %%
resid_norms = np.zeros((model.cfg.n_layers, prompt_tokens.shape[1]))
for layer in range(model.cfg.n_layers):
    resid_norms[layer] = (
        cache["resid_post", layer].norm(dim=-1).squeeze(0).cpu().numpy()
    )
# %%
norms_df = pd.DataFrame(
    data=resid_norms.T,
    columns=[f"L{layer}" for layer in range(model.cfg.n_layers)],
)
fig = px.imshow(
    norms_df,
    labels={"x": "Layer", "y": "Position", "color": "Residual stream norm"},
    zmin=0,
    zmax=500,
)
fig.show()
# %%
norms_df.mean(axis=0).values
# %%
"""
TODO:
* DLA of raw answer logit
* Ablation by token (metric = CE loss)
* Setup counterfactual prompt substituting answer with random token
"""

# %%
