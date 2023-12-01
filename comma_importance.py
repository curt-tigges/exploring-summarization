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
device = torch.device("cuda")
MODEL_NAME = "gpt2-small"
TOKEN = ":"
SPLIT = "train"
DATA_FILES = [
    "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/ArXiv/train/data-00000-of-00222.arrow",
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/BookCorpus2/train/data-00000-of-00096.arrow",
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/Gutenberg%20(PG-19)/train/data-00000-of-00096.arrow",
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/PhilPapers/train/data-00000-of-00096.arrow",
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/PubMed%20Abstracts/train/data-00000-of-00096.arrow",
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/Wikipedia%20(en)/train/data-00000-of-00101.arrow",
]
NAME = DATA_FILES[0].split("/")[-3].replace("%20", " ")
OVERWRITE = False
print(NAME)
# %%
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
# %%
TOKEN_ID = model.to_single_token(TOKEN)
assert isinstance(TOKEN_ID, int)
print(TOKEN_ID)
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
exp_data = PileSplittedData.from_model(
    model,
    name=NAME,
    split=SPLIT,
    data_files=DATA_FILES,
    verbose=True,
)
exp_data.preprocess_datasets(token_to_ablate=TOKEN_ID)
exp_data.dataset_dict
# %%
data_loader = exp_data.get_dataloaders(batch_size=64)[SPLIT]
print(data_loader.name, data_loader.batch_size)
token_mean_values: Float[
    Tensor, "layer d_model"
] = get_layerwise_token_mean_activations(
    model, data_loader, token_id=TOKEN_ID, device=device, overwrite=OVERWRITE
)
# %%
data_loader = exp_data.get_dataloaders(batch_size=32)[SPLIT]
print(data_loader.name, data_loader.batch_size)
losses = compute_ablation_modified_loss(
    model,
    data_loader,
    cached_means=token_mean_values,
    vocab_mask=vocab_mask,
    device=device,
    overwrite=OVERWRITE,
)
losses.shape
# %%
orig_losses = losses[0]
orig_losses_flat = orig_losses.flatten()
orig_losses_flat = orig_losses_flat[orig_losses_flat > 0]
orig_losses_sample = torch.randperm(len(orig_losses_flat))[:1_000]
orig_losses_flat = orig_losses_flat[orig_losses_sample]
fig = px.histogram(
    orig_losses_flat,
    nbins=100,
    marginal="box",
    title=f"Original loss distribution (model: {MODEL_NAME}, data: {data_loader.name}))",
)
fig.show()
# %%
MAX_ORIG_LOSS = 4
orig_loss_filter = orig_losses > MAX_ORIG_LOSS
# %%
loss_mask = mask_positions(
    data_loader,
    model,
    exclude_following_token=TOKEN_ID,
    exclude_regex=exclude_regex,
)
loss_mask |= orig_loss_filter
# %%
ablated_loss_diffs = torch.where(loss_mask, 0, losses[1])
# %%
plot_top_onesided(
    ablated_loss_diffs,
    data_loader,
    model,
    k=50,
    # window_size=30,
    centred=False,
    local=True,
)


# %%
def vectorized_positions_since_last_token(
    batch_tensor: Int[Tensor, "batch pos"], token_id: int
) -> Int[Tensor, "batch pos"]:
    # Create a tensor for the position indices
    positions = (
        torch.arange(batch_tensor.size(1), dtype=torch.long, device=batch_tensor.device)
        .unsqueeze(0)
        .expand_as(batch_tensor)
    )
    # Mask where the token_id appears
    mask = batch_tensor == token_id
    # Use cummax to get the index of the last appearance of token_id
    last_appearance = (mask * positions).cummax(dim=1).values
    # Calculate the distances since the last appearance of token_id
    distances = positions - last_appearance
    # Set the distances to -1 where the token_id has not appeared yet
    distances[~mask.cummax(dim=1).values] = -1
    # Correct distances for positions where token_id appears to 0
    distances[mask] = 0
    return distances


# %%
def compute_distances_since_token(
    token_id: int,
    dataloader: DataLoader,
    model: HookedTransformer,
    device: torch.device,
):
    assert dataloader.batch_size is not None
    distances = torch.zeros(
        (dataloader.dataset.num_rows, model.cfg.n_ctx), dtype=torch.long
    )
    for idx, batch in enumerate(tqdm(dataloader)):
        tokens = batch["tokens"].to(device)
        batch_start = idx * dataloader.batch_size
        batch_end = batch_start + dataloader.batch_size
        distances[batch_start:batch_end] = vectorized_positions_since_last_token(
            tokens, token_id
        )
    return distances


# %%
token_distances = compute_distances_since_token(TOKEN_ID, data_loader, model, device)
print(
    token_distances.shape,
    torch.where(token_distances[0, :100] == 0),
    torch.where(data_loader.dataset[0]["tokens"][:100] == TOKEN_ID),
)

# %%
torch.manual_seed(0)
n_samples = 10_000
max_distance = 10
token_distances_flat = token_distances.flatten()
ablated_loss_diffs_flat = ablated_loss_diffs.flatten()
distance_mask = (0 < token_distances_flat) & (token_distances_flat < max_distance)
token_distances_flat, ablated_loss_diffs_flat = (
    token_distances_flat[distance_mask],
    ablated_loss_diffs_flat[distance_mask],
)
sample_indices = torch.randperm(len(token_distances_flat))[:n_samples]
token_distances_flat, ablated_loss_diffs_flat = (
    token_distances_flat[sample_indices],
    ablated_loss_diffs_flat[sample_indices],
)
# %%
fig = px.box(
    x=token_distances_flat,
    y=ablated_loss_diffs_flat,
    # opacity=0.1,
    # trendline="ols",
    # trendline_color_override="black",
)
fig.update_layout(
    xaxis_title=f"Tokens since last {TOKEN}",
    yaxis_title="Loss difference",
    title=dict(
        text=f"Loss difference vs. tokens since last {TOKEN} (model: {MODEL_NAME}, data: {data_loader.name}))",
        x=0.5,
        xanchor="center",
    ),
)
boxfile = ResultsFile(
    "loss_diff_by_token_distance_boxplot",
    model=MODEL_NAME,
    data=data_loader.name,
    token=TOKEN_ID,
    result_type="plots",
    extension="html",
)
fig.write_html(boxfile.path)
# boxfile.save(fig)
fig.show()
# %%
snippet = " Zeldovich pancakes"
prompt_answer = " pancakes"
instance_index = 1
answer_token = model.to_single_token(prompt_answer)
test_elements = model.to_tokens(snippet, prepend_bos=False)
for batch_idx, batch in enumerate(data_loader):
    tokens = batch["tokens"].to(device)
    if torch.isin(test_elements, tokens).all():
        where = torch.where(tokens == answer_token)
        bat = where[0][instance_index]
        pos = where[1][instance_index]
        prompt_tokens = tokens[bat : bat + 1, :pos]
        prompt = model.to_string(prompt_tokens[0])
        break
else:
    raise ValueError("Snippet not found")
print(orig_losses[batch_idx * data_loader.batch_size + bat, pos])
print(ablated_loss_diffs[batch_idx * data_loader.batch_size + bat, pos])
print(prompt)
# %%
# prompt = """
# ---
# abstract: 'If the large scale structure of the Universe was created, even partially, via Zeldovich pancakes, than the fluctuations of the CMB radiation should be formed due to bulk comptonization of black body spectrum on the contracting pancake. Approximate formulaes for the CMB energy spectrum after bulk comptonization are obtained. The difference between comptonized energy spectra of the CMB due to thermal and bulk comptonozation may be estimated by comparison of the plots for the spectra in these two cases.'
# author:
# - 'G.S. Bisnovatyi-Kogan [^1]'
# title: Spectral distortions in CMB by the bulk Comptonization due to Zeldovich
# """
# prompt_tokens = model.to_tokens(prompt)
top_k = 30
model.reset_hooks()
prompt_mask = prompt_tokens == TOKEN_ID
prompt_str_tokens = model.to_str_tokens(prompt_tokens)
test_prompt(prompt, prompt_answer, model, top_k=top_k, prepend_space_to_answer=False)
# %%
ablation_hook = AblationHook(model, prompt_mask, cached_means=token_mean_values)
with ablation_hook:
    test_prompt(
        prompt, prompt_answer, model, top_k=top_k, prepend_space_to_answer=False
    )
# %%
minimal_mask = torch.zeros_like(prompt_mask)
minimal_mask[:, torch.where(prompt_mask)[-1][-1]] = 1
print(
    torch.where(prompt_mask),
    torch.where(minimal_mask),
    [prompt_str_tokens[i - 1 : i + 1] for i in torch.where(prompt_mask)[-1]],
    [prompt_str_tokens[i - 1 : i + 1] for i in torch.where(minimal_mask)[-1]],
)
# %%
minimal_hook = AblationHook(model, minimal_mask, cached_means=token_mean_values)
with minimal_hook:
    test_prompt(
        prompt, prompt_answer, model, top_k=top_k, prepend_space_to_answer=False
    )
# %%
"""
TODO:
* Debug Zeldovich pancakes snippet
    Ablate non-punctuation tokens? 
    Try zero ablation? 
    Try patching?
* Run on other models
"""

# %%
