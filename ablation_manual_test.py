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
from typing import Dict, Iterable, List, Tuple, Union, Literal, Optional
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
from utils.store import ResultsFile

# %%
torch.set_grad_enabled(False)
torch.manual_seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODEL_NAME = "gpt2-small"
# %%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
# %%
list_of_prompts = [
    (
        "pronoun inference",
        "The neighbor John is celebrating the birthday of Sara, so he bought a cake for",
        " her",
    ),
    (
        "IOI",
        "John and Mary went to the shop, and John decided to leave the decision-making to",
        " Mary",
    ),
    (
        "sentiment inference",
        "I loved the movie for its clever plot and sharp dialogue, it was overall very",
        " entertaining",
    ),
    (
        "induction",
        "After discovering the Summarization Motif, we decided to write a paper titled Exploring The Summarization",
        " Mot",
    ),
    (
        "knowledge extraction",
        "The Eiffel Tower is iconic, famously located in the city of",
        " Paris",
    ),
    (
        "detokenization",
        "As a bilingual educator, Mr. Chen teaches both English and Mandarin, demonstrating his proficiency in both",
        " languages",
    ),
]
# %%
top_k = 5
for _, prompt, answer in list_of_prompts:
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
# ##############################################
# LOSS BY POSITION
# ##############################################
# %%
def plot_loss_by_position(
    prompt_name: str, prompt: str, answer: str, prepend_bos: bool = True
) -> go.Figure:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    answer_id = model.to_single_token(answer)
    prompt_str_tokens = model.to_str_tokens(prompt_tokens, prepend_bos=prepend_bos)
    base_logits: Float[Tensor, "1 seq_len d_vocab"] = model.forward(
        prompt_tokens,
        prepend_bos=False,
        return_type="logits",
    )[:, -1, :]
    base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
    base_loss = -base_log_probs[:, answer_id].item()
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
    loss_chg_by_pos = np.array(loss_by_pos) - base_loss
    fig = px.bar(
        y=loss_chg_by_pos,
        title=f"Loss increase by position: {model.cfg.model_name}, {prompt_name}",
        labels={"x": "Position", "y": "Loss"},
        # use prompt_str_tokens as x axis labels
        x=[f"{i}: {t}" for i, t in enumerate(prompt_str_tokens)],
    )
    fig.update_layout(
        title_x=0.5,
    )
    return fig


# %%
figs = []
for prompt_name, prompt, answer in list_of_prompts:
    fig = plot_loss_by_position(prompt_name, prompt, answer)
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
    yaxis_title="Loss",
    width=800,
    height=400 * len(figs),
)
fig.show()


# %%
# ##############################################
# LOSS BY LAYER
# ##############################################
def plot_loss_by_layer(
    prompt_name: str, prompt: str, answer: str, prepend_bos: bool = True
):
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    answer_id = model.to_single_token(answer)
    prompt_str_tokens = model.to_str_tokens(prompt_tokens, prepend_bos=prepend_bos)
    base_logits: Float[Tensor, "1 seq_len d_vocab"] = model.forward(
        prompt_tokens,
        prepend_bos=False,
        return_type="logits",
    )[:, -1, :]
    base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
    base_loss = -base_log_probs[:, answer_id].item()
    losses = torch.zeros(
        (model.cfg.n_layers, prompt_tokens.shape[1]), dtype=torch.float32
    )
    ablation_values = torch.zeros(
        (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
    )
    for hook in AblationHookIterator(
        model, prompt_tokens, cached_means=ablation_values
    ):
        with hook:
            logits: Float[Tensor, "1 seq_len d_vocab"] = model.forward(
                prompt_tokens,
                prepend_bos=False,
                return_type="logits",
            )[:, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -log_probs[:, answer_id].item()
        losses[hook.layer(), hook.position()] = loss
    loss_chg_by_layer = losses.cpu().numpy() - base_loss
    fig = go.Figure(
        data=go.Heatmap(
            z=loss_chg_by_layer,
            x=[f"{i}: {t}" for i, t in enumerate(prompt_str_tokens)],
            y=[f"{i}" for i in range(model.cfg.n_layers)],
            colorscale="Reds",
            zmin=0,
            zmax=10,
            hovertemplate="Layer %{y}<br>Position %{x}<br>Loss Increase %{z}<extra></extra>",
        ),
        layout=dict(
            title=f"Loss increase by layer: {model.cfg.model_name}, {prompt_name}",
            xaxis_title="Position",
            yaxis_title="Layer",
            title_x=0.5,
        ),
    )
    return fig


# %%
figs = []
for prompt_name, prompt, answer in list_of_prompts:
    fig = plot_loss_by_layer(prompt_name, prompt, answer)
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
    width=1000,
    height=600 * len(figs),
)
fig.show()


# %%
# ##############################################
# DIG INTO EXAMPLES
# ##############################################
# %%
top_k = 10
PROMPT_NAME = "knowledge extraction"
ABLATION_TOKEN = ","
layer = 0
for prompt_name, prompt, answer in list_of_prompts:
    if prompt_name != PROMPT_NAME:
        continue
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
    prompt_tokens = model.to_tokens(prompt, prepend_bos=False)
    prompt_str_tokens = model.to_str_tokens(prompt_tokens, prepend_bos=False)
    ablation_pos = prompt_str_tokens.index(ABLATION_TOKEN)  # type: ignore
    ablation_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
    ablation_mask[:, ablation_pos] = True
    ablation_values = torch.zeros(
        (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
    )
    ablation_hook = AblationHook(
        model, ablation_mask, cached_means=ablation_values, layers_to_ablate=layer
    )
    print("baseline")
    my_test_prompt()
    print(f"Ablating '{ABLATION_TOKEN}'")
    with ablation_hook:
        my_test_prompt()

# %%
"""
TODO:
* Try counterfactual patching?
"""

# %%
