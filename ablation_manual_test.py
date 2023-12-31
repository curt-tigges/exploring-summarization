# %%
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

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.set_grad_enabled(False)
torch.manual_seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODEL_NAME = "gpt2-small"
ABLATION_TYPE = "resample"
ABLATION_TOKEN = ","
ABLATION_SAMPLES = 10
# %%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
# %%
TOKEN_ID = model.to_single_token(ABLATION_TOKEN)
TOKEN_RESIDUAL_FILE = f"token_{TOKEN_ID}_residual_stream_{MODEL_NAME}_owt"
manager = TensorBlockManager(block_size=1024, tensor_prefix=TOKEN_RESIDUAL_FILE)
ABLATION_VALUES = torch.zeros(
    (model.cfg.n_layers, model.cfg.d_model), dtype=torch.float32
)
if ABLATION_TYPE == "zero":
    pass
elif ABLATION_TYPE == "mean":
    for block in tqdm(manager.read_all(), total=len(manager)):
        ABLATION_VALUES += block[:, :-1, :].mean(dim=0)
    ABLATION_VALUES /= len(manager)
elif ABLATION_TYPE == "resample":
    random_blocks = torch.randint(0, len(manager), (ABLATION_SAMPLES,)).tolist()
    random_indices = torch.randint(0, manager.block_size, (ABLATION_SAMPLES,)).tolist()
    ABLATION_VALUES = torch.stack(
        [
            manager.load(block)[index]
            for block, index in zip(random_blocks, random_indices)
        ]
    )
# %%
list_of_prompts = [
    # (
    #     "pronoun inference",
    #     "The neighbor John is celebrating the birthday of Sara, so he bought a cake for",
    #     " her",
    #     "The neighbor Sara is celebrating the birthday of John, so she bought a cake for",
    #     " him",
    # ),
    # (
    #     "IOI",
    #     "When Mary and John went to the store, John gave a drink to",
    #     " Mary",
    #     "When Alice and Bob went to the store, Bob gave a drink to",
    #     " Alice",
    # ),
    # (
    #     "sentiment inference",
    #     "I loved the movie for its clever plot and sharp dialogue, it was overall very",
    #     " entertaining",
    #     "I hated the movie for its boring plot and awkward dialogue, it was overall very",
    #     " boring",
    # ),
    # (
    #     "induction",
    #     "After discovering the Summarization Motif, we decided to write a paper titled Exploring The Summarization",
    #     " Mot",
    #     "After discovering the Summarization Representation, we decided to write a paper titled Exploring The Summarization",
    #     " Represent",
    # ),
    (
        "knowledge extraction",
        "The Eiffel Tower is iconic, famously located in the city of",
        " Paris",
        "The Burj Khalifa is iconic, famously located in the city of",
        " Dubai",
    ),
]
# %%
top_k = 5
for prompt_name, prompt, answer, cf_prompt, cf_answer in list_of_prompts:
    assert len(model.to_str_tokens(prompt)) == len(model.to_str_tokens(cf_prompt)), (
        f"Prompt and counterfactual prompt must have the same length, "
        f"for prompt {prompt_name} "
        f"got {len(model.to_str_tokens(prompt))} and {len(model.to_str_tokens(cf_prompt))}"
    )
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
    my_test_prompt = partial(
        test_prompt,
        prompt=cf_prompt,
        answer=cf_answer,
        model=model,
        top_k=top_k,
        prepend_space_to_answer=False,
        prepend_bos=False,
    )
    my_test_prompt()


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
    n_samples = ABLATION_SAMPLES if ABLATION_TYPE == "resample" else 1
    if ABLATION_TYPE == "zero":
        positions = list(range(len(prompt_str_tokens)))
    else:
        positions = torch.where(prompt_tokens == TOKEN_ID)[1].tolist()
    losses = torch.zeros(
        (n_samples, model.cfg.n_layers, prompt_tokens.shape[1]), dtype=torch.float32
    )
    for hook in AblationHookIterator(
        model=model,
        tokens=prompt_tokens,
        positions=positions,
        ablation_values=ABLATION_VALUES,
    ):
        with hook:
            logits: Float[Tensor, "1 seq_len d_vocab"] = model.forward(
                prompt_tokens,
                prepend_bos=False,
                return_type="logits",
            )[:, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -log_probs[:, answer_id].item()
        losses[hook.sample_index(), hook.layer(), hook.position()] = loss
    losses = losses[..., positions]
    losses = losses.mean(dim=0)
    loss_chg_by_layer = losses.cpu().numpy() - base_loss
    fig = go.Figure(
        data=go.Heatmap(
            z=loss_chg_by_layer,
            x=[f"{i}: {prompt_str_tokens[i]}" for i in positions],
            y=[f"L{i}" for i in range(model.cfg.n_layers)],
            colorscale="Reds",
            zmin=0,
            zmax=10,
            hovertemplate="Layer %{y}<br>Position %{x}<br>Loss Increase %{z}<extra></extra>",
        ),
        layout=dict(
            title=f"Loss increase by layer: {model.cfg.model_name}, {prompt_name}, {ABLATION_TYPE} ablation",
            xaxis_title="Position",
            yaxis_title="Layer",
            title_x=0.5,
        ),
    )
    return fig


# %%
figs = []
for prompt_name, prompt, answer, _, _ in list_of_prompts:
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
# # ##############################################
# # ACTIVATION PATCHING
# # ##############################################
# %%
def plot_patch_by_layer(
    prompt_name: str,
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
        f"for prompt {prompt_name} "
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
            title=f"Patching metric by layer: {model.cfg.model_name}, {prompt_name}",
            xaxis_title="Position",
            yaxis_title="Layer",
            title_x=0.5,
        ),
    )
    return fig


# %%
figs = []
for prompt_name, prompt, answer, cf_prompt, cf_answer in list_of_prompts:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    base_logits_by_pos: Float[Tensor, "1 seq_len d_vocab"] = model(
        prompt_tokens,
        prepend_bos=False,
        return_type="logits",
    )
    # if prompt_name == "sentiment inference":
    #     break
    fig = plot_patch_by_layer(prompt_name, prompt, answer, cf_prompt, cf_answer)
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
# ##############################################
# DIG INTO EXAMPLES
# ##############################################
# %%
top_k = 10
PROMPT_NAME = "knowledge extraction"
ABLATION_TOKEN = ","
layer = 0
for prompt_name, prompt, answer, _, _ in list_of_prompts:
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
        model, ablation_mask, ablation_values=ablation_values, layers_to_ablate=layer
    )
    print("baseline")
    my_test_prompt()
    print(f"Zero Ablating '{ABLATION_TOKEN}'")
    with ablation_hook:
        my_test_prompt()

# %%
