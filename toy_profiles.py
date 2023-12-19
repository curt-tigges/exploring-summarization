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
    "pythia-2.8b",
    torch_dtype=torch.float32,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
model = model.to(device)
assert model.tokenizer is not None
# %%
PREPEND_SPACE_TO_ANSWER = False
PROMPT_TEMPLATE = "Profile: {NAME} was born in {CITY}. {NAME} works as a {JOB}. {QUERY}: {NAME} {CONJ}"
QUERIES = [
    "Nationality",
    "Occupation",
]
CONJUGATIONS = [
    "is from the country of",
    "is a",
]
QUERY_TO_CONJ = dict(zip(QUERIES, CONJUGATIONS))
NAMES = [
    "Anne",
    "Bob",
    "Carol",
    "David",
    "Emma",
    "Mike",
    "Sarah",
    "John",
    "Linda",
    "Peter",
    "Grace",
    "Oliver",
    "Sophie",
    "Josh",
    "Tom",
    "Rachel",
    "Henry",
    "Alice",
    "George",
]
CITIES = [
    "Paris",
    "London",
    "Berlin",
    "Tokyo",
    "Moscow",
    "Beijing",
    "Sydney",
    "Madrid",
    "Rome",
]
COUNTRIES = [
    "France",
    "England",
    "Germany",
    "Japan",
    "Russia",
    "China",
    "Australia",
    "Spain",
    "Italy",
]
CITY_TO_NATIONALITY = dict(zip(CITIES, COUNTRIES))
JOBS = [
    "teacher",
    "doctor",
    "lawyer",
    "scientist",
    "writer",
    "singer",
    "dancer",
    "dentist",
    "pilot",
]


# %%
def get_counterfactual_tuples(
    prompt_tuples: List[Tuple[str, str, str, str]]
) -> List[Tuple[str, str, str, str]]:
    cf_tuples = []
    for _, (name, city, job, query) in enumerate(prompt_tuples):
        if query == "Nationality":
            new_city = random.choice([c for c in CITIES if c != city])
            cf_tuples.append((name, new_city, job, query))
        elif query == "Occupation":
            new_job = random.choice([j for j in JOBS if j != job])
            cf_tuples.append((name, city, new_job, query))
        else:
            raise ValueError(f"Unknown query {query}")
    return cf_tuples


# %%
def get_answers(prompt_tuples: List[Tuple[str, str, str, str]]) -> List[str]:
    answers = []
    for _, city, job, query in prompt_tuples:
        if query == "Nationality":
            answers.append(" " + CITY_TO_NATIONALITY[city])
        elif query == "Occupation":
            answers.append(" " + job)
    return answers


# %%
PROMPT_TUPLES = list(itertools.product(NAMES, CITIES, JOBS, QUERIES))
random.shuffle(PROMPT_TUPLES)
PROMPT_TUPLES = PROMPT_TUPLES[:1000]
PROMPTS = [
    PROMPT_TEMPLATE.format(
        NAME=name,
        CITY=city,
        JOB=job,
        QUERY=query,
        CONJ=QUERY_TO_CONJ[query],
    )
    for name, city, job, query in PROMPT_TUPLES
]
CF_TUPLES = get_counterfactual_tuples(PROMPT_TUPLES)
CF_PROMPTS = [
    PROMPT_TEMPLATE.format(
        NAME=name,
        CITY=city,
        JOB=job,
        QUERY=query,
        CONJ=QUERY_TO_CONJ[query],
    )
    for name, city, job, query in CF_TUPLES
]
ANSWERS = get_answers(PROMPT_TUPLES)
CF_ANSWERS = get_answers(CF_TUPLES)
PROMPTS = PROMPTS[:100]
CF_PROMPTS = CF_PROMPTS[:100]
ANSWERS = ANSWERS[:100]
CF_ANSWERS = CF_ANSWERS[:100]
# %%
for prompt, cf_prompt in zip(PROMPTS, CF_PROMPTS):
    prompt_str_tokens = model.to_str_tokens(prompt)
    cf_str_tokens = model.to_str_tokens(cf_prompt)
    assert len(prompt_str_tokens) == len(cf_str_tokens), (
        f"Prompt and counterfactual prompt must have the same length, "
        f"for prompt \n{prompt_str_tokens} \n and counterfactual\n{cf_str_tokens} \n"
        f"got {len(prompt_str_tokens)} and {len(cf_str_tokens)}"
    )
# %%
i = 0
for prompt, answer, cf_prompt, cf_answer in zip(
    PROMPTS, ANSWERS, CF_PROMPTS, CF_ANSWERS
):
    print(prompt)
    test_prompt(
        prompt,
        answer,
        model,
        top_k=10,
        prepend_space_to_answer=PREPEND_SPACE_TO_ANSWER,
    )
    print(cf_prompt)
    test_prompt(
        cf_prompt,
        cf_answer,
        model,
        top_k=10,
        prepend_space_to_answer=PREPEND_SPACE_TO_ANSWER,
    )
    i += 2
    if i > 10:
        break
# %%
all_logit_diffs = []
cf_logit_diffs = []
for prompt, answer, cf_prompt, cf_answer in zip(
    PROMPTS, ANSWERS, CF_PROMPTS, CF_ANSWERS
):
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    cf_tokens = model.to_tokens(cf_prompt, prepend_bos=True)
    answer_id = model.to_single_token(answer)
    cf_answer_id = model.to_single_token(cf_answer)
    answer_tokens = torch.tensor(
        [answer_id, cf_answer_id], dtype=torch.int64, device=device
    ).unsqueeze(0)
    assert prompt_tokens.shape == cf_tokens.shape, (
        f"Prompt and counterfactual prompt must have the same shape, "
        f"for prompt {prompt} counterfactual prompt {cf_prompt} "
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
print(f"Original accuracy: {(all_logit_diffs > 0).float().mean():.2f}")
print(f"Counterfactual accuracy: {(cf_logit_diffs < 0).float().mean():.2f}")


# %%
# # ##############################################
# # ACTIVATION PATCHING
# # ##############################################
# %%
def patch_by_layer(
    prompt: str,
    answer: str,
    cf_prompt: str,
    cf_answer: str,
    prepend_bos: bool = True,
) -> Tuple[List[str], Float[np.ndarray, "n_layers seq_len"]]:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    prompt_str_tokens = model.to_str_tokens(prompt_tokens, prepend_bos=prepend_bos)
    cf_tokens = model.to_tokens(cf_prompt, prepend_bos=prepend_bos)
    answer_id = model.to_single_token(answer)
    cf_answer_id = model.to_single_token(cf_answer)
    assert answer_id != cf_answer_id, (
        f"Answer and counterfactual answer must be different, "
        f"for prompt {prompt} counterfactual prompt {cf_prompt}"
        f"got {answer_id} and {cf_answer_id}"
    )
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
    assert (
        cf_ldiff < base_ldiff
    ), f"Counterfactual logit diff {cf_ldiff} must be less than base logit diff {base_ldiff}"
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
    assert not np.isnan(results).any(), f"NaNs in results: {results}"
    return prompt_str_tokens, results


# %%
prompts = []
data = []
for patch_idx, (prompt, answer, cf_prompt, cf_answer) in enumerate(
    zip(PROMPTS, ANSWERS, CF_PROMPTS, CF_ANSWERS)
):
    # if patch_idx != 4:
    #     continue
    if patch_idx >= 5:
        break
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    base_logits_by_pos: Float[Tensor, "1 seq_len d_vocab"] = model(
        prompt_tokens,
        prepend_bos=False,
        return_type="logits",
    )
    prompt_str_tokens, results = patch_by_layer(prompt, answer, cf_prompt, cf_answer)
    data.append(results)
    prompts.append(prompt_str_tokens)
# %%
figs = []
for prompt, r in zip(prompts, data):
    fig = go.Figure(
        data=go.Heatmap(
            z=r,
            x=[f"{i}: {t}" for i, t in enumerate(prompt)],
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
