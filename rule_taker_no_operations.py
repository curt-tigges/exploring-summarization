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
    "pythia-1.4b",
    torch_dtype=torch.bfloat16,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
model = model.to(device)
assert model.tokenizer is not None
# %%
test_prompt(
    "Here are examples of red fruit: apple, cherry, strawberry,", " raspberry", model
)
# %%
PREFIX = (
    # "Question: Anne is quiet. Anne is young. Anne is asleep. Is Anne loud? Answer: No\n"
    # "Question: Anne is quiet. Anne is young. Anne is asleep. Is Anne quiet? Answer: Yes\n"
    # "Question: Anne is quiet. Anne is young. Anne is asleep. Is Anne young? Answer: Yes\n"
    # "Question: Anne is quiet. Anne is young. Anne is asleep. Is Anne old? Answer: No\n"
    # "Question: Anne is quiet. Anne is young. Anne is asleep. Is Anne awake? Answer: No\n"
    # "Question: Anne is quiet. Anne is young. Anne is asleep. Is Anne asleep? Answer: Yes\n"
    "<|SYSTEM|>Answer questions with Yes/No.<|USER|>Question: "
)
SUFFIX = " Answer (Yes/No): <|ASSISTANT|>"
PREPEND_SPACE_TO_ANSWER = False
PROMPT_TEMPLATE = (
    "{NAME} is {ATTR1}. {NAME} is {ATTR2}. {NAME} is {ATTR3}. Is {NAME} {ATTR_R}?"
)
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
    "Mia",
    "Tom",
    "Rachel",
    "Henry",
    "Alice",
    "George",
]
POSITIVE_ATTRIBUTES = [
    "loud",
    "fast",
    "tall",
    "fat",
    "young",
    "strong",
    "smart",
    "happy",
    "kind",
    "funny",
    "curious",
    "calm",
    "pretty",
]
NEGATIVE_ATTRIBUTES = [
    "quiet",
    "slow",
    "short",
    "thin",
    "old",
    "weak",
    "dumb",
    "sad",
    "mean",
    "serious",
    "dull",
    "nervous",
    "ugly",
]


# %%
def get_attribute_sign_and_index(attr: str) -> Tuple[bool, int]:
    if attr in POSITIVE_ATTRIBUTES:
        return True, POSITIVE_ATTRIBUTES.index(attr)
    elif attr in NEGATIVE_ATTRIBUTES:
        return False, NEGATIVE_ATTRIBUTES.index(attr)
    else:
        raise ValueError(f"Unknown attribute {attr}")


# %%
def get_answers_for_prompt_tuples(
    prompt_tuples: List[Tuple[str, str, str, str, str]]
) -> List[str]:
    answers = []
    for _, attr1, attr2, attr3, attr_r in prompt_tuples:
        attr1_sign, attr1_idx = get_attribute_sign_and_index(attr1)
        attr2_sign, attr2_idx = get_attribute_sign_and_index(attr2)
        attr3_sign, attr3_idx = get_attribute_sign_and_index(attr3)
        _, attr_r_idx = get_attribute_sign_and_index(attr_r)
        if attr_r_idx == attr2_idx:
            answer = attr2_sign
        elif attr_r_idx == attr1_idx:
            answer = attr1_sign
        elif attr_r_idx == attr3_idx:
            answer = attr3_sign
        else:
            raise ValueError(f"Unknown attribute {attr_r}")
        answers.append("Yes" if answer else "No")
    return answers


# %%
def get_counterfactual_tuples(
    prompt_tuples: List[Tuple[str, str, str, str, str]], seed: int = 0
) -> List[Tuple[str, str, str, str, str]]:
    random.seed(seed)
    cf_tuples = []
    for name, attr1, attr2, attr3, attr_r in prompt_tuples:
        idx_to_change = random.choice([0, 1, 2])
        attr_sign, attr_idx = get_attribute_sign_and_index(
            [attr1, attr2, attr3][idx_to_change]
        )
        cf_attr = (
            POSITIVE_ATTRIBUTES[attr_idx]
            if not attr_sign
            else NEGATIVE_ATTRIBUTES[attr_idx]
        )
        cf_attr1, cf_attr2, cf_attr3 = (
            cf_attr if idx_to_change == 0 else attr1,
            cf_attr if idx_to_change == 1 else attr2,
            cf_attr if idx_to_change == 2 else attr3,
        )
        cf_tuples.append((name, cf_attr1, cf_attr2, cf_attr3, attr_r))
    return cf_tuples


# %%
PROMPT_TUPLES = [
    (
        name,
        attr1_list[attr1_idx],
        attr2_list[attr2_idx],
        attr3_list[attr3_idx],
        attr_r,
    )
    for name in NAMES
    for attr1_idx, attr2_idx, attr3_idx in itertools.combinations(
        range(len(POSITIVE_ATTRIBUTES)), 3
    )
    for attr1_list in [POSITIVE_ATTRIBUTES, NEGATIVE_ATTRIBUTES]
    for attr2_list in [POSITIVE_ATTRIBUTES, NEGATIVE_ATTRIBUTES]
    for attr3_list in [POSITIVE_ATTRIBUTES, NEGATIVE_ATTRIBUTES]
    for attr_r in [
        POSITIVE_ATTRIBUTES[attr1_idx],
        POSITIVE_ATTRIBUTES[attr2_idx],
        POSITIVE_ATTRIBUTES[attr3_idx],
    ]
]
random.shuffle(PROMPT_TUPLES)
PROMPT_TUPLES = PROMPT_TUPLES[:1000]
PROMPTS = [
    PROMPT_TEMPLATE.format(
        NAME=name,
        ATTR1=attr1,
        ATTR2=attr2,
        ATTR3=attr3,
        ATTR_R=attr_r,
    )
    for name, attr1, attr2, attr3, attr_r in PROMPT_TUPLES
]
CF_TUPLES = get_counterfactual_tuples(PROMPT_TUPLES)
CF_PROMPTS = [
    PROMPT_TEMPLATE.format(
        NAME=name,
        ATTR1=attr1,
        ATTR2=attr2,
        ATTR3=attr3,
        ATTR_R=attr_r,
    )
    for name, attr1, attr2, attr3, attr_r in CF_TUPLES
]
ANSWERS = get_answers_for_prompt_tuples(PROMPT_TUPLES)
CF_ANSWERS = get_answers_for_prompt_tuples(CF_TUPLES)
to_keep = [answer != cf_answer for answer, cf_answer in zip(ANSWERS, CF_ANSWERS)]
PROMPTS = [p for p, keep in zip(PROMPTS, to_keep) if keep]
CF_PROMPTS = [p for p, keep in zip(CF_PROMPTS, to_keep) if keep]
ANSWERS = [a for a, keep in zip(ANSWERS, to_keep) if keep]
CF_ANSWERS = [a for a, keep in zip(CF_ANSWERS, to_keep) if keep]
PROMPTS = PROMPTS[:100]
CF_PROMPTS = CF_PROMPTS[:100]
ANSWERS = ANSWERS[:100]


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
        PREFIX + prompt + SUFFIX,
        answer,
        model,
        top_k=10,
        prepend_space_to_answer=PREPEND_SPACE_TO_ANSWER,
    )
    print(cf_prompt)
    test_prompt(
        PREFIX + cf_prompt + SUFFIX,
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
print(f"Original accuracy: {(all_logit_diffs > 0).float().mean():.2f}")
print(f"Counterfactual accuracy: {(cf_logit_diffs < 0).float().mean():.2f}")


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
            title=f"Patching metric by layer, {model.cfg.model_name}",
            xaxis_title="Position",
            yaxis_title="Layer",
            title_x=0.5,
        ),
    )
    return fig


# %%
figs = []
patch_idx = 0
for prompt, answer, cf_prompt, cf_answer in zip(
    PROMPTS, ANSWERS, CF_PROMPTS, CF_ANSWERS
):
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    base_logits_by_pos: Float[Tensor, "1 seq_len d_vocab"] = model(
        prompt_tokens,
        prepend_bos=False,
        return_type="logits",
    )
    # if prompt_name == "sentiment inference":
    #     break
    fig = plot_patch_by_layer(prompt, answer, cf_prompt, cf_answer)
    figs.append(fig)
    patch_idx += 1
    if patch_idx > 5:
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
