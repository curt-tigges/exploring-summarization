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
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# %%
model = HookedTransformer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=torch.bfloat16,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
model = model.to(device)
assert model.tokenizer is not None
# %%
DATASET = [
    (
        "Anne is quiet. Anne is not young. Anne is sleepy. Anne is smart if she is loud. True or False: is Anne smart?",
        "False",
        "Anne is loud. Anne is not young. Anne is sleepy. Anne is smart if she is loud. True or False: is Anne smart?",
        "True",
    ),
    (
        "Anne is quiet. Anne is not young. Anne is sleepy. Anne is smart if she is old and quiet. True or False: is Anne smart?",
        "True",
        "Anne is loud. Anne is not young. Anne is sleepy. Anne is smart if she is old and quiet. True or False: is Anne smart?",
        "False",
    ),
    (
        "Bob is fast. Bob is not tall. Bob is happy. Bob is strong if he is not fast. True or False: is Bob strong?",
        "False",
        "Bob is slow. Bob is not tall. Bob is happy. Bob is strong if he is not fast. True or False: is Bob strong?",
        "True",
    ),
    (
        "Carol is tall. Carol is fat. Carol is tired. Carol is clever if she is thin and tall. True or False: is Carol clever?",
        "False",
        "Carol is tall. Carol is thin. Carol is tired. Carol is clever if she is thin and tall. True or False: is Carol clever?",
        "True",
    ),
    (
        "David is young. David is not weak. David is sad. David is wise if he is old and sad. True or False: is David wise?",
        "False",
        "David is old. David is not weak. David is sad. David is wise if he is old and sad. True or False: is David wise?",
        "True",
    ),
    (
        "Emma is funny. Emma is not short. Emma is cheerful. Emma is kind if she is not funny. True or False: is Emma kind?",
        "False",
        "Emma is serious. Emma is not short. Emma is cheerful. Emma is kind if she is not funny. True or False: is Emma kind?",
        "True",
    ),
    (
        "Mike is happy and young. Mike is short. Mike is clever if he is young XOR tall. True or False: is Mike clever?",
        "False",
        "Mike is happy and old. Mike is tall. Mike is clever if he is young XOR tall. True or False: is Mike clever?",
        "True",
    ),
    (
        "Sarah is friendly and smart. Sarah is slow. Sarah is wise if she is smart AND fast. True or False: is Sarah wise?",
        "False",
        "Sarah is friendly and smart. Sarah is fast. Sarah is wise if she is smart AND fast. True or False: is Sarah wise?",
        "True",
    ),
    (
        "John is strong and not agile. John is young. John is skilled if he is strong XOR agile. True or False: is John skilled?",
        "True",
        "John is weak and not agile. John is young. John is skilled if he is strong XOR agile. True or False: is John skilled?",
        "False",
    ),
    (
        "Emma is strong and not agile. Emma is young. Emma is skilled if he is strong XOR agile. True or False: is Emma skilled?",
        "True",
        "Emma is strong and not young. Emma is agile. Emma is skilled if he is strong XOR agile. True or False: is Emma skilled?",
        "False",
    ),
    (
        "Lucy is kind and honest. Lucy is not cheerful. Lucy is beloved if she is kind AND cheerful. True or False: is Lucy beloved?",
        "False",
        "Lucy is kind and cheerful. Lucy is not honest. Lucy is beloved if she is kind AND cheerful. True or False: is Lucy beloved?",
        "True",
    ),
    (
        "Tom is brave and dumb. Tom is old. Tom is respected if he is brave XOR wise. True or False: is Tom respected?",
        "True",
        "Tom is brave and wise. Tom is old. Tom is respected if he is brave XOR wise. True or False: is Tom respected?",
        "False",
    ),
    (
        "Emma is calm and thoughtful. Emma is sad. Emma is admired if she is calm AND happy. True or False: is Emma admired?",
        "False",
        "Emma is calm and happy. Emma is thoughtful. Emma is admired if she is calm AND happy. True or False: is Emma admired?",
        "True",
    ),
    # (
    #     "Alex is energetic and impatient. Alex is young. Alex is successful if he is energetic XOR patient. True or False: is Alex successful?",
    #     "True",
    #     "Alex is energetic and patient. Alex is young. Alex is successful if he is energetic XOR patient. True or False: is Alex successful?",
    #     "False",
    # ),
    (
        "Linda is creative and intelligent. Linda is not enthusiastic. Linda is accomplished if she is creative AND enthusiastic. True or False: is Linda accomplished?",
        "False",
        "Linda is creative and enthusiastic. Linda is not intelligent. Linda is accomplished if she is creative AND enthusiastic. True or False: is Linda accomplished?",
        "True",
    ),
    # (
    #     "Peter is friendly and introverted. Peter is smart. Peter is popular if he is friendly XOR outgoing. True or False: is Peter popular?",
    #     "True",
    #     "Peter is friendly and outgoing. Peter is smart. Peter is popular if he is friendly XOR outgoing. True or False: is Peter popular?",
    #     "False",
    # ),
    (
        "Rachel is diligent and focused. Rachel is uncomfortable. Rachel is efficient if she is diligent AND relaxed. True or False: is Rachel efficient?",
        "False",
        "Rachel is diligent and relaxed. Rachel is focused. Rachel is efficient if she is diligent AND relaxed. True or False: is Rachel efficient?",
        "True",
    ),
    # (
    #     "Henry is curious and alert. Henry is playful. Henry is creative if he is curious OR playful. True or False: is Henry creative?",
    #     "True",
    #     "Henry is sleepy and playful. Henry is curious. Henry is creative if he is curious OR playful. True or False: is Henry creative?",
    #     "False",
    # ),
    (
        "Alice is energetic and not friendly. Alice is serious. Alice is sociable if she is friendly OR energetic. True or False: is Alice sociable?",
        "True",
        "Alice is serious and not energetic. Alice is friendly. Alice is sociable if she is friendly OR energetic. True or False: is Alice sociable?",
        "False",
    ),
    (
        "George is tall and strong. George is not fast. George is athletic if he is fast OR strong. True or False: is George athletic?",
        "True",
        "George is short and weak. George is not fast. George is athletic if he is fast OR strong. True or False: is George athletic?",
        "False",
    ),
    (
        "Mia is creative and not organized. Mia is not thoughtful. Mia is resourceful if she is organized OR creative. True or False: is Mia resourceful?",
        "True",
        "Mia is thoughtful and not creative. Mia is not organized. Mia is resourceful if she is organized OR creative. True or False: is Mia resourceful?",
        "False",
    ),
    (
        "Oliver is careful and not friendly. Oliver is not outgoing. Oliver is approachable if he is outgoing OR friendly. True or False: is Oliver approachable?",
        "False",
        "Oliver is friendly and not careful. Oliver is not outgoing. Oliver is approachable if he is outgoing OR friendly. True or False: is Oliver approachable?",
        "True",
    ),
    # (
    #     "Sophie is patient and not calm. Sophie is kind. Sophie is reassuring if she is calm OR patient. True or False: is Sophie reassuring?",
    #     "True",
    #     "Sophie is impatient and not calm. Sophie is kind. Sophie is reassuring if she is calm OR patient. True or False: is Sophie reassuring?",
    #     "False",
    # ),
    (
        "Sam is witty and not dumb. Sam is not humorous. Sam is charming if he is humorous OR witty. True or False: is Sam charming?",
        "True",
        "Sam is dumb and not humorous. Sam is not witty. Sam is charming if he is humorous OR witty. True or False: is Sam charming?",
        "False",
    ),
    # (
    #     "Jane is pretty and enthusiastic. Jane is not focused. Jane is productive if she is focused OR enthusiastic. True or False: is Jane productive?",
    #     "True",
    #     "Jane is pretty and distracted. Jane is not enthusiastic. Jane is productive if she is focused OR enthusiastic. True or False: is Jane productive?",
    #     "False",
    # ),
    (
        "Josh is risky and not strong. Josh is cautious. Josh is daring if he is strong OR risky. True or False: is Josh daring?",
        "True",
        "Josh is weak and not risky. Josh is cautious. Josh is daring if he is strong OR risky. True or False: is Josh daring?",
        "False",
    ),
    # (
    #     "Grace is cheerful and optimistic. Grace is not lively. Grace is pleasant if she is lively OR cheerful. True or False: is Grace pleasant?",
    #     "True",
    #     "Grace is sad and lifeless. Grace is not cheerful. Grace is pleasant if she is lively OR cheerful. True or False: is Grace pleasant?",
    #     "False",
    # ),
]
# %%
for prompt, answer, cf_prompt, cf_answer in DATASET:
    prompt_str_tokens = model.to_str_tokens(prompt)
    cf_str_tokens = model.to_str_tokens(cf_prompt)
    assert len(prompt_str_tokens) == len(cf_str_tokens), (
        f"Prompt and counterfactual prompt must have the same length, "
        f"for prompt \n{prompt_str_tokens} \n and counterfactual\n{cf_str_tokens} \n"
        f"got {len(prompt_str_tokens)} and {len(cf_str_tokens)}"
    )
# %%
test_prompt(
    "[INST] Anne is not loud. Anne is not young. Anne is sleepy. Anne is smart if she is loud. True or False: is Anne smart? [/INST]",
    "False",
    model,
    top_k=10,
    prepend_space_to_answer=False,
)
# %%
for test_idx in range(1):
    test_prompt(
        "[INST] " + DATASET[test_idx][0] + " [/INST]",
        DATASET[test_idx][1],
        model,
        top_k=10,
        prepend_space_to_answer=False,
    )
# %%
model.tokenizer.padding_side = "left"
all_tokens = model.to_tokens([d[0] for d in DATASET], prepend_bos=True)
attention_mask = get_attention_mask(model.tokenizer, all_tokens, prepend_bos=False)
answer_tokens = torch.tensor(
    [(model.to_single_token(d[1]), model.to_single_token(d[3])) for d in DATASET],
    device=device,
    dtype=torch.int64,
)
cf_tokens = model.to_tokens([d[2] for d in DATASET], prepend_bos=True)
assert all_tokens.shape == cf_tokens.shape
assert (all_tokens == model.tokenizer.pad_token_id).sum() == (
    cf_tokens == model.tokenizer.pad_token_id
).sum()
print(all_tokens.shape, answer_tokens.shape)
# %%
all_logits: Float[Tensor, "batch pos d_vocab"] = model(
    all_tokens, prepend_bos=False, return_type="logits", attention_mask=attention_mask
)
all_logit_diffs = get_logit_diff(
    all_logits, answer_tokens=answer_tokens, per_prompt=True
)
print(all_logit_diffs)
# %%
cf_logits: Float[Tensor, "batch pos d_vocab"] = model(
    cf_tokens, prepend_bos=False, return_type="logits", attention_mask=attention_mask
)
cf_logit_diffs = get_logit_diff(cf_logits, answer_tokens=answer_tokens, per_prompt=True)
print(cf_logit_diffs)
# %%
print(f"Original mean logit diff: {all_logit_diffs.mean():.2f}")
print(f"Counterfactual mean logit diff: {cf_logit_diffs.mean():.2f}")
# %%
print(f"Original accuracy: {(all_logit_diffs > 0).float().mean():.2f}")
print(f"Counterfactual accuracy: {(cf_logit_diffs < 0).float().mean():.2f}")
# %%
"""
TODO:
* Ensure cf prompts are same length
* Compute accuracy on original and flipped prompts
* Perform activation patching on periods and visualise results
"""


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
