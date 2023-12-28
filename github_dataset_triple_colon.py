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
from summarization_utils.toy_datasets import (
    CounterfactualDataset,
    ToyDeductionTemplate,
    ToyBindingTemplate,
    ToyProfilesTemplate,
)
from summarization_utils.counterfactual_patching import (
    patch_by_position_group,
    patch_by_layer,
    plot_layer_results_per_batch,
)

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    "santacoder",
    torch_dtype=torch.float32,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device=device,
)
assert model.tokenizer is not None
# %%
DATASET = [
    (
        "for i in range(0, len(data_list)):\n   ",
        " data",
        "for j in range(len(im2.imvec)):\n   ",
        " im",
        "for i in range(len(fig.layout.annotations)):\n   ",
        " fig",
    ),
    (
        "for target_idx in range(1, len(arr)):\n   ",
        " target",
        "for i in range(len(records)):\n    qid = int(",
        " records",
        "for i in range(len(U)):\n    color_map[",
        "U",
    ),
    (
        "for i in range(1, rgsize + 1):\n   ",
        " rg",
        "for idx, xml_file_name in enumerate(files):\n   ",
        " xml",
        "for n, d in G.nodes(data=True):\n   ",
        " G",
    ),
    (
        "for r in pool.imap_unordered(fetch, requested):\n   ",
        " r",
        "for i in xrange(-10, 10):\n    if",
        " i",
        "for x in range(1, 101):\n    if",
        " x",
    ),
    (
        "for i, frame in enumerate(iter_frames(im)):\n    if",
        " i",
        "for i in range(0, len(jsonButtonData)):\n    if",
        " json",
        "for factor in range(x, 0, -1):\n    if",
        " x",
    ),  # 16
    (
        "for i in range(len(model_infos)):\n   ",
        " model",
        "for i in range(0, len(requested)):\n   ",
        " requested",
        "for i in range(1,len(y)):\n   ",
        " y",
    ),
    (
        "for i in range(len(short_answers)):\n   ",
        " short",
        "for i in range(len(split_code)):\n   ",
        " split",
        "for i in range(len(text_to)):\n   ",
        " text",
    ),
    (
        "for i in range(len(features)):\n   tr[",
        "features",
        "for i in range(len(vocab)):\n    word =",
        " vocab",
        "for i in range(0,numRuns):\n    #",
        "i",
    ),
    (
        "for button in (left, right, forward, backward):\n   ",
        " button",
        "for iidx, video_name in enumerate(videos):\n   ",
        " video",
        "for test in sorted(all_doctests):    if",
        " test",
    ),
    (
        "for i in xrange(30000):\n    if",
        " i",
        "for t in (list, dict, set):\n    d[",
        "t",
        "for value in range(1,11):\n    square =",
        " value",
    ),
]
DATASET_NAME = "github"
# %%
for p1, a1, p2, a2, p3, a3 in DATASET:
    assert len(model.to_str_tokens(p1)) == len(model.to_str_tokens(p2))
    assert len(model.to_str_tokens(p1)) == len(model.to_str_tokens(p3))
# %%
for batch, (p1, a1, p2, a2, p3, a3) in enumerate(DATASET):
    test_prompt(p1, a1, model, prepend_space_to_answer=False)
    test_prompt(p2, a2, model, prepend_space_to_answer=False)
    test_prompt(p3, a3, model, prepend_space_to_answer=False)
    if batch > 1:
        break
# %%
logit_diffs = []
for i, (p1, a1, p2, a2, p3, a3) in enumerate(DATASET):
    orig_input = model.to_tokens(p1, prepend_bos=True)
    orig_logits = model(orig_input, return_type="logits")
    for cf in (0, 1):
        cf_logits = model(p2 if cf == 0 else p3, return_type="logits", prepend_bos=True)
        answer_tokens = torch.tensor(
            [
                model.to_single_token(a1),
                model.to_single_token(a2 if cf == 0 else a3),
            ],
            device=device,
        ).unsqueeze(0)
        cf_answer_tokens = torch.tensor(
            [
                model.to_single_token(a2 if cf == 0 else a3),
                model.to_single_token(a1),
            ],
            device=device,
        ).unsqueeze(0)
        orig_logit_diff = get_logit_diff(orig_logits, answer_tokens=answer_tokens)
        cf_logit_diff = get_logit_diff(cf_logits, answer_tokens=cf_answer_tokens)
        logit_diffs.append(orig_logit_diff)
        logit_diffs.append(cf_logit_diff)
        # print(orig_logits[-1, -1, model.to_single_token(a1)])
        # print(cf_logits[-1, -1, model.to_single_token(a1)])
        # print(orig_logits[-1, -1, model.to_single_token(a2 if cf == 0 else a3)])
        # print(cf_logits[-1, -1, model.to_single_token(a2 if cf == 0 else a3)])
logit_diffs = torch.stack(logit_diffs, dim=0)
print(logit_diffs.mean().item(), (logit_diffs > 0).float().mean().item())


# %%
def triple_metric_base(
    logits: Float[Tensor, "batch seq vocab"],
    model: HookedTransformer,
    a1: str,
    a2: str,
    a3: str,
    orig12: Optional[Float[Tensor, ""]] = None,
    orig13: Optional[Float[Tensor, ""]] = None,
    new12: Optional[Float[Tensor, ""]] = None,
    new13: Optional[Float[Tensor, ""]] = None,
) -> Float[Tensor, "2"]:
    answers12 = torch.tensor(
        [model.to_single_token(a1), model.to_single_token(a2)], device=device
    ).unsqueeze(0)
    answers13 = torch.tensor(
        [model.to_single_token(a1), model.to_single_token(a3)], device=device
    ).unsqueeze(0)
    logit_diff12 = get_logit_diff(logits, answer_tokens=answers12)
    logit_diff13 = get_logit_diff(logits, answer_tokens=answers13)
    if orig12 is not None:
        assert orig13 is not None
        assert new12 is not None
        assert new13 is not None
        logit_diff12 = (logit_diff12 - orig12) / (new12 - orig12)
        logit_diff13 = (logit_diff13 - orig13) / (new13 - orig13)
    return torch.stack(
        [logit_diff12, logit_diff13],
        dim=0,
    )


# %%
results_list = []
for p1, a1, p2, a2, p3, a3 in DATASET[:2]:
    orig_input = model.to_tokens(p1, prepend_bos=True)
    orig_logits = model(orig_input, return_type="logits")
    orig12, orig13 = triple_metric_base(orig_logits, model, a1, a2, a3)
    prompt_str_tokens = model.to_str_tokens(p1, prepend_bos=True)
    new_logit_diffs = []
    for cf_idx in (0, 1):
        new_input = [p2, p3][cf_idx]
        new_logits = model(new_input, return_type="logits", prepend_bos=True)
        new_logit_diff = triple_metric_base(
            new_logits,
            model,
            a1,
            a2,
            a3,
        )[cf_idx]
        new_logit_diffs.append(new_logit_diff)
    new12, new13 = new_logit_diffs
    metric = partial(
        triple_metric_base,
        model=model,
        a1=a1,
        a2=a2,
        a3=a3,
        orig12=orig12,
        orig13=orig13,
        new12=new12,
        new13=new13,
    )
    print(orig12, orig13, new12, new13)
    print(metric(orig_logits), metric(new_logits))
    seq_pos = [i for i, s in enumerate(prompt_str_tokens) if ":" in s]
    assert len(seq_pos) == 1
    nodes = IterNode(node_names=["resid_pre"], seq_pos=seq_pos)
    for cf_idx in (0, 1):
        new_input = [p2, p3][cf_idx]
        result = act_patch(
            model, orig_input, nodes, metric, new_input=new_input, verbose=True
        )[
            "resid_pre"
        ]  # type: ignore
        result: Float[Tensor, "layer"] = torch.stack(result, dim=0)  # type: ignore
        results_list.append(result)
# %%
results: Float[Tensor, "batch cf layer answer"] = einops.rearrange(
    torch.stack(results_list, dim=0),
    "(batch cf) layer answer -> batch cf layer answer",
    cf=2,
)
print(results.shape)
# %%
fig = make_subplots(
    rows=results.shape[0],
    cols=results.shape[1],
)
for batch in range(results.shape[0]):
    for cf in range(results.shape[1]):
        fig.add_trace(
            go.Heatmap(
                z=results[batch, cf].cpu().numpy(),
                name=f"batch={batch}, cf={cf+1}",
                colorscale="RdBu",
                zmin=0,
                zmax=1,
                hovertemplate="answer=%{x}<br>layer=%{y}<br>logit diff=%{z:.1%}",
            ),
            row=batch + 1,
            col=cf + 1,
        )
        fig.update_xaxes(
            title_text="answer", row=batch + 1, col=cf + 1, tickvals=[0, 1]
        )
        fig.update_yaxes(title_text="layer", row=batch + 1, col=cf + 1)
fig.update_layout(
    height=results.shape[0] * 400,
    width=results.shape[1] * 400,
)

# %%
