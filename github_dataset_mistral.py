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
    patch_by_position,
    plot_position_results_per_batch,
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
    "mistral-7b",
    torch_dtype=torch.bfloat16,
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
    ),
    # (
    #     "for i, frame in enumerate(iter_frames(im)):\n    if",
    #     " i",
    #     "for i in range(0, len(jsonButtonData)):\n    if",
    #     " json",
    # ),
    (
        "for i in range(len(fig.layout.annotations)):\n   ",
        " fig",
        "for target_idx in range(1, len(arr)):\n   ",
        " target",
    ),
    (
        "for i in range(len(model_infos)):\n   ",
        " model",
        "for i in range(0, len(requested)):\n   ",
        " requested",
    ),
    # (
    #     "for i in range(1,len(y)):\n   ",
    #     " y",
    #     "for i in range(len(short_answers)):\n   ",
    #     " short",
    # ),
    (
        "for i in range(len(split_code)):\n   ",
        " split",
        "for i in range(len(text_to)):\n   ",
        " text",
    ),
    # (
    #     "for i in range(len(features)):\n   tr[",
    #     "features",
    #     "for i in range(len(vocab)):\n    word =",
    #     " vocab",
    # ),
    # (
    #     "for i in range(len(records)):\n    qid = int(",
    #     " records",
    #     "for i in range(len(U)):\n    color_map[",
    #     "U",
    # ),
    (
        "for i in range(1, rgsize + 1):\n   ",
        " r",  # " rg",
        "for idx, xml_file_name in enumerate(files):\n   ",
        " xml",
    ),
    # (
    #     "for n, d in G.nodes(data=True):\n   ",
    #     " G",
    #     "for r in pool.imap_unordered(fetch, requested):\n   ",
    #     " r",
    # ),
    # (
    #     "for factor in range(x, 0, -1):\n    if",
    #     " x",
    #     "for module, items in iteritems(all_by_module):\n    for",
    #     " item",
    # ),
    # (
    #     "for button in (left, right, forward, backward):\n   ",
    #     " button",
    #     "for iidx, video_name in enumerate(videos):\n   ",
    #     " video",
    # ),
    # (
    #     "for i in xrange(-10, 10):\n    if",
    #     " i",
    #     "for x in range(1, 101):\n    if",
    #     " x",
    # ),
    # (
    #     "for test in sorted(all_doctests):    if",
    #     " test",
    #     "for i in xrange(30000):\n    if",
    #     " i",
    # ),
    (
        "for cell in dolfin.cells(mesh):\n    contains =",
        " cell",
        "for i in range(0, width * height):\n    if(",
        "i",
    ),
    # (
    #     "for i in range(0,numRuns):\n    #",
    #     "i",
    #     "for x in range(0,15):\n    print",
    #     " x",
    # ),
    (
        "for t in (list, dict, set):\n    d[",
        "t",
        "for value in range(1,11):\n    square =",
        " value",
    ),
    # (
    #     "for i in range(1001):\n    x =",
    #     " i",
    #     "for match in regexp.finditer(data):\n    repr =",
    #     " match",
    # ),
]
DATASET = [
    (prompt, ans.strip(), cf_prompt, cf_ans.strip())
    for prompt, ans, cf_prompt, cf_ans in DATASET
]
DATASET_NAME = "github"
# %%
dataset = CounterfactualDataset.from_tuples(DATASET, model=model)
print(len(dataset))
# %%
dataset.check_lengths_match()
# %%
dataset.test_prompts(max_prompts=4, top_k=10)
# %%
all_logit_diffs, cf_logit_diffs = dataset.compute_logit_diffs()
# %%
print(f"Original mean: {all_logit_diffs.mean():.2f}")
print(f"Counterfactual mean: {cf_logit_diffs.mean():.2f}")
# %%
assert (all_logit_diffs > 0).all()
assert (cf_logit_diffs < 0).all()
# %%
pos_results = patch_by_position(dataset)
# %%
plot_position_results_per_batch(dataset, pos_results)
# %%
# pos_layer_results = patch_by_layer(dataset)
# # %%
# plot_layer_results_per_batch(dataset, pos_layer_results)
# %%
