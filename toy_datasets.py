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
    "pythia-2.8b",
    torch_dtype=torch.float32,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device=device,
)
assert model.tokenizer is not None
# %%
dataset = CounterfactualDataset.from_name(
    "KnownFor", model
) + CounterfactualDataset.from_name("OfCourse", model)
# dataset = ToyDeductionTemplate(model, max=100, dataset_size=10).to_counterfactual()
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
results_pd = dataset.patch_by_position_group(sep=",")
fig = px.bar(
    results_pd.mean(axis=0), labels={"index": "Position", "value": "Patching metric"}
)
fig.update_layout(showlegend=False)
fig.show()
# %%
pos_layer_results = dataset.patch_by_layer()
# %%
dataset.plot_layer_results_per_batch(pos_layer_results)
# %%
