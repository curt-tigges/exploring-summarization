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
    USE_DEFAULT_VALUE,
)
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM
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
#%%
class TokenSafeTransformer(HookedTransformer):
    def to_tokens(
        self,
        string: str | List[str],
        prepend_bos=USE_DEFAULT_VALUE,
        padding_side: Literal["left", "right"] | None = USE_DEFAULT_VALUE,
        move_to_device: bool | None = True,
        truncate: bool | None = True,
        prepend_blank: bool = True,
        verbose: bool = False,
        blank_char: str = "|",
    ):
        """Map a string to the ids for that string.

        Handles the case of mistral which has an extra BOS-like token prepended to the start
        of the output of tokenizer.encode().
        """
        if "mistral" not in self.cfg.model_name.lower() or not prepend_blank:
            return super().to_tokens(
                string,
                prepend_bos=prepend_bos,
                padding_side=padding_side,
                move_to_device=move_to_device,
                truncate=truncate,
            )
        if prepend_bos is USE_DEFAULT_VALUE:
            prepend_bos = self.cfg.default_prepend_bos
        if isinstance(string, str):
            string = blank_char + string
        else:
            string = [blank_char + s for s in string]

        tokens = super().to_tokens(
            string,
            prepend_bos=True,
            padding_side=padding_side,
            move_to_device=move_to_device,
            truncate=truncate,
        )
        if verbose:
            print(f"Tokenizing string={string}")
            print(f"Tokens={tokens}")
            print(f"verbose={verbose}")
        # Remove the artificial token
        if prepend_bos:
            return torch.cat([tokens[:, :1], tokens[:, 2:]], dim=1)
        else:
            return tokens[:, 2:]
#%%
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
torch_dtype = "torch.bfloat16"
# %%
model = TokenSafeTransformer.from_pretrained(
    model_name, 
    fold_ln=False, 
    center_writing_weights=False, 
    center_unembed=False, 
    device="cuda",
    dtype=torch_dtype,
)
assert model.tokenizer is not None
#%%

# %%
# dataset = CounterfactualDataset.from_name(
#     "KnownFor", model
# ) + CounterfactualDataset.from_name("OfCourse", model)
dataset = CounterfactualDataset.from_name("BooleanOperator", model, dataset_size=10)
# %%
dataset.check_lengths_match()
# %%
dataset.test_prompts(max_prompts=4, top_k=10, prepend_space_to_answer=False)
# %%
all_logit_diffs, cf_logit_diffs = dataset.compute_logit_diffs()
# %%
print(f"Original mean: {all_logit_diffs.mean():.2f}")
print(f"Counterfactual mean: {cf_logit_diffs.mean():.2f}")
# %%
assert (all_logit_diffs > 0).all()
assert (cf_logit_diffs < 0).all()
# %%
results_pd = patch_by_position_group(dataset, sep=",")
fig = px.bar(
    results_pd.mean(axis=0), 
    labels={"index": "Position", "value": "Patching metric"}
)
fig.update_layout(showlegend=False)
fig.show()
# %%
pos_layer_results = patch_by_layer(dataset)
# %%
plot_layer_results_per_batch(dataset, pos_layer_results)
# %%
