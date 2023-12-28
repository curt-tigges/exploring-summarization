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
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from transformer_lens.utils import (
    get_dataset,
    tokenize_and_concatenate,
    get_act_name,
    get_attention_mask,
    remove_batch_dim,
    USE_DEFAULT_VALUE,
    rprint,
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
from summarization_utils.toy_datasets import (
    CounterfactualDataset,
    ToyDeductionTemplate,
    ToyBindingTemplate,
    ToyProfilesTemplate,
)
from summarization_utils.counterfactual_patching import (
    patch_by_position_group,
    patch_at_position,
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
class HookedMistral(HookedTransformer):
    def to_tokens(
        self,
        string: str | List[str],
        prepend_bos=USE_DEFAULT_VALUE,
        padding_side: Literal["left", "right"] | None = USE_DEFAULT_VALUE,
        move_to_device: bool | None = True,
        truncate: bool | None = True,
        prepend_blank: bool = True,
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
            string = "_" + string
        else:
            string = ["_" + s for s in string]

        tokens = super().to_tokens(
            string,
            prepend_bos=prepend_bos,
            padding_side=padding_side,
            move_to_device=move_to_device,
            truncate=truncate,
        )
        # Remove the artificial token
        if prepend_bos:
            tokens = torch.cat((tokens[:, :1], tokens[:, 2:]), dim=1)
        else:
            tokens = tokens[:, 1:]
        return tokens


# %%
model = HookedMistral.from_pretrained(
    "mistral-7b-instruct",
    torch_dtype=torch.bfloat16,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device=device,
)
assert model.tokenizer is not None
# %%
# test_prompt(
#     "[INST] Anne carried her cane, hat and glasses. Anne lost her cane on the walk. Anne found her cane again in some bushes. Question: Does Anne have her cane? Answer (Yes/No): [/INST]",
#     " Yes",
#     model,
#     prepend_space_to_answer=False,
# )
# %%
PROMPT_ANSWER_PAIRS = [
    (
        "[INST] Anne carried her stick, hat and glasses. Anne lost her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
        " No",
    ),
    (
        "[INST] Anne carried her stick, hat and glasses. Anne used her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
        " Yes",
    ),
    (
        "[INST] Anne carried her stick, hat and glasses. Anne lost her hat on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
        " Yes",
    ),
    (
        "[INST] Anne carried her stick, hat and glasses. Anne lost her glasses on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
        " Yes",
    ),
    (
        "[INST] Anne carried her stick, hat and glasses. Anne used her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
        " Yes",
    ),
    (
        "[INST] Anne carried her stick, hat and glasses. Anne lost her hat on the walk. Question: Does Anne have her hat? Answer (Yes/No): [/INST]",
        " No",
    ),
    (
        "[INST] Anne carried her stick, hat and glasses. Anne lost her glasses on the walk. Question: Does Anne have her glasses? Answer (Yes/No): [/INST]",
        " No",
    ),
]
# %%
# for prompt, answer in PROMPT_ANSWER_PAIRS:
#     test_prompt(prompt, answer, model, prepend_space_to_answer=False)
# %%
DATA_TUPLES = [
    (prompt, answer, cf_prompt, cf_answer)
    for (prompt, answer), (cf_prompt, cf_answer) in itertools.combinations(
        PROMPT_ANSWER_PAIRS, 2
    )
    if answer != cf_answer
]
# %%
dataset = CounterfactualDataset.from_tuples(DATA_TUPLES, model)
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
print(len(dataset))
# %%
pos_layer_results = patch_by_layer(dataset)
# %%
plot_layer_results_per_batch(dataset, pos_layer_results)
# %%
