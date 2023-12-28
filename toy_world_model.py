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
from typing import Sequence
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
DATA_TUPLES: List[Tuple[str, str, str, str]] = [
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " No",
    #     "[INST] Anne carried her stick, hat and glasses. Anne used her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    # ),
    (
        "[INST] Anne carried her stick, hat and glasses. Anne lost her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
        " No",
        "[INST] Anne carried her stick, hat and glasses. Anne lost her hat on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
        " Yes",
    ),
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " No",
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her glasses on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    # ),
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " No",
    #     "[INST] Anne carried her stick, hat and glasses. Anne used her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    # ),
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne used her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her hat on the walk. Question: Does Anne have her hat? Answer (Yes/No): [/INST]",
    #     " No",
    # ),
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne used her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her glasses on the walk. Question: Does Anne have her glasses? Answer (Yes/No): [/INST]",
    #     " No",
    # ),
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her hat on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her hat on the walk. Question: Does Anne have her hat? Answer (Yes/No): [/INST]",
    #     " No",
    # ),
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her hat on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her glasses on the walk. Question: Does Anne have her glasses? Answer (Yes/No): [/INST]",
    #     " No",
    # ),
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her glasses on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her hat on the walk. Question: Does Anne have her hat? Answer (Yes/No): [/INST]",
    #     " No",
    # ),
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her glasses on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her glasses on the walk. Question: Does Anne have her glasses? Answer (Yes/No): [/INST]",
    #     " No",
    # ),
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne used her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her hat on the walk. Question: Does Anne have her hat? Answer (Yes/No): [/INST]",
    #     " No",
    # ),
    # (
    #     "[INST] Anne carried her stick, hat and glasses. Anne used her stick on the walk. Question: Does Anne have her stick? Answer (Yes/No): [/INST]",
    #     " Yes",
    #     "[INST] Anne carried her stick, hat and glasses. Anne lost her glasses on the walk. Question: Does Anne have her glasses? Answer (Yes/No): [/INST]",
    #     " No",
    # ),
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
# #############################################################################
# Patching across positions
# #############################################################################
# %%
prompt = dataset.prompts[0]
answer = dataset.answers[0]
cf_prompt = dataset.cf_prompts[0]
cf_answer = dataset.cf_answers[0]
prepend_bos: bool = True
# %%
[f"{i}: {t}" for i, t in enumerate(model.to_str_tokens(prompt))]
# %%
prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
cf_tokens = model.to_tokens(cf_prompt, prepend_bos=prepend_bos)
answer_id = model.to_single_token(answer)
cf_answer_id = model.to_single_token(cf_answer)
answer_tokens = torch.tensor(
    [answer_id, cf_answer_id], dtype=torch.int64, device=model.cfg.device
).unsqueeze(0)
assert prompt_tokens.shape == cf_tokens.shape, (
    f"Prompt and counterfactual prompt must have the same shape, "
    f"for prompt {prompt} "
    f"got {prompt_tokens.shape} and {cf_tokens.shape}"
)
model.reset_hooks(including_permanent=True)
base_logits_by_pos, base_cache = model.run_with_cache(
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
metric = lambda logits: (
    get_logit_diff(logits, answer_tokens=answer_tokens) - base_ldiff
) / (cf_ldiff - base_ldiff)
# %%
stick_positions = [7, 16, 27]
# %%
resids = []
labels = []
for cache_str in ("base", "cf"):
    cache = base_cache if cache_str == "base" else cf_cache
    for pos_idx, pos in enumerate(stick_positions):
        for layer in range(model.cfg.n_layers):
            resids.append(cache["resid_pre", layer][:, pos, :])
            labels.append(f"{cache_str} L{layer} P{pos_idx+1}")
resids = torch.cat(resids, dim=0)
cosine_sims = (resids @ resids.T) / (
    resids.norm(dim=-1, keepdim=True) @ resids.norm(dim=-1, keepdim=True).T
)
cosine_sims = cosine_sims.cpu().to(dtype=torch.float32).numpy()
print(resids.shape, cosine_sims.shape)
# %%
# plot cosine similarity matrix between the residuals
fig = px.imshow(
    cosine_sims,
    x=labels,
    y=labels,
    width=1200,
    height=1200,
    title="Cosine Similarity Matrix",
)
fig.update_layout(
    title_x=0.5,
)
fig.show()
# %%
# What is up with the weird discontinuity at L20?
# cf middle position is very different to other positions as expected
