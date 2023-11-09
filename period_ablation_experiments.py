# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import einops
from functools import partial
import torch
import datasets
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union
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
from utils.circuit_analysis import get_logit_diff

from utils.tokenwise_ablation import (
<<<<<<< HEAD
    compute_ablation_modified_loss,
=======
    compute_ablation_modified_metric,
>>>>>>> 59a3578 (Started analyzing period ablations)
    load_directions,
    get_random_directions,
    get_zeroed_dir_vector,
    get_layerwise_token_mean_activations,
)
from utils.datasets import OWTData, PileFullData, PileSplittedData
<<<<<<< HEAD
from utils.neuroscope import plot_top_onesided
=======
from utils.neuroscope import plot_topk_onesided
>>>>>>> 59a3578 (Started analyzing period ablations)

# %%
device = torch.device("cuda")
MODEL_NAME = "gpt2-small"
BATCH_SIZE = 8
TOKEN = "."
SPLIT = "train"
NAME = None
# %%
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
# %%
TOKEN_ID = model.to_single_token(TOKEN)
assert isinstance(TOKEN_ID, int)
print(TOKEN_ID)
# %%
exp_data = OWTData.from_model(model)
exp_data.preprocess_datasets(token_to_ablate=TOKEN_ID)
# %%
data_loader = exp_data.get_dataloaders(batch_size=BATCH_SIZE)[SPLIT]
print(data_loader.name)
# %%
comma_mean_values = get_layerwise_token_mean_activations(
    model, data_loader, token_id=TOKEN_ID, device=device
)
# %%
<<<<<<< HEAD
smaller_owt = OWTData.from_model(model)
smaller_owt.dataset_dict[SPLIT] = smaller_owt.dataset_dict[SPLIT].select(
    list(range(100))
)
smaller_owt.preprocess_datasets(token_to_ablate=TOKEN_ID)
smaller_data_loader = smaller_owt.get_dataloaders(batch_size=BATCH_SIZE)[SPLIT]

# %%
losses = compute_ablation_modified_loss(
    model,
    smaller_data_loader,
    cached_means=comma_mean_values,
=======
losses = compute_ablation_modified_metric(
    model,
    data_loader,
    cached_means=comma_mean_values,
    metric="loss",
>>>>>>> 59a3578 (Started analyzing period ablations)
    device=device,
)
# %%
ablated_loss_diffs = losses[1]
losses.shape
# %%
<<<<<<< HEAD
plot_top_onesided(ablated_loss_diffs, smaller_data_loader, model, k=10)
# %%
plot_top_onesided(
    ablated_loss_diffs,
    smaller_data_loader,
    model,
    k=10,
    p=0.1,
=======
plot_topk_onesided(
    ablated_loss_diffs,
    data_loader,
    model,
    k=10,
>>>>>>> 59a3578 (Started analyzing period ablations)
    window_size=50,
)
# %%
ablated_loss_diffs[0][15:30]

# %%
batch = next(iter(data_loader))
<<<<<<< HEAD
batch["tokens"][0][15:30]

# %%
model.to_str_tokens(batch["tokens"][0][15:30])
=======
batch['tokens'][0][15:30]

# %%
model.to_str_tokens(batch['tokens'][0][15:30])
>>>>>>> 59a3578 (Started analyzing period ablations)

# %%
exp_data = OWTData.from_model(model)

# %%
ds = exp_data.get_datasets()

# %%
<<<<<<< HEAD
ds["train"][1]["text"]
=======
ds['train'][1]['text']
>>>>>>> 59a3578 (Started analyzing period ablations)

# %%
