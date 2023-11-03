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
from utils.circuit_analysis import get_logit_diff

from utils.tokenwise_ablation import (
    compute_ablation_modified_metric,
    load_directions,
    get_random_directions,
    get_zeroed_dir_vector,
    get_layerwise_token_mean_activations,
)
from utils.datasets import OWTData, PileFullData, PileSplittedData
from utils.neuroscope import plot_topk_onesided

# %%
device = torch.device("cuda")
MODEL_NAME = "gpt2-small"
BATCH_SIZE = 8
TOKEN = ","
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
losses = compute_ablation_modified_metric(
    model,
    data_loader,
    cached_means=comma_mean_values,
    metric="loss",
    device=device,
)
# %%
losses.shape
# %%
# We're interested in the first experiment's data
seq_len = model.cfg.n_ctx
ablated_loss_diffs = losses[1]

# Flatten the first_experiment_data tensor to apply topk
flattened_data = ablated_loss_diffs.view(-1)

# Find the top 10 values and their indices in the flattened tensor
top_values, flat_indices = torch.topk(flattened_data, 10)

# Convert the flat indices back into 2D indices corresponding to [batch, pos]
batch_indices = flat_indices // seq_len
pos_indices = flat_indices % seq_len

# Print the results
print(
    "Top 10 largest values and their batch/position indices for the ablation experiment:"
)
for i in range(top_values.size(0)):
    value = top_values[i].item()
    batch_idx = batch_indices[i].item()
    pos_idx = pos_indices[i].item()
    context_tokens = data_loader.dataset[batch_idx]["tokens"][
        pos_idx - 10 : pos_idx + 10
    ]
    context_string = model.to_str_tokens(context_tokens)
    print(
        f"Value: {value:.2f}, Batch index: {batch_idx}, Position index: {pos_idx}, "
        f"Context: {context_string}"
    )
# %%
plot_topk_onesided(
    ablated_loss_diffs,
    data_loader,
    model,
    k=10,
)
