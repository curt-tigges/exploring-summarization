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
ablated_loss_diffs = losses[1]
losses.shape
# %%
plot_topk_onesided(
    ablated_loss_diffs,
    data_loader,
    model,
    k=10,
    window_size=30,
)
# %%
