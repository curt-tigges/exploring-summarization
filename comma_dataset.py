# %%
import einops
from functools import partial
import numpy as np
import torch
import datasets
import re
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
from transformer_lens.evals import make_owt_data_loader
from tqdm.notebook import tqdm
import pandas as pd
from circuitsvis.activations import text_neuron_activations
from circuitsvis.topk_samples import topk_samples
from IPython.display import HTML, display
import plotly.express as px
import plotly.graph_objects as go
from summarization_utils.circuit_analysis import get_logit_diff
from summarization_utils.tokenwise_ablation import (
    compute_ablation_modified_loss,
    load_directions,
    get_random_directions,
    get_zeroed_dir_vector,
    get_layerwise_token_mean_activations,
    AblationHook,
    get_batch_token_mean_activations,
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
from summarization_utils.residual_stream import get_resid_name

# %%
device = torch.device("cuda")
MODEL_NAME = "gpt2-small"
BATCH_SIZE = 64
BUFFER_SIZE = 1024
TOKEN = ","
# %%
# %%
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
TOKEN_ID = model.to_single_token(TOKEN)
assert isinstance(TOKEN_ID, int)
# %%
dataloader = make_owt_data_loader(model.tokenizer, BATCH_SIZE)
# %%
file_name = f"token_{TOKEN_ID}_residual_stream_{MODEL_NAME}_owt"
manager = TensorBlockManager(block_size=BUFFER_SIZE, tensor_prefix=file_name)
manager.clear()
# %%
NAMES = [get_resid_name(layer, model)[0] for layer in range(model.cfg.n_layers + 1)]
print(NAMES)
# %%
buffer = torch.zeros(
    (BUFFER_SIZE, model.cfg.n_layers + 1, model.cfg.d_model),
    dtype=torch.float32,
)
buffer_idx = 0
block = 0
for batch in tqdm(dataloader):
    tokens: Int[Tensor, "batch pos"] = batch["tokens"].to(device)
    is_token = einops.rearrange(tokens == TOKEN_ID, "batch pos -> (batch pos)")
    _, cache = model.run_with_cache(
        tokens, return_type=None, names_filter=lambda name: name in NAMES
    )
    full_residual_stream: Float[Tensor, "batch layer pos d_model"] = torch.stack(
        [cache[name] for name in NAMES], dim=1
    )
    flattened_residual_stream: Float[
        Tensor, "batch_and_pos layer d_model"
    ] = einops.rearrange(
        full_residual_stream, "batch layer pos d_model -> (batch pos) layer d_model"
    )
    to_append = flattened_residual_stream[is_token]
    to_append = to_append[: min(BUFFER_SIZE - buffer_idx, len(to_append))]
    buffer[buffer_idx : buffer_idx + len(to_append)] = to_append.cpu()
    buffer_idx += len(to_append)
    if buffer_idx >= BUFFER_SIZE:
        manager.save(buffer, block)
        buffer_idx = 0
        block += 1
        buffer.zero_()
# %%
