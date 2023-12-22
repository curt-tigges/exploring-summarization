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
from typing import Dict, Iterable, List, Tuple, Union, Literal, Optional, Generator
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
from typing import Generator
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
SPLIT = "train"
DATA_FILES = [
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/ArXiv/train/data-00000-of-00222.arrow",
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/BookCorpus2/train/data-00000-of-00096.arrow",
    "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/Github/train/data-00000-of-00191.arrow"
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/Gutenberg%20(PG-19)/train/data-00000-of-00096.arrow",
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/PhilPapers/train/data-00000-of-00096.arrow",
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/PubMed%20Abstracts/train/data-00000-of-00096.arrow",
    # "https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/Wikipedia%20(en)/train/data-00000-of-00101.arrow",
]
NAME = DATA_FILES[0].split("/")[-3].replace("%20", " ")
BATCH_SIZE = 64
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
exp_data = PileSplittedData.from_model(
    model,
    name=NAME,
    split=SPLIT,
    data_files=DATA_FILES,
    verbose=True,
)
ablation_token: int = model.to_single_token(":")  # type: ignore
exp_data.preprocess_datasets(token_to_ablate=ablation_token)
# %%
data_loader = exp_data.get_dataloaders(batch_size=BATCH_SIZE)[SPLIT]
# %%
template = "for i in range(x"  # x: \n    print( TODO: start simple here and then move to more complex
tokens = model.to_tokens(template, prepend_bos=False)
str_tokens = model.to_str_tokens(tokens, prepend_bos=False)
search_tokens = [
    t if "x" not in s else -1 for t, s in zip(tokens.tolist()[0], str_tokens)
]
print(search_tokens)


# %%
def find_subtensor(
    main_tensor: Int[Tensor, "*batch pos"],
    sub_tensor: Int[Tensor, "sub_pos"],
    wildcard_value: int = -1,
    verbose: bool = False,
) -> Generator[Tensor, None, None]:
    """
    main_tensor: main tensor to search e.g. torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    sub_tensor: sub-tensor to search for in main_tensor, e.g. torch.tensor([3, 0, 5])
    wildcard_tensor: treat this value as a wildcard, e.g. 0
    out: list of found sub-tensors, e.g. [torch.tensor([3, 4, 5])]
    """
    if main_tensor.ndim == 1:
        main_tensor = main_tensor.unsqueeze(0)
    is_wildcard = sub_tensor == wildcard_value
    for batch_idx, batch in enumerate(main_tensor):
        for i in range(len(batch) - len(sub_tensor) + 1):
            is_match = batch[i : i + len(sub_tensor)] == sub_tensor
            if torch.all(is_match | is_wildcard):
                if verbose:
                    print(
                        f"Found match at batch {batch_idx}, pos {i}: {batch[i:i+len(sub_tensor)]}"
                    )
                yield batch[i : i + len(sub_tensor)]


# def find_subtensor(
#     main_tensor: torch.Tensor,
#     sub_tensor: torch.Tensor,
#     wildcard_value: int = -1,
#     verbose: bool = False,
# ) -> torch.Tensor:

#     # Move tensors to GPU
#     main_tensor = main_tensor.to('cuda')
#     sub_tensor = sub_tensor.to('cuda')

#     if main_tensor.ndim == 1:
#         main_tensor = main_tensor.unsqueeze(0)

#     is_wildcard = sub_tensor == wildcard_value
#     is_wildcard = is_wildcard.unsqueeze(1)

#     # Use unfold to generate all sub-tensors
#     sub_tensors = main_tensor.unfold(1, sub_tensor.size(0), 1)
#     is_match = sub_tensors == sub_tensor
#     indices = torch.all(is_match | is_wildcard, dim=2)
#     result = torch.masked_select(sub_tensors, indices.unsqueeze(2)).view(-1, sub_tensor.size(0))

#     if verbose:
#         print(f"Found match : {result}")
#     return result


# # %%
# out = find_subtensor(
#     torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 2, 5, 6, 7, 8, 9, 0]]),
#     torch.tensor([3, -1, 5]),
#     -1,
# )
# list(out)
# # %%
# out = find_subtensor(
#     torch.tensor([[1, 2, 3, 4, 5, 6, 3, 8, 5], [1, 3, 2, 5, 6, 7, 8, 9, 0]]),
#     torch.tensor([3, -1, 5]),
#     -1,
# )
# list(out)
# # %%
# search_function = partial(
#     find_subtensor,
#     sub_tensor=torch.tensor(search_tokens, device=device),
#     wildcard_value=-1,
#     verbose=True,
# )
# # %%
# out = search_function(model.to_tokens("for x in x: \n    print("))
# list(out)
# %%
MAX_RESULTS = 100
results = []
for batch in tqdm(data_loader, total=len(data_loader)):
    for result in find_subtensor(
        batch["tokens"], torch.tensor(search_tokens), -1, verbose=True
    ):
        results.append(result)
    if len(results) >= MAX_RESULTS:
        break
# %%
for result in results:
    print(model.to_str_tokens(result))
# %%
