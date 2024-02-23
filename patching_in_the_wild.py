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
    f"https://huggingface.co/datasets/ArmelR/the-pile-splitted/blob/main/data/Github/train/data-{i:05d}-of-00191.arrow"
    for i in range(10)
]
NAME = DATA_FILES[0].split("/")[-3].replace("%20", " ")
BATCH_SIZE = 1024
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
template = "for i in range(1, len(sys.argv)):\n   "
tokens = model.to_tokens(template, prepend_bos=False)
str_tokens = model.to_str_tokens(tokens, prepend_bos=False)
print([f"{s}:{t}" for s, t in zip(str_tokens, tokens[0].tolist())])
search_tokens = [
    996,  # for
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    # -1,
    # -1,
    # -1,
    # Adding wildcards here
    # -1,
    # -1,
    # -1,
    399,  # )): is 3554, ): is 399, : is 25
    258,  # 258 is 3, 246 is 4 spaces
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
]
# %%


def find_subtensor(
    main_tensor: torch.Tensor,
    sub_tensor: torch.Tensor,
    wildcard_value: int = -1,
    device: torch.device = torch.device("cuda"),
    return_full_batch: bool = False,
) -> torch.Tensor:
    # Move tensors to GPU
    main_tensor = main_tensor.to(device)
    sub_tensor = sub_tensor.to(device)

    if main_tensor.ndim == 1:
        main_tensor = main_tensor.unsqueeze(0)

    is_wildcard = sub_tensor == wildcard_value

    # Use unfold to generate all sub-tensors
    sub_tensors = main_tensor.unfold(1, sub_tensor.size(0), 1)
    is_match = sub_tensors == sub_tensor
    indices = torch.all(is_match | is_wildcard, dim=2)
    if return_full_batch:
        return torch.where(indices)[0]
    result = torch.masked_select(sub_tensors, indices.unsqueeze(2)).view(
        -1, sub_tensor.size(0)
    )
    return result


# %%
search_function = partial(
    find_subtensor,
    sub_tensor=torch.tensor(search_tokens, device=device),
    wildcard_value=-1,
    return_full_batch=False,
)
# %%
MAX_RESULTS = 100
results = []
for batch_num, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
    for result in search_function(batch["tokens"]):
        results.append(result)
        # results.append(batch_num * BATCH_SIZE + result)
    if len(results) >= MAX_RESULTS:
        break
# %%
print(len(results))
# %%
for result in results:
    print(model.to_str_tokens(result))
    print(model.to_string(result))
    # tokens = exp_data.dataset_dict[SPLIT][result.item()]["tokens"]
    # print(model.to_string(tokens))
# %%
DATASET = [
    (
        "for i in range(0, len(data_list)):\n   ",
        " data",
        "for j in range(len(im2.imvec)):\n   ",
        " im",
    ),
    (
        "for i, frame in enumerate(iter_frames(im)):\n    if",
        " i",
        "for i in range(0, len(jsonButtonData)):\n    if",
        " json",
    ),
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
    (
        "for i in range(1,len(y)):\n   ",
        " y",
        "for i in range(len(short_answers)):\n   ",
        " short",
    ),
    (
        "for i in range(len(split_code)):\n   ",
        " split",
        "for i in range(len(text_to)):\n   ",
        " text",
    ),
    (
        "for i in range(len(features)):\n   tr[",
        "features",
        "for i in range(len(vocab)):\n    word =",
        " vocab",
    ),
    (
        "for i in range(len(records)):\n    qid = int(",
        " records",
        "for i in range(len(U)):\n    color_map[",
        "U",
    ),
    (
        "for i in range(1, rgsize + 1):\n   ",
        " rg",
        "for idx, xml_file_name in enumerate(files):\n   ",
        " xml",
    ),
    (
        "for n, d in G.nodes(data=True):\n   ",
        " G",
        "for r in pool.imap_unordered(fetch, requested):\n   ",
        " r",
    ),
    (
        "for factor in range(x, 0, -1):\n    if",
        " x",
        "for module, items in iteritems(all_by_module):\n    for",
        " item",
    ),
    (
        "for button in (left, right, forward, backward):\n   ",
        " button",
        "for iidx, video_name in enumerate(videos):\n   ",
        " video",
    ),
    (
        "for i in xrange(-10, 10):\n    if",
        " i",
        "for x in range(1, 101):\n    if",
        " x",
    ),
    (
        'for index, item in enumerate(my_list):\n    print(f"Item" {',
        "index",
        "for temp in convert_temp(adc.values):\n    print('The temperature is',)",
        " temp",
    ),
    (
        "for test in sorted(all_doctests):    if",
        " test",
        "for i in xrange(30000):\n    if",
        " i",
    ),
    (
        "for cell in dolfin.cells(mesh):\n    contains =",
        " cell",
        "for i in range(0, width * height):\n    if(",
        "i",
    ),
    (
        "for i in range(0,numRuns):\n    #",
        "i",
        "for x in range(0,15):\n    print",
        " x",
    ),
    (
        "for index, rank in enumerate(precedence):\n    for token in",
        " rank",
    ),
    (
        "for t in (list, dict, set):\n    d[",
        "t",
        "for value in range(1,11):\n    square =",
        " value",
    ),
    (
        "for i in range(1000):\n    power = 10 if",
        " i",
    ),
    (
        "for i in range(1001):\n    x =",
        " i",
        "for match in regexp.finditer(data):\n    repr =",
        " match",
    ),
]

# %%
