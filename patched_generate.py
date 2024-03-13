# %%
from functools import partial
from datasets import load_dataset
from jaxtyping import Float
import random
from summarization_utils.counterfactual_patching import patch_by_position_group
from summarization_utils.toy_datasets import (
    CounterfactualDataset,
    TemplaticDataset,
    HookedTransformer,
    wrap_instruction,
    itertools,
    List,
    Tuple,
)
from summarization_utils.models import TokenSafeTransformer
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt
from transformers import AutoTokenizer, AutoConfig
import torch
from torch import Tensor
import warnings

# %%
model = TokenSafeTransformer.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device="cuda",
    dtype="bfloat16",
)

# %%
base_prompt = "Alice loves the outdoors and sports."
cf_prompt = "Alice loves books, learning and quiet."
base_str_tokens = model.to_str_tokens(base_prompt)
cf_str_tokens = model.to_str_tokens(cf_prompt)

# %%
_, base_cache = model.run_with_cache(
    base_prompt, return_type=None, names_filter=lambda name: "resid_post" in name
)
_, cf_cache = model.run_with_cache(
    cf_prompt, return_type=None, names_filter=lambda name: "resid_post" in name
)
print(len(base_cache), len(cf_cache))


# %%
def hook_fn_base(
    activation: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    cache: ActivationCache,
    position: int,
):
    assert hook.name is not None and "resid_post" in hook.name
    activation[:, position] = cache[hook.name][:, position]
    return activation


# %%
patching_position = base_str_tokens.index(".")  # type: ignore
print(patching_position)


# %%
class PatchingContextManager:
    def __init__(self, model: HookedTransformer, cache: ActivationCache, position: int):
        self.model = model
        self.cache = cache
        self.position = position
        self.hook_fn = partial(hook_fn_base, cache=cache, position=position)

    def __enter__(self):
        for layer in range(model.cfg.n_layers):
            act_name = get_act_name("resid_post", layer)
            self.model.add_hook(act_name, self.hook_fn)

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.reset_hooks()


# %%
base_completions = []
for seed in range(10):
    torch.manual_seed(seed)
    out = model.generate(
        base_prompt,
        max_new_tokens=10,
        stop_at_eos=True,
        temperature=1.0,
    )
    base_completions.append(out)
print(base_completions)

# %%
cf_completions = []
for seed in range(10):
    torch.manual_seed(seed)
    out = model.generate(
        cf_prompt,
        max_new_tokens=10,
        stop_at_eos=True,
        temperature=1.0,
    )
    cf_completions.append(out)
print(cf_completions)


# %%
patched_completions = []
for seed in range(10):
    with PatchingContextManager(model, cf_cache, patching_position):
        torch.manual_seed(seed)
        out = model.generate(
            base_prompt,
            max_new_tokens=10,
            stop_at_eos=True,
            temperature=1.0,
        )
        patched_completions.append(out)
print(patched_completions)
