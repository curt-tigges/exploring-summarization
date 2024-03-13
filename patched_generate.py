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
base_prompt = "Alice loves sports, adventure and nature. Alice"
cf_prompt = "Alice loves books, learning and quiet. Alice"
erased_prompt = " Alice"
erased_str_tokens = model.to_str_tokens(erased_prompt)
base_str_tokens = model.to_str_tokens(base_prompt)
cf_str_tokens = model.to_str_tokens(cf_prompt)
source_position = base_str_tokens.index(" Alice")  # type: ignore
erased_patch_position = erased_str_tokens.index(" Alice")  # type: ignore
assert len(base_str_tokens) == len(cf_str_tokens)

# %%
_, base_cache = model.run_with_cache(
    base_prompt, return_type=None, names_filter=lambda name: "resid_post" in name
)
_, cf_cache = model.run_with_cache(
    cf_prompt, return_type=None, names_filter=lambda name: "resid_post" in name
)
print(len(base_cache), len(cf_cache))

# %%
print(
    base_cache["blocks.0.hook_resid_post"].shape,
    cf_cache["blocks.0.hook_resid_post"].shape,
)


# %%
base_completions = []
for seed in range(10):
    torch.manual_seed(seed)
    out = model.generate(
        base_prompt,
        max_new_tokens=10,
        stop_at_eos=True,
        temperature=1.0,
        verbose=False,
    )
    base_completions.append(out)
print("\n".join(base_completions))

# %%
cf_completions = []
for seed in range(10):
    torch.manual_seed(seed)
    out = model.generate(
        cf_prompt,
        max_new_tokens=10,
        stop_at_eos=True,
        temperature=1.0,
        verbose=False,
    )
    cf_completions.append(out)
print("\n".join(cf_completions))


# %%
def hook_fn_base(
    activation: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    cache: ActivationCache,
    src_position: int,
    dest_position: int,
):
    assert hook.name is not None and "resid_post" in hook.name
    assert activation.shape[0] == 1, "Batch size should be 1"
    assert activation.shape[2] == model.cfg.d_model
    activation[:, dest_position] = cache[hook.name][:, src_position]
    return activation


# %%
class PatchingContextManager:
    def __init__(
        self,
        model: HookedTransformer,
        cache: ActivationCache,
        src_position: int,
        dest_position: int,
    ):
        self.model = model
        self.cache = cache
        self.src_position = src_position
        self.dest_position = dest_position
        self.hook_fn = partial(
            hook_fn_base,
            cache=cache,
            src_position=src_position,
            dest_position=dest_position,
        )

    def __enter__(self):
        for layer in range(model.cfg.n_layers):
            act_name = get_act_name("resid_post", layer)
            self.model.add_hook(act_name, self.hook_fn)

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.reset_hooks()


# %%
patched_completions = []
for seed in range(10):
    with PatchingContextManager(model, cf_cache, source_position, source_position):
        torch.manual_seed(seed)
        out = model.generate(
            base_prompt,
            max_new_tokens=10,
            stop_at_eos=True,
            temperature=1.0,
            use_past_kv_cache=False,
            verbose=False,
        )
        patched_completions.append(out)
print("\n".join(patched_completions))
# %%
erased_base_completions = []
for seed in range(10):
    with PatchingContextManager(
        model, base_cache, source_position, erased_patch_position
    ):
        torch.manual_seed(seed)
        out = model.generate(
            erased_prompt,
            max_new_tokens=10,
            stop_at_eos=True,
            temperature=1.0,
            use_past_kv_cache=False,
            verbose=False,
        )
        erased_base_completions.append(out)
print("\n".join(erased_base_completions))
# %%
erased_cf_completions = []
for seed in range(10):
    with PatchingContextManager(
        model, cf_cache, source_position, erased_patch_position
    ):
        torch.manual_seed(seed)
        out = model.generate(
            erased_prompt,
            max_new_tokens=10,
            stop_at_eos=True,
            temperature=1.0,
            use_past_kv_cache=False,
            verbose=False,
        )
        erased_cf_completions.append(out)
print("\n".join(erased_cf_completions))
# %%
