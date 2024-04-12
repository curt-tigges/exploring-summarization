# %% [markdown]
# # Summarization (Day 2)

# %%
import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import torch
import plotly_express as px

from transformer_lens import HookedTransformer
from sae_lens import SparseAutoencoder, ActivationsStore

# Model Loading
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes


def get_sae_out_all_layers(cache, gpt2_small_sparse_autoencoders):

    sae_outs = []
    feature_actss = []
    for hook_point in gpt2_small_sparse_autoencoders.keys():
        sae_out, feature_acts, loss, mse_loss, l1_loss, _ = (
            gpt2_small_sparse_autoencoders[hook_point](cache[hook_point])
        )
        sae_outs.append(sae_out)
        feature_actss.append(feature_acts)

    sae_outs = torch.stack(sae_outs, dim=0)
    feature_actss = torch.stack(feature_actss, dim=0)
    return sae_outs, feature_actss


import json
import urllib.parse
import webbrowser


def open_neuronpedia(features: list[int], layer: int, name: str = "temporary_list"):
    url = "https://neuronpedia.org/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {"modelId": "gpt2-small", "layer": f"{layer}-res-jb", "index": str(feature)}
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    webbrowser.open(url)


# %% [markdown]
# # Set Up

# %%
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
gpt2_small_sparse_autoencoders, gpt2_small_sae_sparsities = get_gpt2_res_jb_saes()

sparse_autoencoder = gpt2_small_sparse_autoencoders["blocks.5.hook_resid_pre"]
sparse_autoencoder.eval()

activation_store = ActivationsStore.from_config(model, sparse_autoencoder.cfg)


# %%
class InferenceSparseAutoencoder(SparseAutoencoder):

    def forward(self, x: torch.Tensor, dead_neuron_mask: torch.Tensor | None = None):
        return super(InferenceSparseAutoencoder, self).forward(x).sae_out


def cast_to_inference_sparse_autoencoder(sparse_autoencoder: SparseAutoencoder):
    print("Casting to InferenceSparseAutoencoder...")
    inference_sparse_autoencoder = InferenceSparseAutoencoder(sparse_autoencoder.cfg)
    print("Copying state dict...")
    inference_sparse_autoencoder.load_state_dict(sparse_autoencoder.state_dict())
    return inference_sparse_autoencoder


inference_sparse_autoencoder = cast_to_inference_sparse_autoencoder(sparse_autoencoder)
# %%
# inference_sparse_autoencoder(activation_store.next_batch()).shape

# %%
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from jaxtyping import Float

from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint  # Hooking utilities

# from transformer_lens.HookedSAE import HookedSAE

SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]


def get_deep_attr(obj: Any, path: str):
    """Helper function to get a nested attribute from a object.
    In practice used to access HookedTransformer HookPoints (eg model.blocks[0].attn.hook_z)

    Args:
        obj: Any object. In practice, this is a HookedTransformer (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")

    returns:
        Any. The attribute at the end of the path
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts:
        if part.isdigit():  # This is a list index
            obj = obj[int(part)]
        else:  # This is an attribute
            obj = getattr(obj, part)
    return obj


def set_deep_attr(obj: Any, path: str, value: Any):
    """Helper function to change the value of a nested attribute from a object.
    In practice used to swap HookedTransformer HookPoints (eg model.blocks[0].attn.hook_z) with HookedSAEs

    Args:
        obj: Any object. In practice, this is a HookedTransformer (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")
        value: Any. The value you want to set the attribute to (eg a HookedSAE object)
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts[:-1]:
        if part.isdigit():  # This is a list index
            obj = obj[int(part)]
        else:  # This is an attribute
            obj = getattr(obj, part)
    # Set the value on the final attribute
    setattr(obj, parts[-1], value)


class HookedSAETransformer(HookedTransformer):
    def __init__(
        self,
        *model_args,
        **model_kwargs,
    ):
        """Model initialization. Just HookedTransformer init, but adds a dictionary to attach SAEs.

        Note that if you want to load the model from pretrained weights, you should use
        :meth:`from_pretrained` instead.

        Args:
            *model_args: Positional arguments for HookedTransformer initialization
            **model_kwargs: Keyword arguments for HookedTransformer initialization
        """
        super().__init__(*model_args, **model_kwargs)
        self.acts_to_saes: Dict[str, SparseAutoencoder] = {}

    def attach_sae(
        self,
        sae: SparseAutoencoder,
        turn_on: bool = True,
        hook_name: Optional[str] = None,
    ):
        """Attach an SAE to the model.

        By default, it will use the hook_name from the SAE's HookedSAEConfig. If you want to use a different hook_name, you can pass it in as an argument.
        By default, the SAE will be turned on. If you want to attach the SAE without turning it on, you can pass in turn_on=False.

        Args:
            sae: (HookedAutoEncoder) SAE that you want to attach
            turn_on: if true, turn on the SAE (default: True)
            hook_name: (Optional[str]) The hook name to attach the SAE to (default: None)
        """
        act_name = hook_name or sae.cfg.hook_point
        if (act_name not in self.acts_to_saes) and (act_name not in self.hook_dict):
            logging.warning(
                f"No hook found for {act_name}. Skipping. Check model.hook_dict for available hooks."
            )
            return
        if act_name in self.acts_to_saes:
            logging.warning(
                f"SAE already attached to {act_name}. This will be replaced."
            )
        self.acts_to_saes[act_name] = sae
        if turn_on:
            self.turn_saes_on([act_name])

    def turn_saes_on(self, act_names: Optional[Union[str, List[str]]] = None):
        """
        Turn on the attached SAEs for the given act_name(s)

        Note they will stay on you turn them off

        Args:
            act_names: (Union[str, List[str]]) The act_names for the SAEs to turn on
        """
        if isinstance(act_names, str):
            act_names = [act_names]

        for act_name in act_names or self.acts_to_saes.keys():
            if act_name not in self.acts_to_saes:
                logging.warning(f"No SAE is attached to {act_name}. Skipping.")
            else:
                set_deep_attr(self, act_name, self.acts_to_saes[act_name])

        self.setup()

    def turn_saes_off(self, act_names: Optional[Union[str, List[str]]] = None):
        """
        Turns off the SAEs for the given act_name(s)

        If no act_names are given, will turn off all SAEs

        Args:
            act_names: (Optional[Union[str, List[str]]]) The act_names for the SAEs to turn off. Defaults to None.
        """
        if isinstance(act_names, str):
            act_names = [act_names]

        for act_name in act_names or self.acts_to_saes.keys():
            if act_name not in self.acts_to_saes:
                logging.warning(
                    f"No SAE is attached to {act_name}. There's nothing to turn off."
                )
            else:
                set_deep_attr(self, act_name, HookPoint())

        self.setup()

    def run_with_saes(
        self,
        *model_args,
        act_names: List[str] = [],
        **model_kwargs,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Loss,
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    ]:
        """Wrapper around HookedTransformer forward pass.

        Runs the model with the SAEs for given act_names turned on.
        Note this will turn off all other SAEs that are not in act_names before running
        After running, it will turn off all SAEs

        Args:
            *model_args: Positional arguments for the model forward pass
            act_names: (List[str]) The act_names for the SAEs to turn on for this forward pass
            **model_kwargs: Keyword arguments for the model forward pass
        """
        self.turn_saes_off()
        try:
            self.turn_saes_on(act_names)
            out = self(*model_args, **model_kwargs)
        finally:
            self.turn_saes_off()
        return out

    def run_with_cache_with_saes(
        self,
        *model_args,
        act_names: List[str] = [],
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """Wrapper around 'run_with_cache' in HookedTransformer.

        Turns on the SAEs for the given act_names before running the model with cache and then turns them off after
        Note this will turn off all other SAEs that are not in act_names before running
        After running, it will turn off all SAEs

        Args:
            *model_args: Positional arguments for the model forward pass
            act_names: (List[str]) The act_names for the SAEs to turn on for this forward pass
            return_cache_object: (bool) if True, this will return an ActivationCache object, with a bunch of
                useful HookedTransformer specific methods, otherwise it will return a dictionary of
                activations as in HookedRootModule.
            remove_batch_dim: (bool) Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            **kwargs: Keyword arguments for the model forward pass
        """
        self.turn_saes_off()
        try:
            self.turn_saes_on(act_names)
            out = self.run_with_cache(
                *model_args,
                return_cache_object=return_cache_object,
                remove_batch_dim=remove_batch_dim,
                **kwargs,
            )
        finally:
            self.turn_saes_off()
        return out

    def run_with_hooks_with_saes(
        self,
        *model_args,
        act_names: List[str] = [],
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end=True,
        clear_contexts=False,
        **model_kwargs,
    ):
        """Wrapper around 'run_with_hooks' in HookedTransformer.

        Turns on the SAEs for the given act_names before running the model with hooks and then turns them off after
        Note this will turn off all other SAEs that are not in act_names before running
        After running, it will turn off all SAEs

        ARgs:
            *model_args: Positional arguments for the model forward pass
            act_names: (List[str]) The act_names for the SAEs to turn on for this forward pass
            fwd_hooks: (List[Tuple[Union[str, Callable], Callable]]) List of forward hooks to apply
            bwd_hooks: (List[Tuple[Union[str, Callable], Callable]]) List of backward hooks to apply
            reset_hooks_end: (bool) Whether to reset the hooks at the end of the forward pass (default: True)
            clear_contexts: (bool) Whether to clear the contexts at the end of the forward pass (default: False)
            **model_kwargs: Keyword arguments for the model forward pass
        """
        self.turn_saes_off()
        try:
            self.turn_saes_on(act_names)
            out = self.run_with_hooks(
                *model_args,
                fwd_hooks=fwd_hooks,
                bwd_hooks=bwd_hooks,
                reset_hooks_end=reset_hooks_end,
                clear_contexts=clear_contexts,
                **model_kwargs,
            )
        finally:
            self.turn_saes_off()
        return out

    def get_saes_status(self):
        """
        Helper function to check which SAEs attached to the model are currently turned on / off

        Returns:
            Dict[str, bool]: A dictionary of act_name to whether the corresponding SAE is turned on
        """
        return {
            act_name: (
                False if isinstance(get_deep_attr(self, act_name), HookPoint) else True
            )
            for act_name in self.acts_to_saes.keys()
        }


# %% [markdown]
# # Using HookedSAETransformer

# %%
device = "cuda:0"
model = model.to(device)


prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
names = [
    (
        " John",
        " Mary",
    ),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]
# List of prompts
prompts = []
# List of answers, in the format (correct, incorrect)
answers = []
# List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
answer_tokens = []
for i in range(len(prompt_format)):
    for j in range(2):
        answers.append((names[i][j], names[i][1 - j]))
        answer_tokens.append(
            (
                model.to_single_token(answers[-1][0]),
                model.to_single_token(answers[-1][1]),
            )
        )
        # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
        prompts.append(prompt_format[i].format(answers[-1][1]))
answer_tokens = torch.tensor(answer_tokens).to(device)
print(prompts)
print(answers)


# %%
def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()


tokens = model.to_tokens(prompts, prepend_bos=True)
original_logits, cache = model.run_with_cache(tokens)
original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
print(f"Original average logit diff: {original_average_logit_diff}")
original_per_prompt_logit_diff = logits_to_ave_logit_diff(
    original_logits, answer_tokens, per_prompt=True
)
print(f"Original per prompt logit diff: {original_per_prompt_logit_diff}")

# %%
model: HookedSAETransformer = HookedSAETransformer.from_pretrained("gpt2-small").to(
    device
)

# %%
# HookedSAETransformer will have this method.
model.attach_sae(inference_sparse_autoencoder)
# %%
assert model.cfg.device == str(tokens.device), (
    f"Model device: {model.cfg.device}, " f"tokens device: {tokens.device}"
)
# %%
logits_with_saes = model(tokens)
average_logit_diff_with_saes = logits_to_ave_logit_diff(logits_with_saes, answer_tokens)
print(f"Average logit diff with SAEs: {average_logit_diff_with_saes}")
per_prompt_diff_with_saes = logits_to_ave_logit_diff(
    logits_with_saes, answer_tokens, per_prompt=True
)

# %%
for sae in gpt2_small_sparse_autoencoders.values():
    sae.cfg.device = device
    sae.to(device)
    model.attach_sae(cast_to_inference_sparse_autoencoder(sae))
print("SAEs turned on before:", model.get_saes_status())
# model.turn_saes_off()
# print("SAEs turned on after:", model.get_saes_status())

# %%
answer_tokens.device

# %%
from transformer_lens import utils


def zero_ablate_resid(resid, hook, pos=None):
    if pos is None:
        resid[:] = 0.0
    else:
        resid[:, pos, :] = 0.0
    return resid


layers = [5, 6]
act_names = [
    utils.get_act_name("resid_pre", layer) + ".hook_sae_out" for layer in layers
]
zero_abl_logits = model.run_with_hooks(
    tokens,
    return_type="logits",
    fwd_hooks=[(act_name, zero_ablate_resid) for act_name in act_names],
)

per_prompt_zero_abl_logit_diff = logits_to_ave_logit_diff(
    zero_abl_logits, answer_tokens, per_prompt=True
)
avg_zero_abl_logit_diff = logits_to_ave_logit_diff(zero_abl_logits, answer_tokens)
print(f"Zero ablated logit diff: {avg_zero_abl_logit_diff}")

# %%
from typing import List
import plotly.graph_objects as go


def show_avg_logit_diffs(x_axis: List[str], per_prompt_logit_diffs: List[torch.tensor]):

    y_data = [
        per_prompt_logit_diff.mean().item()
        for per_prompt_logit_diff in per_prompt_logit_diffs
    ]
    error_y_data = [
        per_prompt_logit_diff.std().item()
        for per_prompt_logit_diff in per_prompt_logit_diffs
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=x_axis,
                y=y_data,
                error_y=dict(
                    type="data",  # specifies that the actual values are given
                    array=error_y_data,  # the magnitudes of the errors
                    visible=True,  # make error bars visible
                ),
            )
        ]
    )

    # Customize layout
    fig.update_layout(
        title_text=f"Logit Diff after Interventions",
        xaxis_title_text="Intervention",
        yaxis_title_text="Logit diff",
        plot_bgcolor="white",
    )

    # Show the figure
    fig.show()


all_layers = [[i] for i in range(12)]
x_axis = ["Clean Baseline", "Zero Abl L[5, 6]"]
per_prompt_logit_diffs = [
    original_per_prompt_logit_diff,
    per_prompt_zero_abl_logit_diff,
]

for layers in all_layers:
    act_names = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
    logits_with_saes = model.run_with_saes(tokens, act_names=act_names)
    average_logit_diff_with_saes = logits_to_ave_logit_diff(
        logits_with_saes, answer_tokens
    )
    per_prompt_diff_with_saes = logits_to_ave_logit_diff(
        logits_with_saes, answer_tokens, per_prompt=True
    )

    x_axis.append(f"With SAEs L{layers}")
    per_prompt_logit_diffs.append(per_prompt_diff_with_saes)

show_avg_logit_diffs(x_axis, per_prompt_logit_diffs)

# %%


# %%
model.turn_saes_off()
model.get_saes_status()
model.turn_saes_on([f"blocks.{layer}.hook_resid_pre" for layer in [6]])
_, cache = model.run_with_cache(tokens)
model.get_saes_status()

layer = 6
_, cache = model.run_with_cache(tokens)
s2_pos = 10
print(model.to_str_tokens(prompts[0])[s2_pos])
sae_acts = cache[utils.get_act_name("resid_pre", layer) + ".hook_hidden_post"][
    :, s2_pos, :
]
live_feature_mask = sae_acts > 0.5
live_feature_union = live_feature_mask.any(dim=0)
print(f"Live features: {live_feature_union.sum()}")

# %%
display(prompts)
px.imshow(
    sae_acts[:, live_feature_union].detach().cpu(),
    title=f"Activations of Live SAE features at L{layer} S2 position per prompt",
    # xaxis="Feature Id", yaxis="Prompt",
    x=list(map(str, live_feature_union.nonzero().flatten().tolist())),
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
)

# %%
vals, inds = torch.topk(sae_acts[:, :].detach().cpu().sum(dim=0), 8)
open_neuronpedia(inds.tolist(), layer=layer, name="IOI S2 Names")

# %% [markdown]
# # Ablation + Measure Logit Difference

# %%
from tqdm import tqdm
from functools import partial

LAYER = 6
model.turn_saes_off()
model.get_saes_status()
model.turn_saes_on([f"blocks.{i}.hook_resid_pre" for i in [LAYER]])
_, cache = model.run_with_cache(tokens)
model.get_saes_status()

logits_with_saes = model(tokens)
clean_sae_baseline_avg = logits_to_ave_logit_diff(logits_with_saes, answer_tokens)
clean_sae_baseline_per_prompt = logits_to_ave_logit_diff(
    logits_with_saes, answer_tokens, per_prompt=True
)

# %%


def ablate_sae_feature(sae_acts, hook, pos, feature_id):
    if pos is None:
        sae_acts[:, :, feature_id] = 0.0
    else:
        sae_acts[:, pos, feature_id] = 0.0
    return sae_acts


layer = LAYER
hooked_encoder = model.acts_to_saes[utils.get_act_name("resid_pre", layer)]
all_live_features = torch.arange(hooked_encoder.cfg.d_sae)[live_feature_union.cpu()]


causal_effects = torch.zeros((len(prompts), all_live_features.shape[0]))
fid_to_idx = {fid.item(): idx for idx, fid in enumerate(all_live_features)}


abl_layer, abl_pos = LAYER, 10
for feature_id in tqdm(all_live_features):
    feature_id = feature_id.item()
    abl_feature_logits = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=[
            (
                utils.get_act_name("resid_pre", abl_layer) + ".hook_hidden_post",
                partial(ablate_sae_feature, pos=abl_pos, feature_id=feature_id),
            )
        ],
    )  # [batch, seq, vocab]

    abl_feature_logit_diff = logits_to_ave_logit_diff(
        abl_feature_logits, answer_tokens, per_prompt=True
    )  # [batch]
    del abl_feature_logits
    torch.cuda.empty_cache()
    causal_effects[:, fid_to_idx[feature_id]] = (
        abl_feature_logit_diff - clean_sae_baseline_per_prompt
    )

# %%

fig = px.imshow(
    causal_effects.detach().cpu(),
    title=f"Change in logit diff when ablating L{abl_layer} SAE features for all prompts at pos {abl_pos}",
    x=list(map(str, all_live_features.tolist())),
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
)

# label x and y axis
fig.update_xaxes(title_text="Feature Id")
fig.update_yaxes(title_text="Prompt")
fig.show()

# %%
vals, inds = torch.topk(causal_effects.abs().sum(dim=0), 8)
print(vals, inds.tolist())

# %%
open_neuronpedia(
    [int(i) for i in all_live_features[inds]],
    layer=LAYER,
)

# %% [markdown]
# # Ablating SAE Features and looking at the downstream effects

# %%
model.turn_saes_off()
model.get_saes_status()

# %%
prompt = "I loved this movie. It was great. \nIn summary: This movie is"
# prompt = "John and Mary went to the store and then John said to"
utils.test_prompt(prompt, " great", model)

layer, position = 3, 11
model.turn_saes_off()
model.get_saes_status()
model.turn_saes_on([f"blocks.{layer}.hook_resid_pre" for layer in [layer]])
model.get_saes_status()
print(f"token at position {position}: '{model.to_str_tokens(prompt)[position]}'")
_, cache = model.run_with_cache(prompt)

sae_acts = cache[utils.get_act_name("resid_pre", layer) + ".hook_hidden_post"][
    :, position, :
]
print(sae_acts.shape)
live_feature_mask = sae_acts > 0
live_feature_union = live_feature_mask.any(dim=0)
all_live_features = torch.arange(live_feature_union.shape[-1])[live_feature_union.cpu()]
print(f"Live features: {live_feature_union.sum()}")

# %%
specific_token = model.to_str_tokens(prompt)[position]
px.line(
    sae_acts.squeeze().detach().cpu(),
    title=f"Activations of Live SAE features at L{layer} token {position}/{specific_token} per prompt",
).show()
n = 5
vals, inds = torch.topk(sae_acts.squeeze().detach().cpu(), n)
print(inds.tolist())
open_neuronpedia(
    inds.tolist(),
    layer=layer,
    name=f"Top {n} SAE features at L{layer} token {position}/{specific_token} per prompt",
)

# %%
import pandas as pd
import circuitsvis as cv

model.turn_saes_off()
prompt = "I loved this movie. It was great. \nIn summary: This movie is"
prompt = model.generate(
    prompt, max_new_tokens=40, stop_at_eos=False, temperature=0.7, verbose=False
)
(original_logits, original_loss), clean_cache = model.run_with_cache(
    prompt, return_type="both", loss_per_token=True, prepend_bos=True
)
model.turn_saes_on([f"blocks.{layer}.hook_resid_pre" for layer in [layer]])
(sae_logits, sae_loss), sae_cache = model.run_with_cache(
    prompt, return_type="both", loss_per_token=True, prepend_bos=True
)
# Let's make a longer prompt and see the log probabilities of the tokens
cv.logits.token_log_probs(
    model.to_tokens(prompt),
    model(prompt)[0].log_softmax(dim=-1),
    model.to_string,
)

# %%
tmp_feature_acts_df = pd.DataFrame(
    sae_cache[utils.get_act_name("resid_pre", abl_layer) + ".hook_hidden_post"]
    .squeeze()[:, live_feature_union.to("cpu")]
    .to("cpu")
    .numpy(),
    index=[f"{i}/{tok}" for i, tok in enumerate(model.to_str_tokens(prompt))],
    columns=all_live_features.tolist(),
)
px.line(tmp_feature_acts_df)

# %%
from torch.nn.functional import kl_div

str_tokens = model.to_str_tokens(prompt)[:-1]
unique_tokens = [f"{i}/{token}" for i, token in enumerate(str_tokens)]
ablation_df = pd.DataFrame(
    {
        "unique_tokens": unique_tokens,
        "original_loss": original_loss.squeeze().detach().cpu().numpy(),
        "sae_loss": sae_loss.squeeze().detach().cpu().numpy(),
    }
)


measure_position = 17
print(model.to_str_tokens(prompt)[measure_position])
vocab_df = pd.DataFrame(model.tokenizer.vocab, index=["token"]).T
vocab_df = vocab_df.sort_values("token")
vocab_df["original_logits"] = (
    original_logits[:, measure_position].squeeze().detach().cpu().numpy()
)
vocab_df["original_logprobs"] = (
    original_logits[:, measure_position]
    .squeeze()
    .detach()
    .cpu()
    .log_softmax(dim=-1)
    .numpy()
)
vocab_df["sae_logits"] = (
    sae_logits[:, measure_position].squeeze().detach().cpu().numpy()
)
vocab_df["sae_logprobs"] = (
    sae_logits[:, measure_position].squeeze().detach().cpu().log_softmax(dim=-1).numpy()
)

vocab_df.sort_values("original_logits", ascending=False).head(10)


# %%
def ablate_sae_feature(sae_acts, hook, pos, feature_id, inclusive=False):
    if pos is None:
        sae_acts[:, :, feature_id] = 0.0
    else:
        if not inclusive:
            sae_acts[:, pos, feature_id] = 0.0
        else:
            # ablated from that pos onwards
            pos = 1 + pos  # inclusive
            sae_acts[:, :pos, feature_id] = 0.0

    return sae_acts


abl_layer, abl_pos = layer, position


# features_to_ablate = all_live_features.tolist()
features_to_ablate = [12508, 12266, 22477]

for feature_id in tqdm(features_to_ablate):
    abl_logits, abl_loss = model.run_with_hooks(
        prompt,
        return_type="both",
        loss_per_token=True,
        fwd_hooks=[
            (
                utils.get_act_name("resid_pre", abl_layer) + ".hook_hidden_post",
                partial(
                    ablate_sae_feature,
                    pos=abl_pos,
                    feature_id=feature_id,
                    inclusive=True,
                ),
            )
        ],
    )
    ablation_df[f"abl_loss_{feature_id}"] = abl_loss.squeeze().detach().cpu().numpy()
    ablation_df[f"m_abl_{feature_id}"] = (
        abl_loss - original_loss
    ).squeeze().detach().cpu().numpy() - (
        sae_loss - original_loss
    ).squeeze().detach().cpu().numpy()
    ablation_df[f"kl_div_{feature_id}"] = (
        kl_div(
            sae_logits[:, measure_position].softmax(dim=-1),
            abl_logits[:, measure_position].softmax(dim=-1),
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    vocab_df[f"abl_logits_{feature_id}"] = (
        abl_logits[:, measure_position].squeeze().detach().cpu().numpy()
    )
    vocab_df[f"m_abl_{feature_id}"] = (abl_logits - original_logits)[
        :, measure_position
    ].squeeze().detach().cpu().numpy() - (sae_logits - original_logits)[
        :, measure_position
    ].squeeze().detach().cpu().numpy()
    vocab_df[f"log_prob_abl_{feature_id}"] = (
        abl_logits[:, measure_position]
        .squeeze()
        .detach()
        .cpu()
        .log_softmax(dim=-1)
        .numpy()
    )
    vocab_df[f"m_log_prob_abl_{feature_id}"] = (
        vocab_df[f"log_prob_abl_{feature_id}"] - vocab_df["sae_logprobs"]
    )


ablation_df.head()
px.line(
    ablation_df,
    x="unique_tokens",
    y=["original_loss", "sae_loss"] + [f"abl_loss_{i}" for i in features_to_ablate],
    title="Loss per token with and without SAEs",
    labels={"value": "Loss", "variable": "Model"},
)

# %%
tmp_df = ablation_df.filter(like="m_abl").T.astype(float)
# tmp_df = ablation_df.filter(like="kl_div").T.astype(float)
tmp_df.columns = unique_tokens
# tmp_df.style.format("{:.2f}").background_gradient(cmap='RdBu', axis=1)
tmp_df.style.format("{:.5f}").background_gradient(cmap="RdBu", axis=1)

# %%
# convert long table to wide table and have a column for index, column and value
wide_df = tmp_df.stack().reset_index()
wide_df.columns = ["feature_id", "unique_tokens", "m_abl"]
wide_df["feature_id"] = wide_df["feature_id"].str.replace("m_abl_", "").astype(int)

# wide_df.columns = ["feature_id", "unique_tokens", "kl_div"]
# wide_df["feature_id"] = wide_df["feature_id"].str.replace("kl_div_", "").astype(int)

fig = px.strip(
    wide_df,
    x="unique_tokens",
    y="m_abl",
    color="feature_id",
    hover_data=["feature_id"],
    height=600,
    title="Difference in Loss per token after ablating SAE features",
)
# rotate x-axis labels
fig.update_xaxes(tickangle=45)
fig.show()

# %% [markdown]
# Experiment to get logit rankings

# %%
vocab_df

# %%
tmp_df = vocab_df.sort_values("original_logits", ascending=False)[
    ["original_logprobs", "sae_logprobs", "m_log_prob_abl_12266"]
].head(1000)
px.parallel_coordinates(
    tmp_df,
    # color="original_logits",
    title="Top 30 tokens by original logits",
)

# %%
feature_id = 12266
top_n = 1000
top_n_tokens_and_feature_column = vocab_df.sort_values(
    "original_logits", ascending=False
)[
    [
        "original_logprobs",
        "original_logits",
        "sae_logits",
        f"m_log_prob_abl_{feature_id}",
    ]
].head(
    top_n
)


display(
    top_n_tokens_and_feature_column.sort_values("original_logits", ascending=False)
    .head(30)
    .style.format("{:.2f}")
    .background_gradient(cmap="RdBu", axis=0)
)

for ascending_effect in [True, False]:
    result = (
        top_n_tokens_and_feature_column.sort_values("original_logits", ascending=False)
        .sort_values(f"m_log_prob_abl_{feature_id}", ascending=ascending_effect)
        .head(30)
    )
    result = result.style.format("{:.2f}").background_gradient(cmap="RdBu", axis=0)
    display(result)


# %%


# %%
# Libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt

plt.figure(figsize=(30, 30))
word_cloud_frequencies = (
    vocab_df.sort_values("original_logits", ascending=False)
    .head(10000)
    .sort_values("m_abl_12266", ascending=True)["m_abl_12266"]
)


display(word_cloud_frequencies.head())
display(word_cloud_frequencies.tail())

wordcloud = WordCloud(width=1200, height=400, margin=0).generate_from_frequencies(
    word_cloud_frequencies.apply(lambda x: -1 * x).to_dict()
)
# wordcloud.
# # Display the generated image:
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.margins(x=0, y=0)
plt.show()

# reset plt
plt.close()

# plt.figure(figsize=(30, 30))
#

# display(word_cloud_frequencies.head())
# display(word_cloud_frequencies.tail())

# wordcloud = WordCloud(width=1200, height=400, margin=0).generate_from_frequencies(word_cloud_frequencies.apply(lambda x: -1*x).to_dict())

# # wordcloud.
# # # Display the generated image:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# # plt.margins(x=0, y=0)
# plt.show()

# # reset plt
# plt.close()

# %%
vocab_df.sort_values("original_logits", ascending=False).head(1000).sort_values(
    "m_abl_15006", ascending=True
).head(100)[["m_abl_15006"]].style.format("{:.2f}").background_gradient(
    cmap="RdBu", axis=0
)
