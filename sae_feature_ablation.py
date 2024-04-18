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
from transformer_lens import utils
from sae_lens import SparseAutoencoder

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
import pandas as pd
import circuitsvis as cv
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from jaxtyping import Float, Int

from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint  # Hooking utilities
from torch.nn.functional import kl_div
from transformer_lens.evals import load_dataset, DataLoader
from transformer_lens.utils import tokenize_and_concatenate
from typing import List
import plotly.graph_objects as go
from tqdm.auto import tqdm
from functools import partial
import einops
import itertools
import plotly.io as pio


# %%
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
torch.set_grad_enabled(False)
device = "cuda:0"
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
# LAYER = 8
# features_to_ablate = [12477, 6614, 13563, 8341, 7276, 6508, 16021, 2761, 20847, 23510]
# LAYER = 3
# features_to_ablate = [12508, 12266, 22477, 3121, 4135, 7838, 4494]
LAYER = 7
features_to_ablate = [22595, 23063]
gpt2_small_sparse_autoencoders, gpt2_small_sae_sparsities = get_gpt2_res_jb_saes()

sparse_autoencoder = gpt2_small_sparse_autoencoders[
    utils.get_act_name("resid_pre", LAYER)
]
sparse_autoencoder.eval()

model = model.to(device)
assert isinstance(sparse_autoencoder.cfg.d_sae, int)


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


# %%
def ablate_sae_feature(sae_acts, hook, pos, feature_id):
    if pos is None:
        sae_acts[:, :, feature_id] = 0.0
    else:
        sae_acts[:, pos, feature_id] = 0.0

    return sae_acts


# %%
def ablate_sae_features_for_prompt(
    tokens: Int[Tensor, "n_ctx"],
    abl_layer: int,
    abl_pos: int,
    sae_features: List[int],
    model: HookedSAETransformer,
    prepend_bos: bool = True,
) -> pd.DataFrame:
    resid_pre_name = f"blocks.{abl_layer}.hook_resid_pre"
    hidden_post_name = utils.get_act_name("resid_pre", abl_layer) + ".hook_hidden_post"
    unique_tokens = [
        f"{i+1}/{token}" for i, token in enumerate(model.to_str_tokens(tokens)[1:])
    ]
    model.turn_saes_off()
    original_loss = model(
        tokens, return_type="loss", loss_per_token=True, prepend_bos=prepend_bos
    ).squeeze(0)
    model.turn_saes_on([resid_pre_name])
    sae_loss, sae_cache = model.run_with_cache(
        tokens, return_type="loss", loss_per_token=True, prepend_bos=prepend_bos
    )
    assert isinstance(sae_loss, torch.Tensor)
    sae_loss = sae_loss.squeeze(0)
    ablation_df = pd.DataFrame(
        columns=[
            "unique_tokens",
            "original_loss",
            "sae_loss",
            *[f"abl_loss_{feature}" for feature in sae_features],
            *[f"abl_loss_diff_{feature}" for feature in sae_features],
            *[f"act_{feature}" for feature in sae_features],
        ],
        index=range(len(tokens) - 1),
    )
    ablation_df["unique_tokens"] = unique_tokens
    ablation_df["original_loss"] = original_loss.detach().cpu().numpy()
    ablation_df["sae_loss"] = sae_loss.detach().cpu().numpy()
    feature_iter = tqdm(sae_features) if len(sae_features) > 1 else sae_features
    for feature_id in feature_iter:
        abl_loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            loss_per_token=True,
            fwd_hooks=[
                (
                    hidden_post_name,
                    partial(
                        ablate_sae_feature,
                        pos=abl_pos,
                        feature_id=feature_id,
                    ),
                )
            ],
            prepend_bos=prepend_bos,
        ).squeeze(0)
        ablation_df[f"abl_loss_{feature_id}"] = abl_loss.detach().cpu().numpy()
        ablation_df[f"abl_loss_diff_{feature_id}"] = (
            (abl_loss - sae_loss).detach().cpu().numpy()
        )
        ablation_df[f"abl_loss_pct_diff_{feature_id}"] = (
            ((abl_loss - sae_loss) / sae_loss).detach().cpu().numpy()
        )
        ablation_df[f"act_{feature_id}"] = (
            sae_cache[hidden_post_name][0, :-1, feature_id].detach().cpu().numpy()
        )

    return ablation_df


# %%
def plot_ablation_results(
    ablation_df: pd.DataFrame,
    feature: int,
    activation: Optional[int],
    position: Optional[int],
    loss_info: str = "diff",
) -> go.Figure:
    assert isinstance(feature, int)
    fig = go.Figure()
    x_labels = ablation_df["unique_tokens"].values
    x_values = np.arange(len(x_labels))
    # Add traces for original_loss, sae_loss, and abl_loss_{feature}
    if loss_info == "diff":
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=ablation_df[f"abl_loss_diff_{feature}"],
                mode="lines",
                name=f"abl_loss_diff_{feature}",
                yaxis="y",
                text=[
                    f"SAE Loss: {loss:.2f}" for loss in ablation_df["sae_loss"]
                ],  # Formatting hover text with label
                hoverinfo="x+y+text",  # Show x, y values, and the text in hover
            )
        )
        yaxis_title = "Loss Difference"
    elif loss_info == "pct_diff":
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=ablation_df[f"abl_loss_pct_diff_{feature}"],
                mode="lines",
                name=f"abl_loss_pct_diff_{feature}",
                yaxis="y",
                text=[
                    f"SAE Loss: {loss:.2f}" for loss in ablation_df["sae_loss"]
                ],  # Formatting hover text with label
                hoverinfo="x+y+text",  # Show x, y values, and the text in hover
            )
        )
        yaxis_title = "Loss Percentage Difference"
    else:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=ablation_df["original_loss"],
                mode="lines",
                name="original_loss",
                yaxis="y",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=ablation_df["sae_loss"],
                mode="lines",
                name="sae_loss",
                yaxis="y",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=ablation_df[f"abl_loss_{feature}"],
                mode="lines",
                name=f"abl_loss_{feature}",
                yaxis="y",
            )
        )
        yaxis_title = "Loss"

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=ablation_df[f"act_{feature}"].shift(-1),
            mode="lines",
            name=f"act_{feature}",
            yaxis="y2",  # Assign this trace to the second y-axis
            # make the line dashed
            line_dash="dash",
        )
    )

    if position is not None:
        # Add a vertical dotted line at act_pos
        fig.add_vline(
            x=position,
            line_dash="dot",  # Makes the line dotted
            annotation_text="first ablation position",
        )
    title = f"Loss per token with and without SAEs: feature {feature}"
    if activation is not None:
        title += f", activation {activation}"
    fig.update_layout(
        title=title,
        xaxis_title="Unique Tokens",
        yaxis_title=yaxis_title,
        yaxis2=dict(title="Activation", overlaying="y", side="right"),
        xaxis=dict(tickvals=x_values, ticktext=x_labels),
    )
    return fig


# %% [markdown]
# # Using HookedSAETransformer


# %%
model = HookedSAETransformer.from_pretrained("gpt2-small").to(device)
# %%
# HookedSAETransformer will have this method.
inference_sparse_autoencoder.to(device)
model.attach_sae(inference_sparse_autoencoder)
# %%
prompt = "I loved this movie.\nConclusion: this movie was great"
prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
abl_pos = torch.where(prompt_tokens == model.to_single_token("."))[1].item()
print(abl_pos)
# %%
_, prompt_cache = model.run_with_cache(
    prompt_tokens,
    return_type=None,
    names_filter=lambda name: "hidden_post" in name,
    prepend_bos=False,  # already tokenized
)
feature_prompt_activations: Float[Tensor, "pos feature"] = prompt_cache[
    utils.get_act_name("resid_pre", LAYER) + ".hook_hidden_post"
].squeeze(0)
prompt_features = torch.where(feature_prompt_activations[abl_pos])[0].tolist()
# %%
prompt_ablation_df = ablate_sae_features_for_prompt(
    prompt_tokens,
    LAYER,
    abl_pos,
    prompt_features,
    model,
    prepend_bos=True,
)
# %%
loss_diff_cols = [col for col in prompt_ablation_df.columns if "abl_loss_diff" in col]
prompt_ablation_df[loss_diff_cols].iloc[-1].describe()
# %%
k_features = 10
topk_features = (
    prompt_ablation_df.iloc[-1][loss_diff_cols]
    .astype(float)
    .nlargest(k_features)
    .index.str.split("_")
    .str[-1]
    .astype(int)
    .values.tolist()
)
print(topk_features)
# %%
prompt_ablation_df[[f"abl_loss_diff_{f}" for f in topk_features]].iloc[-1]
# %%
for feature in topk_features:
    fig = plot_ablation_results(prompt_ablation_df, feature, None, abl_pos)
    fig.show()


# %% [markdown]
# # OpenWebText activations
def make_owt_data_loader(tokenizer, batch_size=8):
    """
    Evaluate on OpenWebText an open source replication of the GPT-2 training corpus (Reddit links with >3 karma)

    I think the Mistral models were trained on this dataset, so they get very good performance.
    """
    owt_data = load_dataset("stas/openwebtext-10k", split="train")
    print(len(owt_data))
    dataset = tokenize_and_concatenate(
        owt_data, tokenizer, max_length=sparse_autoencoder.cfg.context_size
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return data_loader


# %%
batch_size = 8
model.tokenizer.model_max_length = sparse_autoencoder.cfg.context_size
dataloader = make_owt_data_loader(model.tokenizer, batch_size=batch_size)
# %%
is_comma_mask = torch.zeros(
    (
        len(dataloader.dataset),
        sparse_autoencoder.cfg.context_size,
    ),
    dtype=torch.bool,
)
for batch_idx, batch in enumerate(tqdm(dataloader)):
    is_comma_mask[batch_idx * batch_size : (batch_idx + 1) * batch_size] = batch[
        "tokens"
    ] == model.to_single_token(",")
# %%
feature_owt_activations = torch.empty(
    (
        len(dataloader.dataset),
        sparse_autoencoder.cfg.context_size,
        len(features_to_ablate),
    ),
    dtype=torch.float32,
)
for batch_idx, batch in enumerate(tqdm(dataloader)):
    _, batch_cache = model.run_with_cache(
        batch["tokens"],
        return_type=None,
        names_filter=lambda name: "hidden_post" in name,
        prepend_bos=False,  # already tokenized
    )
    batch_acts = batch_cache[
        utils.get_act_name("resid_pre", LAYER) + ".hook_hidden_post"
    ][:, :, features_to_ablate]
    feature_owt_activations[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
        batch_acts.detach().cpu()
    )
# %%
# max activations per feature
feature_owt_activations.max(dim=0).values.max(dim=0).values


# %%
def get_topk_activation_positions(
    activations: Float[Tensor, "batch pos feature"],
    topk: int = 10,
) -> Tuple[Int[Tensor, "feature topk"], Int[Tensor, "feature topk"]]:
    max_act_per_batch, max_pos_per_batch = activations.max(dim=1)  # [batch, feature]
    topk_batches: Int[Tensor, "topk feature"] = max_act_per_batch.topk(
        k=topk, dim=0, largest=True
    ).indices
    n_features = activations.shape[-1]
    topk_positions: Int[Tensor, "topk feature"] = max_pos_per_batch[topk_batches][
        :, torch.arange(n_features), torch.arange(n_features)
    ]
    return topk_batches.T, topk_positions.T


# %%
k_activations = 10  # number of top activations to look at
# weighted_owt_activations = feature_owt_activations * einops.repeat(
#     is_comma_mask, "batch pos -> batch pos feature", feature=len(features_to_ablate)
# )
weighted_owt_activations = feature_owt_activations
top_activation_batches, top_activation_positions = get_topk_activation_positions(
    weighted_owt_activations,
    k_activations,
)
assert top_activation_batches.shape == (len(features_to_ablate), k_activations)
assert top_activation_positions.shape == (len(features_to_ablate), k_activations)
assert top_activation_batches.max() < len(dataloader.dataset)
assert top_activation_positions.max() < model.cfg.n_ctx
# %% [markdown]
# # Ablation + Measure Loss Difference

# %%
left_width = 10
right_width = 20
min_loss_diff = 0.1
activation_feature_iter = tqdm(
    list(itertools.product(range(len(features_to_ablate)), range(k_activations))),
    total=len(features_to_ablate) * k_activations,
)
for feature_i, act_idx in activation_feature_iter:
    act_batch = top_activation_batches[feature_i, act_idx].item()
    act_pos = top_activation_positions[feature_i, act_idx].item()
    assert isinstance(act_batch, int)
    assert isinstance(act_pos, int)
    # assert feature_owt_activations[
    #     act_batch, act_pos, feature_i
    # ] > feature_owt_activations.quantile(0.99)
    start_pos = max(0, act_pos - left_width)
    end_pos = min(act_pos + right_width, model.cfg.n_ctx)
    tokens = dataloader.dataset[act_batch]["tokens"]
    feature = features_to_ablate[feature_i]
    ablation_df = ablate_sae_features_for_prompt(
        tokens, LAYER, act_pos, [feature], model, prepend_bos=False
    )
    if (
        ablation_df[f"abl_loss_diff_{feature}"].iloc[act_pos + 1 : end_pos].max()
        < min_loss_diff
    ):
        continue
    fig = plot_ablation_results(
        ablation_df.iloc[start_pos:end_pos], feature, act_idx, act_pos - start_pos
    )
    fig.show()

# %%
