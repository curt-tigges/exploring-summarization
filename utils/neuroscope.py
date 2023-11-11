from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import einops
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate, get_act_name
from circuitsvis.activations import text_neuron_activations
from circuitsvis.topk_samples import topk_samples
from IPython.display import display
from utils.cache import resid_names_filter
from utils.store import ResultsFile


# Harry Potter in English
harry_potter_start = """
    Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense.

    Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere.

    The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn’t think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley’s sister, but they hadn’t met for several years; in fact, Mrs. Dursley pretended she didn’t have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn’t want Dudley mixing with a child like that.

    When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair.

    None of them noticed a large, tawny owl flutter past the window.

    At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs. Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls. “Little tyke,” chortled Mr. Dursley as he left the house. He got into his car and backed out of number four’s drive.

    It was on the corner of the street that he noticed the first sign of something peculiar — a cat reading a map. For a second, Mr. Dursley didn’t realize what he had seen — then he jerked his head around to look again. There was a tabby cat standing on the corner of Privet Drive, but there wasn’t a map in sight. What could he have been thinking of? It must have been a trick of the light. Mr. Dursley blinked and stared at the cat. It stared back. As Mr. Dursley drove around the corner and up the road, he watched the cat in his mirror. It was now reading the sign that said Privet Drive — no, looking at the sign; cats couldn’t read maps or signs. Mr. Dursley gave himself a little shake and put the cat out of his mind. As he drove toward town he thought of nothing except a large order of drills he was hoping to get that day.

    But on the edge of town, drills were driven out of his mind by something else. As he sat in the usual morning traffic jam, he couldn’t help noticing that there seemed to be a lot of strangely dressed people about. People in cloaks. Mr. Dursley couldn’t bear people who dressed in funny clothes — the getups you saw on young people! He supposed this was some stupid new fashion. He drummed his fingers on the steering wheel and his eyes fell on a huddle of these weirdos standing quite close by. They were whispering excitedly together. Mr. Dursley was enraged to see that a couple of them weren’t young at all; why, that man had to be older than he was, and wearing an emerald-green cloak! The nerve of him! But then it struck Mr. Dursley that this was probably some silly stunt — these people were obviously collecting for something . . . yes, that would be it. The traffic moved on and a few minutes later, Mr. Dursley arrived in the Grunnings parking lot, his mind back on drills.

    Mr. Dursley always sat with his back to the window in his office on the ninth floor. If he hadn’t, height have found it harder to concentrate on drills that morning. He didn’t see the owls swooping past in broad daylight, though people down in the street did; they pointed and gazed open-mouthed as owl after owl sped overhead. Most of them had never seen an owl even at nighttime. Mr. Dursley, however, had a perfectly normal, owl-free morning. He yelled at ﬁve diﬀerent people. He made several important telephone calls and shouted a bit more. He was in a very good mood until lunchtime, when he thought he’d stretch his legs and walk across the road to buy himself a bun from the bakery.

    He’d forgotten all about the people in cloaks until he passed a group of them next to the baker’s. He eyed them angrily as he passed. He didn’t know why, but they made him uneasy. This bunch were whispering excitedly, too, and he couldn’t see a single collecting tin. It was on his way back past them, clutching a large doughnut in a bag, that he caught a few words of what they were saying.
"""

# Harry Potter in French
harry_potter_fr_start = """
Mr et Mrs Dursley, qui habitaient au 4, Privet Drive, avaient toujours affirmé avec la plus grande
fierté qu'ils étaient parfaitement normaux, merci pour eux. Jamais quiconque n'aurait imaginé qu'ils
puissent se trouver impliqués dans quoi que ce soit d'étrange ou de mystérieux. Ils n'avaient pas de
temps à perdre avec des sornettes.
Mr Dursley dirigeait la Grunnings, une entreprise qui fabriquait des perceuses. C'était un homme
grand et massif, qui n'avait pratiquement pas de cou, mais possédait en revanche une moustache de
belle taille. Mrs Dursley, quant à elle, était mince et blonde et disposait d'un cou deux fois plus long
que la moyenne, ce qui lui était fort utile pour espionner ses voisins en regardant par-dessus les
clôtures des jardins. Les Dursley avaient un petit garçon prénommé Dudley et c'était à leurs yeux le
plus bel enfant du monde.
Les Dursley avaient tout ce qu'ils voulaient. La seule chose indésirable qu'ils possédaient, c'était un
secret dont ils craignaient plus que tout qu'on le découvre un jour. Si jamais quiconque venait à
entendre parler des Potter, ils étaient convaincus qu'ils ne s'en remettraient pas. Mrs Potter était la
soeur de Mrs Dursley, mais toutes deux ne s'étaient plus revues depuis des années. En fait, Mrs
Dursley faisait comme si elle était fille unique, car sa soeur et son bon à rien de mari étaient aussi
éloignés que possible de tout ce qui faisait un Dursley. Les Dursley tremblaient d'épouvante à la
pensée de ce que diraient les voisins si par malheur les Potter se montraient dans leur rue. Ils savaient
que les Potter, eux aussi, avaient un petit garçon, mais ils ne l'avaient jamais vu. Son existence
constituait une raison supplémentaire de tenir les Potter à distance: il n'était pas question que le petit
Dudley se mette à fréquenter un enfant comme celui-là.
Lorsque Mr et Mrs Dursley s'éveillèrent, au matin du mardi où commence cette histoire, il faisait gris
et triste et rien dans le ciel nuageux ne laissait prévoir que des choses étranges et mystérieuses allaient
bientôt se produire dans tout le pays. Mr Dursley fredonnait un air en nouant sa cravate la plus sinistre
pour aller travailler et Mrs Dursley racontait d'un ton badin les derniers potins du quartier en
s'efforçant d'installer sur sa chaise de bébé le jeune Dudley qui braillait de toute la force de ses
poumons.
"""


def get_projections_for_text(
    tokens: Int[Tensor, "batch pos"],
    special_dir: Float[Tensor, "d_model"],
    model: HookedTransformer,
) -> Float[Tensor, "batch pos layer"]:
    """Computes residual stream projections across all layers and positions for a given text."""
    _, cache = model.run_with_cache(tokens, names_filter=resid_names_filter)
    acts_by_layer = []
    for layer in range(-1, model.cfg.n_layers):
        if layer >= 0:
            emb: Int[Tensor, "batch pos d_model"] = cache["resid_post", layer]
        else:
            emb: Int[Tensor, "batch pos d_model"] = cache["resid_pre", 0]
        emb /= emb.norm(dim=-1, keepdim=True)
        act: Float[Tensor, "batch pos"] = einops.einsum(
            emb, special_dir, "batch pos d_model, d_model -> batch pos"
        ).cpu()
        acts_by_layer.append(act)
    return torch.stack(acts_by_layer, dim=2)


def get_activations_from_direction(
    tokens: Int[Tensor, "1 pos"],
    special_dir: Float[Tensor, "d_model"],
    model: HookedTransformer,
) -> Float[Tensor, "pos layer 1"]:
    projections: Float[Tensor, "1 pos layer"] = get_projections_for_text(
        tokens, special_dir=special_dir, model=model
    )
    activations = einops.rearrange(projections, "1 pos layer -> pos layer 1")
    return activations


def plot_activations(
    text: Union[str, List[str]],
    model: HookedTransformer,
    centered: bool,
    activations: Optional[Float[Tensor, "pos ..."]] = None,
    special_dir: Optional[Float[Tensor, "d_model"]] = None,
    verbose=False,
    first_dimension_name: Optional[str] = None,
    second_dimension_name: Optional[str] = None,
    first_dimension_labels: Optional[List[str]] = None,
    second_dimension_labels: Optional[List[str]] = None,
    show_selectors: bool = True,
):
    """
    Wrapper around CircuitVis's `text_neuron_activations`.
    Performs centering if `centered` is True.
    Computes activations based on projecting onto a resid stream direction if not provided.
    """
    assert activations is not None or special_dir is not None
    assert model.tokenizer is not None
    tokens: Int[Tensor, "1 pos"] = model.to_tokens(text)
    str_tokens: List[str]
    if isinstance(text, str):
        str_tokens = model.to_str_tokens(tokens, prepend_bos=False)  # type: ignore
    else:
        str_tokens = text
    if verbose:
        print(f"Tokens shape: {tokens.shape}")
    if activations is None:
        assert special_dir is not None
        if verbose:
            print("Computing activations")
        activations = get_activations_from_direction(
            tokens, special_dir=special_dir, model=model
        )
    else:
        activations = add_layer_neuron_dims(activations)
    if verbose:
        print(f"Activations shape: {activations.shape}")
    if centered:
        if verbose:
            print("Centering activations")
        layer_means = einops.reduce(
            activations, "pos layer neuron -> 1 layer neuron", reduction="mean"
        )
        layer_means = einops.repeat(
            layer_means, "1 layer neuron -> pos layer neuron", pos=activations.shape[0]
        )
        activations -= layer_means
    elif verbose:
        print("Activations already centered")
    assert (
        activations.ndim == 3
    ), f"activations must be of shape [tokens x layers x neurons], found {activations.shape}"
    assert len(str_tokens) == activations.shape[0], (
        "tokens and activations must have the same length, found "
        f"tokens={len(str_tokens)} and acts={activations.shape[0]}, "
        f"tokens={str_tokens}, "
        f"activations={activations.shape}"
    )
    if second_dimension_name is None and activations.shape[-1] == 1:
        second_dimension_name = "Model"
        second_dimension_labels = [model.cfg.model_name]
    return text_neuron_activations(
        tokens=str_tokens,
        activations=activations,
        first_dimension_name=first_dimension_name,
        first_dimension_labels=first_dimension_labels,
        second_dimension_name=second_dimension_name,
        second_dimension_labels=second_dimension_labels,
        show_selectors=show_selectors,
    )


def get_window(
    batch: int,
    pos: int,
    dataloader: torch.utils.data.DataLoader,
    window_size: int = 10,
) -> Tuple[int, int]:
    """Helper function to get the window around a position in a batch (used in topk plotting))"""
    lb = max(0, pos - window_size)
    ub = min(len(dataloader.dataset[batch]["tokens"]), pos + window_size + 1)
    return lb, ub


def extract_text_window(
    batch: int,
    pos: int,
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    window_size: int = 10,
) -> List[str]:
    """Helper function to get the text window around a position in a batch (used in topk plotting)"""
    assert model.tokenizer is not None
    expected_size = 2 * window_size + 1
    lb, ub = get_window(batch, pos, dataloader=dataloader, window_size=window_size)
    tokens = dataloader.dataset[batch]["tokens"][lb:ub]
    str_tokens = model.to_str_tokens(tokens, prepend_bos=False)
    padding_to_add = expected_size - len(str_tokens)
    if padding_to_add > 0 and model.tokenizer.padding_side == "right":
        str_tokens += [model.tokenizer.bos_token] * padding_to_add
    elif padding_to_add > 0 and model.tokenizer.padding_side == "left":
        str_tokens = [model.tokenizer.bos_token] * padding_to_add + str_tokens
    assert len(str_tokens) == expected_size, (
        f"Expected text window of size {expected_size}, "
        f"found {len(str_tokens)}: {str_tokens}"
    )
    return str_tokens  # type: ignore


def extract_activations_window(
    activations: Float[Tensor, "row pos ..."],
    batch: int,
    pos: int,
    model: HookedTransformer,
    dataloader: torch.utils.data.DataLoader,
    window_size: int = 10,
) -> Float[Tensor, "pos ..."]:
    """Helper function to get the activations window around a position in a batch (used in topk plotting)"""
    assert model.tokenizer is not None
    expected_size = 2 * window_size + 1
    lb, ub = get_window(batch, pos, dataloader=dataloader, window_size=window_size)
    acts_window: Float[Tensor, "pos ..."] = activations[batch, lb:ub]
    padding_to_add = expected_size - len(acts_window)
    if padding_to_add > 0:
        padding_shape = [padding_to_add] + list(acts_window.shape[1:])
        padding_tensor = torch.zeros(
            padding_shape, dtype=acts_window.dtype, device=acts_window.device
        )
        if model.tokenizer.padding_side == "right":
            acts_window = torch.cat([acts_window, padding_tensor], dim=0)
        elif model.tokenizer.padding_side == "left":
            acts_window = torch.cat([padding_tensor, acts_window], dim=0)
    assert len(acts_window) == expected_size, (
        f"Expected activations window of size {expected_size}, "
        f"found {len(acts_window)}: {acts_window}"
    )
    return acts_window


def get_batch_pos_mask(
    tokens: Union[str, List[str], Tensor],
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    activations: Optional[Float[Tensor, "row pos ..."]] = None,
    device: Optional[torch.device] = None,
):
    """Helper function to get a mask for inclusions/exclusions (used in topk plotting)"""
    if device is None and activations is not None:
        device = activations.device
    elif device is None:
        assert model.cfg.device is not None
        device = torch.device(model.cfg.device)
    mask_values: Int[Tensor, "words"] = torch.unique(
        model.to_tokens(tokens, prepend_bos=False).flatten()  # type: ignore
    ).to(device)
    masks = []
    for _, batch_value in enumerate(dataloader):
        batch_tokens: Int[Tensor, "batch_size pos 1"] = (
            batch_value["tokens"].to(device).unsqueeze(-1)
        )
        batch_mask: Bool[Tensor, "batch_size pos"] = torch.any(
            batch_tokens == mask_values, dim=-1
        )
        masks.append(batch_mask)
    mask: Bool[Tensor, "row pos"] = torch.cat(masks, dim=0)
    if activations is not None:
        assert mask.shape == activations.shape[:2]
        assert mask.device == activations.device
    return mask


def remove_layer_neuron_dims(
    all_activations: Float[Tensor, "row pos ..."],
    layer: Optional[int] = None,
    neuron: Optional[int] = None,
    base_layer: Optional[int] = None,
) -> Float[Tensor, "row pos"]:
    """Helper function to remove layer and neuron dimensions from activations (used in topk plotting)"""
    activations: Float[Tensor, "row pos"]
    if all_activations.ndim == 4:
        assert layer is not None
        assert neuron is not None
        activations = all_activations[:, :, layer, neuron]
    elif all_activations.ndim == 3:
        assert layer is not None
        assert neuron is None
        activations = all_activations[:, :, layer]
        if base_layer is not None:
            base_activations: Float[Tensor, "row pos"] = all_activations[
                :, :, base_layer
            ]
            activations = activations - base_activations
    elif all_activations.ndim == 2:
        assert layer is None
        assert neuron is None
        activations = all_activations
    else:
        raise ValueError(
            f"Activations must be of shape [batch x pos x layer x neuron], "
            f"found {all_activations.shape}"
        )
    return activations


def add_layer_neuron_dims(
    activations: Float[Tensor, "..."],
) -> Float[Tensor, "*row pos layer neuron"]:
    if activations.ndim == 4:
        return activations
    elif activations.ndim == 3:
        return activations.unsqueeze(-1)
    elif activations.ndim <= 2:
        return activations.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(
            f"Activations must be of shape [batch x pos x layer x neuron], "
            f"found {activations.shape}"
        )


def mask_activations(
    activations: Float[Tensor, "row pos ..."],
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    device: torch.device,
    k: int,
    largest: bool,
    inclusions: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
    verbose: bool = False,
) -> Float[Tensor, "row pos ..."]:
    assert not (inclusions is not None and exclusions is not None)

    if largest:
        ignore_value = torch.tensor(-np.inf, device=device, dtype=torch.float32)
    else:
        ignore_value = torch.tensor(np.inf, device=device, dtype=torch.float32)
    # create a mask for the inclusions/exclusions
    if exclusions is not None:
        mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(
            exclusions, dataloader, model, activations
        )
        masked_activations = activations.where(~mask, other=ignore_value)
    elif inclusions is not None:
        mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(
            inclusions, dataloader, model, activations
        )
        assert (
            mask.sum() >= k
        ), f"Only {mask.sum()} positions match the inclusions, but {k} are required"
        if verbose:
            print(f"Including {mask.sum()} positions")
        masked_activations = activations.where(mask, other=ignore_value)
    else:
        masked_activations = activations
    return masked_activations


def plot_top_onesided(
    all_activations: Float[Tensor, "row pos ..."],
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    layer: Optional[int] = None,
    neuron: Optional[int] = None,
    k: int = 10,
    p: Optional[float] = None,
    largest: bool = True,
    window_size: Optional[int] = None,
    centred: bool = True,
    base_layer: Optional[int] = None,
    local: bool = True,
):
    """
    One-sided topk plotting.
    Main entrypoint should be `plot_topk` for two-sided.
    """
    activations: Float[Tensor, "row pos"] = remove_layer_neuron_dims(
        all_activations, layer=layer, neuron=neuron, base_layer=base_layer
    )
    activations_flat: Float[Tensor, "(batch pos)"] = activations.flatten()
    if p is not None:
        # Take a random k from the top p% of activations
        sample_size = int(p * len(activations_flat))
        top_indices = torch.topk(
            activations_flat, k=sample_size, largest=largest
        ).indices
        top_indices = top_indices[torch.randperm(sample_size)[:k]]
    else:
        # Take the top k overall activations
        top_k_return = torch.topk(activations_flat, k=k, largest=largest)
        assert torch.isfinite(top_k_return.values).all()
        top_indices = top_k_return.indices
    top_indices = np.array(
        np.unravel_index(top_indices.cpu().numpy(), activations.shape)
    ).T.tolist()

    # Construct nested list of texts and activations for plotting
    assert model.tokenizer is not None
    texts = []
    text_to_not_repeat = set()
    seq_len = activations.shape[1] if window_size is None else window_size * 2 + 1
    acts_to_plot = torch.zeros(
        (1, 1, k, seq_len),
        dtype=torch.float32,
    )
    for sample, (batch, pos) in enumerate(top_indices):
        if window_size is None:
            text_window: List[str] = model.to_str_tokens(dataloader.dataset[batch]["tokens"])  # type: ignore
            activation_window: Float[Tensor, "pos"] = activations[batch]
        else:
            text_window: List[str] = extract_text_window(
                batch, pos, dataloader=dataloader, model=model, window_size=window_size
            )
            activation_window: Float[Tensor, "pos"] = extract_activations_window(
                activations,
                batch,
                pos,
                window_size=window_size,
                model=model,
                dataloader=dataloader,
            )
        text_flat = "".join(text_window)
        if text_flat in text_to_not_repeat:
            continue
        text_to_not_repeat.add(text_flat)
        texts.append(text_window)
        acts_to_plot[0, 0, sample, :] = activation_window
    rendered_html = topk_samples(
        tokens=[[texts]],  # convert texts from 2D to 4D
        activations=acts_to_plot,  # type: ignore
        zeroth_dimension_name="Model",
        zeroth_dimension_labels=[model.cfg.model_name],
        first_dimension_name="Side",
        first_dimension_labels=["Largest" if largest else "Smallest"],
    )
    html = rendered_html.local_src if local else rendered_html.cdn_src
    file = ResultsFile(
        "top_activations",
        model=model.cfg.model_name,
        dataloader=dataloader,
        layer=layer,
        neuron=neuron,
        k=k,
        largest=largest,
        centred=centred,
        base_layer=base_layer,
        local=local,
        extension="html",
        result_type="plots",
    )
    with open(file.path, "w") as f:
        f.write(html)
    display(rendered_html)


def plot_top_twosided(
    activations: Float[Tensor, "row pos ..."],
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    k: int = 10,
    layer: int = 0,
    window_size: int = 10,
    centred: bool = True,
    base_layer: Optional[int] = None,
):
    """
    Main entrypoint for topk plotting. Plots both positive and negative examples.
    Finds topk in a tensor of activations, matches them up against the text from the dataset,
    and plots them neuroscope-style.
    """
    plot_top_onesided(
        activations,
        dataloader=dataloader,
        model=model,
        layer=layer,
        k=k,
        largest=True,
        window_size=window_size,
        centred=centred,
        base_layer=base_layer,
    )
    plot_top_onesided(
        activations,
        dataloader=dataloader,
        model=model,
        layer=layer,
        k=k,
        largest=False,
        window_size=window_size,
        centred=centred,
        base_layer=base_layer,
    )
