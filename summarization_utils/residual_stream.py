from typing import List, Tuple
from jaxtyping import Int, Float, Bool
from typeguard import typechecked
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from summarization_utils.prompts import get_dataset, PromptType


def get_resid_name(layer: int, model: HookedTransformer) -> Tuple[str, int]:
    resid_type = "resid_pre"
    if layer == model.cfg.n_layers:
        resid_type = "resid_post"
        layer -= 1
    return get_act_name(resid_type, layer), layer


class ResidualStreamDataset:
    @typechecked
    def __init__(
        self,
        prompt_strings: List[str],
        prompt_tokens: Int[Tensor, "batch pos"],
        binary_labels: Bool[Tensor, "batch"],
        model: HookedTransformer,
        prompt_type: PromptType,
    ) -> None:
        assert len(prompt_strings) == len(prompt_tokens)
        assert len(prompt_strings) == len(binary_labels)
        self.prompt_strings = prompt_strings
        self.prompt_tokens = prompt_tokens
        self._binary_labels = binary_labels
        self.model = model
        self.prompt_type = prompt_type
        example_str_tokens = model.to_str_tokens(prompt_strings[0])
        self.example = [f"{i}:{tok}" for i, tok in enumerate(example_str_tokens)]
        self.placeholder_dict = prompt_type.get_placeholder_positions(
            example_str_tokens
        )
        label_positions = [
            pos for _, positions in self.placeholder_dict.items() for pos in positions
        ]
        self.str_labels = [
            "".join([model.to_str_tokens(prompt)[pos] for pos in label_positions])
            for prompt in prompt_strings
        ]

    @property
    def binary_labels(self) -> Bool[Tensor, "batch"]:
        return self._binary_labels.cpu().detach()

    def __len__(self) -> int:
        return len(self.prompt_strings)

    def __eq__(self, other: object) -> bool:
        return set(self.prompt_strings) == set(other.prompt_strings)

    @typechecked
    def embed(self, position_type: str, layer: int) -> Float[Tensor, "batch d_model"]:
        """
        Returns a dataset of embeddings at the specified position and layer.
        Useful for training classifiers on the residual stream.
        """
        assert 0 <= layer <= self.model.cfg.n_layers
        assert position_type in self.placeholder_dict.keys(), (
            f"Position type {position_type} not found in {self.placeholder_dict.keys()} "
            f"for prompt type {self.prompt_type}"
        )
        embed_position = self.placeholder_dict[position_type][-1]
        hook, _ = get_resid_name(layer, self.model)
        _, cache = self.model.run_with_cache(
            self.prompt_tokens, return_type=None, names_filter=lambda name: hook == name
        )
        out: Float[Tensor, "batch pos d_model"] = cache[hook]
        return out[:, embed_position, :].cpu().detach()

    @classmethod
    def get_dataset(
        cls,
        model: HookedTransformer,
        device: torch.device,
        prompt_type: str = "simple_train",
    ) -> "ResidualStreamDataset":
        clean_corrupt_data = get_dataset(model, device, prompt_type=prompt_type)
        clean_labels = (
            clean_corrupt_data.answer_tokens[:, 0, 0]
            == clean_corrupt_data.answer_tokens[0, 0, 0]
        )

        assert len(clean_corrupt_data.all_prompts) == len(
            clean_corrupt_data.answer_tokens
        )
        assert len(clean_corrupt_data.all_prompts) == len(
            clean_corrupt_data.clean_tokens
        )
        return cls(
            clean_corrupt_data.all_prompts,
            clean_corrupt_data.clean_tokens,
            clean_labels,
            model,
            prompt_type,
        )

    def _get_labels_by_class(self, label: int) -> List[str]:
        return [
            string.strip()
            for string, one_hot in zip(self.str_labels, self.binary_labels)
            if one_hot == label
        ]

    def get_positive_negative_labels(self):
        return (
            self._get_labels_by_class(1),
            self._get_labels_by_class(0),
        )
