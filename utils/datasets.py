from abc import ABC, abstractmethod
import einops
from functools import partial
import torch
import datasets
from torch import Tensor
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import (
    load_dataset,
    concatenate_datasets,
    load_from_disk,
    DatasetDict,
    Dataset as HFDataset,
)
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Optional, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import (
    tokenize_and_concatenate,
    get_act_name,
    test_prompt,
)
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd
from circuitsvis.activations import text_neuron_activations
from utils.store import (
    load_array,
    save_html,
    save_array,
    is_file,
    get_model_name,
    clean_label,
    save_text,
)
from utils.circuit_analysis import get_logit_diff


class ExperimentData(ABC):
    """Abstract Base class for experiment data."""

    def __init__(
        self,
        dataset_dict: DatasetDict,
        model: HookedTransformer,
        text_column: str = "text",
        label_column: Optional[str] = None,
    ):
        """
        dataset_dict:
            A dictionary of datasets, with keys 'train', 'validation', and 'test'.
            This can be created using datasets.load_dataset().
        model:
            A hooked transformer model. Instantiate using HookedTransformer.from_pretrained().
        """
        self.dataset_dict = dataset_dict
        self.model = model
        self.text_column = text_column
        self.label_column = label_column
        self.token_to_ablate = None

    @classmethod
    def from_string(
        cls,
        dataset_name: str,
        model: HookedTransformer,
        text_column: str = "text",
        label_column: Optional[str] = None,
    ):
        dataset_dict = load_dataset(dataset_name)
        assert isinstance(dataset_dict, DatasetDict), "Dataset is not a DatasetDict"
        return cls(dataset_dict, model, text_column, label_column)

    def apply_function(self, function, **kwargs):
        """Applies an arbitrary function to the datasets"""
        for split in self.dataset_dict.keys():
            self.dataset_dict[split] = self.dataset_dict[split].map(function, **kwargs)

    def preprocess_datasets(self, token_to_ablate: Optional[int] = None):
        """Preprocesses the dataset. This function can be overridden by subclasses, but should always result in a dataset with a 'tokens' column"""
        self._tokenize()
        self.apply_function(self._create_attention_mask)

        if token_to_ablate is not None:
            find_dataset_positions = partial(
                self._find_dataset_positions, token_to_ablate=token_to_ablate
            )
            self.apply_function(find_dataset_positions, batched=False)

        example_ds = list(self.dataset_dict.values())[0]
        assert (
            "tokens" in example_ds.column_names
        ), "Dataset does not have a 'tokens' column"
        assert (
            "positions" in example_ds.column_names
        ), "Dataset does not have a 'positions' column in the train split"

    def get_dataloaders(self, batch_size: int) -> Dict[str, DataLoader]:
        """Returns a dictionary of dataloaders for each split"""

        if "positions" not in self.dataset_dict["train"].column_names:
            print(
                "Warning: 'positions' column not found in dataset. Returning dataloaders without positions. \n"
                "To add positions, run find_dataset_positions() first."
            )

        dataloaders = {}
        for split in self.dataset_dict.keys():
            dataloaders[split] = DataLoader(
                self.dataset_dict[split], batch_size=batch_size, shuffle=True
            )
        return dataloaders

    def get_datasets(self) -> Dict[str, datasets.Dataset]:
        return self.dataset_dict

    @staticmethod
    def _find_dataset_positions(example, token_to_ablate: int):
        # Create a tensor of zeros with the same shape as example['tokens']
        positions = torch.zeros_like(torch.tensor(example["tokens"]), dtype=torch.int)

        # Find positions where tokens match the given token_id
        positions[example["tokens"] == token_to_ablate] = 1
        has_token = True if positions.sum() > 0 else False

        return {"positions": positions, "has_token": has_token}

    @staticmethod
    @abstractmethod
    def _create_attention_mask(example: Dict) -> Dict:
        pass

    @abstractmethod
    def _tokenize(self) -> None:
        pass


class OWTData(ExperimentData):
    """Class for the OpenWebText dataset

    When using this class, first instantiate and then call preprocess_datasets() to tokenize the dataset.

    Next, call find_dataset_positions() to find the positions of the token to ablate in the dataset.
    """

    def __init__(
        self,
        dataset_dict: DatasetDict,
        model,
        text_column: str = "text",
        label_column: Optional[str] = None,
    ):
        super().__init__(dataset_dict, model, text_column, label_column)

    def _tokenize(self):
        """Preprocesses the dataset by tokenizing and concatenating the text column"""
        for split in self.dataset_dict.keys():
            self.dataset_dict[split] = tokenize_and_concatenate(
                self.dataset_dict[split], self.model.tokenizer
            )

    def _create_attention_mask(self, example: Dict):
        attention_mask = torch.ones_like(
            torch.tensor(example["tokens"]), dtype=torch.int
        )
        return {"attention_mask": attention_mask}

    @classmethod
    def from_model(
        cls,
        model: HookedTransformer,
        text_column: str = "text",
        label_column: Optional[str] = None,
    ):
        return cls.from_string(
            "stas/openwebtext-10k",
            model,
            text_column=text_column,
            label_column=label_column,
        )
