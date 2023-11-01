from abc import ABC, abstractmethod
import einops
from functools import partial
import torch
import datasets
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Sampler,
)
from datasets import (
    load_dataset,
    DatasetDict,
    Dataset as HFDataset,
)
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Optional, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import (
    tokenize_and_concatenate,
)
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd


class ExperimentDataLoader(DataLoader):
    COLUMN_NAMES = ["tokens", "attention_mask", "positions", "has_token"]

    def __init__(
        self,
        dataset: HFDataset,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[List] | Iterable[List] | None = None,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        # Assert that dataset has the required columns
        for col_name in self.COLUMN_NAMES:
            assert (
                col_name in dataset.column_names
            ), f"Dataset does not have a '{col_name}' column"
        dataset.set_format(
            type="torch",
            columns=self.COLUMN_NAMES,
        )
        super().__init__(
            dataset,  # type: ignore
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )


class ExperimentData(ABC):
    """Abstract Base class for experiment data."""

    def __init__(
        self,
        dataset_dict: DatasetDict,
        model: HookedTransformer,
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
        self.token_to_ablate = None

    @classmethod
    def from_string(
        cls,
        path: str,
        model: HookedTransformer,
        name: str | None = None,
        split: str | None = None,
        num_proc: int | None = None,
    ):
        dataset_dict = load_dataset(path, name, split=split, num_proc=num_proc)
        if split is not None:
            dataset_dict = DatasetDict(
                {
                    split: dataset_dict,
                }
            )
        assert isinstance(dataset_dict, DatasetDict), "Dataset is not a DatasetDict"
        return cls(dataset_dict, model)

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

    def get_dataloaders(self, batch_size: int) -> Dict[str, ExperimentDataLoader]:
        """Returns a dictionary of dataloaders for each split"""

        dataloaders = {}
        for split in self.dataset_dict.keys():
            dataloaders[split] = ExperimentDataLoader(
                self.dataset_dict[split], batch_size=batch_size, shuffle=True
            )
        return dataloaders

    def get_datasets(self) -> Dict[str, datasets.Dataset]:
        return self.dataset_dict

    @staticmethod
    def _find_dataset_positions(example, token_to_ablate: int):
        # Create a tensor of zeros with the same shape as example['tokens']
        positions = torch.zeros_like(example["tokens"])

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


class HFData(ExperimentData):
    def __init__(
        self,
        dataset_dict: DatasetDict,
        model,
    ):
        super().__init__(dataset_dict, model)

    def _tokenize(self):
        """Preprocesses the dataset by tokenizing and concatenating the text column"""
        for split in self.dataset_dict.keys():
            self.dataset_dict[split] = tokenize_and_concatenate(
                self.dataset_dict[split], self.model.tokenizer  # type: ignore
            )

    def _create_attention_mask(self, example: Dict):
        attention_mask = torch.ones_like(example["tokens"])
        return {"attention_mask": attention_mask}


class OWTData(HFData):
    """Class for the OpenWebText dataset"""

    @classmethod
    def from_model(
        cls,
        model: HookedTransformer,
    ):
        return cls.from_string(
            "stas/openwebtext-10k",
            model,
        )


class PileFullData(HFData):
    """Class for the Pile dataset"""

    @classmethod
    def from_model(
        cls,
        model: HookedTransformer,
    ):
        return cls.from_string(
            "monology/pile-uncopyrighted",
            model,
        )


class PileSplittedData(HFData):
    """Class for the Pile dataset"""

    @classmethod
    def from_model(
        cls,
        model: HookedTransformer,
        name: str,
        split: Optional[str] = None,
    ):
        return cls.from_string(
            "ArmelR/the-pile-splitted",
            model,
            name=name,
            split=split,
        )
