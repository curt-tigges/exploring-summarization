from abc import ABC, abstractmethod
import einops
from functools import partial
import re
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


DEFAULT_EXCLUDE_CHARACTERS = [
    "]",
    "[",
    "(",
    ")",
    ",",
    ":",
    ";",
    "``",
    "''",
    ".",
    "!",
    "?",
    "“",
]


def construct_exclude_list(
    model: HookedTransformer,
    characters: List[str] = DEFAULT_EXCLUDE_CHARACTERS,
) -> List[int]:
    assert model.tokenizer is not None
    exclude_list = []
    for vocab_str, token_id in model.tokenizer.vocab.items():
        if any([p in vocab_str for p in characters]):
            exclude_list.append(token_id)
    return exclude_list


def mask_positions(
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    exclude_following_token: Optional[int] = None,
    exclude_characters: Optional[List[str]] = None,
) -> Float[Tensor, "row pos ..."]:
    """Returns a mask of the same shape as the dataset, with True values at positions to be excluded."""
    num_rows = dataloader.dataset.num_rows
    seq_len = dataloader.dataset.seq_len
    mask = torch.ones((num_rows, seq_len), dtype=torch.bool)
    if exclude_characters is not None:
        exclude_list = construct_exclude_list(model, exclude_characters)
        exclude_pt = torch.tensor(exclude_list, device=mask.device)
    else:
        exclude_pt = None

    for batch_idx, batch in enumerate(dataloader):
        batch_tokens: Int[Tensor, "batch_size pos"] = batch["tokens"]
        batch_start = batch_idx * dataloader.batch_size
        batch_end = batch_start + dataloader.batch_size
        batch_mask = torch.zeros_like(batch_tokens, dtype=torch.bool)
        batch_mask[batch["attention_mask"] == 0] = 1
        if exclude_pt is not None:
            batch_mask[torch.isin(batch_tokens, exclude_pt)] = 1
        if exclude_following_token is not None:
            # Exclude positions directly following token to ablate
            shifted_tokens = torch.roll(batch_tokens, shifts=1, dims=1)
            shifted_tokens[
                :, 0
            ] = 0  # Set the first column to zero because roll is circular
            batch_mask[shifted_tokens == exclude_following_token] = 1
        mask[batch_start:batch_end] = batch_mask
    return mask


class ExperimentDataLoader(DataLoader):
    COLUMN_NAMES = ["tokens", "attention_mask", "positions", "has_token"]

    def __init__(
        self,
        dataset: HFDataset,
        batch_size: int | None = 1,
        shuffle: bool = False,
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
        if dataset.builder_name is not None:
            self._name = dataset.builder_name
        elif dataset.info.homepage:
            self._name = dataset.info.homepage.split("/")[-2]
        else:
            pattern = r"/huggingface/datasets/([^/]+/[^-]+)"
            # Performing the regex search
            match = re.search(pattern, dataset.cache_files[0]["filename"])
            assert match
            self._name = match.group(1)
        self._name += f"_{dataset.split}"
        self.seq_len = dataset[0]["tokens"].shape[0]

    @property
    def name(self):
        return self._name


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

    @classmethod
    def from_string(
        cls,
        path: str,
        model: HookedTransformer,
        name: str | None = None,
        split: str | None = None,
        num_proc: int | None = None,
        data_files: List[str] | None = None,
        verbose: bool = False,
    ):
        if data_files is not None:
            data_files = [
                file_url.replace("blob", "resolve") for file_url in data_files
            ]
            data_name = data_files[0].split("/")[-2]
            if name is None:
                name = data_name
            assert (
                name == data_name
            ), f"Name {name} does not match data name {data_name}"
        if verbose:
            print(
                f"load_dataset: path={path}, name={name}, split={split}, num_proc={num_proc}, data_files={data_files}"
            )
        dataset_dict = load_dataset(
            path=path, name=name, split=split, num_proc=num_proc, data_files=data_files
        )
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

    def preprocess_datasets(
        self,
        token_to_ablate: Optional[int] = None,
    ):
        """Preprocesses the dataset. This function can be overridden by subclasses, but should always result in a dataset with a 'tokens' column"""
        self._tokenize()
        self.apply_function(self._create_attention_mask)

        if token_to_ablate is not None:
            find_dataset_positions = partial(
                self._find_dataset_positions, token_to_ablate=token_to_ablate
            )
            self.apply_function(find_dataset_positions, batched=False)

        for split in self.dataset_dict.keys():
            if self.dataset_dict[split].split is None:
                self.dataset_dict[split]._split = split

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
                self.dataset_dict[split], batch_size=batch_size
            )
        return dataloaders

    def get_datasets(self) -> Dict[str, datasets.Dataset]:
        return self.dataset_dict

    @staticmethod
    def _find_dataset_positions(example: dict, token_to_ablate: int) -> dict:
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
                self.dataset_dict[split],
                self.model.tokenizer,  # type: ignore
                max_length=self.model.cfg.n_ctx,
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
        split: Optional[str] = None,
        num_proc: int | None = None,
        data_files: List[str] | None = None,
        verbose: bool = False,
    ):
        return cls.from_string(
            "stas/openwebtext-10k",
            model,
            split=split,
            num_proc=num_proc,
            data_files=data_files,
            verbose=verbose,
        )


class PileFullData(HFData):
    """Class for the Pile dataset"""

    @classmethod
    def from_model(
        cls,
        model: HookedTransformer,
        split: Optional[str] = None,
        num_proc: int | None = None,
        data_files: List[str] | None = None,
        verbose: bool = False,
    ):
        return cls.from_string(
            "monology/pile-uncopyrighted",
            model,
            split=split,
            num_proc=num_proc,
            data_files=data_files,
            verbose=verbose,
        )


class PileSplittedData(HFData):
    """Class for the Pile dataset"""

    @classmethod
    def from_model(
        cls,
        model: HookedTransformer,
        name: Optional[str] = None,
        split: Optional[str] = None,
        num_proc: int | None = None,
        data_files: List[str] | None = None,
        verbose: bool = False,
    ):
        return cls.from_string(
            "ArmelR/the-pile-splitted",
            model,
            name=name,
            split=split,
            num_proc=num_proc,
            data_files=data_files,
            verbose=verbose,
        )
