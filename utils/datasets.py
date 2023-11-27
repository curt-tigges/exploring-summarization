from abc import ABC, abstractmethod
import einops
from functools import partial
import re
import numpy as np
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
    Dataset,
)
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Optional, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import (
    tokenize_and_concatenate,
    keep_single_column,
    AutoTokenizer,
)
from tqdm.notebook import tqdm


DEFAULT_EXCLUDE_REGEX = [
    r"\]",
    r"\[",
    r"\(",
    r"\)",
    r",",
    r":",
    r";",
    r"`",
    r"'",
    r"\.",
    r"!",
    r"\?",
    r"“",
    r"{",
    r"}",
    r"\\",
    r"/",
    r"^g$",
    r"[0-9]",
]


def tokenize_truncate_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
    padding: bool = True,
    truncation: bool = True,
) -> Dataset:
    """
    Helper function to tokenize, truncate and concatenate a dataset of text.
    This converts the text to tokens and truncates them to max_length (provided truncation=True).

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.
        num_proc (int, optional): The number of processes to use. Defaults to 10.
        padding (bool, optional): Whether to pad the sequences. Defaults to True.
        truncation (bool, optional): Whether to truncate the sequences. Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it just outputs nothing. I'm not super sure why
    """
    if not truncation:
        return tokenize_and_concatenate(
            dataset,
            tokenizer,
            streaming=streaming,
            max_length=max_length,
            column_name=column_name,
            add_bos_token=add_bos_token,
            num_proc=num_proc,
        )
    dataset = keep_single_column(dataset, column_name)
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(
        examples: Dict[str, List[str]],
    ) -> Dict[str, Int[np.ndarray, "batch seq"]]:
        text = examples[column_name]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens: Int[np.ndarray, "batch seq"] = tokenizer(
            text,
            return_tensors="np",
            padding=padding,
            truncation=truncation,
            max_length=seq_len,
        )[  # type: ignore
            "input_ids"
        ]
        if add_bos_token:
            prefix = np.full((len(tokens), 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=(num_proc if not streaming else None),
        remove_columns=[column_name],
    )
    tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset


def construct_exclude_list(
    model: HookedTransformer,
    regex: List[str] = DEFAULT_EXCLUDE_REGEX,
) -> List[int]:
    assert model.tokenizer is not None
    exclude_list = []
    for vocab_str, token_id in model.tokenizer.vocab.items():
        if any([re.search(p, vocab_str) for p in regex]):
            exclude_list.append(token_id)
    return exclude_list


def mask_positions(
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    exclude_following_token: Optional[int] = None,
    exclude_regex: Optional[List[str]] = None,
) -> Float[Tensor, "row pos ..."]:
    """
    Returns a mask of the same shape as the dataset, with True values at positions to be excluded.
    TODO:
        - Add option to change number of following positions to mask
        - Unify list of regex to single string
    """
    num_rows = dataloader.dataset.num_rows
    seq_len = dataloader.dataset[0]["tokens"].shape[0]
    mask = torch.ones((num_rows, seq_len), dtype=torch.bool)
    if exclude_regex is not None:
        exclude_list = construct_exclude_list(model, exclude_regex)
        exclude_pt = torch.tensor(exclude_list, device=mask.device)
    else:
        exclude_pt = None

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
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


class ExperimentDataLoader(DataLoader):
    COLUMN_NAMES = ["tokens", "attention_mask", "positions", "has_token"]

    def __init__(
        self,
        dataset: Dataset,
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
        exclude_characters: List[str] = DEFAULT_EXCLUDE_CHARACTERS,
    ,
        truncation: bool = True,
    ):
        """Preprocesses the dataset. This function can be overridden by subclasses, but should always result in a dataset with a 'tokens' column"""
        self._tokenize(truncation=truncation)
        self.apply_function(self._create_attention_mask)

        if token_to_ablate is not None:
            find_dataset_positions = partial(
                self._find_dataset_positions, token_to_ablate=token_to_ablate
            )
            self.apply_function(find_dataset_positions, batched=False)
            find_exclusions = partial(
                self._find_exclude_positions, exclude_characters=exclude_characters
            )
            self.apply_function(find_exclusions, batched=False)

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

    def _find_exclude_positions(
        self, example: dict, exclude_characters: List[str] = DEFAULT_EXCLUDE_CHARACTERS
    ) -> dict:
        tokens = example["tokens"]
        positions = example["positions"]
        attention_mask = example["attention_mask"]
        exclude_list = construct_exclude_list(self.model, exclude_characters)
        exclude_pt = torch.tensor(exclude_list, device=tokens.device)
        exclusions = torch.isin(tokens, exclude_pt)

        # Exclude positions directly following token to ablate
        shifted_positions = torch.roll(positions, shifts=1, dims=1)
        shifted_positions[
            :, 0
        ] = 0  # Set the first column to zero because roll is circular
        exclusions[shifted_positions == 1] = 1

        # Exclude all masked positions
        exclusions[attention_mask == 0] = 1

        return {"exclusions": exclusions}

    @staticmethod
    @abstractmethod
    def _create_attention_mask(example: Dict) -> Dict:
        pass

    @abstractmethod
    def _tokenize(self, truncation: bool = True) -> None:
        pass


class HFData(ExperimentData):
    def __init__(
        self,
        dataset_dict: DatasetDict,
        model,
    ):
        super().__init__(dataset_dict, model)

    def _tokenize(self, truncation: bool = True):
        """Preprocesses the dataset by tokenizing and concatenating the text column"""
        for split in self.dataset_dict.keys():
            self.dataset_dict[split] = tokenize_truncate_concatenate(
                self.dataset_dict[split],
                self.model.tokenizer,  # type: ignore
                max_length=self.model.cfg.n_ctx,
                truncation=truncation,
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
