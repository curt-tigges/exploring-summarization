import numpy as np
import pandas as pd
import glob
import os
from transformer_lens import HookedTransformer
from typing import Iterable, List, Union
import torch
import plotly.graph_objects as go
from circuitsvis.utils.render import RenderedHTML
from pandas.io.formats.style import Styler
import re
import pickle
from datasets import dataset_dict

from summarization_utils.datasets import ExperimentDataLoader


ARGS_TO_INGORE = ["device", "batch_size", "overwrite"]


def add_styling(html: str):
    # Extract the table ID from the HTML using regex
    table_id_match = re.search(r'<table id="([^"]+)">', html)
    if not table_id_match:
        return "Invalid HTML: Table ID not found"

    table_id = table_id_match.group(1)

    # Define the general styles using the extracted table ID
    styles = f"""
    /* General styles */
    #{table_id} {{
        border-collapse: collapse;
        width: 300px; /* Specify the width you want */
        height: 200px; /* Specify the height you want */
        overflow: auto;
        position: relative;
    }}

    #{table_id} th, #{table_id} td {{
        padding: 8px 12px;
        border: 1px solid #d4d4d4;
        text-align: center;
        min-width: 50px;
        box-sizing: border-box;
        position: relative;
    }}

    /* Freeze first column */   
    #{table_id} .level0 {{
        background-color: #ddd;
        position: -webkit-sticky;
        position: sticky;
        left: 0;
        z-index: 1;
    }}

    /* Freeze first row */
    #{table_id} thead {{
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        z-index: 2;
    }}

    #{table_id} thead {{
        background-color: #ddd;
    }}
    """

    # Insert the general styles into the existing style section of the HTML
    style_start_index = html.find('<style type="text/css">')
    if style_start_index == -1:
        return "Invalid HTML: Style section not found"

    style_start_index += len('<style type="text/css">')
    html_with_styles = html[:style_start_index] + styles + html[style_start_index:]

    return html_with_styles


def nested_list_to_string(nested_list):
    # Flatten the nested list using recursion
    def flatten(lst):
        for elem in lst:
            if isinstance(elem, list) or isinstance(elem, tuple):
                yield from flatten(elem)
            else:
                yield elem

    # Flatten the nested list and join using underscores
    return "_".join(map(str, flatten(nested_list)))


def assert_alphanumeric_underscore(string):
    # Check if all characters in the string are either alphanumeric or underscores
    assert all(
        c.isalnum() or c == "_" for c in string
    ), f"String contains non-alphanumeric and non-underscore characters: {string}"
    assert string == string.lower(), f"String contains uppercase characters: {string}"


def clean_string(string: str):
    # Remove all non-alphanumeric characters from the string
    return re.sub(r"[^a-zA-Z0-9_]+", "_", string.lower())


def args_to_file_name(**kwargs):
    """Converts a dictionary of arguments to a file name"""
    file_name = ""
    for key, value in kwargs.items():
        if value is None:
            continue
        elif hasattr(value, "nelement") and value.nelement() == 0:
            continue
        elif hasattr(value, "nelement") and value.nelement() == 1:
            value = value.item()
            if isinstance(value, int):
                value = str(value)
            else:
                value = "{:.2f}".format(value).replace(".", "_")
        elif hasattr(value, "__len__") and len(value) == 0:
            continue
        elif key in ARGS_TO_INGORE:
            continue
        elif isinstance(value, list) or isinstance(value, tuple):
            value = f"{len(value):d}"
        elif isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, HookedTransformer):
            value = value.cfg.model_name
        elif isinstance(value, ExperimentDataLoader):
            value = value.name
        elif isinstance(value, torch.Tensor):
            value = "_".join([str(d) for d in value.shape])
        elif isinstance(value, int):
            value = str(value)
        elif isinstance(value, float):
            value = "{:.2f}".format(value).replace(".", "_")
        elif isinstance(value, str):
            pass
        else:
            raise ValueError(f"Unimplemented type: {type(value)}")
        assert isinstance(value, str), f"Value is not a string: {value}"
        value = clean_string(value)
        file_name += f"{key}__{value}__"
    assert_alphanumeric_underscore(file_name)
    return file_name[:-2]


def create_file_name(name: str, extension: str, **kwargs):
    """Creates a file name from a name and a dictionary of arguments"""
    extension = extension.replace(".", "")
    file_name = args_to_file_name(**kwargs)
    return f"{name}_{file_name}.{extension}"


class TensorBlockManager:
    def __init__(
        self,
        block_size: int,
        tensor_prefix: str = "tensor_block",
        root: str = "results/cache",
    ):
        self.block_size = block_size
        self.tensor_prefix = tensor_prefix
        self.root = root

    def _get_filename(self, block_index: int):
        return f"{self.root}/{self.tensor_prefix}_{block_index}.pt"

    def _get_block_and_index(self, index: int):
        block_index = index // self.block_size
        index_in_block = index % self.block_size
        return block_index, index_in_block

    def save(self, block: torch.Tensor, block_index: int):
        filename = self._get_filename(block_index)
        torch.save(block, filename)

    def load(self, block_index: int):
        filename = self._get_filename(block_index)
        return torch.load(filename)

    def slice_indices(self, indices: List[int]):
        blocks_and_indices = [self._get_block_and_index(idx) for idx in indices]
        # group by block
        block_indices = {}
        for block_id, index in blocks_and_indices:
            if block_id not in block_indices:
                block_indices[block_id] = []
            block_indices[block_id].append(index)
        # load blocks and get slices
        slices = []
        for block_id, index_list in block_indices.items():
            block = self.load(block_id)
            slices += [block[idx] for idx in index_list]

        # Concatenate all slices
        full_slice = torch.cat(slices, dim=0)
        return full_slice

    def list(self):
        return glob.glob(f"{self.root}/{self.tensor_prefix}_*.pt")

    def read_all(self):
        return (torch.load(filename) for filename in self.list())

    def clear(self):
        for filename in self.list():
            os.remove(filename)

    def delete(self, block_index):
        filename = self._get_filename(block_index)
        os.remove(filename)

    def __len__(self):
        return len(self.list())


class ResultsFile:
    def __init__(
        self,
        name: str,
        extension: str,
        root: str = "results",
        result_type="cache",
        **kwargs,
    ):
        if not os.path.exists(root):
            os.mkdir(root)
        if not os.path.exists(f"{root}/{result_type}"):
            os.mkdir(f"{root}/{result_type}")
        self.name = name
        self.extension = extension.replace(".", "")
        self.file_name = create_file_name(name, extension, **kwargs)
        self.path = f"{root}/{result_type}/{self.file_name}"

    def exists(self):
        return os.path.exists(self.path)

    def save(self, data):
        if isinstance(data, str):
            with open(self.path, "w") as f:
                f.write(data)
        elif isinstance(data, torch.Tensor):
            torch.save(data, self.path)
        elif isinstance(data, go.Figure):
            data.write_html(self.path)
        else:
            raise ValueError(f"Unimplemented type for save: {type(data)}")

    def load(self):
        if self.extension in ("txt", "html"):
            with open(self.path, "r") as f:
                return f.read()
        elif self.extension == "pt":
            return torch.load(self.path)
        else:
            raise ValueError(f"Unimplemented extension for load: {self.extension}")
