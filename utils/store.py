import numpy as np
import pandas as pd
import glob
import os
from transformer_lens import HookedTransformer
from typing import Iterable, Union
import torch
import plotly.graph_objects as go
from circuitsvis.utils.render import RenderedHTML
from pandas.io.formats.style import Styler
import re
import pickle
from datasets import dataset_dict


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
            if isinstance(elem, list):
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


def args_to_file_name(**kwargs):
    """Converts a dictionary of arguments to a file name"""
    file_name = ""
    for key, value in kwargs.items():
        if isinstance(value, list) or isinstance(value, tuple):
            value = nested_list_to_string(value)
        elif isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, HookedTransformer):
            value = value.cfg.model_name
        elif isinstance(value, torch.Tensor):
            value = "_".join([str(d) for d in value.shape])
        elif value is None:
            value = "none"
        elif isinstance(value, str):
            value = value.lower()
            value = value.replace(" ", "_")
            value = value.replace("/", "_")
        elif isinstance(value, int):
            value = str(value)
        elif isinstance(value, float):
            value = "{:.2f}".format(value).replace(".", "_")
        else:
            raise ValueError(f"Unimplemented type: {type(value)}")

        file_name += f"{key}__{value}__"
    assert_alphanumeric_underscore(file_name)
    return file_name[:-2]


def create_file_name(name: str, extension: str, **kwargs):
    """Creates a file name from a name and a dictionary of arguments"""
    extension = extension.replace(".", "")
    file_name = args_to_file_name(**kwargs)
    return f"{name}_{file_name}.{extension}"
