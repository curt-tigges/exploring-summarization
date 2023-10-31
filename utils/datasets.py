import einops
from functools import partial
import torch
import datasets
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, load_from_disk, DatasetDict
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_dataset, tokenize_and_concatenate, get_act_name, test_prompt
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd
from circuitsvis.activations import text_neuron_activations
from utils.store import load_array, save_html, save_array, is_file, get_model_name, clean_label, save_text
from utils.circuit_analysis import get_logit_diff

class ExperimentData:
    def __init__(self, dataset_dict: DatasetDict, model, text_column: str = 'text', label_column: str = None):
        self.dataset_dict = dataset_dict
        self.model = model
        self.text_column = text_column
        self.label_column = label_column


    def preprocess_datasets(self, token_to_ablate: int = None):
        """Preprocesses the dataset. This function can be overridden by subclasses, but should always result in a dataset with a 'tokens' column"""
        for split in self.dataset_dict.keys():
            self.dataset_dict[split] = self.dataset_dict[split].map(self._preprocess_function)

        assert 'tokens' in self.dataset_dict['train'].column_names, "Dataset does not have a 'tokens' column"

        if token_to_ablate is not None:
            for split in self.dataset_dict.keys():
                self.dataset_dict[split] = self.dataset_dict[split].map(self._find_dataset_positions)

    def find_dataset_positions(self, token_to_ablate: int = 13):
        """Finds the positions of a given token in the dataset"""
        for split in self.dataset_dict.keys():
            self.dataset_dict[split] = self.dataset_dict[split].map(lambda example: self._find_dataset_positions(example, token_id=token_to_ablate))

        assert 'positions' in self.dataset_dict['train'].column_names, "Dataset does not have a 'positions' column"


    def apply_function(self, function):
        """Applies an arbitrary function to the datasets"""
        for split in self.dataset_dict.keys():
            self.dataset_dict[split] = self.dataset_dict[split].map(function)


    def get_dataloaders(self, batch_size: int) -> Dict[str, DataLoader]:
        """Returns a dictionary of dataloaders for each split"""
        
        if 'positions' not in self.dataset_dict['train'].column_names:
            print("Warning: 'positions' column not found in dataset. Returning dataloaders without positions. \nTo add positions, run find_dataset_positions() first.")
        
        dataloaders = {}
        for split in self.dataset_dict.keys():
            dataloaders[split] = DataLoader(self.dataset_dict[split], batch_size=batch_size, shuffle=True)
        return dataloaders


    def get_datasets(self) -> Dict[str, datasets.Dataset]:
        return self.dataset_dict


    def _find_dataset_positions(example, token_id=13):
        # Create a tensor of zeros with the same shape as example['tokens']
        positions = torch.zeros_like(torch.tensor(example['tokens']), dtype=torch.int)

        # Find positions where tokens match the given token_id
        positions[example['tokens'] == token_id] = 1
        has_token = True if positions.sum() > 0 else False

        return {'positions': positions, 'has_token': has_token}


    def _preprocess_function(example: Dict):
        return example


class OWTData(ExperimentData):
    def __init__(self, dataset_dict: DatasetDict, model, text_column: str = 'text', label_column: str = None):
        super().__init__(dataset_dict, model, text_column, label_column)


    def preprocess_datasets(self, token_to_ablate: int = None):
        """Preprocesses the dataset by tokenizing and concatenating the text column"""
        for split in self.dataset_dict.keys():
            self.dataset_dict[split] = tokenize_and_concatenate(self.dataset_dict[split], self.model.tokenizer)

        assert 'tokens' in self.dataset_dict['train'].column_names, "Dataset does not have a 'tokens' column"

        if token_to_ablate is not None:
            for split in self.dataset_dict.keys():
                self.dataset_dict[split] = self.dataset_dict[split].map(self._find_dataset_positions)

        # Create full attention mask column
        for split in self.dataset_dict.keys():
            self.dataset_dict[split] = self.dataset_dict[split].map(self._create_attention_mask)


    def _create_attention_mask(self, example: Dict):
        attention_mask = torch.ones_like(torch.tensor(example['tokens']), dtype=torch.int)
        return {'attention_mask': attention_mask}


#class CEBaBData(ExperimentData):


# TODO: Finish this class stub if we want to use it
# class SSTData(ExperimentData):
#     def __init__(self, dataset_dict: DatasetDict, text_column: str = 'text', label_column: str = 'label'):
#         super().__init__(dataset_dict, text_column, label_column)
#         self.dataset_dict = dataset_dict

#     def _filter_function(example: Dict):
#         prompt = model.to_tokens(example['text'] + " Review Sentiment:", prepend_bos=False)
#         answer = torch.tensor([29071, 32725]).unsqueeze(0).unsqueeze(0).to(device) if example['label'] == 1 else torch.tensor([32725, 29071]).unsqueeze(0).unsqueeze(0).to(device)
#         logits, cache = model.run_with_cache(prompt)
#         logit_diff = get_logit_diff(logits, answer)
        
#         # Determine if the top answer (index 0) token is in top 10 logits
#         _, top_indices = logits.topk(10, dim=-1)  # Get indices of top 10 logits
#         top_answer_token = answer[0, 0, 0]  # Assuming answer is of shape (1, 1, 2) and the top answer token is at index 0
#         is_top_answer_in_top_10_logits = (top_indices == top_answer_token).any()
        
#         # Add a new field 'keep_example' to the example
#         example['keep_example'] = (logit_diff > 0.0) and is_top_answer_in_top_10_logits.item()
#         return example