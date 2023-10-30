import unittest
from unittest.mock import patch, Mock
import torch
import numpy as np
from typing import List
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from utils.tokenwise_ablation import (
    find_positions,
    load_directions,
    get_random_directions,
    get_zeroed_dir_vector,
    get_layerwise_token_mean_activations,
    zero_attention_pos_hook,
)


class HookPoint:
    def __init__(self, name=None):
        self.name = name


# More Mock objects and constants for testing
class MockBatch:
    def __init__(self, tokens):
        self.tokens = tokens


class MockDataLoader(DataLoader):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class TestTokenwise(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = HookedTransformer.from_pretrained("gelu-1l")

    def test_find_positions(self):
        tensor = torch.tensor([[1, 2, 3, 13, 5], [13, 7, 13, 9, 10]])
        result = find_positions(tensor, [13])
        self.assertEqual(result, [[3], [0, 2]])

    @patch("numpy.load")
    def test_load_directions(self, mock_load):
        # Mocking numpy.load to return a predefined array
        mock_load.return_value = np.array([1, 2, 3])
        result = load_directions(self.model)
        expected_result = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(result, expected_result)

    def test_get_random_directions(self):
        result = get_random_directions(self.model)
        self.assertEqual(result.shape, (1, 512))

    def test_get_zeroed_dir_vector(self):
        result = get_zeroed_dir_vector(self.model)
        expected_result = torch.zeros((1, 512))
        torch.testing.assert_close(result, expected_result)

    @patch.object(HookedTransformer, "run_with_cache")
    def test_get_layerwise_token_mean_activations(self, mock_run_with_cache):
        # Mocking the return value for `model.run_with_cache`
        mock_run_with_cache.return_value = (
            None,
            {
                "blocks.0.hook_resid_post": torch.tensor(
                    [[[1], [2], [3], [4]], [[5], [6], [7], [8]]]
                ),
            },
        )
        self.model.cfg.d_model = 1

        data_loader = MockDataLoader(
            [
                {"tokens": torch.tensor([[1, 2, 3, 4], [5, 6, 2, 7]])},
            ]
        )
        result = get_layerwise_token_mean_activations(
            self.model, data_loader, token_id=2
        )
        expected_result = torch.tensor([[4.5]])
        torch.testing.assert_close(result, expected_result)

    def test_zero_attention_pos_hook(self):
        pattern = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
        hook = self.model.hook_dict[get_act_name("pattern", 0)]
        result = zero_attention_pos_hook(pattern, hook, pos_by_batch=[[1]], head_idx=1)
        expected_result = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 0]]]])
        torch.testing.assert_close(result, expected_result)


if __name__ == "__main__":
    unittest.main()
