import unittest
from unittest.mock import patch, Mock
import torch
import numpy as np
from typing import List
from transformer_lens import HookedTransformer
from utils.tokenwise_ablation import (
    find_positions,
    load_directions,
    get_random_directions,
    get_zeroed_dir_vector,
)


class TestUtils(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
