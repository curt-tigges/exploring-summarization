import unittest
from unittest.mock import patch, Mock
import torch
import numpy as np
from typing import List
from torch.utils.data import Dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from utils.datasets import ExperimentDataLoader
from utils.tokenwise_ablation import (
    find_positions,
    load_directions,
    get_random_directions,
    get_zeroed_dir_vector,
    get_layerwise_token_mean_activations,
    zero_attention_pos_hook,
    compute_ablation_modified_metric,
    compute_zeroed_attn_modified_loss,
    mask_loss_at_next_positions,
)


class HookPoint:
    def __init__(self, name=None):
        self.name = name


# More Mock objects and constants for testing
class MockBatch:
    def __init__(self, tokens):
        self.tokens = tokens


class MockDataLoader(ExperimentDataLoader):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    @property
    def name(self):
        return "MockDataLoader"


class DummyDataset(Dataset):
    def __init__(self, seed: int = 0):
        torch.manual_seed(seed)
        self.tokens = torch.tensor([[14, 9, 8], [0, 13, 9], [3, 11, 11], [6, 9, 13]])
        self.attention_mask = torch.ones((4, 3), dtype=torch.long)
        self.positions = torch.ones((4, 3), dtype=torch.long)
        self.answers = torch.randint(0, 15, (4, 2))
        self.has_token = torch.ones((4,), dtype=torch.long)
        self.column_names = [
            "tokens",
            "attention_mask",
            "positions",
            "answers",
            "has_token",
        ]
        self.builder_name = "dummy"
        self.split = "dummy"

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "tokens": self.tokens[idx],
            "attention_mask": self.attention_mask[idx],
            "positions": self.positions[idx],
            "has_token": self.has_token[idx],
            "answers": self.answers[idx],
        }

    def set_format(self, type, columns):
        pass


class TestTokenwise(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = HookedTransformer.from_pretrained("gelu-1l")
        torch.manual_seed(0)

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
        torch.manual_seed(0)
        result = get_random_directions(self.model)
        self.assertEqual(result.shape, (1, 512))

    def test_get_zeroed_dir_vector(self):
        torch.manual_seed(0)
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
                    [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
                    dtype=torch.float32,
                ),
            },
        )
        model = HookedTransformer.from_pretrained("gelu-1l")
        model.cfg.d_model = 1

        data_loader = MockDataLoader(
            [
                {"tokens": torch.tensor([[1, 2, 3, 4], [5, 6, 2, 7]])},
            ]
        )
        result = get_layerwise_token_mean_activations(model, data_loader, token_id=2)
        expected_result = torch.tensor([[4.5]])
        torch.testing.assert_close(result, expected_result)

    def test_zero_attention_pos_hook(self):
        pattern = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
        hook = self.model.hook_dict[get_act_name("pattern", 0)]
        result = zero_attention_pos_hook(pattern, hook, pos_by_batch=[[1]], head_idx=1)
        expected_result = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 0]]]])
        torch.testing.assert_close(result, expected_result)

    def test_compute_ablation_modified_logit_diff(self):
        dataset = DummyDataset()
        data_loader = ExperimentDataLoader(dataset, batch_size=2)
        layers_to_ablate = [
            0,
        ]
        heads_to_freeze = [
            (0, 0),
        ]
        cached_means = torch.zeros((1, 512))

        metrics = compute_ablation_modified_metric(
            self.model,
            data_loader,
            layers_to_ablate=layers_to_ablate,
            heads_to_freeze=list(heads_to_freeze),
            cached_means=cached_means,
        )

        self.assertEqual(metrics.shape[1], len(data_loader))

    def test_compute_zeroed_attn_modified_loss(self):
        dataset = DummyDataset()
        data_loader = ExperimentDataLoader(dataset, batch_size=1)
        heads_to_ablate = [
            (0, 0),
        ]
        token_ids = [13]

        loss_diff = compute_zeroed_attn_modified_loss(
            self.model, data_loader, list(heads_to_ablate), token_ids
        )

        torch.testing.assert_close(
            loss_diff,
            torch.tensor(
                [[0.0000, 0.0000], [0.0000, 0.0593], [0.0000, 0.0000], [0.0000, 0.0000]]
            ),
            atol=1e-4,
            rtol=1e-4,
        )


class TestMaskLossAtNextPositions(unittest.TestCase):
    def test_mask_loss(self):
        # Setup tensors for loss, positions, and attention_mask
        loss = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        positions = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

        # Expected output after mask_loss_at_next_positions is applied
        expected_output = torch.tensor([[0.1, 0.0, 0.3, 0.0], [0.5, 0.6, 0.0, 0.0]])

        # Apply mask_loss_at_next_positions
        result = mask_loss_at_next_positions(loss, positions, attention_mask)

        # Assert the result is as expected
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-4, rtol=1e-4))

    def test_attention_mask(self):
        # Setup tensors for loss, positions, and attention_mask
        loss = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        positions = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
        attention_mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])

        # Expected output after mask_loss_at_next_positions is applied
        expected_output = torch.tensor([[0.1, 0.0, 0.3, 0.0], [0.0, 0.6, 0.0, 0.8]])

        # Apply mask_loss_at_next_positions
        result = mask_loss_at_next_positions(loss, positions, attention_mask)

        # Assert the result is as expected
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-4, rtol=1e-4))

    def test_no_positions_masked(self):
        # Test when no positions should be masked
        loss = torch.randn(2, 4)
        positions = torch.zeros(2, 4)
        attention_mask = torch.ones(2, 4)

        # The result should be identical to loss as no positions are masked
        expected_output = loss.clone()

        # Apply mask_loss_at_next_positions
        result = mask_loss_at_next_positions(loss, positions, attention_mask)

        # Assert the result is as expected
        self.assertTrue(torch.equal(result, expected_output))


if __name__ == "__main__":
    unittest.main()
