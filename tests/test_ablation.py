import unittest
import torch
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache
from utils.ablation import (
    handle_position,
    resample_cache_component,
    mean_over_cache_component,
    zero_cache_component,
    freeze_layer_pos_hook,
    convert_to_tensors,
)


class TestAblation(unittest.TestCase):
    def setUp(self):
        # Create mock data for testing
        # The shape of the tensor is (batch_size, seq_len, hidden_dim)
        # batch_size = 2, seq_len = 1, hidden_dim = 3
        self.mock_tensor = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
        self.mock_model = None
        self.mock_activation_cache = ActivationCache(
            {
                "blocks.0.hook_resid_post": torch.tensor(
                    [[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]
                ),
            },
            model=self.mock_model,
        )

    def test_handle_position(self):
        pos = [0, 1]
        component = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = handle_position(pos, component)
        expected = torch.tensor([0, 1])
        self.assertTrue(torch.equal(result, expected))

    def test_resample_cache_component(self):
        result = resample_cache_component(self.mock_tensor)
        # The output should have the same shape as the input
        self.assertEqual(result.shape, self.mock_tensor.shape)

    def test_mean_over_cache_component(self):
        result = mean_over_cache_component(self.mock_tensor)
        expected = torch.tensor([[[2.5, 3.5, 4.5]], [[2.5, 3.5, 4.5]]])
        self.assertTrue(torch.equal(result, expected))

    def test_zero_cache_component(self):
        result = zero_cache_component(self.mock_tensor)
        expected = torch.zeros_like(self.mock_tensor)
        self.assertTrue(torch.equal(result, expected))

    # ... [add more unit tests for other functions] ...

    def test_freeze_layer_pos_hook(self):
        mock_hook = HookPoint()
        mock_hook.name = "blocks.0.hook_resid_post"
        result = freeze_layer_pos_hook(
            self.mock_tensor, mock_hook, self.mock_activation_cache
        )
        # Check if the tensor is correctly updated from the cache
        expected = self.mock_activation_cache["blocks.0.hook_resid_post"]
        self.assertTrue(torch.equal(result, expected))

    def test_convert_to_tensors(self):
        dataset = [{"tokens": [1, 2]}, {"tokens": [3, 4]}]
        result = convert_to_tensors(dataset)
        # Check if the tensor values are the same as the input dataset
        self.assertTrue(torch.equal(result["tokens"][0], torch.tensor([1, 2])))
        self.assertTrue(torch.equal(result["tokens"][1], torch.tensor([3, 4])))


if __name__ == "__main__":
    unittest.main()
