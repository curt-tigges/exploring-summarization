from typing import List, Tuple
import numpy as np
import torch
from transformer_lens import HookedTransformer
import unittest
from summarization_utils.toy_datasets import CounterfactualDataset, get_position_dict


TEST_TUPLES: List[Tuple[str, str, str, str]] = [
    (
        "Known for being the first to walk on the moon, Neil",
        " Armstrong",
        "Known for being the star of the movie Jazz Singer, Neil",
        " Diamond",
    ),
]


class TestToyDatasets(unittest.TestCase):
    def test_compute_logit_diff(self):
        model = HookedTransformer.from_pretrained("attn-only-1l", device="cpu")
        ds = CounterfactualDataset.from_tuples(TEST_TUPLES, model)
        base, cf = ds.compute_logit_diffs(vectorized=False)
        base_vec, cf_vec = ds.compute_logit_diffs(vectorized=True)
        self.assertTrue(torch.allclose(base, base_vec))
        self.assertTrue(torch.allclose(cf, cf_vec))

    def test_patch_position(self):
        sep = ","
        key = "sep_0"
        model = HookedTransformer.from_pretrained("pythia-70m", device="cpu")
        ds = CounterfactualDataset.from_tuples(TEST_TUPLES, model)
        pos_dict = get_position_dict(ds.prompt_tokens, model=ds.model, sep=sep)
        positions = pos_dict[key]
        results_pd = ds.patch_by_position_group(sep=sep)
        results = ds.patch_at_position(positions)
        self.assertTrue(np.isclose(results_pd[key], results.item()))


if __name__ == "__main__":
    unittest.main()
