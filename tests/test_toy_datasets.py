from jaxtyping import Float, Int, Bool
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from typing import Tuple, Dict
import unittest
from summarization_utils.circuit_analysis import get_logit_diff
from summarization_utils.path_patching import act_patch, Node
from summarization_utils.toy_datasets import CounterfactualDataset


torch.set_grad_enabled(False)


TEST_TUPLES: List[Tuple[str, str, str, str]] = [
    (
        "Known for being the first to walk on the moon, Neil",
        " Armstrong",
        "Known for being the star of the movie Jazz Singer, Neil",
        " Diamond",
    ),
]


def get_position_dict(
    tokens: Float[Tensor, "batch seq_len"], model: HookedTransformer, sep=","
) -> Dict[str, List[List[int]]]:
    """
    Returns a dictionary of positions of clauses and separators
    e.g. {"clause_0": [[0, 1, 2]], "sep_0": [[3]], "clause_1": [[4, 5, 6]]}
    """
    sep_id = model.to_single_token(sep)
    batch_size, seq_len = tokens.shape
    full_pos_dict = dict()
    for b in range(batch_size):
        batch_pos_dict = {}
        sep_count = 0
        current_clause = []
        for s in range(seq_len):
            if tokens[b, s] == sep_id:
                batch_pos_dict[f"clause_{sep_count}"] = current_clause
                batch_pos_dict[f"sep_{sep_count}"] = [s]
                sep_count += 1
                current_clause = []
                continue
            else:
                current_clause.append(s)
        if current_clause:
            batch_pos_dict[f"clause_{sep_count}"] = current_clause
        for k, v in batch_pos_dict.items():
            if k in full_pos_dict:
                full_pos_dict[k].append(v)
            else:
                full_pos_dict[k] = [v]
    return full_pos_dict


class TestToyDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = HookedTransformer.from_pretrained("pythia-70m", device="cpu")
        return super().setUpClass()

    def test_compute_logit_diff(self):
        ds = CounterfactualDataset.from_tuples(TEST_TUPLES, self.model)
        base, cf = ds.compute_logit_diffs(vectorized=False)
        base_vec, cf_vec = ds.compute_logit_diffs(vectorized=True)
        self.assertTrue(torch.allclose(base, base_vec))
        self.assertTrue(torch.allclose(cf, cf_vec))

    def test_patch_position(self):
        sep = ","
        key = "sep_0"

        ds = CounterfactualDataset.from_tuples(TEST_TUPLES, self.model)
        pos_dict = get_position_dict(ds.prompt_tokens, model=ds.model, sep=sep)
        positions = pos_dict[key]
        results_pd = ds.patch_by_position_group(sep=sep)
        results = ds.patch_at_position(positions)
        self.assertTrue(np.isclose(results_pd[key], results.item()))

    def test_reconciliation(self):
        PROMPTS = [d[0] for d in TEST_TUPLES]
        CF_PROMPTS = [d[2] for d in TEST_TUPLES]
        prompt_tokens = self.model.to_tokens(PROMPTS, prepend_bos=True)
        cf_tokens = self.model.to_tokens(CF_PROMPTS, prepend_bos=True)
        comma_id = self.model.to_single_token(",")
        pos_dict = get_position_dict(prompt_tokens, self.model, sep=",")
        answer_tokens = torch.tensor(
            [
                (self.model.to_single_token(d[1]), self.model.to_single_token(d[3]))
                for d in TEST_TUPLES
            ],
            device=self.model.cfg.device,
        )
        mask = get_attention_mask(
            self.model.tokenizer, prompt_tokens, prepend_bos=False
        )
        cf_mask = get_attention_mask(self.model.tokenizer, cf_tokens, prepend_bos=False)
        assert prompt_tokens.shape == cf_tokens.shape
        self.model.reset_hooks()
        base_logits_by_pos = self.model(
            prompt_tokens,
            attention_mask=mask,
            prepend_bos=False,
            return_type="logits",
        )
        base_ldiff = get_logit_diff(
            base_logits_by_pos,
            answer_tokens=answer_tokens,
            mask=mask,
            per_prompt=True,
        )
        cf_logits, cf_cache = self.model.run_with_cache(
            cf_tokens, prepend_bos=False, return_type="logits"
        )
        assert isinstance(cf_logits, torch.Tensor)
        cf_ldiff = get_logit_diff(
            cf_logits,
            answer_tokens=answer_tokens,
            mask=cf_mask,
            per_prompt=True,
        )
        metric = lambda logits: (
            get_logit_diff(
                logits,
                answer_tokens=answer_tokens,
                mask=mask,
                per_prompt=True,
            )
            - base_ldiff
        ) / (cf_ldiff - base_ldiff)
        results_dict = dict()
        for pos_label, positions in pos_dict.items():
            positions = pos_dict[pos_label]
            # pos_tensor = torch.zeros_like(prompt_tokens, dtype=torch.bool)
            nodes = [
                Node(node_name="resid_pre", layer=layer, seq_pos=positions)
                for layer in range(self.model.cfg.n_layers)
            ]
            pos_results = (
                act_patch(
                    self.model, prompt_tokens, nodes, metric, new_cache=cf_cache, verbose=True  # type: ignore
                )
                .cpu()
                .numpy()
            )
            results_dict[pos_label] = pos_results
        results_pd_manual = pd.DataFrame(results_dict)
        ds = CounterfactualDataset.from_tuples(TEST_TUPLES, self.model)
        results_pd_new = ds.patch_by_position_group(sep=",")
        self.assertTrue(np.allclose(results_pd_manual, results_pd_new))


if __name__ == "__main__":
    unittest.main()
